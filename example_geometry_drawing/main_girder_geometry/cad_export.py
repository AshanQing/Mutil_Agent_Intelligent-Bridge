from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

from .geometry_config import get_cad_config, get_layer_style


class CadExportError(ValueError):
    """Raised when a geometry document cannot be exported to a CAD script."""


CAD_CFG = get_cad_config()
DEFAULT_LAYOUT_GAP_MM = CAD_CFG.layout_gap_mm
COMBINED_LAYOUT_VIEW = CAD_CFG.combined_view_name
COMBINED_LAYOUT_SECTION = CAD_CFG.combined_section_id


def load_geometry_document(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a UTF-8 geometry JSON document for SCR export."""
    source_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CadExportError(f"几何JSON读取失败：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise CadExportError(f"几何JSON格式错误：{exc}") from exc
    if not isinstance(document, dict):
        raise CadExportError("几何JSON顶层必须是对象。")
    return document


def export_geometry_to_cad(
    document_or_path: Union[Dict[str, Any], str, Path],
    output_dir: Union[str, Path],
    *,
    section_id: str = "ALL",
    view_name: str = "cross_section",
) -> List[Path]:
    """Write one AutoCAD SCR file for every selected geometry section."""
    document = (
        load_geometry_document(document_or_path)
        if isinstance(document_or_path, (str, Path))
        else document_or_path
    )
    sections = _select_sections(document, section_id, view_name)

    rendered: List[Tuple[str, str]] = []
    filenames = set()
    for section in sections:
        current_section_id = str(section["id"])
        filename = f"{_safe_filename(current_section_id)}.scr"
        if filename in filenames:
            raise CadExportError(f"图形ID生成了重复文件名：{filename}")
        filenames.add(filename)
        rendered.append(
            (
                filename,
                render_section_script(
                    document,
                    current_section_id,
                    view_name=view_name,
                ),
            )
        )

    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []
    for filename, content in rendered:
        target = target_dir / filename
        target.write_text(content, encoding="utf-8")
        output_paths.append(target)
    return output_paths


def build_combined_layout(
    documents: Sequence[Dict[str, Any]],
    *,
    gap_mm: float = DEFAULT_LAYOUT_GAP_MM,
) -> Dict[str, Any]:
    """Translate elevation, plan and sections onto one 1:1 drawing canvas."""
    gap = _number(gap_mm, "gap_mm")
    if gap < 0.0:
        raise CadExportError("gap_mm不能小于0。")
    source_documents = list(documents)
    if not source_documents or not all(
        isinstance(document, dict) for document in source_documents
    ):
        raise CadExportError("合图至少需要一个几何文档。")

    units = str(source_documents[0].get("units", "unitless"))
    angle_unit = str(source_documents[0].get("angle_unit", "degree"))
    for document in source_documents[1:]:
        if str(document.get("units", "unitless")) != units:
            raise CadExportError("合图中的几何文档必须使用相同单位。")
        if str(document.get("angle_unit", "degree")) != angle_unit:
            raise CadExportError("合图中的几何文档必须使用相同角度单位。")

    sections_by_view: Dict[str, List[Dict[str, Any]]] = {
        "cross_section": [],
        "plan": [],
        "elevation": [],
    }
    for document in source_documents:
        views = document.get("views")
        if not isinstance(views, dict):
            raise CadExportError("合图中的几何文档缺少views对象。")
        for view_name in sections_by_view:
            if view_name in views:
                sections_by_view[view_name].extend(
                    _select_sections(document, "ALL", view_name)
                )
    for view_name, sections in sections_by_view.items():
        if not sections:
            raise CadExportError(f"整套CAD缺少视图：{view_name}")

    rows = {
        view_name: _pack_layout_row(sections, gap)
        for view_name, sections in sections_by_view.items()
    }
    canvas_width = max(row[1] for row in rows.values())
    cross_height = rows["cross_section"][2]
    plan_height = rows["plan"][2]
    row_bottom = {
        "cross_section": 0.0,
        "plan": cross_height + gap,
        "elevation": cross_height + gap + plan_height + gap,
    }

    combined_entities: Dict[str, List[Dict[str, Any]]] = {
        "lines": [],
        "arcs": [],
        "regions": [],
    }
    origin_x = CAD_CFG.layout_origin_x
    origin_y = CAD_CFG.layout_origin_y

    for view_name in ("elevation", "plan", "cross_section"):
        packed_sections, row_width, _ = rows[view_name]
        centered_x = (canvas_width - row_width) / 2.0
        for index, (section, local_x, bounds) in enumerate(packed_sections, 1):
            min_x, min_y, _, _ = bounds
            translated = _translate_layout_section(
                section,
                origin_x + centered_x + local_x - min_x,
                origin_y + row_bottom[view_name] - min_y,
                prefix=f"layout_{view_name}_{index:03d}_",
            )
            for kind in combined_entities:
                combined_entities[kind].extend(translated[kind])

    ox, oy = _format_number(origin_x), _format_number(origin_y)
    return {
        "schema_version": str(source_documents[0].get("schema_version", "2.0.0")),
        "units": units,
        "angle_unit": angle_unit,
        "source": {"type": "combined_geometry_layout"},
        "views": {
            COMBINED_LAYOUT_VIEW: {
                "coordinate_system": {
                    "origin": "({}, {})".format(ox, oy),
                    "x_axis": "+X 向右",
                    "y_axis": "+Y 向上",
                },
                "sections": [
                    {
                        "id": COMBINED_LAYOUT_SECTION,
                        "entities": combined_entities,
                    }
                ],
            }
        },
    }


def render_section_script(
    document: Dict[str, Any],
    section_id: str,
    *,
    view_name: str = "cross_section",
) -> str:
    """Render one geometry section as an AutoCAD coordinate command script.

    Every layer property configured in ``geometry_config.json`` (colour, lineweight,
    linetype) is written into the SCR so that the resulting drawing faithfully
    reflects the configuration.

    Colour is applied via the ``CECOLOR`` system variable (rather than ``-LAYER
    Color``) because ``CECOLOR`` accepts ``R,G,B`` TrueColor format reliably
    across all AutoCAD versions.  Lineweight and linetype are set as layer
    properties via ``-LAYER LW`` / ``-LAYER LT``.
    """
    section = _select_section(document, section_id, view_name)
    lines, arcs = _section_entities(section)
    entities: List[Tuple[str, Dict[str, Any]]] = [
        *(("line", entity) for entity in lines),
        *(("arc", entity) for entity in arcs),
    ]
    layers = _ordered_layers(entity for _, entity in entities)
    style_by_layer = {layer: get_layer_style(layer) for layer in layers}
    commands: List[str] = []

    # ── 1. Pre-load non-Continuous linetypes ────────────────────────────
    preloaded: set[str] = set()
    for layer in layers:
        lt = style_by_layer[layer].linetype
        if lt.lower() not in ("continuous", "bylayer") and lt not in preloaded:
            commands.extend(["_.-LINETYPE", "_L", lt, "", ""])
            preloaded.add(lt)

    # ── 2. Create layers & set lineweight + linetype ────────────────────
    # Colour is NOT set via -LAYER because its TrueColor sub-command is
    # version-dependent and causes "invalid colour" prompts.  Entity colour
    # is applied via CECOLOR in phase 3 instead.
    for layer in layers:
        style = style_by_layer[layer]
        commands.extend(["_.-LAYER", "_M", layer, ""])
        # Lineweight (mm)
        lw_str = _lineweight_for_scr(style.lineweight)
        if lw_str:
            commands.extend(["_.-LAYER", "_LW", lw_str, layer, ""])
        # Linetype
        if style.linetype.lower() not in ("continuous", "bylayer"):
            commands.extend(["_.-LAYER", "_LT", style.linetype, layer, ""])

    # ── 3. Draw entities — colour via CECOLOR for reliable TrueColor ────
    current_layer: str | None = None
    for kind, entity in entities:
        layer = _layer_name(entity)
        if layer != current_layer:
            commands.extend(["_.-LAYER", "_S", layer, ""])
            # CECOLOR as belt-and-suspenders: even if -LAYER C failed
            # silently, the entity still gets the right colour.
            r, g, b = _hex_to_rgb(style_by_layer[layer].color)
            commands.extend(["_.CECOLOR", f"{r},{g},{b}"])
            current_layer = layer
        if kind == "line":
            commands.extend(
                [
                    "_.LINE",
                    _format_point(_point(entity, "start")),
                    _format_point(_point(entity, "end")),
                    "",
                ]
            )
            continue

        center = _point(entity, "center")
        radius = _positive_number(entity.get("radius"), "arc.radius")
        start_angle = _number(
            entity.get("start_angle_deg"), "arc.start_angle_deg"
        )
        end_angle = _number(entity.get("end_angle_deg"), "arc.end_angle_deg")
        if bool(entity.get("clockwise", False)):
            start_angle, end_angle = end_angle, start_angle
        commands.extend(
            [
                "_.ARC",
                "_C",
                _format_point(center),
                _format_point(_arc_point(center, radius, start_angle)),
                _format_point(_arc_point(center, radius, end_angle)),
            ]
        )

    commands.extend(["_.ZOOM", "_E"])
    return "\n".join(commands) + "\n"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert ``"#172033"`` → ``(23, 32, 51)`` for AutoCAD TrueColor."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _lineweight_for_scr(lineweight: str) -> str:
    """Return a lineweight string suitable for ``_.-LAYER _LW``, or ``""``."""
    try:
        value = float(lineweight)
    except (ValueError, TypeError):
        return ""
    if value <= 0.0:
        return ""
    # AutoCAD accepts mm values with up to two decimal places
    return f"{value:.2f}"


def _select_section(
    document: Dict[str, Any], section_id: str, view_name: str
) -> Dict[str, Any]:
    if not isinstance(document, dict):
        raise CadExportError("几何JSON顶层必须是对象。")
    views = document.get("views")
    if not isinstance(views, dict) or view_name not in views:
        raise CadExportError(f"几何JSON不存在视图：{view_name}")
    view = views[view_name]
    if not isinstance(view, dict) or not isinstance(view.get("sections"), list):
        raise CadExportError(f"视图{view_name}缺少sections数组。")
    for section in view["sections"]:
        if isinstance(section, dict) and str(section.get("id")) == section_id:
            return section
    raise CadExportError(f"视图{view_name}不存在图形：{section_id}")


def _select_sections(
    document: Dict[str, Any], section_id: str, view_name: str
) -> List[Dict[str, Any]]:
    if section_id != "ALL":
        return [_select_section(document, section_id, view_name)]
    if not isinstance(document, dict):
        raise CadExportError("几何JSON顶层必须是对象。")
    views = document.get("views")
    if not isinstance(views, dict) or view_name not in views:
        raise CadExportError(f"几何JSON不存在视图：{view_name}")
    view = views[view_name]
    if not isinstance(view, dict) or not isinstance(view.get("sections"), list):
        raise CadExportError(f"视图{view_name}缺少sections数组。")
    sections = view["sections"]
    if not sections:
        raise CadExportError(f"视图{view_name}没有可导出的图形。")
    if not all(
        isinstance(section, dict) and section.get("id") for section in sections
    ):
        raise CadExportError("sections中的每个图形都必须具有非空id。")
    return sections


def _section_entities(
    section: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entities = section.get("entities")
    if not isinstance(entities, dict):
        raise CadExportError(f"图形{section.get('id', '')}缺少entities对象。")
    lines = entities.get("lines", [])
    arcs = entities.get("arcs", [])
    if not isinstance(lines, list) or not isinstance(arcs, list):
        raise CadExportError("entities.lines和entities.arcs必须是数组。")
    if not all(isinstance(item, dict) for item in [*lines, *arcs]):
        raise CadExportError("line和arc图元必须是对象。")
    if not lines and not arcs:
        raise CadExportError(f"图形{section.get('id', '')}没有可导出的直线或圆弧。")
    return lines, arcs


def _section_regions(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities = section.get("entities")
    if not isinstance(entities, dict):
        raise CadExportError(f"图形{section.get('id', '')}缺少entities对象。")
    regions = entities.get("regions", [])
    if not isinstance(regions, list) or not all(
        isinstance(item, dict) for item in regions
    ):
        raise CadExportError("entities.regions必须是对象数组。")
    return regions


def _pack_layout_row(
    sections: Sequence[Dict[str, Any]], gap: float
) -> Tuple[
    List[Tuple[Dict[str, Any], float, Tuple[float, float, float, float]]],
    float,
    float,
]:
    packed = []
    current_x = 0.0
    row_height = 0.0
    for section in sections:
        bounds = _section_bounds(section)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        packed.append((section, current_x, bounds))
        current_x += width + gap
        row_height = max(row_height, height)
    row_width = current_x - gap if packed else 0.0
    return packed, row_width, row_height


def _section_bounds(section: Dict[str, Any]) -> Tuple[float, float, float, float]:
    lines, arcs = _section_entities(section)
    x_values: List[float] = []
    y_values: List[float] = []
    for line in lines:
        for point_name in ("start", "end"):
            x_value, y_value = _point(line, point_name)
            x_values.append(x_value)
            y_values.append(y_value)
    for arc in arcs:
        center_x, center_y = _point(arc, "center")
        radius = _positive_number(arc.get("radius"), "arc.radius")
        x_values.extend([center_x - radius, center_x + radius])
        y_values.extend([center_y - radius, center_y + radius])
    return min(x_values), min(y_values), max(x_values), max(y_values)


def _translate_layout_section(
    section: Dict[str, Any],
    dx: float,
    dy: float,
    *,
    prefix: str,
) -> Dict[str, List[Dict[str, Any]]]:
    lines, arcs = _section_entities(section)
    regions = _section_regions(section)
    _entity_lookup(lines, arcs)
    translated: Dict[str, List[Dict[str, Any]]] = {
        "lines": [],
        "arcs": [],
        "regions": [],
    }
    id_mapping: Dict[str, str] = {}

    for kind, entities in (("lines", lines), ("arcs", arcs)):
        for entity in entities:
            item = copy.deepcopy(entity)
            old_id = str(item["id"])
            new_id = f"{prefix}{old_id}"
            id_mapping[old_id] = new_id
            item["id"] = new_id
            point_names = ("start", "end") if kind == "lines" else ("center",)
            for point_name in point_names:
                x_value, y_value = _point(item, point_name)
                item[point_name] = {
                    "x": round(x_value + dx, 6),
                    "y": round(y_value + dy, 6),
                }
            translated[kind].append(item)

    for region in regions:
        item = copy.deepcopy(region)
        old_region_id = str(item.get("id", "")).strip()
        if not old_region_id:
            raise CadExportError("region缺少非空id。")
        item["id"] = f"{prefix}{old_region_id}"
        loops = [item.get("outer"), *item.get("holes", [])]
        for loop in loops:
            if not isinstance(loop, list):
                raise CadExportError(f"region {old_region_id}的边界环必须是数组。")
            for reference in loop:
                if not isinstance(reference, dict):
                    raise CadExportError("region边界引用必须是对象。")
                old_entity_id = str(reference.get("entity_id", "")).strip()
                if old_entity_id not in id_mapping:
                    raise CadExportError(
                        f"region引用了不存在的图元：{old_entity_id or '<empty>'}"
                    )
                reference["entity_id"] = id_mapping[old_entity_id]
        translated["regions"].append(item)
    return translated


def _entity_lookup(
    lines: Sequence[Dict[str, Any]], arcs: Sequence[Dict[str, Any]]
) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    lookup: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for kind, entities in (("line", lines), ("arc", arcs)):
        for entity in entities:
            identifier = str(entity.get("id", "")).strip()
            if not identifier:
                raise CadExportError(f"{kind}图元缺少非空id。")
            if identifier in lookup:
                raise CadExportError(f"图元id重复：{identifier}")
            lookup[identifier] = (kind, entity)
    return lookup


def _ordered_layers(entities: Iterable[Dict[str, Any]]) -> List[str]:
    layers: List[str] = []
    for entity in entities:
        layer = _layer_name(entity)
        if layer not in layers:
            layers.append(layer)
    return layers or [get_cad_config().fallback_layer]


def _layer_name(entity: Dict[str, Any]) -> str:
    fallback = get_cad_config().fallback_layer
    layer = str(entity.get("layer", fallback)).strip() or fallback
    if any(character in layer for character in "\r\n"):
        raise CadExportError("图层名不能包含换行符。")
    return layer


def _point(entity: Dict[str, Any], field_name: str) -> Tuple[float, float]:
    value = entity.get(field_name)
    if not isinstance(value, dict):
        raise CadExportError(f"{field_name}必须是坐标对象。")
    return (
        _number(value.get("x"), f"{field_name}.x"),
        _number(value.get("y"), f"{field_name}.y"),
    )


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise CadExportError(f"{field_name}必须是有限数值。")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CadExportError(f"{field_name}必须是有限数值。") from exc
    if not math.isfinite(result):
        raise CadExportError(f"{field_name}必须是有限数值。")
    return result


def _positive_number(value: Any, field_name: str) -> float:
    result = _number(value, field_name)
    if result <= 0.0:
        raise CadExportError(f"{field_name}必须大于0。")
    return result


def _safe_filename(section_id: str) -> str:
    characters: List[str] = []
    previous_was_separator = False
    for character in section_id.strip().lower():
        if character.isalnum() or character in {"_", "-"}:
            characters.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            characters.append("-")
            previous_was_separator = True
    filename = "".join(characters).strip("-_")
    if not filename:
        raise CadExportError(f"图形ID不能生成有效文件名：{section_id}")
    return filename


def _arc_point(
    center: Tuple[float, float], radius: float, angle_deg: float
) -> Tuple[float, float]:
    angle_rad = math.radians(angle_deg)
    return (
        center[0] + radius * math.cos(angle_rad),
        center[1] + radius * math.sin(angle_rad),
    )


def _format_point(point: Sequence[float]) -> str:
    return f"{_format_number(point[0])},{_format_number(point[1])}"


def _format_number(value: float) -> str:
    normalized = 0.0 if abs(float(value)) < 1e-10 else float(value)
    return format(normalized, get_cad_config().format_precision)
