from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from .geometry_config import get_layer_style


# ── SVG 预览硬编码参数（仅效果展示，不开放配置） ──────────────────────────

_SVG = {
    "width_px": 1400,
    "height_px": 820,
    "arc_step_deg": 4.0,
    "padding": {"left": 70, "right": 70, "top": 90, "bottom": 70},
    "background_color": "#f7f9fc",
    "title": {"font_family": "Microsoft YaHei, sans-serif", "font_size": 20, "font_weight": "600", "color": "#E7E9F0"},
    "subtitle": {"font_family": "Microsoft YaHei, sans-serif", "font_size": 13, "font_weight": "normal", "color": "#5f6b7a"},
    "legend_text": {"font_family": "Microsoft YaHei, sans-serif", "font_size": 12, "font_weight": "normal", "color": "#364152"},
    "mono_text": {"font_family": "Consolas, monospace", "font_size": 12, "font_weight": "normal", "color": "#5f6b7a"},
    "origin": {"dot_radius": 4, "dot_color": "#21885b", "label_color": "#21885b"},
    "axis": {"stroke": "#35a06f", "stroke_width": "1.2", "dash": "8 6"},
    "hatch": {"tile_width": 14, "tile_height": 14, "rotation": 35, "background": "#dbe4ee", "line_color": "#8aa0b8", "line_width": "2"},
    "view_labels": {"cross_section": "横断面", "elevation": "立面", "plan": "平面"},
}


class GeometryPreviewError(ValueError):
    pass


def load_geometry_json(path: str | Path) -> dict[str, Any]:
    source_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeometryPreviewError(f"几何JSON读取失败：{exc}") from exc
    if not isinstance(document, dict):
        raise GeometryPreviewError("几何JSON顶层必须是对象。")
    return document


def write_geometry_preview(
    document_or_path: dict[str, Any] | str | Path,
    output_path: str | Path,
    *,
    section_id: str | None = None,
    view_name: str = "cross_section",
) -> Path:
    document = (
        load_geometry_json(document_or_path)
        if isinstance(document_or_path, (str, Path))
        else document_or_path
    )
    svg = render_geometry_svg(
        document,
        section_id=section_id,
        view_name=view_name,
    )
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg, encoding="utf-8")
    return target


def render_geometry_svg(
    document: dict[str, Any],
    *,
    section_id: str | None = None,
    view_name: str = "cross_section",
    width_px: int | None = None,
    height_px: int | None = None,
) -> str:
    wp = width_px or _SVG["width_px"]
    hp = height_px or _SVG["height_px"]

    section = _select_section(document, section_id, view_name=view_name)
    entities = section.get("entities")
    if not isinstance(entities, dict):
        raise GeometryPreviewError("断面缺少entities对象。")
    lines = entities.get("lines", [])
    arcs = entities.get("arcs", [])
    regions = entities.get("regions", [])
    if not isinstance(lines, list) or not isinstance(arcs, list) or not isinstance(regions, list):
        raise GeometryPreviewError("lines、arcs和regions必须是数组。")

    entity_index: dict[str, dict[str, Any]] = {}
    all_points: list[tuple[float, float]] = []
    for line in lines:
        entity_index[str(line["id"])] = {"kind": "line", **line}
        all_points.extend((_point(line["start"]), _point(line["end"])))
    for arc in arcs:
        entity_index[str(arc["id"])] = {"kind": "arc", **arc}
        all_points.extend(_arc_points(arc))
    if not all_points:
        raise GeometryPreviewError("断面没有可绘制的line或arc。")

    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    drawing_width = max(max_x - min_x, 1.0)
    drawing_height = max(max_y - min_y, 1.0)
    pad = _SVG["padding"]
    left_padding = pad["left"]
    right_padding = pad["right"]
    top_padding = pad["top"]
    bottom_padding = pad["bottom"]
    scale = min(
        (wp - left_padding - right_padding) / drawing_width,
        (hp - top_padding - bottom_padding) / drawing_height,
    )

    def screen(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (
            left_padding + (x - min_x) * scale,
            top_padding + (max_y - y) * scale,
        )

    section_name = escape(str(section.get("id", section_id or view_name)))
    view_label = _SVG["view_labels"].get(view_name, view_name)
    has_box_void = any(
        str(entity.get("layer", "")) == "XS-VOID"
        for entity in [*lines, *arcs]
    )
    if view_name in {"elevation", "plan"}:
        void_label = "箱室及人洞空腔"
    else:
        void_label = "箱室孔洞" if has_box_void else "人洞孔洞"
    source_name = escape(Path(str(document.get("source", {}).get("path", ""))).name)

    hatch = _SVG["hatch"]
    title_style = _SVG["title"]
    sub_style = _SVG["subtitle"]
    legend_style = _SVG["legend_text"]
    mono_style = _SVG["mono_text"]
    origin_cfg = _SVG["origin"]
    axis_cfg = _SVG["axis"]

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{wp}" '
            f'height="{hp}" viewBox="0 0 {wp} {hp}" '
            'role="img" aria-labelledby="title desc">'
        ),
        f"<title id=\"title\">{section_name} 几何图元预览</title>",
        (
            f'<desc id="desc">由{len(lines)}条直线、{len(arcs)}条圆弧和'
            f"{len(regions)}个区域拼合的{view_label}预览。</desc>"
        ),
        "<defs>",
        (
            f'<pattern id="concrete-hatch" width="{hatch["tile_width"]}" '
            f'height="{hatch["tile_height"]}" patternUnits="userSpaceOnUse" '
            f'patternTransform="rotate({hatch["rotation"]})">'
            f'<rect width="{hatch["tile_width"]}" height="{hatch["tile_height"]}" '
            f'fill="{hatch["background"]}"/>'
            f'<line x1="0" y1="0" x2="0" y2="{hatch["tile_height"]}" '
            f'stroke="{hatch["line_color"]}" stroke-width="{hatch["line_width"]}"/></pattern>'
        ),
        "</defs>",
        f'<rect width="100%" height="100%" fill="{_SVG["background_color"]}"/>',
        (
            f'<text x="{wp / 2:.3f}" y="34" text-anchor="middle" '
            f'font-family="{title_style["font_family"]}" font-size="{title_style["font_size"]}" '
            f'font-weight="{title_style["font_weight"]}" fill="{title_style["color"]}">'
            f"{section_name} {view_label}图元预览</text>"
        ),
        (
            f'<text x="{wp / 2:.3f}" y="58" text-anchor="middle" '
            f'font-family="{sub_style["font_family"]}" font-size="{sub_style["font_size"]}" '
            f'fill="{sub_style["color"]}">来源：{source_name or "geometry JSON"}；单位：'
            f'{escape(str(document.get("units", "mm")))}</text>'
        ),
    ]

    if min_x <= 0.0 <= max_x:
        x0, _ = screen((0.0, 0.0))
        parts.append(
            f'<line x1="{x0:.3f}" y1="{top_padding - 10:.3f}" '
            f'x2="{x0:.3f}" y2="{hp - bottom_padding + 10:.3f}" '
            f'stroke="{axis_cfg["stroke"]}" stroke-width="{axis_cfg["stroke_width"]}" '
            f'stroke-dasharray="{axis_cfg["dash"]}"/>'
        )
    if min_y <= 0.0 <= max_y:
        _, y0 = screen((0.0, 0.0))
        parts.append(
            f'<line x1="{left_padding - 10:.3f}" y1="{y0:.3f}" '
            f'x2="{wp - right_padding + 10:.3f}" y2="{y0:.3f}" '
            f'stroke="{axis_cfg["stroke"]}" stroke-width="{axis_cfg["stroke_width"]}" '
            f'stroke-dasharray="{axis_cfg["dash"]}"/>'
        )

    for region in regions:
        region_path = _region_path(region, entity_index, screen)
        parts.append(
            f'<path id="{escape(str(region["id"]))}" d="{region_path}" '
            'fill="url(#concrete-hatch)" fill-rule="evenodd" '
            'stroke="none"/>'
        )

    for line in lines:
        start = screen(_point(line["start"]))
        end = screen(_point(line["end"]))
        layer = str(line.get("layer", ""))
        stroke, w, dash = _layer_style(layer)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line id="{escape(str(line["id"]))}" '
            f'x1="{start[0]:.3f}" y1="{start[1]:.3f}" '
            f'x2="{end[0]:.3f}" y2="{end[1]:.3f}" '
            f'stroke="{stroke}" stroke-width="{w}" '
            f'fill="none" vector-effect="non-scaling-stroke"{dash_attr}/>'
        )

    for arc in arcs:
        points = [screen(point) for point in _arc_points(arc)]
        point_text = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        stroke, w, dash = _layer_style(str(arc.get("layer", "")))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline id="{escape(str(arc["id"]))}" points="{point_text}" '
            f'stroke="{stroke}" stroke-width="{w}" fill="none" '
            f'vector-effect="non-scaling-stroke"{dash_attr}/>'
        )

    origin_x, origin_y = screen((0.0, 0.0))
    if min_x <= 0.0 <= max_x and min_y <= 0.0 <= max_y:
        parts.extend(
            [
                (
                    f'<circle cx="{origin_x:.3f}" cy="{origin_y:.3f}" '
                    f'r="{origin_cfg["dot_radius"]}" fill="{origin_cfg["dot_color"]}"/>'
                ),
                (
                    f'<text x="{origin_x + 8:.3f}" y="{origin_y - 8:.3f}" '
                    f'font-family="{mono_style["font_family"]}" '
                    f'font-size="{mono_style["font_size"]}" '
                    f'fill="{origin_cfg["label_color"]}">O(0,0)</text>'
                ),
            ]
        )

    legend_y = hp - 26
    parts.extend(
        [
            (
                f'<line x1="70" y1="{legend_y}" x2="105" y2="{legend_y}" '
                f'stroke="#172033" stroke-width="2"/>'
            ),
            (
                f'<text x="114" y="{legend_y + 4}" '
                f'font-family="{legend_style["font_family"]}" '
                f'font-size="{legend_style["font_size"]}" '
                f'fill="{legend_style["color"]}">主梁外轮廓</text>'
            ),
            (
                f'<line x1="250" y1="{legend_y}" x2="285" y2="{legend_y}" '
                f'stroke="#172033" stroke-width="1.6"/>'
            ),
            (
                f'<text x="294" y="{legend_y + 4}" '
                f'font-family="{legend_style["font_family"]}" '
                f'font-size="{legend_style["font_size"]}" '
                f'fill="{legend_style["color"]}">{void_label}</text>'
            ),
            (
                f'<rect x="435" y="{legend_y - 9}" width="28" height="14" '
                f'fill="url(#concrete-hatch)" stroke="{hatch["line_color"]}" stroke-width="1"/>'
            ),
            (
                f'<text x="472" y="{legend_y + 4}" '
                f'font-family="{legend_style["font_family"]}" '
                f'font-size="{legend_style["font_size"]}" '
                f'fill="{legend_style["color"]}">混凝土region（孔洞扣除）</text>'
            ),
            (
                f'<text x="{wp - 70}" y="{legend_y + 4}" text-anchor="end" '
                f'font-family="{mono_style["font_family"]}" '
                f'font-size="{mono_style["font_size"]}" '
                f'fill="{mono_style["color"]}">'
                f"line {len(lines)} / arc {len(arcs)} / region {len(regions)}</text>"
            ),
            "</svg>",
            "",
        ]
    )
    return "\n".join(parts)


def _select_section(
    document: dict[str, Any],
    section_id: str | None,
    *,
    view_name: str,
) -> dict[str, Any]:
    try:
        sections = document["views"][view_name]["sections"]
    except (KeyError, TypeError) as exc:
        raise GeometryPreviewError(
            f"几何JSON缺少views.{view_name}.sections。"
        ) from exc
    if not isinstance(sections, list) or not sections:
        raise GeometryPreviewError(f"{view_name}.sections不能为空。")
    if section_id is None:
        section = sections[0]
    else:
        section = next(
            (item for item in sections if str(item.get("id")) == section_id),
            None,
        )
        if section is None:
            raise GeometryPreviewError(f"几何JSON中不存在断面：{section_id}")
    if not isinstance(section, dict):
        raise GeometryPreviewError("断面数据必须是对象。")
    return section


def _point(value: dict[str, Any]) -> tuple[float, float]:
    try:
        return float(value["x"]), float(value["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GeometryPreviewError(f"无效点坐标：{value!r}") from exc


def _arc_points(arc: dict[str, Any]) -> list[tuple[float, float]]:
    center_x, center_y = _point(arc["center"])
    radius = float(arc["radius"])
    start = math.radians(float(arc["start_angle_deg"]))
    end = math.radians(float(arc["end_angle_deg"]))
    clockwise = bool(arc["clockwise"])
    delta = _directed_angle_delta(start, end, clockwise)
    steps = max(4, int(math.ceil(abs(math.degrees(delta)) / _SVG["arc_step_deg"])))
    return [
        (
            center_x + radius * math.cos(start + delta * index / steps),
            center_y + radius * math.sin(start + delta * index / steps),
        )
        for index in range(steps + 1)
    ]


def _directed_angle_delta(start: float, end: float, clockwise: bool) -> float:
    if clockwise:
        return -((start - end) % (2.0 * math.pi))
    return (end - start) % (2.0 * math.pi)


def _region_path(
    region: dict[str, Any],
    entities: dict[str, dict[str, Any]],
    screen: Any,
) -> str:
    loops = [region.get("outer", [])] + list(region.get("holes", []))
    path_parts: list[str] = []
    for loop in loops:
        points = _loop_points(loop, entities)
        if not points:
            continue
        screen_points = [screen(point) for point in points]
        path_parts.append(
            "M "
            + " L ".join(f"{x:.3f} {y:.3f}" for x, y in screen_points)
            + " Z"
        )
    if not path_parts:
        raise GeometryPreviewError(f"区域{region.get('id')}没有有效闭环。")
    return " ".join(path_parts)


def _loop_points(
    loop: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for reference in loop:
        entity_id = str(reference.get("entity_id"))
        entity = entities.get(entity_id)
        if entity is None:
            raise GeometryPreviewError(f"区域引用了不存在的图元：{entity_id}")
        reverse = reference.get("direction") == "reverse"
        if entity["kind"] == "line":
            points = [_point(entity["start"]), _point(entity["end"])]
        else:
            points = _arc_points(entity)
        if reverse:
            points.reverse()
        if result and _same_point(result[-1], points[0]):
            result.extend(points[1:])
        else:
            result.extend(points)
    if result and _same_point(result[0], result[-1]):
        result.pop()
    return result


def _same_point(
    first: tuple[float, float],
    second: tuple[float, float],
    tolerance: float = 1e-5,
) -> bool:
    return math.hypot(first[0] - second[0], first[1] - second[1]) <= tolerance


# SVG 虚线样式（仅预览用，不参与 CAD 输出）
_DASH: dict[str, str | None] = {
    "XS-MANHOLE-REF": "7 5",
    "EL-CUT": "6 4",
    "PL-CUT": "6 4",
}


def _layer_style(layer: str) -> tuple[str, str, str | None]:
    style = get_layer_style(layer)
    return style.color, style.lineweight, _DASH.get(layer)
