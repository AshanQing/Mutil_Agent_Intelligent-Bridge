from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .geometry_config import (
    get_layer_style,
)
from .geometry_models import (
    Arc,
    DesignParameters,
    Diagnostic,
    Direction,
    EntityRef,
    GeometryDocument,
    GeometryView,
    Line,
    Point,
    Region,
    SectionGeometry,
)

CLOSURE_TOLERANCE_MM = 0.01
GEOMETRY_EPSILON = 1e-9
KEY_SECTION_IDS = ("END-DIAPHRAGM", "CONSTANT-SECTION", "MID-DIAPHRAGM", "PIER-SECTION")
DEFAULT_KEY_SECTION_IDS = KEY_SECTION_IDS[:3]


class GeometryInputError(ValueError):
    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = diagnostics
        super().__init__("；".join(item.message for item in diagnostics))


@dataclass(frozen=True)
class _Fillet:
    center: Point
    start: Point
    end: Point
    radius: float
    clockwise: bool


class _EntityBuilder:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.lines: list[Line] = []
        self.arcs: list[Arc] = []

    @staticmethod
    def _resolve_linetype(layer: str) -> str:
        return get_layer_style(layer).linetype

    def add_line(
        self,
        start: Point,
        end: Point,
        *,
        layer: str,
        linetype: str | None = None,
    ) -> EntityRef:
        identifier = f"{self.prefix}_line_{len(self.lines) + 1:03d}"
        self.lines.append(
            Line(
                id=identifier,
                start=start,
                end=end,
                layer=layer,
                linetype=linetype if linetype is not None else self._resolve_linetype(layer),
            )
        )
        return EntityRef(identifier)

    def add_arc(
        self,
        fillet: _Fillet,
        *,
        layer: str,
        linetype: str | None = None,
    ) -> EntityRef:
        identifier = f"{self.prefix}_arc_{len(self.arcs) + 1:03d}"
        self.arcs.append(
            Arc(
                id=identifier,
                center=fillet.center,
                radius=fillet.radius,
                start_angle_deg=_point_angle_deg(fillet.center, fillet.start),
                end_angle_deg=_point_angle_deg(fillet.center, fillet.end),
                clockwise=fillet.clockwise,
                layer=layer,
                linetype=linetype if linetype is not None else self._resolve_linetype(layer),
            )
        )
        return EntityRef(identifier)

    def add_line_loop(
        self,
        points: list[Point],
        *,
        layer: str,
    ) -> tuple[EntityRef, ...]:
        result: list[EntityRef] = []
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            result.append(
                self.add_line(
                    start,
                    end,
                    layer=layer,
                )
            )
        return tuple(result)


def load_section_spec(
    section_id: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    spec_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "data" / "geometry_section_specs.json"
    )
    try:
        document = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"断面配置读取失败：{exc}") from exc
    sections = document.get("sections")
    if not isinstance(sections, dict) or section_id not in sections:
        raise ValueError(f"断面配置不存在：{section_id}")
    result = sections[section_id]
    if not isinstance(result, dict):
        raise ValueError(f"断面配置无效：{section_id}")
    _validate_section_spec_shape(section_id, result)
    return result


def _validate_section_spec_shape(
    section_id: str,
    spec: dict[str, Any],
) -> None:
    topology = spec.get("topology")
    expected = {"topology", "height"}
    if topology == "box_girder":
        expected.update(
            {
                "top_thickness",
                "bottom_thickness",
                "web_thickness",
                "box_haunch",
            }
        )
        nested_key = "box_haunch"
        nested_expected = {"top", "bottom"}
    elif topology == "solid_diaphragm_with_manhole":
        expected.add("manhole")
        nested_key = "manhole"
        nested_expected = {"width", "height", "chamfer", "bottom_offset"}
    else:
        raise ValueError(f"断面配置{section_id}的topology无效：{topology}")

    actual = set(spec)
    if actual != expected:
        raise ValueError(
            f"断面配置{section_id}字段不匹配："
            f"缺少{sorted(expected - actual)}，多余{sorted(actual - expected)}"
        )
    nested = spec.get(nested_key)
    if not isinstance(nested, dict) or set(nested) != nested_expected:
        actual_nested = set(nested) if isinstance(nested, dict) else set()
        raise ValueError(
            f"断面配置{section_id}.{nested_key}字段不匹配："
            f"缺少{sorted(nested_expected - actual_nested)}，"
            f"多余{sorted(actual_nested - nested_expected)}"
        )


def generate_cross_section(
    parameters: DesignParameters,
    *,
    section_id: str = "CONSTANT-SECTION",
    section_spec: dict[str, Any] | None = None,
) -> GeometryDocument:
    spec = section_spec or load_section_spec(section_id)
    parameter_diagnostics = validate_section_parameters(parameters, spec)
    if parameter_diagnostics:
        raise GeometryInputError(parameter_diagnostics)

    longitudinal = parameters.longitudinal
    cross = parameters.cross_section
    topology = str(spec["topology"])
    height_key = str(spec["height"])
    height = float(longitudinal[height_key])
    top_width = float(cross["B_top"])
    bottom_width = float(cross["B_bottom"])
    top_half = top_width / 2.0
    bottom_half = bottom_width / 2.0

    left_top = _profile_points(
        cross["profile_top_L"],
        start_x=-top_half,
        direction=1.0,
    )
    right_top = _profile_points(
        cross["profile_top_R"],
        start_x=top_half,
        direction=-1.0,
    )
    top_points_for_lookup = sorted(
        left_top + right_top[:-1],
        key=lambda item: item.x,
    )
    left_tip_top = _interpolate_y(top_points_for_lookup, -top_half)
    right_tip_top = _interpolate_y(top_points_for_lookup, top_half)
    left_root_top = _interpolate_y(top_points_for_lookup, -bottom_half)
    right_root_top = _interpolate_y(top_points_for_lookup, bottom_half)

    tip_thickness = float(cross["t_cant_tip"])
    root_thickness = float(cross["t_cant_root"])
    left_tip = Point(-top_half, left_tip_top)
    left_tip_under = Point(-top_half, left_tip_top - tip_thickness)
    left_top_corner = Point(-bottom_half, left_root_top - root_thickness)
    left_bottom_corner = Point(-bottom_half, -height)
    right_bottom_corner = Point(bottom_half, -height)
    right_top_corner = Point(bottom_half, right_root_top - root_thickness)
    right_tip_under = Point(top_half, right_tip_top - tip_thickness)
    right_tip = Point(top_half, right_tip_top)

    top_radius = float(cross["R_web_top"])
    bottom_radius = float(cross["R_web_bottom"])
    left_top_fillet = _fillet(
        left_tip_under,
        left_top_corner,
        left_bottom_corner,
        top_radius,
    )
    left_bottom_fillet = _fillet(
        left_top_corner,
        left_bottom_corner,
        right_bottom_corner,
        bottom_radius,
    )
    right_bottom_fillet = _fillet(
        left_bottom_corner,
        right_bottom_corner,
        right_top_corner,
        bottom_radius,
    )
    right_top_fillet = _fillet(
        right_bottom_corner,
        right_top_corner,
        right_tip_under,
        top_radius,
    )

    prefix = "xs_{}".format(section_id.lower().replace("-", "_"))
    builder = _EntityBuilder(prefix)
    outer: list[EntityRef] = []
    outer.append(
        builder.add_line(
            left_tip,
            left_tip_under,
            layer="XS-OUTLINE",
        )
    )
    outer.append(
        builder.add_line(
            left_tip_under,
            left_top_fillet.start,
            layer="XS-OUTLINE",
        )
    )
    outer.append(
        builder.add_arc(
            left_top_fillet,
            layer="XS-OUTLINE",
        )
    )
    outer.append(
        builder.add_line(
            left_top_fillet.end,
            left_bottom_fillet.start,
            layer="XS-OUTLINE",
        )
    )
    outer.append(
        builder.add_arc(
            left_bottom_fillet,
            layer="XS-OUTLINE",
        )
    )
    outer.append(
        builder.add_line(
            left_bottom_fillet.end,
            right_bottom_fillet.start,
            layer="XS-OUTLINE",
        )
    )
    outer.append(
        builder.add_arc(
            right_bottom_fillet,
            layer="XS-OUTLINE",
        )
    )
    outer.append(
        builder.add_line(
            right_bottom_fillet.end,
            right_top_fillet.start,
            layer="XS-OUTLINE",
        )
    )
    outer.append(
        builder.add_arc(
            right_top_fillet,
            layer="XS-OUTLINE",
        )
    )
    outer.append(
        builder.add_line(
            right_top_fillet.end,
            right_tip_under,
            layer="XS-OUTLINE",
        )
    )
    outer.append(
        builder.add_line(
            right_tip_under,
            right_tip,
            layer="XS-OUTLINE",
        )
    )

    top_path = right_top + list(reversed(left_top[:-1]))
    for start, end in zip(top_path, top_path[1:]):
        outer.append(
            builder.add_line(
                start,
                end,
                layer="XS-OUTLINE",
            )
        )

    region_holes: tuple[tuple[EntityRef, ...], ...]
    if topology == "box_girder":
        top_thickness_key = str(spec["top_thickness"])
        bottom_thickness_key = str(spec["bottom_thickness"])
        web_thickness_key = str(spec["web_thickness"])
        top_thickness = float(longitudinal[top_thickness_key])
        bottom_thickness = float(longitudinal[bottom_thickness_key])
        web_thickness = float(longitudinal[web_thickness_key])
        inner_half = bottom_half - web_thickness
        chamber_top = -top_thickness
        chamber_bottom = -height + bottom_thickness
        top_haunch_key = str(spec["box_haunch"]["top"])
        bottom_haunch_key = str(spec["box_haunch"]["bottom"])
        top_haunch = _pair(_parameter_value(parameters, top_haunch_key))
        bottom_haunch = _pair(_parameter_value(parameters, bottom_haunch_key))
        chamber_points = [
            Point(-inner_half + top_haunch[0], chamber_top),
            Point(inner_half - top_haunch[0], chamber_top),
            Point(inner_half, chamber_top - top_haunch[1]),
            Point(inner_half, chamber_bottom + bottom_haunch[1]),
            Point(inner_half - bottom_haunch[0], chamber_bottom),
            Point(-inner_half + bottom_haunch[0], chamber_bottom),
            Point(-inner_half, chamber_bottom + bottom_haunch[1]),
            Point(-inner_half, chamber_top - top_haunch[1]),
        ]
        chamber_loop = builder.add_line_loop(
            chamber_points,
            layer="XS-VOID",
        )
        region_holes = (chamber_loop,)
    elif topology == "solid_diaphragm_with_manhole":
        manhole_spec = spec["manhole"]
        manhole_width_key = str(manhole_spec["width"])
        manhole_height_key = str(manhole_spec["height"])
        manhole_chamfer_key = str(manhole_spec["chamfer"])
        manhole_offset_key = str(manhole_spec["bottom_offset"])
        manhole_width = float(cross[manhole_width_key])
        manhole_height = float(cross[manhole_height_key])
        manhole_chamfer = _pair(cross[manhole_chamfer_key])
        manhole_bottom_offset = float(cross[manhole_offset_key])
        manhole_bottom = -height + manhole_bottom_offset
        manhole_top = manhole_bottom + manhole_height
        manhole_half = manhole_width / 2.0
        manhole_points = [
            Point(-manhole_half + manhole_chamfer[0], manhole_top),
            Point(manhole_half - manhole_chamfer[0], manhole_top),
            Point(manhole_half, manhole_top - manhole_chamfer[1]),
            Point(manhole_half, manhole_bottom + manhole_chamfer[1]),
            Point(manhole_half - manhole_chamfer[0], manhole_bottom),
            Point(-manhole_half + manhole_chamfer[0], manhole_bottom),
            Point(-manhole_half, manhole_bottom + manhole_chamfer[1]),
            Point(-manhole_half, manhole_top - manhole_chamfer[1]),
        ]
        manhole_loop = builder.add_line_loop(
            manhole_points,
            layer="XS-MANHOLE",
        )
        region_holes = (manhole_loop,)
    else:
        raise GeometryInputError(
            [Diagnostic("unsupported_topology", f"不支持的断面拓扑：{topology}")]
        )

    region = Region(
        id=f"{prefix}_region_001",
        layer="XS-CONCRETE",
        outer=tuple(outer),
        holes=region_holes,
    )

    section = SectionGeometry(
        id=section_id,
        lines=builder.lines,
        arcs=builder.arcs,
        regions=[region],
    )
    geometry_diagnostics = validate_geometry(section)
    if geometry_diagnostics:
        raise GeometryInputError(geometry_diagnostics)
    return GeometryDocument(
        source=dict(parameters.source),
        section=section,
        coordinate_system={
            "origin": "梁中心线与顶面中心交点",
            "x_axis": "+X 向右",
            "y_axis": "+Y 向上",
        },
    )


def generate_key_sections(parameters: DesignParameters) -> GeometryDocument:
    section_ids = list(DEFAULT_KEY_SECTION_IDS)
    if int(parameters.longitudinal.get("n_box_0_side") or 0) > 0:
        section_ids.append("PIER-SECTION")
    documents = [
        generate_cross_section(parameters, section_id=section_id)
        for section_id in section_ids
    ]
    return GeometryDocument(
        source=dict(parameters.source),
        section=documents[0].section,
        additional_sections=[item.section for item in documents[1:]],
        coordinate_system={
            "origin": "梁中心线与顶面中心交点",
            "x_axis": "+X 向右",
            "y_axis": "+Y 向上",
        },
    )


def generate_geometry(parameters: DesignParameters) -> GeometryDocument:
    from .elevation_generator import generate_elevation
    from .plan_generator import generate_plan

    document = generate_key_sections(parameters)
    elevation = generate_elevation(parameters)
    document.additional_views["elevation"] = GeometryView(
        coordinate_system=dict(elevation.coordinate_system or {}),
        sections=elevation.sections,
    )
    plan = generate_plan(parameters)
    document.additional_views["plan"] = GeometryView(
        coordinate_system=dict(plan.coordinate_system or {}),
        sections=plan.sections,
    )
    return document


def validate_section_parameters(
    parameters: DesignParameters,
    spec: dict[str, Any],
) -> list[Diagnostic]:
    longitudinal = parameters.longitudinal
    cross = parameters.cross_section
    topology = str(spec.get("topology", ""))
    long_keys = [str(spec["height"])]
    cross_keys = [
        "B_top",
        "B_bottom",
        "L_cant_L",
        "L_cant_R",
        "profile_top_L",
        "profile_top_R",
        "t_cant_tip",
        "t_cant_root",
        "R_web_top",
        "R_web_bottom",
        "k_web",
    ]
    pair_keys: list[str] = []
    zero_allowed = {"k_web"}
    if topology == "box_girder":
        long_keys.extend(
            [
                str(spec["top_thickness"]),
                str(spec["bottom_thickness"]),
                str(spec["web_thickness"]),
            ]
        )
        pair_keys.extend(
            [
                str(spec["box_haunch"]["top"]),
                str(spec["box_haunch"]["bottom"]),
            ]
        )
    elif topology == "solid_diaphragm_with_manhole":
        manhole_spec = spec["manhole"]
        pair_keys.append(str(manhole_spec["chamfer"]))
        cross_keys.extend(
            [
                str(manhole_spec["width"]),
                str(manhole_spec["height"]),
                str(manhole_spec["chamfer"]),
                str(manhole_spec["bottom_offset"]),
            ]
        )
        zero_allowed.add(str(manhole_spec["bottom_offset"]))
    else:
        return [
            Diagnostic("unsupported_topology", f"不支持的断面拓扑：{topology}")
        ]

    diagnostics: list[Diagnostic] = []
    for key in long_keys:
        if key not in longitudinal or longitudinal[key] is None:
            diagnostics.append(Diagnostic("missing_parameter", f"缺少纵向参数：{key}"))
    for key in cross_keys:
        if key not in cross or cross[key] is None:
            diagnostics.append(Diagnostic("missing_parameter", f"缺少横断面参数：{key}"))
    for key in pair_keys:
        if _parameter_value_or_none(parameters, key) is None:
            diagnostics.append(Diagnostic("missing_parameter", f"缺少尺寸对参数：{key}"))
    if diagnostics:
        return diagnostics

    numeric_values = {
        key: longitudinal[key] for key in long_keys
    }
    numeric_values.update(
        {
            key: cross[key]
            for key in cross_keys
            if key not in {"profile_top_L", "profile_top_R"}
            and key not in pair_keys
        }
    )
    for key, value in numeric_values.items():
        if not _finite_number(value):
            diagnostics.append(Diagnostic("invalid_number", f"参数{key}必须是有限数值。"))
        elif key not in zero_allowed and float(value) <= 0:
            diagnostics.append(Diagnostic("non_positive", f"参数{key}必须大于0。"))
        elif key in zero_allowed and float(value) < 0:
            diagnostics.append(Diagnostic("negative_value", f"参数{key}不能小于0。"))

    if _finite_number(cross["k_web"]) and not math.isclose(
        float(cross["k_web"]), 0.0, abs_tol=GEOMETRY_EPSILON
    ):
        diagnostics.append(
            Diagnostic("unsupported_web_slope", "首版断面生成仅支持k_web=0的直腹板。")
        )

    for key in pair_keys:
        value = _parameter_value_or_none(parameters, key)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(_finite_number(item) and float(item) > 0 for item in value)
        ):
            diagnostics.append(
                Diagnostic("invalid_pair", f"参数{key}必须是两个正数构成的尺寸对。")
            )

    if diagnostics:
        return diagnostics

    top_width = float(cross["B_top"])
    bottom_width = float(cross["B_bottom"])
    if not math.isclose(
        top_width,
        bottom_width + float(cross["L_cant_L"]) + float(cross["L_cant_R"]),
        abs_tol=1.0,
    ):
        diagnostics.append(
            Diagnostic(
                "width_reconciliation",
                "B_top必须等于B_bottom + L_cant_L + L_cant_R（允许1 mm误差）。",
            )
        )

    for key in ("profile_top_L", "profile_top_R"):
        profile = cross[key]
        if not isinstance(profile, list) or not profile:
            diagnostics.append(Diagnostic("invalid_profile", f"参数{key}必须是非空横坡序列。"))
            continue
        total = 0.0
        for item in profile:
            if (
                not isinstance(item, dict)
                or not _finite_number(item.get("length_mm"))
                or not _finite_number(item.get("slope_percent"))
                or float(item["length_mm"]) <= 0
            ):
                diagnostics.append(
                    Diagnostic("invalid_profile", f"参数{key}包含无效横坡分段。")
                )
                break
            total += float(item["length_mm"])
        else:
            if not math.isclose(total, top_width / 2.0, abs_tol=1.0):
                diagnostics.append(
                    Diagnostic(
                        "profile_length",
                        f"{key}长度合计必须等于B_top/2（允许1 mm误差）。",
                    )
                )

    height = float(longitudinal[str(spec["height"])])
    if topology == "box_girder":
        top_t = float(longitudinal[str(spec["top_thickness"])])
        bottom_t = float(longitudinal[str(spec["bottom_thickness"])])
        web_t = float(longitudinal[str(spec["web_thickness"])])
        if top_t + bottom_t >= height:
            diagnostics.append(
                Diagnostic(
                    "empty_chamber_height",
                    "顶板厚度与底板厚度之和必须小于梁高。",
                )
            )
        if web_t >= bottom_width / 2.0:
            diagnostics.append(
                Diagnostic("empty_chamber_width", "腹板厚度必须小于B_bottom/2。")
            )

        inner_half = bottom_width / 2.0 - web_t
        chamber_height = height - top_t - bottom_t
        for label, pair_key in (
            ("顶部箱室倒角", str(spec["box_haunch"]["top"])),
            ("底部箱室倒角", str(spec["box_haunch"]["bottom"])),
        ):
            value = _parameter_value(parameters, pair_key)
            if 2.0 * float(value[0]) >= 2.0 * inner_half:
                diagnostics.append(
                    Diagnostic("haunch_width", f"{label}横向尺寸无法装入箱室。")
                )
            if float(value[1]) >= chamber_height:
                diagnostics.append(
                    Diagnostic("haunch_height", f"{label}竖向尺寸无法装入箱室。")
                )
    else:
        manhole_spec = spec["manhole"]
        manhole_width = float(cross[str(manhole_spec["width"])])
        manhole_height = float(cross[str(manhole_spec["height"])])
        manhole_offset = float(cross[str(manhole_spec["bottom_offset"])])
        manhole_chamfer = cross[str(manhole_spec["chamfer"])]
        if manhole_width >= bottom_width:
            diagnostics.append(
                Diagnostic("manhole_width", "人洞宽度必须小于梁底宽度。")
            )
        if manhole_offset + manhole_height > height:
            diagnostics.append(
                Diagnostic("manhole_vertical", "人洞必须完整位于横墙实体轮廓内部。")
            )
        if 2.0 * float(manhole_chamfer[0]) >= manhole_width:
            diagnostics.append(Diagnostic("manhole_chamfer", "人洞横向倒角过大。"))
        if 2.0 * float(manhole_chamfer[1]) >= manhole_height:
            diagnostics.append(Diagnostic("manhole_chamfer", "人洞竖向倒角过大。"))
    return diagnostics


def validate_geometry(
    section: SectionGeometry,
    *,
    tolerance: float = CLOSURE_TOLERANCE_MM,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    entities: dict[str, Line | Arc] = {
        item.id: item for item in section.lines
    }
    entities.update({item.id: item for item in section.arcs})
    for line in section.lines:
        if line.start.distance_to(line.end) <= GEOMETRY_EPSILON:
            diagnostics.append(
                Diagnostic("zero_length_line", "直线长度必须大于0。", entity_id=line.id)
            )
    for arc in section.arcs:
        if not math.isfinite(arc.radius) or arc.radius <= 0:
            diagnostics.append(
                Diagnostic("invalid_arc_radius", "圆弧半径必须为有限正数。", entity_id=arc.id)
            )
        if arc.start.distance_to(arc.end) <= GEOMETRY_EPSILON:
            diagnostics.append(
                Diagnostic("zero_sweep_arc", "圆弧起终点不能重合。", entity_id=arc.id)
            )

    for region in section.regions:
        loops = [("outer", region.outer), *[
            (f"hole_{index + 1}", loop)
            for index, loop in enumerate(region.holes)
        ]]
        for loop_name, loop in loops:
            if not loop:
                diagnostics.append(
                    Diagnostic("empty_loop", f"区域{region.id}的{loop_name}为空。", region.id)
                )
                continue
            diagnostics.extend(
                _validate_loop_continuity(region.id, loop_name, loop, entities, tolerance)
            )
            if any(ref.entity_id not in entities for ref in loop):
                continue
            area = _loop_signed_area(loop, entities)
            if loop_name == "outer" and area <= 0:
                diagnostics.append(
                    Diagnostic("outer_orientation", "region外环必须为逆时针。", region.id)
                )
            if loop_name != "outer" and area >= 0:
                diagnostics.append(
                    Diagnostic("hole_orientation", "region孔洞环必须为顺时针。", region.id)
                )
            flattened = _flatten_loop(loop, entities)
            if _has_self_intersection(flattened, tolerance):
                diagnostics.append(
                    Diagnostic("self_intersection", f"{loop_name}存在自交。", region.id)
                )
            diagnostics.extend(
                _validate_arc_tangency(region.id, loop_name, loop, entities)
            )

        if region.outer and all(ref.entity_id in entities for ref in region.outer):
            outer_points = _flatten_loop(region.outer, entities)
            for index, hole in enumerate(region.holes):
                if not hole or any(ref.entity_id not in entities for ref in hole):
                    continue
                hole_points = _flatten_loop(hole, entities)
                if hole_points and not _point_in_polygon(hole_points[0], outer_points):
                    diagnostics.append(
                        Diagnostic(
                            "hole_outside",
                            f"孔洞环{index + 1}不在外环内部。",
                            region.id,
                        )
                    )
    return diagnostics


def _profile_points(
    profile: list[dict[str, Any]],
    *,
    start_x: float,
    direction: float,
) -> list[Point]:
    points = [Point(start_x, 0.0)]
    x = start_x
    y = 0.0
    for item in profile:
        dx = direction * float(item["length_mm"])
        x += dx
        y += dx * float(item["slope_percent"]) / 100.0
        points.append(Point(x, y))
    center_shift = points[-1].y
    return [Point(item.x, item.y - center_shift) for item in points]


def _interpolate_y(points: list[Point], target_x: float) -> float:
    for first, second in zip(points, points[1:]):
        if first.x - GEOMETRY_EPSILON <= target_x <= second.x + GEOMETRY_EPSILON:
            if math.isclose(first.x, second.x, abs_tol=GEOMETRY_EPSILON):
                return first.y
            ratio = (target_x - first.x) / (second.x - first.x)
            return first.y + ratio * (second.y - first.y)
    raise ValueError(f"顶面横坡范围不包含x={target_x}。")


def _fillet(p0: Point, corner: Point, p2: Point, radius: float) -> _Fillet:
    if radius <= 0:
        raise GeometryInputError([Diagnostic("invalid_fillet_radius", "圆角半径必须大于0。")])
    incoming = _unit(corner - p0)
    outgoing = _unit(p2 - corner)
    turn = _cross(incoming, outgoing)
    if abs(turn) <= GEOMETRY_EPSILON:
        raise GeometryInputError([Diagnostic("parallel_fillet_lines", "圆角相邻母线不能平行。")])
    # A CCW outer loop can contain both convex and re-entrant corners.  Convex
    # corners use the left offsets; re-entrant corners use the right offsets
    # so that both tangent points remain on the original finite segments.
    offset_side = 1.0 if turn > 0.0 else -1.0
    incoming_normal = Point(-incoming.y, incoming.x).scale(offset_side)
    outgoing_normal = Point(-outgoing.y, outgoing.x).scale(offset_side)
    center = _line_intersection(
        corner + incoming_normal.scale(radius),
        incoming,
        corner + outgoing_normal.scale(radius),
        outgoing,
    )
    start = _project_to_line(center, p0, incoming)
    end = _project_to_line(center, corner, outgoing)
    if not _point_on_segment(start, p0, corner) or not _point_on_segment(end, corner, p2):
        raise GeometryInputError(
            [Diagnostic("fillet_does_not_fit", f"半径{radius:g}的圆角无法装入相邻线段。")]
        )
    radial = _unit(start - center)
    tangent_ccw = Point(-radial.y, radial.x)
    tangent_cw = Point(radial.y, -radial.x)
    clockwise = _dot(tangent_cw, incoming) > _dot(tangent_ccw, incoming)
    return _Fillet(center, start, end, radius, clockwise)


def _line_intersection(p: Point, u: Point, q: Point, v: Point) -> Point:
    denominator = _cross(u, v)
    if abs(denominator) <= GEOMETRY_EPSILON:
        raise GeometryInputError([Diagnostic("parallel_fillet_lines", "圆角相邻母线不能平行。")])
    factor = _cross(q - p, v) / denominator
    return p + u.scale(factor)


def _project_to_line(point: Point, line_point: Point, unit: Point) -> Point:
    return line_point + unit.scale(_dot(point - line_point, unit))


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    length = start.distance_to(end)
    return (
        point.distance_to(start) <= length + CLOSURE_TOLERANCE_MM
        and point.distance_to(end) <= length + CLOSURE_TOLERANCE_MM
    )


def _parameter_value(parameters: DesignParameters, key: str) -> Any:
    value = _parameter_value_or_none(parameters, key)
    if value is None:
        raise KeyError(key)
    return value


def _parameter_value_or_none(
    parameters: DesignParameters,
    key: str,
) -> Any:
    if key in parameters.longitudinal and parameters.longitudinal[key] is not None:
        return parameters.longitudinal[key]
    if key in parameters.cross_section:
        return parameters.cross_section[key]
    return None


def _pair(value: Any) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit(vector: Point) -> Point:
    length = math.hypot(vector.x, vector.y)
    if length <= GEOMETRY_EPSILON:
        raise GeometryInputError([Diagnostic("zero_vector", "几何母线长度必须大于0。")])
    return Point(vector.x / length, vector.y / length)


def _dot(first: Point, second: Point) -> float:
    return first.x * second.x + first.y * second.y


def _cross(first: Point, second: Point) -> float:
    return first.x * second.y - first.y * second.x


def _point_angle_deg(center: Point, point: Point) -> float:
    return math.degrees(math.atan2(point.y - center.y, point.x - center.x)) % 360.0


def _directed_entity(
    ref: EntityRef,
    entities: dict[str, Line | Arc],
) -> tuple[Point, Point, bool | None]:
    entity = entities[ref.entity_id]
    if isinstance(entity, Line):
        return (
            (entity.start, entity.end, None)
            if ref.direction == "forward"
            else (entity.end, entity.start, None)
        )
    return (
        (entity.start, entity.end, entity.clockwise)
        if ref.direction == "forward"
        else (entity.end, entity.start, not entity.clockwise)
    )


def _validate_loop_continuity(
    region_id: str,
    loop_name: str,
    loop: tuple[EntityRef, ...],
    entities: dict[str, Line | Arc],
    tolerance: float,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for ref in loop:
        if ref.entity_id not in entities:
            diagnostics.append(
                Diagnostic("missing_entity", f"区域引用不存在：{ref.entity_id}", region_id)
            )
    if diagnostics:
        return diagnostics
    for index, ref in enumerate(loop):
        next_ref = loop[(index + 1) % len(loop)]
        _, end, _ = _directed_entity(ref, entities)
        start, _, _ = _directed_entity(next_ref, entities)
        if end.distance_to(start) > tolerance:
            diagnostics.append(
                Diagnostic(
                    "open_loop",
                    f"{loop_name}在{ref.entity_id}与{next_ref.entity_id}之间未闭合。",
                    region_id,
                )
            )
    return diagnostics


def _loop_signed_area(
    loop: tuple[EntityRef, ...],
    entities: dict[str, Line | Arc],
) -> float:
    area = 0.0
    for ref in loop:
        entity = entities[ref.entity_id]
        start, end, clockwise = _directed_entity(ref, entities)
        if isinstance(entity, Line):
            area += 0.5 * (start.x * end.y - end.x * start.y)
            continue
        start_angle = math.atan2(start.y - entity.center.y, start.x - entity.center.x)
        end_angle = math.atan2(end.y - entity.center.y, end.x - entity.center.x)
        delta = _directed_angle_delta(start_angle, end_angle, bool(clockwise))
        area += 0.5 * (
            entity.radius
            * entity.center.x
            * (math.sin(end_angle) - math.sin(start_angle))
            - entity.radius
            * entity.center.y
            * (math.cos(end_angle) - math.cos(start_angle))
            + entity.radius * entity.radius * delta
        )
    return area


def _directed_angle_delta(start: float, end: float, clockwise: bool) -> float:
    if clockwise:
        return -((start - end) % (2.0 * math.pi))
    return (end - start) % (2.0 * math.pi)


def _flatten_loop(
    loop: tuple[EntityRef, ...],
    entities: dict[str, Line | Arc],
) -> list[Point]:
    points: list[Point] = []
    for ref in loop:
        entity = entities[ref.entity_id]
        start, end, clockwise = _directed_entity(ref, entities)
        if not points:
            points.append(start)
        if isinstance(entity, Line):
            points.append(end)
            continue
        start_angle = math.atan2(start.y - entity.center.y, start.x - entity.center.x)
        end_angle = math.atan2(end.y - entity.center.y, end.x - entity.center.x)
        delta = _directed_angle_delta(start_angle, end_angle, bool(clockwise))
        steps = max(2, int(math.ceil(abs(delta) / math.radians(10.0))))
        for index in range(1, steps + 1):
            angle = start_angle + delta * index / steps
            points.append(
                Point(
                    entity.center.x + entity.radius * math.cos(angle),
                    entity.center.y + entity.radius * math.sin(angle),
                )
            )
    if points and points[0].distance_to(points[-1]) > GEOMETRY_EPSILON:
        points.append(points[0])
    return points


def _has_self_intersection(points: list[Point], tolerance: float) -> bool:
    segments = list(zip(points, points[1:]))
    count = len(segments)
    for first_index, first in enumerate(segments):
        for second_index in range(first_index + 1, count):
            if second_index in {first_index, first_index + 1}:
                continue
            if first_index == 0 and second_index == count - 1:
                continue
            if _segments_intersect(first[0], first[1], segments[second_index][0], segments[second_index][1], tolerance):
                return True
    return False


def _segments_intersect(
    a: Point,
    b: Point,
    c: Point,
    d: Point,
    tolerance: float,
) -> bool:
    def orientation(p: Point, q: Point, r: Point) -> float:
        return _cross(q - p, r - p)

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    return (
        (o1 > tolerance and o2 < -tolerance or o1 < -tolerance and o2 > tolerance)
        and (o3 > tolerance and o4 < -tolerance or o3 < -tolerance and o4 > tolerance)
    )


def _point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    inside = False
    for first, second in zip(polygon, polygon[1:]):
        if (first.y > point.y) != (second.y > point.y):
            crossing_x = (
                (second.x - first.x)
                * (point.y - first.y)
                / (second.y - first.y)
                + first.x
            )
            if point.x < crossing_x:
                inside = not inside
    return inside


def _validate_arc_tangency(
    region_id: str,
    loop_name: str,
    loop: tuple[EntityRef, ...],
    entities: dict[str, Line | Arc],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for index, ref in enumerate(loop):
        next_ref = loop[(index + 1) % len(loop)]
        first = entities.get(ref.entity_id)
        second = entities.get(next_ref.entity_id)
        if first is None or second is None:
            continue
        if not isinstance(first, Arc) and not isinstance(second, Arc):
            continue
        first_tangent = _entity_tangent(ref, first, at_end=True)
        second_tangent = _entity_tangent(next_ref, second, at_end=False)
        if abs(_cross(first_tangent, second_tangent)) > 1e-6 or _dot(
            first_tangent, second_tangent
        ) < 0.999999:
            diagnostics.append(
                Diagnostic(
                    "non_tangent_arc",
                    f"{loop_name}中的{ref.entity_id}与{next_ref.entity_id}不相切。",
                    region_id,
                )
            )
    return diagnostics


def _entity_tangent(
    ref: EntityRef,
    entity: Line | Arc,
    *,
    at_end: bool,
) -> Point:
    start, end, clockwise = _directed_entity(ref, {ref.entity_id: entity})
    if isinstance(entity, Line):
        return _unit(end - start)
    point = end if at_end else start
    radial = _unit(point - entity.center)
    return Point(radial.y, -radial.x) if clockwise else Point(-radial.y, radial.x)
