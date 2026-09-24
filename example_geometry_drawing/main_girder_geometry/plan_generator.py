from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .geometry_config import (
    get_layer_style,
)
from .geometry_models import (
    DesignParameters,
    Diagnostic,
    EntityRef,
    GeometryDocument,
    Line,
    Point,
    Region,
    SectionGeometry,
)
from .longitudinal_layout import (
    BOXED_ZERO_BLOCK,
    THICKENED_ZERO_BLOCK,
    TRANSITION_ZERO_BLOCK,
    LongitudinalLayout,
    resolve_longitudinal_layout,
)


@dataclass(frozen=True)
class PlanStation:
    id: str
    x: float
    web_thickness: float


class _PlanBuilder:
    def __init__(self) -> None:
        self.prefix = "pl_half_girder_plan"
        self.lines: list[Line] = []

    @staticmethod
    def _resolve_linetype(layer: str) -> str:
        return get_layer_style(layer).linetype

    def add_line(
        self, start: Point, end: Point, layer: str, linetype: str | None = None
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

    def add_polygon(
        self,
        points: list[Point],
        layers: list[str],
    ) -> tuple[EntityRef, ...]:
        if len(points) != len(layers):
            raise ValueError("多边形每条边都必须指定图层。")
        return tuple(
            self.add_line(
                start,
                points[(index + 1) % len(points)],
                layers[index],
            )
            for index, start in enumerate(points)
        )


def validate_plan_parameters(
    parameters: DesignParameters,
) -> list[Diagnostic]:
    global_parameters = parameters.global_parameters
    longitudinal = parameters.longitudinal
    cross = parameters.cross_section
    diagnostics: list[Diagnostic] = []

    spans = global_parameters.get("span_layout")
    if (
        not isinstance(spans, list)
        or len(spans) not in {2, 3}
        or not all(_finite_positive(item) for item in spans)
    ):
        diagnostics.append(
            Diagnostic(
                "unsupported_span_layout",
                "平面首版要求span_layout为两个或三个正跨度。",
            )
        )

    required_longitudinal = {
        "L_end_offset",
        "L_cast_side",
        "L_diaphragm_end",
        "L_w_end_var",
        "t_w_end",
        "t_w_std",
        "L_closure_side",
        "L_seg_var_seq",
        "t_w_var_seq",
        "L_H_var",
        "L_0_half",
        "n_box_0_side",
    }
    required_cross = {"B_bottom", "B_hole_end", "B_hole_0"}
    for key in sorted(required_longitudinal):
        if key not in longitudinal or longitudinal[key] is None:
            diagnostics.append(
                Diagnostic("missing_parameter", f"平面缺少纵向参数：{key}")
            )
    for key in sorted(required_cross):
        if key not in cross or cross[key] is None:
            diagnostics.append(
                Diagnostic("missing_parameter", f"平面缺少横断面参数：{key}")
            )
    if diagnostics:
        return diagnostics

    try:
        layout = resolve_longitudinal_layout(parameters)
    except ValueError as exc:
        return [Diagnostic("invalid_zero_block_layout", str(exc))]

    variant_required = {
        THICKENED_ZERO_BLOCK: {
            "L_wall_0",
            "t_w_ch_0_s",
            "t_w_ch_1_s",
            "t_w_ch_2_s",
            "x_b_lin_end_0",
            "L_trans_0",
        },
        TRANSITION_ZERO_BLOCK: {
            "L_wall_0",
            "ch_0_p",
            "L_trans_0",
        },
        BOXED_ZERO_BLOCK: {
            "L_wall_0_side",
            "L_box_0_half",
            "t_w_box_0",
            "ch_box_0_p",
            "t_w_ch_0_s",
            "t_w_ch_1_s",
            "t_w_ch_2_s",
            "x_b_lin_end_0",
        },
    }[layout.kind]
    for key in sorted(variant_required):
        if key not in longitudinal or longitudinal[key] is None:
            diagnostics.append(
                Diagnostic("missing_parameter", f"平面缺少纵向参数：{key}")
            )
    if diagnostics:
        return diagnostics

    segments = longitudinal["L_seg_var_seq"]
    web_values = longitudinal["t_w_var_seq"]
    if (
        not isinstance(segments, list)
        or not segments
        or not all(_finite_positive(item) for item in segments)
    ):
        diagnostics.append(
            Diagnostic("invalid_segment_sequence", "L_seg_var_seq必须是正数序列。")
        )
    if (
        not isinstance(web_values, list)
        or not all(_finite_positive(item) for item in web_values)
        or not isinstance(segments, list)
        or len(web_values) != len(segments) + 1
    ):
        diagnostics.append(
            Diagnostic(
                "invalid_web_sequence",
                "t_w_var_seq长度必须等于L_seg_var_seq长度加1。",
            )
        )

    numeric_longitudinal = required_longitudinal - {
        "L_seg_var_seq",
        "t_w_var_seq",
        "n_box_0_side",
    }
    for key in sorted(numeric_longitudinal):
        if not _finite_positive(longitudinal[key]):
            diagnostics.append(
                Diagnostic("invalid_number", f"平面参数{key}必须是有限正数。")
            )
    for key in sorted(required_cross):
        if not _finite_positive(cross[key]):
            diagnostics.append(
                Diagnostic("invalid_number", f"平面参数{key}必须是有限正数。")
            )
    pair_keys = set()
    if layout.kind == TRANSITION_ZERO_BLOCK:
        pair_keys.add("ch_0_p")
    elif layout.kind == BOXED_ZERO_BLOCK:
        pair_keys.add("ch_box_0_p")
    for key in sorted(variant_required - pair_keys):
        if not _finite_positive(longitudinal[key]):
            diagnostics.append(
                Diagnostic("invalid_number", f"平面参数{key}必须是有限正数。")
            )
    for key in sorted(pair_keys):
        value = longitudinal[key]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(_finite_positive(item) for item in value)
        ):
            diagnostics.append(
                Diagnostic("invalid_pair", f"平面参数{key}必须是两个正数组成的尺寸对。")
            )
    end_chamfer = longitudinal.get("ch_end_p")
    if end_chamfer is not None and not _finite_positive_pair(end_chamfer):
        diagnostics.append(
            Diagnostic(
                "invalid_pair",
                "平面参数ch_end_p必须为null或两个正数组成的尺寸对。",
            )
        )
    if (
        layout.kind == BOXED_ZERO_BLOCK
        and _finite_positive_pair(longitudinal.get("ch_box_0_p"))
        and layout.box_start is not None
        and layout.box_end is not None
        and float(longitudinal["ch_box_0_p"][0])
        > layout.box_end - layout.box_start
    ):
        diagnostics.append(
            Diagnostic(
                "box_chamfer_exceeds_half_length",
                "平面参数ch_box_0_p的纵向宽度不能大于箱室半长。",
            )
        )
    if diagnostics:
        return diagnostics

    if layout.kind == THICKENED_ZERO_BLOCK:
        transition_end = float(longitudinal["L_wall_0"]) + float(
            longitudinal["t_w_ch_2_s"]
        )
        if not math.isclose(
            transition_end,
            float(longitudinal["x_b_lin_end_0"]),
            abs_tol=1.0,
        ):
            diagnostics.append(
                Diagnostic(
                    "middle_web_transition_extent_mismatch",
                    "中支点腹板变厚段必须终止于x_b_lin_end_0。",
                )
            )
    elif layout.kind == BOXED_ZERO_BLOCK and layout.wall_end is not None:
        transition_end = layout.wall_end + float(longitudinal["t_w_ch_2_s"])
        if not math.isclose(
            transition_end,
            layout.zero_block_end,
            abs_tol=1.0,
        ):
            diagnostics.append(
                Diagnostic(
                    "middle_web_transition_extent_mismatch",
                    "有箱室0号块外侧腹板变厚段必须终止于L_0_half。",
                )
            )

    if layout.kind in {THICKENED_ZERO_BLOCK, BOXED_ZERO_BLOCK} and not math.isclose(
        float(web_values[0]),
        float(longitudinal["t_w_ch_1_s"]),
        abs_tol=0.01,
    ):
        diagnostics.append(
            Diagnostic(
                "web_sequence_start_mismatch",
                "t_w_var_seq首值必须等于0号块外侧腹板厚度。",
            )
        )
    if not math.isclose(
        float(web_values[-1]),
        float(longitudinal["t_w_std"]),
        abs_tol=0.01,
    ):
        diagnostics.append(
            Diagnostic(
                "web_sequence_end_mismatch",
                "t_w_var_seq末值必须等于一般腹板厚度。",
            )
        )

    sequence_end = layout.sequence_start + sum(float(item) for item in segments)
    if not math.isclose(sequence_end, layout.curve_end, abs_tol=1.0):
        diagnostics.append(
            Diagnostic(
                "web_sequence_extent_mismatch",
                "腹板分段序列必须终止于变高度段终点。",
            )
        )

    half_width = float(cross["B_bottom"]) / 2.0
    thicknesses = [
        float(longitudinal["t_w_end"]),
        float(longitudinal["t_w_std"]),
        *[float(item) for item in web_values],
    ]
    if layout.kind == THICKENED_ZERO_BLOCK:
        thicknesses.extend(
            [
                float(longitudinal["t_w_ch_0_s"]),
                float(longitudinal["t_w_ch_1_s"]),
            ]
        )
    elif layout.kind == TRANSITION_ZERO_BLOCK:
        thicknesses.append(
            float(web_values[0]) + float(longitudinal["ch_0_p"][1])
        )
    else:
        thicknesses.extend(
            [
                float(longitudinal["t_w_ch_0_s"]),
                float(longitudinal["t_w_box_0"])
                + float(longitudinal["ch_box_0_p"][1]),
            ]
        )
    if max(thicknesses) >= half_width:
        diagnostics.append(
            Diagnostic(
                "web_thickness_exceeds_half_width",
                "腹板厚度必须小于梁底半宽。",
            )
        )
    if any(
        float(cross[key]) >= float(cross["B_bottom"])
        for key in ("B_hole_end", "B_hole_0")
    ):
        diagnostics.append(
            Diagnostic(
                "manhole_width_exceeds_plan_width",
                "平面人洞宽度必须小于梁底宽度。",
            )
        )

    if isinstance(spans, list) and len(spans) == 3:
        left_extent = float(spans[0]) * 1000.0 + float(
            longitudinal["L_end_offset"]
        )
        mirrored_extent = left_extent - (
            float(longitudinal["L_cast_side"])
            + float(longitudinal["L_closure_side"]) / 2.0
        )
        middle_half = float(spans[1]) * 500.0
        if not math.isclose(mirrored_extent, middle_half, abs_tol=1.0):
            diagnostics.append(
                Diagnostic(
                    "mirror_extent_mismatch",
                    "边跨合拢段中心距中支点必须等于中跨跨度的一半。",
                )
            )
    return diagnostics


def generate_plan_stations(
    parameters: DesignParameters,
) -> list[PlanStation]:
    diagnostics = validate_plan_parameters(parameters)
    if diagnostics:
        raise ValueError("；".join(item.message for item in diagnostics))

    longitudinal = parameters.longitudinal
    layout = resolve_longitudinal_layout(parameters)
    spans = parameters.global_parameters["span_layout"]
    x_values = _station_x_values(spans, longitudinal, layout)
    beam_end = min(x_values)
    return [
        PlanStation(
            id="pl_station_{:03d}".format(index),
            x=_round(x),
            web_thickness=_round(
                _web_thickness_at(x, beam_end, longitudinal, layout)
            ),
        )
        for index, x in enumerate(x_values, start=1)
    ]


def generate_plan(parameters: DesignParameters) -> GeometryDocument:
    stations = generate_plan_stations(parameters)
    longitudinal = parameters.longitudinal
    cross = parameters.cross_section
    half_width = float(cross["B_bottom"]) / 2.0
    layout = resolve_longitudinal_layout(parameters)
    wall_half = layout.wall_half
    is_three_span = len(parameters.global_parameters["span_layout"]) == 3
    left_end = stations[0].x
    right_end = stations[-1].x
    end_wall_inner = left_end + float(longitudinal["L_diaphragm_end"])

    boundary_function = (
        _three_span_inner_boundary
        if is_three_span
        else (
            _two_span_boxed_inner_boundary
            if layout.kind == BOXED_ZERO_BLOCK
            else _two_span_inner_boundary
        )
    )
    boundary_arguments = {
        "half_width": half_width,
        "end_wall_inner": end_wall_inner,
        "wall_half": wall_half,
        "end_hole_y": float(cross["B_hole_end"]) / 2.0,
        "middle_hole_y": float(cross["B_hole_0"]) / 2.0,
        "upper": True,
        "end_chamfer": longitudinal.get("ch_end_p"),
    }
    if layout.kind == BOXED_ZERO_BLOCK:
        boundary_arguments["box_end"] = layout.box_end
        boundary_arguments["wall_end"] = layout.wall_end
    upper_inner = boundary_function(
        stations,
        **boundary_arguments,
    )
    boundary_arguments.update(
        {
            "end_hole_y": -float(cross["B_hole_end"]) / 2.0,
            "middle_hole_y": -float(cross["B_hole_0"]) / 2.0,
            "upper": False,
        }
    )
    lower_inner = boundary_function(
        stations,
        **boundary_arguments,
    )

    builder = _PlanBuilder()
    upper_points = [
        *upper_inner,
        Point(right_end, half_width),
        Point(left_end, half_width),
    ]
    upper_layers = [
        *["PL-VOID"] * (len(upper_inner) - 1),
        "PL-CUT",      # 跨中 — 取半桥，真正截断
        "PL-OUTLINE",  # 外侧边
        "PL-OUTLINE",  # 梁端 — 实际梁终点，不是截断
    ]
    upper_loop = builder.add_polygon(upper_points, upper_layers)

    lower_points = [
        Point(left_end, -half_width),
        Point(right_end, -half_width),
        *list(reversed(lower_inner)),
    ]
    lower_layers = [
        "PL-OUTLINE",  # 外侧边
        "PL-CUT",      # 跨中 — 取半桥，真正截断
        *["PL-VOID"] * (len(lower_inner) - 1),
        "PL-OUTLINE",  # 梁端 — 实际梁终点，不是截断
    ]
    lower_loop = builder.add_polygon(lower_points, lower_layers)

    prefix = builder.prefix
    section = SectionGeometry(
        id="HALF-GIRDER-PLAN",
        lines=builder.lines,
        arcs=[],
        regions=[
            Region(
                id=f"{prefix}_region_001",
                layer="PL-CONCRETE",
                outer=upper_loop,
            ),
            Region(
                id=f"{prefix}_region_002",
                layer="PL-CONCRETE",
                outer=lower_loop,
            ),
        ],
    )
    return GeometryDocument(
        source=dict(parameters.source),
        section=section,
        view_name="plan",
        coordinate_system={
            "origin": "中支点中心与梁纵向中心线交点",
            "x_axis": "+X 由中支点指向中跨" if is_three_span else "+X 由梁端指向中支点",
            "y_axis": "+Y 横桥向向左",
        },
    )


def _three_span_inner_boundary(
    stations: list[PlanStation],
    *,
    half_width: float,
    end_wall_inner: float,
    wall_half: float,
    end_hole_y: float,
    middle_hole_y: float,
    upper: bool,
    end_chamfer: list[float] | None,
) -> list[Point]:
    left_end = stations[0].x
    right_end = stations[-1].x
    result = [Point(left_end, end_hole_y), Point(end_wall_inner, end_hole_y)]

    def boundary_y(station: PlanStation) -> float:
        return _inner_y(station, half_width, upper)

    cavity_start = _append_left_end_connection(
        result,
        stations,
        end_wall_inner,
        boundary_y,
        upper,
        end_chamfer,
    )
    for station in stations:
        if cavity_start < station.x <= -wall_half:
            _append_point(
                result,
                Point(station.x, boundary_y(station)),
            )
    left_wall_station = _interpolate_station(stations, -wall_half)
    _append_point(
        result,
        Point(-wall_half, _inner_y(left_wall_station, half_width, upper)),
    )
    _append_point(result, Point(-wall_half, middle_hole_y))
    _append_point(result, Point(wall_half, middle_hole_y))
    right_wall_station = _interpolate_station(stations, wall_half)
    _append_point(
        result,
        Point(wall_half, _inner_y(right_wall_station, half_width, upper)),
    )
    for station in stations:
        if wall_half < station.x <= right_end:
            _append_point(
                result,
                Point(station.x, _inner_y(station, half_width, upper)),
            )
    return result


def _two_span_inner_boundary(
    stations: list[PlanStation],
    *,
    half_width: float,
    end_wall_inner: float,
    wall_half: float,
    end_hole_y: float,
    middle_hole_y: float,
    upper: bool,
    end_chamfer: list[float] | None,
) -> list[Point]:
    left_end = stations[0].x
    right_crop = stations[-1].x
    result = [Point(left_end, end_hole_y), Point(end_wall_inner, end_hole_y)]

    def boundary_y(station: PlanStation) -> float:
        return _inner_y(station, half_width, upper)

    cavity_start = _append_left_end_connection(
        result,
        stations,
        end_wall_inner,
        boundary_y,
        upper,
        end_chamfer,
    )
    for station in stations:
        if cavity_start < station.x <= -wall_half:
            _append_point(
                result,
                Point(station.x, boundary_y(station)),
            )
    wall_station = _interpolate_station(stations, -wall_half)
    _append_point(result, Point(-wall_half, boundary_y(wall_station)))
    _append_point(result, Point(-wall_half, middle_hole_y))
    _append_point(result, Point(right_crop, middle_hole_y))
    return result


def _two_span_boxed_inner_boundary(
    stations: list[PlanStation],
    *,
    half_width: float,
    end_wall_inner: float,
    wall_half: float,
    box_end: float | None,
    wall_end: float | None,
    end_hole_y: float,
    middle_hole_y: float,
    upper: bool,
    end_chamfer: list[float] | None,
) -> list[Point]:
    if box_end is None or wall_end is None:
        raise ValueError("有箱室0号块缺少箱室或横墙范围。")
    left_end = stations[0].x
    right_crop = stations[-1].x

    def boundary_y(station: PlanStation) -> float:
        return _inner_y(station, half_width, upper)

    result = [Point(left_end, end_hole_y), Point(end_wall_inner, end_hole_y)]
    cavity_start = _append_left_end_connection(
        result,
        stations,
        end_wall_inner,
        boundary_y,
        upper,
        end_chamfer,
    )
    for station in stations:
        if cavity_start < station.x <= -wall_end:
            _append_point(
                result,
                Point(station.x, boundary_y(station)),
            )
    wall_end_station = _interpolate_station(stations, -wall_end)
    _append_point(
        result,
        Point(-wall_end, boundary_y(wall_end_station)),
    )
    _append_point(result, Point(-wall_end, middle_hole_y))
    _append_point(result, Point(-box_end, middle_hole_y))
    box_end_station = _interpolate_station(stations, -box_end)
    _append_point(result, Point(-box_end, boundary_y(box_end_station)))
    for station in stations:
        if -box_end < station.x <= right_crop:
            _append_point(
                result,
                Point(station.x, boundary_y(station)),
            )
    if not math.isclose(result[-1].x, right_crop, abs_tol=1e-6):
        crop_station = _interpolate_station(stations, right_crop)
        _append_point(result, Point(right_crop, boundary_y(crop_station)))
    return result


def _append_left_end_connection(
    points: list[Point],
    stations: list[PlanStation],
    wall_x: float,
    boundary_y: Any,
    upper: bool,
    chamfer: list[float] | None,
) -> float:
    if chamfer is None:
        wall_station = _interpolate_station(stations, wall_x)
        _append_point(points, Point(wall_x, boundary_y(wall_station)))
        return wall_x

    cavity_x = wall_x + float(chamfer[0])
    cavity_station = _interpolate_station(stations, cavity_x)
    cavity_y = boundary_y(cavity_station)
    wall_y = cavity_y - float(chamfer[1]) if upper else cavity_y + float(
        chamfer[1]
    )
    _append_point(points, Point(wall_x, wall_y))
    _append_point(points, Point(cavity_x, cavity_y))
    return cavity_x


def _inner_y(
    station: PlanStation,
    half_width: float,
    upper: bool,
) -> float:
    value = half_width - station.web_thickness
    return value if upper else -value


def _interpolate_station(
    stations: list[PlanStation],
    x: float,
) -> PlanStation:
    for station in stations:
        if math.isclose(station.x, x, abs_tol=1e-6):
            return station
    for first, second in zip(stations, stations[1:]):
        if first.x < x < second.x:
            return PlanStation(
                id="interpolated",
                x=x,
                web_thickness=_interpolate(
                    x,
                    first.x,
                    second.x,
                    first.web_thickness,
                    second.web_thickness,
                ),
            )
    raise ValueError(f"纵向站点范围不包含x={x}。")


def _append_point(points: list[Point], point: Point) -> None:
    if not points or points[-1].distance_to(point) > 1e-9:
        points.append(point)


def _station_x_values(
    spans: list[Any],
    longitudinal: dict[str, Any],
    layout: LongitudinalLayout,
) -> list[float]:
    if layout.kind == THICKENED_ZERO_BLOCK:
        transition_end = layout.wall_half + float(
            longitudinal["t_w_ch_2_s"]
        )
    elif layout.kind == TRANSITION_ZERO_BLOCK:
        transition_end = layout.wall_half + float(longitudinal["ch_0_p"][0])
    else:
        transition_end = (
            layout.wall_end
            if layout.wall_end is not None
            else layout.zero_block_end
        )
    sequence_start = layout.sequence_start
    common_distances = {
        0.0,
        transition_end,
        sequence_start,
    }
    if layout.kind != BOXED_ZERO_BLOCK:
        common_distances.add(layout.wall_half)
    if layout.kind == BOXED_ZERO_BLOCK:
        if (
            layout.box_start is None
            or layout.box_end is None
            or layout.wall_start is None
            or layout.wall_end is None
        ):
            raise ValueError("有箱室0号块缺少箱室或横墙范围。")
        chamfer_width = float(longitudinal["ch_box_0_p"][0])
        common_distances.update(
            {
                layout.box_start,
                layout.box_end - chamfer_width,
                layout.box_end,
                layout.wall_start,
                layout.wall_end,
                layout.zero_block_end,
            }
        )
    cumulative = sequence_start
    for length in longitudinal["L_seg_var_seq"]:
        cumulative += float(length)
        common_distances.add(cumulative)

    end_offset = float(longitudinal["L_end_offset"])
    if len(spans) == 3:
        left_extent = float(spans[0]) * 1000.0 + end_offset
        right_extent = float(spans[1]) * 500.0
        left_end = -left_extent
        result = {
            left_end,
            right_extent,
            -float(spans[0]) * 1000.0,
            left_end + float(longitudinal["L_diaphragm_end"]),
            left_end
            + float(longitudinal["L_diaphragm_end"])
            + float(longitudinal["L_w_end_var"]),
            left_end + float(longitudinal["L_cast_side"]),
            left_end
            + float(longitudinal["L_cast_side"])
            + float(longitudinal["L_closure_side"]) / 2.0,
            left_end
            + float(longitudinal["L_cast_side"])
            + float(longitudinal["L_closure_side"]),
        }
        for distance in common_distances:
            if distance <= left_extent:
                result.add(-distance)
            if distance <= right_extent:
                result.add(distance)
        return sorted(result)

    right_extent = float(spans[0]) * 1000.0 + end_offset
    result = {0.0, float(spans[0]) * 1000.0, right_extent}
    for distance in common_distances:
        if distance <= right_extent:
            result.add(distance)
    result.update(
        {
            right_extent - float(longitudinal["L_diaphragm_end"]),
            right_extent
            - float(longitudinal["L_diaphragm_end"])
            - float(longitudinal["L_w_end_var"]),
            right_extent - float(longitudinal["L_cast_side"]),
            right_extent
            - float(longitudinal["L_cast_side"])
            - float(longitudinal["L_closure_side"]) / 2.0,
            right_extent
            - float(longitudinal["L_cast_side"])
            - float(longitudinal["L_closure_side"]),
        }
    )
    return sorted(-value for value in result)


def _web_thickness_at(
    x: float,
    beam_end: float,
    longitudinal: dict[str, Any],
    layout: LongitudinalLayout,
) -> float:
    distance_from_beam_end = abs(x - beam_end)
    diaphragm_length = float(longitudinal["L_diaphragm_end"])
    if distance_from_beam_end <= diaphragm_length:
        return float(longitudinal["t_w_end"])

    end_transition = float(longitudinal["L_w_end_var"])
    transition_distance = distance_from_beam_end - diaphragm_length
    if transition_distance <= end_transition:
        return _interpolate(
            transition_distance,
            0.0,
            end_transition,
            float(longitudinal["t_w_end"]),
            float(longitudinal["t_w_std"]),
        )

    distance = abs(x)
    if layout.kind == BOXED_ZERO_BLOCK:
        return _boxed_web_thickness(distance, longitudinal, layout)
    wall_half = layout.wall_half
    if layout.kind == TRANSITION_ZERO_BLOCK:
        chamfer = longitudinal["ch_0_p"]
        middle_transition_end = wall_half + float(chamfer[0])
        middle_start = float(longitudinal["t_w_var_seq"][0]) + float(
            chamfer[1]
        )
        middle_end = float(longitudinal["t_w_var_seq"][0])
    else:
        middle_transition_end = wall_half + float(
            longitudinal["t_w_ch_2_s"]
        )
        middle_start = float(longitudinal["t_w_ch_0_s"])
        middle_end = float(longitudinal["t_w_ch_1_s"])
    if distance <= wall_half:
        return middle_start
    if distance <= middle_transition_end:
        return _interpolate(
            distance,
            wall_half,
            middle_transition_end,
            middle_start,
            middle_end,
        )

    sequence_start = layout.sequence_start
    if distance <= sequence_start:
        return middle_end

    position = sequence_start
    values = longitudinal["t_w_var_seq"]
    for index, length in enumerate(longitudinal["L_seg_var_seq"]):
        next_position = position + float(length)
        if distance <= next_position:
            return _interpolate(
                distance,
                position,
                next_position,
                float(values[index]),
                float(values[index + 1]),
            )
        position = next_position
    return float(longitudinal["t_w_std"])


def _boxed_web_thickness(
    distance: float,
    longitudinal: dict[str, Any],
    layout: LongitudinalLayout,
) -> float:
    if (
        layout.box_start is None
        or layout.box_end is None
        or layout.wall_end is None
    ):
        raise ValueError("有箱室0号块缺少箱室或横墙范围。")
    if distance <= layout.box_end:
        chamfer = longitudinal["ch_box_0_p"]
        edge = float(longitudinal["t_w_box_0"]) + float(chamfer[1])
        flat = float(longitudinal["t_w_box_0"])
        return _boxed_half_chamfer_value(
            distance,
            layout.box_end,
            float(chamfer[0]),
            flat,
            edge,
        )
    if distance < layout.wall_end:
        return float(longitudinal["t_w_ch_0_s"])
    if distance < layout.sequence_start:
        return _interpolate(
            distance,
            layout.wall_end,
            layout.sequence_start,
            float(longitudinal["t_w_ch_0_s"]),
            float(longitudinal["t_w_ch_1_s"]),
        )

    position = layout.sequence_start
    values = longitudinal["t_w_var_seq"]
    for index, length in enumerate(longitudinal["L_seg_var_seq"]):
        next_position = position + float(length)
        if distance <= next_position:
            return _interpolate(
                distance,
                position,
                next_position,
                float(values[index]),
                float(values[index + 1]),
            )
        position = next_position
    return float(longitudinal["t_w_std"])


def _boxed_half_chamfer_value(
    distance: float,
    end: float,
    chamfer_width: float,
    flat_value: float,
    edge_value: float,
) -> float:
    chamfer_start = end - chamfer_width
    if distance >= chamfer_start:
        return _interpolate(
            distance,
            chamfer_start,
            end,
            flat_value,
            edge_value,
        )
    return flat_value


def _interpolate(
    value: float,
    start: float,
    end: float,
    start_value: float,
    end_value: float,
) -> float:
    if math.isclose(start, end):
        return end_value
    ratio = (value - start) / (end - start)
    return start_value + ratio * (end_value - start_value)


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_positive_pair(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_finite_positive(item) for item in value)
    )


def _round(value: float) -> float:
    result = round(float(value), 6)
    return 0.0 if result == -0.0 else result
