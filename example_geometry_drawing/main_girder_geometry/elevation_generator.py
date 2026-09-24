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
class LongitudinalStation:
    id: str
    x: float
    height: float
    top_thickness: float
    bottom_thickness: float

    @property
    def outer_bottom_y(self) -> float:
        return -self.height

    @property
    def void_top_y(self) -> float:
        return -self.top_thickness

    @property
    def void_bottom_y(self) -> float:
        return -self.height + self.bottom_thickness


class _ElevationBuilder:
    def __init__(self) -> None:
        self.prefix = "el_half_girder_elevation"
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
        refs: list[EntityRef] = []
        for index, start in enumerate(points):
            refs.append(
                self.add_line(
                    start,
                    points[(index + 1) % len(points)],
                    layers[index],
                )
            )
        return tuple(refs)


def validate_elevation_parameters(
    parameters: DesignParameters,
) -> list[Diagnostic]:
    global_parameters = parameters.global_parameters
    longitudinal = parameters.longitudinal
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
                "立面首版要求span_layout为两个或三个正跨度。",
            )
        )

    required = {
        "L_end_offset",
        "L_diaphragm_end",
        "L_t_end_var",
        "t_t_end",
        "t_t_std",
        "L_b_end_var",
        "t_b_end",
        "t_b_std",
        "L_closure_side",
        "L_cast_side",
        "L_const_side",
        "L_seg_var_seq",
        "t_b_var_seq",
        "L_H_var",
        "a_H",
        "n_H",
        "H_pier",
        "H_end",
        "L_0_half",
        "n_box_0_side",
    }
    for key in sorted(required):
        if key not in longitudinal or longitudinal[key] is None:
            diagnostics.append(
                Diagnostic("missing_parameter", f"立面缺少纵向参数：{key}")
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
            "t_t_ch_0_s",
            "t_t_ch_0_e",
            "t_b_ch_0_s",
            "t_b_ch_0_m",
            "t_b_trans_0_s",
            "t_b_trans_0_e",
            "x_b_lin_end_0",
            "L_trans_0",
        },
        TRANSITION_ZERO_BLOCK: {
            "L_wall_0",
            "ch_t_0_v",
            "t_t_trans_0",
            "t_b_trans_0_s",
            "t_b_trans_0_e",
            "L_trans_0",
        },
        BOXED_ZERO_BLOCK: {
            "L_wall_0_side",
            "L_box_0_half",
            "t_t_box_0",
            "t_b_box_0",
            "ch_t_box_0_v",
            "ch_b_box_0_v",
            "t_t_ch_0_s",
            "t_t_ch_0_e",
            "t_b_ch_0_s",
            "t_b_ch_0_e",
            "t_b_ch_0_m",
            "x_b_lin_end_0",
        },
    }[layout.kind]
    for key in sorted(variant_required):
        if key not in longitudinal or longitudinal[key] is None:
            diagnostics.append(
                Diagnostic("missing_parameter", f"立面缺少纵向参数：{key}")
            )
    if diagnostics:
        return diagnostics

    segments = longitudinal["L_seg_var_seq"]
    bottom_values = longitudinal["t_b_var_seq"]
    if (
        not isinstance(segments, list)
        or not segments
        or not all(_finite_positive(item) for item in segments)
    ):
        diagnostics.append(
            Diagnostic("invalid_segment_sequence", "L_seg_var_seq必须是正数序列。")
        )
    if (
        not isinstance(bottom_values, list)
        or not all(_finite_positive(item) for item in bottom_values)
        or not isinstance(segments, list)
        or len(bottom_values) != len(segments) + 1
    ):
        diagnostics.append(
            Diagnostic(
                "invalid_bottom_sequence",
                "t_b_var_seq长度必须等于L_seg_var_seq长度加1。",
            )
        )

    numeric_keys = required - {"L_seg_var_seq", "t_b_var_seq"}
    for key in sorted(numeric_keys):
        value = longitudinal[key]
        if key == "n_box_0_side":
            continue
        if not _finite_positive(value):
            diagnostics.append(
                Diagnostic("invalid_number", f"立面参数{key}必须是有限正数。")
            )

    pair_keys = set()
    if layout.kind == TRANSITION_ZERO_BLOCK:
        pair_keys.add("ch_t_0_v")
    elif layout.kind == BOXED_ZERO_BLOCK:
        pair_keys.update({"ch_t_box_0_v", "ch_b_box_0_v"})
    for key in sorted(variant_required - pair_keys):
        if not _finite_positive(longitudinal[key]):
            diagnostics.append(
                Diagnostic("invalid_number", f"立面参数{key}必须是有限正数。")
            )
    for key in sorted(pair_keys):
        value = longitudinal[key]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(_finite_positive(item) for item in value)
        ):
            diagnostics.append(
                Diagnostic("invalid_pair", f"立面参数{key}必须是两个正数组成的尺寸对。")
            )

    end_chamfer = longitudinal.get("ch_end_v")
    if end_chamfer is not None and not _finite_positive_pair(end_chamfer):
        diagnostics.append(
            Diagnostic(
                "invalid_pair",
                "立面参数ch_end_v必须为null或两个正数组成的尺寸对。",
            )
        )
    if (
        layout.kind == BOXED_ZERO_BLOCK
        and _finite_positive_pair(longitudinal.get("ch_t_box_0_v"))
        and _finite_positive_pair(longitudinal.get("ch_b_box_0_v"))
        and layout.box_start is not None
        and layout.box_end is not None
    ):
        box_half_length = layout.box_end - layout.box_start
        for key in ("ch_t_box_0_v", "ch_b_box_0_v"):
            if float(longitudinal[key][0]) > box_half_length:
                diagnostics.append(
                    Diagnostic(
                        "box_chamfer_exceeds_half_length",
                        f"立面参数{key}的纵向宽度不能大于箱室半长。",
                    )
                )

    if diagnostics:
        return diagnostics

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
        if layout.kind == THICKENED_ZERO_BLOCK:
            side_straight = float(spans[0]) * 1000.0 - layout.curve_end
            if not math.isclose(
                side_straight,
                float(longitudinal["L_const_side"]),
                abs_tol=1.0,
            ):
                diagnostics.append(
                    Diagnostic(
                        "side_straight_length_mismatch",
                        "边支点至变高度段终点的距离必须等于L_const_side。",
                    )
                )
            middle_straight = middle_half - layout.curve_end
            if not math.isclose(
                middle_straight,
                float(longitudinal["L_closure_side"]) / 2.0,
                abs_tol=1.0,
            ):
                diagnostics.append(
                    Diagnostic(
                        "middle_straight_length_mismatch",
                        "变高度段终点至中跨跨中的距离必须等于半个合拢段。",
                    )
                )
        elif layout.kind == TRANSITION_ZERO_BLOCK:
            side_straight = left_extent - layout.curve_end
            if not math.isclose(
                side_straight,
                float(longitudinal["L_const_side"]),
                abs_tol=1.0,
            ):
                diagnostics.append(
                    Diagnostic(
                        "side_straight_length_mismatch",
                        "梁端至变高度段终点的距离必须等于L_const_side。",
                    )
                )

    bottom_sequence_end = layout.sequence_start + sum(
        float(item) for item in longitudinal["L_seg_var_seq"]
    )
    if not math.isclose(bottom_sequence_end, layout.curve_end, abs_tol=1.0):
        diagnostics.append(
            Diagnostic(
                "bottom_sequence_extent_mismatch",
                "底板过渡段与分段长度合计必须终止于变高度段终点。",
            )
        )

    expected_sequence_start = (
        float(longitudinal["t_b_ch_0_e"])
        if layout.kind == BOXED_ZERO_BLOCK
        else float(longitudinal["t_b_trans_0_e"])
    )
    if not math.isclose(
        float(bottom_values[0]),
        expected_sequence_start,
        abs_tol=0.01,
    ):
        diagnostics.append(
            Diagnostic(
                "bottom_sequence_start_mismatch",
                "t_b_var_seq首值必须等于0号块过渡终点底板厚度。",
            )
        )

    height_difference = float(longitudinal["H_pier"]) - float(
        longitudinal["H_end"]
    )
    coefficient_height_difference = 1000.0 * float(
        longitudinal["a_H"]
    ) * math.pow(
        float(longitudinal["L_H_var"]) / 1000.0,
        float(longitudinal["n_H"]),
    )
    coefficient_tolerance = max(1.0, abs(height_difference) * 0.01)
    if not math.isclose(
        coefficient_height_difference,
        height_difference,
        abs_tol=coefficient_tolerance,
    ):
        diagnostics.append(
            Diagnostic(
                "height_curve_coefficient_mismatch",
                "a_H、n_H与L_H_var计算的梁高差必须和H_pier-H_end一致。",
            )
        )
    return diagnostics


def generate_elevation_stations(
    parameters: DesignParameters,
) -> list[LongitudinalStation]:
    diagnostics = validate_elevation_parameters(parameters)
    if diagnostics:
        raise ValueError("；".join(item.message for item in diagnostics))

    longitudinal = parameters.longitudinal
    layout = resolve_longitudinal_layout(parameters)
    spans = parameters.global_parameters["span_layout"]
    x_values = _station_x_values(spans, longitudinal, layout)
    left_end = min(x_values)
    beam_end = left_end
    stations: list[LongitudinalStation] = []
    for index, x in enumerate(x_values, start=1):
        stations.append(
            LongitudinalStation(
                id="el_station_{:03d}".format(index),
                x=_round(x),
                height=_round(_height_at(abs(x), longitudinal, layout)),
                top_thickness=_round(
                    _top_thickness_at(x, beam_end, longitudinal, layout)
                ),
                bottom_thickness=_round(
                    _bottom_thickness_at(x, beam_end, longitudinal, layout)
                ),
            )
        )
    return stations


def generate_elevation(parameters: DesignParameters) -> GeometryDocument:
    stations = generate_elevation_stations(parameters)
    longitudinal = parameters.longitudinal
    cross = parameters.cross_section
    required_cross = ("H_hole_end", "Z_hole_end", "H_hole_0", "Z_hole_0")
    missing = [
        key
        for key in required_cross
        if key not in cross or not _finite_positive(cross[key])
    ]
    if missing:
        raise ValueError(f"立面缺少有效人洞参数：{', '.join(missing)}")

    left_end = stations[0].x
    right_end = stations[-1].x
    layout = resolve_longitudinal_layout(parameters)
    wall_half = layout.wall_half
    is_three_span = len(parameters.global_parameters["span_layout"]) == 3
    end_wall_inner = left_end + float(longitudinal["L_diaphragm_end"])

    end_hole_bottom = -float(longitudinal["H_end"]) + float(
        cross["Z_hole_end"]
    )
    end_hole_top = end_hole_bottom + float(cross["H_hole_end"])
    middle_hole_bottom = -float(longitudinal["H_pier"]) + float(
        cross["Z_hole_0"]
    )
    middle_hole_top = middle_hole_bottom + float(cross["H_hole_0"])

    boundary_function = (
        _three_span_void_boundary
        if is_three_span
        else (
            _two_span_boxed_void_boundary
            if layout.kind == BOXED_ZERO_BLOCK
            else _two_span_void_boundary
        )
    )
    boundary_arguments = {
        "end_wall_inner": end_wall_inner,
        "wall_half": wall_half,
        "end_hole_y": end_hole_top,
        "middle_hole_y": middle_hole_top,
        "use_top": True,
        "end_chamfer": longitudinal.get("ch_end_v"),
    }
    if layout.kind == BOXED_ZERO_BLOCK:
        boundary_arguments["box_end"] = layout.box_end
        boundary_arguments["wall_end"] = layout.wall_end
    upper_void = boundary_function(
        stations,
        **boundary_arguments,
    )
    boundary_arguments.update(
        {
            "end_hole_y": end_hole_bottom,
            "middle_hole_y": middle_hole_bottom,
            "use_top": False,
        }
    )
    lower_void = boundary_function(
        stations,
        **boundary_arguments,
    )

    builder = _ElevationBuilder()
    upper_points = [Point(left_end, 0.0), *upper_void, Point(right_end, 0.0)]
    upper_layers = [
        "EL-OUTLINE",  # 梁端 — 实际梁终点，不是截断
        *["EL-VOID"] * (len(upper_void) - 1),
        "EL-CUT",      # 跨中 — 取半桥，真正截断
        "EL-OUTLINE",
    ]
    upper_loop = builder.add_polygon(upper_points, upper_layers)

    outer_bottom = [
        Point(item.x, item.outer_bottom_y)
        for item in stations
    ]
    lower_points = [
        lower_void[0],
        *outer_bottom,
        *list(reversed(lower_void[1:])),
    ]
    lower_layers = [
        "EL-OUTLINE",  # 梁端 — 实际梁终点，不是截断
        *["EL-OUTLINE"] * (len(outer_bottom) - 1),
        "EL-CUT",      # 跨中 — 取半桥，真正截断
        *["EL-VOID"] * (len(lower_void) - 1),
    ]
    lower_loop = builder.add_polygon(lower_points, lower_layers)

    prefix = builder.prefix
    section = SectionGeometry(
        id="HALF-GIRDER-ELEVATION",
        lines=builder.lines,
        arcs=[],
        regions=[
            Region(
                id=f"{prefix}_region_001",
                layer="EL-CONCRETE",
                outer=upper_loop,
            ),
            Region(
                id=f"{prefix}_region_002",
                layer="EL-CONCRETE",
                outer=lower_loop,
            ),
        ],
    )
    return GeometryDocument(
        source=dict(parameters.source),
        section=section,
        view_name="elevation",
        coordinate_system={
            "origin": "中支点中心与梁顶面交点",
            "x_axis": "+X 由中支点指向中跨" if is_three_span else "+X 由梁端指向中支点",
            "y_axis": "+Y 向上",
        },
    )


def _three_span_void_boundary(
    stations: list[LongitudinalStation],
    *,
    end_wall_inner: float,
    wall_half: float,
    end_hole_y: float,
    middle_hole_y: float,
    use_top: bool,
    end_chamfer: list[float] | None,
) -> list[Point]:
    left_end = stations[0].x
    right_end = stations[-1].x
    result = [Point(left_end, end_hole_y), Point(end_wall_inner, end_hole_y)]

    def boundary_y(station: LongitudinalStation) -> float:
        return station.void_top_y if use_top else station.void_bottom_y

    cavity_start = _append_left_end_connection(
        result,
        stations,
        end_wall_inner,
        boundary_y,
        use_top,
        end_chamfer,
    )
    for station in stations:
        if cavity_start < station.x <= -wall_half:
            _append_point(result, Point(station.x, boundary_y(station)))
    left_wall_station = _interpolate_station(stations, -wall_half)
    _append_point(result, Point(-wall_half, boundary_y(left_wall_station)))
    _append_point(result, Point(-wall_half, middle_hole_y))
    _append_point(result, Point(wall_half, middle_hole_y))
    right_wall_station = _interpolate_station(stations, wall_half)
    _append_point(result, Point(wall_half, boundary_y(right_wall_station)))
    for station in stations:
        if wall_half < station.x <= right_end:
            _append_point(result, Point(station.x, boundary_y(station)))
    if not math.isclose(result[-1].x, right_end, abs_tol=1e-6):
        end_station = _interpolate_station(stations, right_end)
        _append_point(result, Point(right_end, boundary_y(end_station)))
    return result


def _two_span_void_boundary(
    stations: list[LongitudinalStation],
    *,
    end_wall_inner: float,
    wall_half: float,
    end_hole_y: float,
    middle_hole_y: float,
    use_top: bool,
    end_chamfer: list[float] | None,
) -> list[Point]:
    left_end = stations[0].x
    right_crop = stations[-1].x

    def boundary_y(station: LongitudinalStation) -> float:
        return station.void_top_y if use_top else station.void_bottom_y

    result = [Point(left_end, end_hole_y), Point(end_wall_inner, end_hole_y)]
    cavity_start = _append_left_end_connection(
        result,
        stations,
        end_wall_inner,
        boundary_y,
        use_top,
        end_chamfer,
    )
    for station in stations:
        if cavity_start < station.x <= -wall_half:
            _append_point(result, Point(station.x, boundary_y(station)))
    wall_station = _interpolate_station(stations, -wall_half)
    _append_point(result, Point(-wall_half, boundary_y(wall_station)))
    _append_point(result, Point(-wall_half, middle_hole_y))
    _append_point(result, Point(right_crop, middle_hole_y))
    return result


def _two_span_boxed_void_boundary(
    stations: list[LongitudinalStation],
    *,
    end_wall_inner: float,
    wall_half: float,
    box_end: float | None,
    wall_end: float | None,
    end_hole_y: float,
    middle_hole_y: float,
    use_top: bool,
    end_chamfer: list[float] | None,
) -> list[Point]:
    if box_end is None or wall_end is None:
        raise ValueError("有箱室0号块缺少箱室或横墙范围。")
    left_end = stations[0].x
    right_crop = stations[-1].x

    def boundary_y(station: LongitudinalStation) -> float:
        return station.void_top_y if use_top else station.void_bottom_y

    result = [Point(left_end, end_hole_y), Point(end_wall_inner, end_hole_y)]
    cavity_start = _append_left_end_connection(
        result,
        stations,
        end_wall_inner,
        boundary_y,
        use_top,
        end_chamfer,
    )
    for station in stations:
        if cavity_start < station.x <= -wall_end:
            _append_point(result, Point(station.x, boundary_y(station)))
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
            _append_point(result, Point(station.x, boundary_y(station)))
    if not math.isclose(result[-1].x, right_crop, abs_tol=1e-6):
        crop_station = _interpolate_station(stations, right_crop)
        _append_point(result, Point(right_crop, boundary_y(crop_station)))
    return result


def _append_left_end_connection(
    points: list[Point],
    stations: list[LongitudinalStation],
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
    wall_y = points[-1].y
    cavity_y = wall_y + float(chamfer[1]) if upper else wall_y - float(
        chamfer[1]
    )
    _append_point(points, Point(cavity_x, cavity_y))
    return cavity_x


def _interpolate_station(
    stations: list[LongitudinalStation],
    x: float,
) -> LongitudinalStation:
    for station in stations:
        if math.isclose(station.x, x, abs_tol=1e-6):
            return station
    for first, second in zip(stations, stations[1:]):
        if first.x < x < second.x:
            return LongitudinalStation(
                id="interpolated",
                x=x,
                height=_interpolate(
                    x, first.x, second.x, first.height, second.height
                ),
                top_thickness=_interpolate(
                    x,
                    first.x,
                    second.x,
                    first.top_thickness,
                    second.top_thickness,
                ),
                bottom_thickness=_interpolate(
                    x,
                    first.x,
                    second.x,
                    first.bottom_thickness,
                    second.bottom_thickness,
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
    curve_start = layout.curve_start
    bottom_transition_end = layout.sequence_start
    curve_end = layout.curve_end
    common_distances = {
        0.0,
        curve_start,
        bottom_transition_end,
        curve_end,
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
        top_chamfer = longitudinal["ch_t_box_0_v"]
        bottom_chamfer = longitudinal["ch_b_box_0_v"]
        for width in (float(top_chamfer[0]), float(bottom_chamfer[0])):
            common_distances.add(layout.box_end - width)
        common_distances.update(
            {
                layout.box_start,
                layout.box_end,
                layout.wall_start,
                layout.wall_end,
                layout.zero_block_end,
            }
        )
    cumulative = bottom_transition_end
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
            + float(longitudinal["L_t_end_var"]),
            left_end
            + float(longitudinal["L_diaphragm_end"])
            + float(longitudinal["L_b_end_var"]),
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
            - float(longitudinal["L_t_end_var"]),
            right_extent
            - float(longitudinal["L_diaphragm_end"])
            - float(longitudinal["L_b_end_var"]),
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


def _height_at(
    distance: float,
    longitudinal: dict[str, Any],
    layout: LongitudinalLayout,
) -> float:
    curve_start = layout.curve_start
    variable_length = float(longitudinal["L_H_var"])
    curve_end = layout.curve_end
    if distance <= curve_start:
        return float(longitudinal["H_pier"])
    if distance >= curve_end:
        return float(longitudinal["H_end"])
    remaining_ratio = (curve_end - distance) / variable_length
    height_difference = float(longitudinal["H_pier"]) - float(
        longitudinal["H_end"]
    )
    return float(longitudinal["H_end"]) + height_difference * math.pow(
        remaining_ratio,
        float(longitudinal["n_H"]),
    )


def _top_thickness_at(
    x: float,
    beam_end: float,
    longitudinal: dict[str, Any],
    layout: LongitudinalLayout,
) -> float:
    distance_from_beam_end = abs(x - beam_end)
    diaphragm_length = float(longitudinal["L_diaphragm_end"])
    if distance_from_beam_end <= diaphragm_length:
        return float(longitudinal["t_t_end"])

    end_transition = float(longitudinal["L_t_end_var"])
    transition_distance = distance_from_beam_end - diaphragm_length
    if transition_distance <= end_transition:
        return _interpolate(
            transition_distance,
            0.0,
            end_transition,
            float(longitudinal["t_t_end"]),
            float(longitudinal["t_t_std"]),
        )

    distance = abs(x)
    if layout.kind == BOXED_ZERO_BLOCK:
        return _boxed_top_thickness(distance, longitudinal, layout)
    wall_half = layout.wall_half
    transition_end = layout.curve_start
    if layout.kind == TRANSITION_ZERO_BLOCK:
        chamfer = longitudinal["ch_t_0_v"]
        start_thickness = float(longitudinal["t_t_trans_0"]) + float(
            chamfer[1]
        )
        end_thickness = float(longitudinal["t_t_trans_0"])
    else:
        start_thickness = float(longitudinal["t_t_ch_0_s"])
        end_thickness = float(longitudinal["t_t_ch_0_e"])
    if distance <= wall_half:
        return start_thickness
    if distance < transition_end:
        return _interpolate(
            distance,
            wall_half,
            transition_end,
            start_thickness,
            end_thickness,
        )
    return float(longitudinal["t_t_std"])


def _bottom_thickness_at(
    x: float,
    beam_end: float,
    longitudinal: dict[str, Any],
    layout: LongitudinalLayout,
) -> float:
    distance_from_beam_end = abs(x - beam_end)
    diaphragm_length = float(longitudinal["L_diaphragm_end"])
    if distance_from_beam_end <= diaphragm_length:
        return float(longitudinal["t_b_end"])

    end_transition = float(longitudinal["L_b_end_var"])
    transition_distance = distance_from_beam_end - diaphragm_length
    if transition_distance <= end_transition:
        return _interpolate(
            transition_distance,
            0.0,
            end_transition,
            float(longitudinal["t_b_end"]),
            float(longitudinal["t_b_std"]),
        )

    distance = abs(x)
    if layout.kind == BOXED_ZERO_BLOCK:
        return _boxed_bottom_thickness(distance, longitudinal, layout)
    wall_half = layout.wall_half
    linear_end = layout.curve_start
    transition_end = layout.sequence_start
    if layout.kind == TRANSITION_ZERO_BLOCK:
        start_thickness = float(longitudinal["t_b_trans_0_s"]) + float(
            longitudinal["ch_t_0_v"][1]
        )
        middle_thickness = float(longitudinal["t_b_trans_0_s"])
    else:
        start_thickness = float(longitudinal["t_b_ch_0_s"])
        middle_thickness = float(longitudinal["t_b_ch_0_m"])
    if distance <= wall_half:
        return start_thickness
    if distance < linear_end:
        return _interpolate(
            distance,
            wall_half,
            linear_end,
            start_thickness,
            middle_thickness,
        )
    if distance <= transition_end:
        return _interpolate(
            distance,
            linear_end,
            transition_end,
            float(longitudinal["t_b_trans_0_s"]),
            float(longitudinal["t_b_trans_0_e"]),
        )

    position = transition_end
    values = longitudinal["t_b_var_seq"]
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
    return float(longitudinal["t_b_std"])


def _boxed_top_thickness(
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
    chamfer = longitudinal["ch_t_box_0_v"]
    edge = float(longitudinal["t_t_box_0"]) + float(chamfer[1])
    flat = float(longitudinal["t_t_box_0"])
    if distance <= layout.box_end:
        return _boxed_half_chamfer_value(
            distance,
            layout.box_end,
            float(chamfer[0]),
            flat,
            edge,
        )
    if distance < layout.wall_end:
        return float(longitudinal["t_t_ch_0_s"])
    if distance <= layout.zero_block_end:
        return _interpolate(
            distance,
            layout.wall_end,
            layout.zero_block_end,
            float(longitudinal["t_t_ch_0_s"]),
            float(longitudinal["t_t_ch_0_e"]),
        )
    return float(longitudinal["t_t_std"])


def _boxed_bottom_thickness(
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
        chamfer = longitudinal["ch_b_box_0_v"]
        edge = float(longitudinal["t_b_box_0"]) + float(chamfer[1])
        flat = float(longitudinal["t_b_box_0"])
        return _boxed_half_chamfer_value(
            distance,
            layout.box_end,
            float(chamfer[0]),
            flat,
            edge,
        )
    if distance < layout.wall_end:
        return float(longitudinal["t_b_ch_0_s"])
    if distance < layout.curve_start:
        return _interpolate(
            distance,
            layout.wall_end,
            layout.curve_start,
            float(longitudinal["t_b_ch_0_s"]),
            float(longitudinal["t_b_ch_0_m"]),
        )
    if distance < layout.sequence_start:
        return _interpolate(
            distance,
            layout.curve_start,
            layout.sequence_start,
            float(longitudinal["t_b_ch_0_m"]),
            float(longitudinal["t_b_ch_0_e"]),
        )

    position = layout.sequence_start
    values = longitudinal["t_b_var_seq"]
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
    return float(longitudinal["t_b_std"])


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
