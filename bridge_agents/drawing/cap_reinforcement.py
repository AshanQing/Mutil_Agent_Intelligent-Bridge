from __future__ import annotations

import math
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .reinforcement_adapter import DrawingExpressionError, evaluate_drawing_expression
from .reinforcement_resolution import ResolvedCap


Point = tuple[float, float]


class _PathModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BarPathSegment(_PathModel):
    kind: Literal["line", "arc"]
    start: Point
    end: Point
    center: Point | None = None
    radius_mm: float | None = Field(default=None, gt=0)
    arc_bulge: Literal["upper", "lower"] | None = None


class BarPathComponent(_PathModel):
    component_id: str
    segments: list[BarPathSegment] = Field(min_length=1)


class ResolvedCapBarPath(_PathModel):
    mark: str
    diameter_mm: float = Field(gt=0)
    z_pattern: str
    components: list[BarPathComponent] = Field(min_length=1)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def bottom_y(cap: ResolvedCap, x_mm: float) -> float:
    """盖梁底面在 x 处的高度（变截面斜面）。

    平底段（|x| <= flat_half = 半长 - 悬臂长）：0
    悬臂斜段（flat_half < |x| <= half）：线性升到 height_mid - height_end
    """
    half = cap.length_mm / 2
    flat_half = half - cap.cantilever_mm
    abs_x = abs(x_mm)
    if abs_x <= flat_half:
        return 0.0
    ratio = (abs_x - flat_half) / (half - flat_half)
    return ratio * (cap.height_mid_mm - cap.height_end_mm)


def _bottom_cover_y(cap: ResolvedCap, x_mm: float, offset: float) -> float:
    """底面保护层线：钢筋中心到底面距离 = cover_y + offset（随 x 变化）。"""
    return bottom_y(cap, x_mm) + cap.cover_y_mm + offset


def _slope_bottom_intersection(
    start: Point,
    *,
    direction: str,
    angle_deg: float,
    cap: ResolvedCap,
    cover_offset: float,
) -> Point:
    """求斜线段与底面保护层线的交点（解析分段求交）。

    斜线从 start 沿 direction（left_down / right_down）向下，角度 angle_deg；
    底面保护层线分段线性：平底段常数 C=cover+offset，左右悬臂斜段线性升高。
    返回 (x, y) 或抛错（起点已在保护层线下 / 斜线未到达保护层线）。
    """
    x0, y0 = start
    tangent = math.tan(math.radians(angle_deg))
    if math.isclose(tangent, 0.0, abs_tol=1e-12):
        raise DrawingExpressionError("钢筋分支角度不能为 0 度。")
    half = cap.length_mm / 2
    flat_half = half - cap.cantilever_mm
    if flat_half <= 0 or flat_half >= half:
        raise DrawingExpressionError("平底段长度异常。")
    slope_k = (cap.height_mid_mm - cap.height_end_mm) / (half - flat_half)
    const_c = cap.cover_y_mm + cover_offset
    # 方向斜率：left_down 斜率 +tan（x 减 y 减），right_down 斜率 -tan
    s = tangent if "left" in direction else -tangent

    def line_y(x: float) -> float:
        return y0 + s * (x - x0)

    def in_range(x: float, x_min: float, x_max: float) -> bool:
        return x_min - 1e-9 <= x <= x_max + 1e-9

    # 1) 平底段 |x|<=flat_half：line = C
    x_flat = x0 + (const_c - y0) / s
    if in_range(x_flat, -flat_half, flat_half) and _direction_ok(x_flat, x0, direction):
        return (x_flat, const_c)
    # 2) 左侧悬臂段 -half<=x<=-flat_half：line = k*(-x-flat)+C
    denom_left = s + slope_k
    if not math.isclose(denom_left, 0.0, abs_tol=1e-12):
        x_left = (const_c - y0 + s * x0 - slope_k * flat_half) / denom_left
        if in_range(x_left, -half, -flat_half) and _direction_ok(x_left, x0, direction):
            return (x_left, line_y(x_left))
    # 3) 右侧悬臂段 flat_half<=x<=half：line = k*(x-flat)+C
    denom_right = s - slope_k
    if not math.isclose(denom_right, 0.0, abs_tol=1e-12):
        x_right = (const_c - y0 + s * x0 + slope_k * flat_half) / denom_right
        if in_range(x_right, flat_half, half) and _direction_ok(x_right, x0, direction):
            return (x_right, line_y(x_right))
    raise DrawingExpressionError(
        f"斜段 ({x0:g},{y0:g}) {direction} 未能与底面保护层线相交。"
    )


def _direction_ok(x: float, x0: float, direction: str) -> bool:
    """交点必须位于从起点出发的方向上（left: x<x0；right: x>x0）。"""
    return x < x0 - 1e-9 if "left" in direction else x > x0 + 1e-9


def _context(cap: ResolvedCap, *, diameter_mm: float) -> dict[str, float]:
    return {
        "cap_length": cap.length_mm,
        "cap_width": cap.width_mm,
        "cap_height_mid": cap.height_mid_mm,
        "cap_height_end": cap.height_end_mm,
        "cantilever_length": cap.cantilever_mm,
        "column_count": float(cap.column_count),
        "column_diameter": cap.column_diameter_mm,
        "column_spacing": cap.column_spacing_mm,
        "cover_x": cap.cover_x_mm,
        "cover_y": cap.cover_y_mm,
        "cover_z": cap.cover_z_mm,
        "pier_centerline_x": (
            -cap.column_spacing_mm / 2 if cap.column_count > 1 else 0.0
        ),
        "cap_centerline_x": 0.0,
        "dia": diameter_mm,
    }


def _y_from_reference(
    definition: Mapping[str, Any],
    *,
    cap: ResolvedCap,
    context: Mapping[str, float],
    x_mm: float | None = None,
) -> float:
    offset = evaluate_drawing_expression(definition.get("y_offset", 0), context)
    reference = str(definition.get("y_ref") or "top_cover")
    if "bottom" in reference or "底" in reference:
        if x_mm is None:
            raise DrawingExpressionError("bottom_cover 参考必须提供 x 坐标。")
        return _bottom_cover_y(cap, x_mm, offset)
    return cap.height_mid_mm - cap.cover_y_mm - offset


def _dedup_points(points: list[Point]) -> list[Point]:
    """去掉相邻重复点（水平连接段与斜段起点重合时产生零长段）。"""
    cleaned: list[Point] = []
    for point in points:
        if not cleaned or not (
            abs(cleaned[-1][0] - point[0]) < 1e-9
            and abs(cleaned[-1][1] - point[1]) < 1e-9
        ):
            cleaned.append(point)
    return cleaned


def _line(start: Point, end: Point) -> BarPathSegment:
    return BarPathSegment(kind="line", start=start, end=end)


def _component(component_id: str, points: list[Point]) -> BarPathComponent:
    if len(points) < 2:
        raise DrawingExpressionError(f"钢筋路径 {component_id} 至少需要两个点。")
    return BarPathComponent(
        component_id=component_id,
        segments=[_line(points[index - 1], points[index]) for index in range(1, len(points))],
    )


def _mirror_component(component: BarPathComponent, component_id: str) -> BarPathComponent:
    mirrored: list[BarPathSegment] = []
    for segment in reversed(component.segments):
        center = (
            (-segment.center[0], segment.center[1])
            if segment.center is not None
            else None
        )
        mirrored.append(
            BarPathSegment(
                kind=segment.kind,
                start=(-segment.end[0], segment.end[1]),
                end=(-segment.start[0], segment.start[1]),
                center=center,
                radius_mm=segment.radius_mm,
                arc_bulge=segment.arc_bulge,
            )
        )
    return BarPathComponent(component_id=component_id, segments=mirrored)


def _until_point(
    start: Point,
    rule: Mapping[str, Any],
    *,
    cap: ResolvedCap,
    context: Mapping[str, float],
    default_angle_deg: float,
) -> Point:
    until = _mapping(rule.get("until"))
    if not until:
        # 容错：部分配筋输出把"左右弯起点之间的水平连接段"写进分支规则
        # （direction=horizontal 且 end_at 引用关键点），此时分支终点即起点，
        # 不是真正缺数据；只有既无 until 也非水平连接段的才判定为数据缺失。
        direction = str(rule.get("direction") or "").lower()
        end_at = rule.get("end_at")
        if "horizontal" in direction or end_at:
            return start
        raise DrawingExpressionError("方向分支缺少 until 定义。")
    direction = str(rule.get("direction") or "")
    angle_deg = evaluate_drawing_expression(
        rule.get("angle", default_angle_deg), context
    )
    reference = str(until.get("y_ref") or "top_cover")
    if "bottom" in reference or "底" in reference:
        # 斜段与悬臂段斜面保护层线求交（随 x 变化，解析二分）
        offset = evaluate_drawing_expression(until.get("y_offset", 0), context)
        return _slope_bottom_intersection(
            start,
            direction=direction,
            angle_deg=angle_deg,
            cap=cap,
            cover_offset=offset,
        )
    # 固定 y 参考（top_cover 等）：保持原水平换算逻辑
    target_y = _y_from_reference(until, cap=cap, context=context)
    vertical_drop = abs(start[1] - target_y)
    tangent = math.tan(math.radians(angle_deg))
    if math.isclose(tangent, 0.0, abs_tol=1e-12):
        raise DrawingExpressionError("钢筋分支角度不能为 0 度。")
    horizontal = vertical_drop / abs(tangent)
    sign = -1.0 if "left" in direction else 1.0
    return (start[0] + sign * horizontal, target_y)


def _compile_range_bar(
    mark: str,
    source: Mapping[str, Any],
    *,
    cap: ResolvedCap,
    context: Mapping[str, float],
) -> list[BarPathComponent]:
    range_definition = _mapping(source.get("range_definition"))
    layer_definition = _mapping(source.get("layer_definition"))
    start_x = evaluate_drawing_expression(
        range_definition.get("x_start_expr"), context
    )
    end_x = evaluate_drawing_expression(range_definition.get("x_end_expr"), context)
    reference = str(layer_definition.get("y_ref") or "top_cover")
    mirror = str(source.get("mirror") or "").lower() == "longitudinal"
    if "bottom" in reference or "底" in reference:
        # 底部通长筋：跟随底面折线（平底段 + 悬臂斜段）
        offset = evaluate_drawing_expression(
            layer_definition.get("y_offset", 0), context
        )
        half = cap.length_mm / 2
        flat_half = half - cap.cantilever_mm

        def bottom_points(x_a: float, x_b: float) -> list[Point]:
            xs = sorted({x_a, x_b, -flat_half, flat_half})
            xs = [
                x for x in xs if min(x_a, x_b) - 1e-9 <= x <= max(x_a, x_b) + 1e-9
            ]
            return [(x, _bottom_cover_y(cap, x, offset)) for x in xs]

        if mirror and math.isclose(end_x, 0.0, abs_tol=1e-9):
            # 半跨定义 + 纵向镜像：整跨折线（途经 ±flat_half，无冗余点）
            full = bottom_points(start_x, -start_x)
            return [_component(f"{mark}:full", full)]
        source_points = bottom_points(start_x, end_x)
        source_component = _component(f"{mark}:source", source_points)
        if not mirror:
            return [source_component]
        mirrored = _mirror_component(source_component, f"{mark}:mirrored")
        return sorted(
            [source_component, mirrored],
            key=lambda item: min(segment.start[0] for segment in item.segments),
        )

    y = _y_from_reference(layer_definition, cap=cap, context=context)
    source_component = _component(f"{mark}:source", [(start_x, y), (end_x, y)])
    if not mirror:
        return [source_component]
    if math.isclose(end_x, 0.0, abs_tol=1e-9):
        return [_component(f"{mark}:full", [(start_x, y), (-start_x, y)])]
    mirrored = _mirror_component(source_component, f"{mark}:mirrored")
    return sorted(
        [source_component, mirrored],
        key=lambda item: min(segment.start[0] for segment in item.segments),
    )


def _compile_bend_pair(
    mark: str,
    source: Mapping[str, Any],
    *,
    cap: ResolvedCap,
    context: Mapping[str, float],
) -> list[BarPathComponent]:
    control = _mapping(source.get("control_definition"))
    key_points = _mapping(control.get("key_points"))
    # 兼容顶部弯下（left/right_bend_point）与底部弯起（left/right_bottom_point）两种命名
    left_definition = _mapping(
        key_points.get("left_bend_point") or key_points.get("left_bottom_point")
    )
    right_definition = _mapping(
        key_points.get("right_bend_point") or key_points.get("right_bottom_point")
    )
    left_x = evaluate_drawing_expression(left_definition.get("x_expr"), context)
    right_x = evaluate_drawing_expression(right_definition.get("x_expr"), context)
    left = (
        left_x,
        _y_from_reference(
            left_definition, cap=cap, context=context, x_mm=left_x
        ),
    )
    right = (
        right_x,
        _y_from_reference(
            right_definition, cap=cap, context=context, x_mm=right_x
        ),
    )
    rules = _mapping(control.get("branch_rules"))
    angle = evaluate_drawing_expression(control.get("bend_angle", 45), context)
    left_end = _until_point(
        left,
        _mapping(rules.get("left_branch")),
        cap=cap,
        context=context,
        default_angle_deg=angle,
    )
    right_rule = _mapping(rules.get("right_branch"))
    step_1 = _mapping(right_rule.get("step_1")) or right_rule
    right_end = _until_point(
        right,
        step_1,
        cap=cap,
        context=context,
        default_angle_deg=angle,
    )
    points = _dedup_points([left_end, left, right, right_end])
    step_2 = _mapping(right_rule.get("step_2"))
    if step_2:
        end_at = step_2.get("end_at")
        # end_at 兼容两种形式：字符串（直接作为 x 表达式）或对象（取 x / x_expr）
        if isinstance(end_at, str):
            end_x = evaluate_drawing_expression(end_at, context)
        else:
            end_at_map = _mapping(end_at)
            end_x = evaluate_drawing_expression(
                end_at_map.get("x", end_at_map.get("x_expr")), context
            )
        points.append(
            (
                end_x,
                _y_from_reference(
                    step_2, cap=cap, context=context, x_mm=end_x
                ),
            )
        )
    points = _dedup_points(points)
    component = _component(f"{mark}:source", points)
    components = [component]
    if str(source.get("mirror") or "").lower() == "longitudinal":
        components.append(_mirror_component(component, f"{mark}:mirrored"))
    return components


def _compile_independent_diagonal(
    mark: str,
    source: Mapping[str, Any],
    *,
    cap: ResolvedCap,
    context: Mapping[str, float],
) -> list[BarPathComponent]:
    control = _mapping(source.get("control_definition"))
    bend = _mapping(_mapping(control.get("key_points")).get("bend_start_point"))
    start = (
        evaluate_drawing_expression(bend.get("x_expr"), context),
        _y_from_reference(bend, cap=cap, context=context),
    )
    rule = _mapping(_mapping(control.get("branch_rules")).get("main_branch"))
    end = _until_point(
        start,
        rule,
        cap=cap,
        context=context,
        default_angle_deg=evaluate_drawing_expression(
            control.get("bend_angle", 45), context
        ),
    )
    component = _component(f"{mark}:source", [start, end])
    components = [component]
    if str(source.get("mirror") or "").lower() == "longitudinal":
        components.append(_mirror_component(component, f"{mark}:mirrored"))
    return components


def _compile_bottom_rise(
    mark: str,
    source: Mapping[str, Any],
    *,
    cap: ResolvedCap,
    context: Mapping[str, float],
) -> list[BarPathComponent]:
    """编译"底部上弯起钢筋"变体：底部水平段（起点在盖梁中线）+ 45° 斜段到顶部端点。

    配筋输出结构为：
      bottom_branch: {start_at: {x: 中线表达式}, end_at: bottom_bend_point,
                      y_ref: bottom_cover, y_offset: ...}   # 底部水平段
      left_branch/right_branch: {start_from: bottom_bend_point,
                      direction: left_up/right_up, angle: 45, end_at: top_end_point}
    关键点 bottom_bend_point 与 top_end_point 由 x_expr + y_ref/y_offset 定义。
    """
    control = _mapping(source.get("control_definition"))
    key_points = _mapping(control.get("key_points"))
    rules = _mapping(control.get("branch_rules"))
    bottom = _mapping(rules.get("bottom_branch"))
    start_at = _mapping(bottom.get("start_at"))
    if not start_at:
        raise DrawingExpressionError(
            f"cap.{mark} 的底部上弯起钢筋缺少 bottom_branch.start_at。"
        )
    start_x = evaluate_drawing_expression(start_at.get("x"), context)
    start_y = _y_from_reference(bottom, cap=cap, context=context, x_mm=start_x)

    points_by_name: dict[str, Point] = {}
    for name in ("bottom_bend_point", "top_end_point"):
        definition = _mapping(key_points.get(name))
        x = evaluate_drawing_expression(definition.get("x_expr"), context)
        points_by_name[str(name)] = (
            x,
            _y_from_reference(definition, cap=cap, context=context, x_mm=x),
        )
    ordered_points = _dedup_points(
        [
            (start_x, start_y),
            points_by_name["bottom_bend_point"],
            points_by_name["top_end_point"],
        ]
    )
    if len(ordered_points) < 2:
        raise DrawingExpressionError(f"cap.{mark} 的底部上弯起钢筋路径为空。")
    component = _component(f"{mark}:source", ordered_points)
    components = [component]
    if str(source.get("mirror") or "").lower() == "longitudinal":
        components.append(_mirror_component(component, f"{mark}:mirrored"))
    return components


def _compile_named_polyline(
    mark: str,
    source: Mapping[str, Any],
    *,
    cap: ResolvedCap,
    context: Mapping[str, float],
) -> list[BarPathComponent]:
    """编译由命名关键点和 start_from/end_at 连续定义的普通折线路径。"""
    control = _mapping(source.get("control_definition"))
    key_points = _mapping(control.get("key_points"))
    rules = _mapping(control.get("branch_rules"))
    points_by_name: dict[str, Point] = {}
    for name, raw_definition in key_points.items():
        definition = _mapping(raw_definition)
        x = evaluate_drawing_expression(definition.get("x_expr"), context)
        points_by_name[str(name)] = (
            x,
            _y_from_reference(definition, cap=cap, context=context, x_mm=x),
        )
    def _resolve_point(rule: Mapping[str, Any], field: str) -> Point:
        raw = rule.get(field)
        if isinstance(raw, dict):
            # 内联点定义：{x: 表达式, y_ref: ..., y_offset: ...}
            definition = _mapping(raw)
            x = evaluate_drawing_expression(definition.get("x"), context)
            return (
                x,
                _y_from_reference(definition, cap=cap, context=context, x_mm=x),
            )
        name = str(raw or "")
        if name not in points_by_name:
            raise DrawingExpressionError(
                f"cap.{mark} 的命名折线路径引用了不存在的关键点。"
            )
        return points_by_name[name]

    ordered_points: list[Point] = []
    for raw_rule in rules.values():
        rule = _mapping(raw_rule)
        start = _resolve_point(rule, "start_from")
        end = _resolve_point(rule, "end_at")
        if ordered_points and ordered_points[-1] != start:
            raise DrawingExpressionError(f"cap.{mark} 的命名折线路径不连续。")
        if not ordered_points:
            ordered_points.append(start)
        ordered_points.append(end)
    if len(ordered_points) < 2:
        raise DrawingExpressionError(f"cap.{mark} 的命名折线路径为空。")
    component = _component(f"{mark}:source", ordered_points)
    components = [component]
    if str(source.get("mirror") or "").lower() == "longitudinal":
        components.append(_mirror_component(component, f"{mark}:mirrored"))
    return components


def _compile_round_transition(
    mark: str,
    source: Mapping[str, Any],
    *,
    cap: ResolvedCap,
    context: Mapping[str, float],
) -> list[BarPathComponent]:
    control = _mapping(source.get("control_definition"))
    key_points = _mapping(control.get("key_points"))
    arc_definition = _mapping(control.get("arc_definition"))
    radius = evaluate_drawing_expression(arc_definition.get("radius"), context)
    angle = evaluate_drawing_expression(control.get("bend_angle", 45), context)
    arc_context = {**context, "radius": radius, "bend_angle": angle}
    left_definition = _mapping(key_points.get("arc_left_point"))
    right_definition = _mapping(key_points.get("arc_right_point"))
    right_bottom_definition = _mapping(key_points.get("right_bottom_end"))
    rules = _mapping(control.get("branch_rules"))
    right_branch = _mapping(rules.get("right_branch"))

    # 1. 右下端点：x 由表达式，y 由底面保护层线（随 x）
    right_bottom_x = evaluate_drawing_expression(
        right_bottom_definition.get("x_expr"), arc_context
    )
    right_bottom = (
        right_bottom_x,
        _y_from_reference(
            right_bottom_definition, cap=cap, context=arc_context, x_mm=right_bottom_x
        ),
    )

    # 2. 圆弧定位（用户确认语义）：
    #    - 弧与两侧 45° 斜线相切，弧角 90°（两端点相对圆心角度 45°/135°）
    #    - 两端点关于过圆心的竖直线对称；圆心在 pier_centerline_x 上
    #    - 弧右端点位于 right_branch（从 right_bottom 出发 left_up 45°）直线上
    #    由此：arc_right = C + r*(cos45°, sin45°)，且 arc_right 在 45° 斜线上
    pier_x = context["pier_centerline_x"]
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # arc_right.y - right_bottom.y = -(arc_right.x - right_bottom.x)
    # => cy = bx + by - cx - r*(sin+cos)
    cy = right_bottom[0] + right_bottom[1] - pier_x - radius * (sin_t + cos_t)
    center = (pier_x, cy)
    arc_right = (pier_x + radius * cos_t, cy + radius * sin_t)
    arc_left = (pier_x - radius * cos_t, cy + radius * sin_t)

    # 3. 左斜段：与底面保护层线求交（相切，45°）
    left_end = _until_point(
        arc_left,
        _mapping(rules.get("left_branch")),
        cap=cap,
        context=arc_context,
        default_angle_deg=angle,
    )

    # 4. 圆弧：凸上（经过圆心正上方 90° 顶点），弧角 90°
    component = BarPathComponent(
        component_id=f"{mark}:source",
        segments=[
            _line(left_end, arc_left),
            BarPathSegment(
                kind="arc",
                start=arc_left,
                end=arc_right,
                center=center,
                radius_mm=radius,
                arc_bulge="upper",
            ),
            _line(arc_right, right_bottom),
        ],
    )
    components = [component]
    if str(source.get("mirror") or "").lower() == "longitudinal":
        components.append(_mirror_component(component, f"{mark}:mirrored"))
    return components


def compile_cap_bar_paths(cap: ResolvedCap) -> dict[str, ResolvedCapBarPath]:
    paths: dict[str, ResolvedCapBarPath] = {}
    for mark, bar in cap.bars.items():
        source = bar.source
        context = _context(cap, diameter_mm=bar.diameter_mm)
        if _mapping(source.get("range_definition")):
            components = _compile_range_bar(
                mark, source, cap=cap, context=context
            )
        else:
            key_points = _mapping(
                _mapping(source.get("control_definition")).get("key_points")
            )
            if (
                key_points.get("left_bend_point") and key_points.get("right_bend_point")
            ) or (
                key_points.get("left_bottom_point") and key_points.get("right_bottom_point")
            ):
                components = _compile_bend_pair(
                    mark, source, cap=cap, context=context
                )
            elif key_points.get("bend_start_point"):
                components = _compile_independent_diagonal(
                    mark, source, cap=cap, context=context
                )
            elif key_points.get("arc_left_point") and key_points.get("arc_right_point"):
                components = _compile_round_transition(
                    mark, source, cap=cap, context=context
                )
            elif key_points.get("bottom_bend_point") and key_points.get("top_end_point"):
                components = _compile_bottom_rise(
                    mark, source, cap=cap, context=context
                )
            elif key_points and all(
                isinstance(rule, Mapping)
                and rule.get("start_from")
                and rule.get("end_at")
                for rule in _mapping(
                    _mapping(source.get("control_definition")).get("branch_rules")
                ).values()
            ):
                components = _compile_named_polyline(
                    mark, source, cap=cap, context=context
                )
            else:
                raise DrawingExpressionError(
                    f"cap.{mark} 的路径定义无法识别，未生成替代线。"
                )
        paths[mark] = ResolvedCapBarPath(
            mark=mark,
            diameter_mm=bar.diameter_mm,
            z_pattern=bar.z_pattern or "",
            components=components,
        )
    return paths


def _line_intersection_y(segment: BarPathSegment, x_mm: float) -> list[float]:
    x1, y1 = segment.start
    x2, y2 = segment.end
    lower = min(x1, x2)
    upper = max(x1, x2)
    if x_mm < lower - 1e-9 or x_mm > upper + 1e-9:
        return []
    if math.isclose(x1, x2, abs_tol=1e-9):
        return [y1, y2] if math.isclose(x_mm, x1, abs_tol=1e-9) else []
    ratio = (x_mm - x1) / (x2 - x1)
    return [y1 + ratio * (y2 - y1)]


def _arc_intersection_y(segment: BarPathSegment, x_mm: float) -> list[float]:
    if segment.center is None or segment.radius_mm is None:
        raise DrawingExpressionError("圆弧路径缺少圆心或半径。")
    dx = x_mm - segment.center[0]
    if abs(dx) > segment.radius_mm + 1e-9:
        return []
    dy = math.sqrt(max(segment.radius_mm**2 - dx**2, 0.0))
    sign = 1.0 if segment.arc_bulge == "upper" else -1.0
    return [segment.center[1] + sign * dy]


def intersections_at_x(path: ResolvedCapBarPath, x_mm: float) -> list[float]:
    intersections: list[float] = []
    for component in path.components:
        for segment in component.segments:
            values = (
                _arc_intersection_y(segment, x_mm)
                if segment.kind == "arc"
                else _line_intersection_y(segment, x_mm)
            )
            for value in values:
                if not any(math.isclose(value, existing, abs_tol=1e-6) for existing in intersections):
                    intersections.append(value)
    return sorted(intersections)

