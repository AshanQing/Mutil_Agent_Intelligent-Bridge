from __future__ import annotations

import math
from typing import Any, Mapping

from .models import (
    ArcEntity,
    CircleEntity,
    DimensionEntity,
    DrawingDocument,
    DrawingGroupSource,
    DrawingLayer,
    GeometryView,
    LineEntity,
    SectionGeometry,
    TextEntity,
)
from .reinforcement_adapter import (
    CapGeometry,
    ColumnGeometry,
    DrawingExpressionError,
    adapt_cap_geometry,
    adapt_column_geometry,
    evaluate_drawing_expression,
    resolve_z_positions,
)


DRAWING_LAYERS = [
    DrawingLayer(name="S-BEAM", color_rgb=(255, 255, 255), lineweight_mm=0.50),
    DrawingLayer(name="S-COLUMN", color_rgb=(180, 180, 180), lineweight_mm=0.50),
    DrawingLayer(name="R-MAIN", color_rgb=(255, 0, 0), lineweight_mm=0.35),
    DrawingLayer(name="R-EXTRA", color_rgb=(255, 128, 0), lineweight_mm=0.30),
    DrawingLayer(name="R-STIRRUP", color_rgb=(0, 180, 255), lineweight_mm=0.25),
    DrawingLayer(name="A-DIMS", color_rgb=(0, 255, 0), lineweight_mm=0.18),
    DrawingLayer(name="A-TEXT", color_rgb=(255, 255, 0), lineweight_mm=0.18),
    DrawingLayer(name="A-TITLE", color_rgb=(255, 255, 255), lineweight_mm=0.25),
]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _reinforcement_parts(
    source: DrawingGroupSource,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    root = _mapping(source.reinforcement.get("reinforcement"))
    return _mapping(root.get("pier_cap")), _mapping(root.get("pier_column"))


def _outline_lines(
    *,
    prefix: str,
    layer: str,
    role: str,
    points: list[tuple[float, float]],
) -> list[LineEntity]:
    return [
        LineEntity(
            entity_id=f"{prefix}-{role}-{index:03d}",
            layer=layer,
            start=points[index - 1],
            end=points[index],
            metadata={"role": role},
        )
        for index in range(1, len(points))
    ]


def _positions(start: float, end: float, spacing: float) -> list[float]:
    if spacing <= 0:
        raise DrawingExpressionError("钢筋间距必须大于 0。")
    lower, upper = sorted((float(start), float(end)))
    count = int(math.floor((upper - lower) / spacing + 1e-9))
    if count > 1000:
        raise DrawingExpressionError("单个钢筋分区生成数量超过 1000。")
    values = [lower + index * spacing for index in range(count + 1)]
    if not values or not math.isclose(values[-1], upper, abs_tol=1e-6):
        values.append(upper)
    return values


def _cap_context(cap: CapGeometry, column: ColumnGeometry) -> dict[str, float]:
    return {
        "cap_length": cap.length_mm,
        "cap_width": cap.width_mm,
        "cap_height_mid": cap.height_mid_mm,
        "cap_height_end": cap.height_end_mm,
        "cantilever_length": cap.cantilever_mm,
        "column_count": float(column.count),
        "column_diameter": column.diameter_mm,
        "column_height": column.height_mm,
        "column_spacing": column.spacing_mm,
        "concrete_cover": column.concrete_cover_mm,
        "cover_x": cap.cover_x_mm,
        "cover_y": cap.cover_y_mm,
        "cover_z": cap.cover_z_mm,
        "pier_centerline_x": (
            -column.spacing_mm / 2.0 if column.count > 1 else 0.0
        ),
        "cap_centerline_x": 0.0,
    }


def _column_context(column: ColumnGeometry, *, dia: float = 0.0) -> dict[str, float]:
    return {
        "column_count": float(column.count),
        "column_diameter": column.diameter_mm,
        "column_height": column.height_mm,
        "column_spacing": column.spacing_mm,
        "concrete_cover": column.concrete_cover_mm,
        "dia": float(dia),
        "cap_centerline_x": 0.0,
        "pier_centerline_x": 0.0,
    }


def _column_centers(column: ColumnGeometry) -> list[float]:
    if column.count == 1:
        return [0.0]
    return [
        (index - (column.count - 1) / 2.0) * column.spacing_mm
        for index in range(column.count)
    ]


def _cap_outline(cap: CapGeometry, column: ColumnGeometry) -> list[tuple[float, float]]:
    half = cap.length_mm / 2.0
    if column.count > 1 and column.spacing_mm > 0:
        flat_half = (
            (column.count - 1) * column.spacing_mm / 2.0
            + column.diameter_mm / 2.0
            - 100.0
        )
    else:
        flat_half = half - max(cap.cantilever_mm, 1000.0)
    flat_half = max(min(flat_half, half - 100.0), half * 0.45)
    return [
        (-half, cap.height_mid_mm),
        (half, cap.height_mid_mm),
        (half, cap.height_end_mm),
        (flat_half, 0.0),
        (-flat_half, 0.0),
        (-half, cap.height_end_mm),
        (-half, cap.height_mid_mm),
    ]


def _bar_y(bar: Mapping[str, Any], cap: CapGeometry) -> float:
    layer = _mapping(bar.get("layer_definition"))
    dia = float(bar.get("dia") or 0.0)
    context = {"dia": dia}
    offset = evaluate_drawing_expression(layer.get("y_offset", 0), context)
    y_ref = str(layer.get("y_ref") or "top_cover")
    if "bottom" in y_ref or "底" in y_ref:
        return cap.cover_y_mm + offset
    return cap.height_mid_mm - cap.cover_y_mm - offset


def _control_point_y(
    point: Mapping[str, Any],
    *,
    dia: float,
    cap: CapGeometry,
) -> float:
    offset = evaluate_drawing_expression(
        point.get("y_offset", 0),
        {"dia": dia},
    )
    y_ref = str(point.get("y_ref") or "top_cover")
    if "bottom" in y_ref or "底" in y_ref:
        return cap.cover_y_mm + offset
    return cap.height_mid_mm - cap.cover_y_mm - offset


def _cap_control_segments(
    *,
    prefix: str,
    bar_index: int,
    bar: Mapping[str, Any],
    cap: CapGeometry,
    context: Mapping[str, float],
) -> list[Any]:
    control = _mapping(bar.get("control_definition"))
    key_points = _mapping(control.get("key_points"))
    left = _mapping(key_points.get("left_bend_point"))
    right = _mapping(key_points.get("right_bend_point"))
    dia = float(bar.get("dia") or 0.0)
    values = {**context, "dia": dia}
    mark = str(bar.get("id") or f"N{bar_index}")
    metadata = {
        "bar_role": "cap_control",
        "bar_mark": mark,
        "diameter_mm": dia,
    }

    bend_start = _mapping(key_points.get("bend_start_point"))
    if bend_start:
        start_x = evaluate_drawing_expression(bend_start.get("x_expr"), values)
        start_y = _control_point_y(bend_start, dia=dia, cap=cap)
        direction = str(
            _mapping(_mapping(control.get("branch_rules")).get("main_branch")).get(
                "direction"
            )
            or "right_down"
        )
        bottom_y = cap.cover_y_mm + dia
        horizontal = max(start_y - bottom_y, dia)
        end_x = start_x - horizontal if "left" in direction else start_x + horizontal
        limit = cap.length_mm / 2.0 - cap.cover_x_mm
        source = [(start_x, start_y), (max(min(end_x, limit), -limit), bottom_y)]
        paths = [("source", source)]
        if str(bar.get("mirror") or "").lower() == "longitudinal":
            paths.append(("mirrored", [(-x, y) for x, y in source]))
        return [
            LineEntity(
                entity_id=f"{prefix}-control-{bar_index:03d}-{side}-01",
                layer="R-EXTRA",
                start=points[0],
                end=points[1],
                metadata={**metadata, "mirror_side": side},
            )
            for side, points in paths
        ]

    arc_left = _mapping(key_points.get("arc_left_point"))
    arc_right = _mapping(key_points.get("arc_right_point"))
    arc_definition = _mapping(control.get("arc_definition"))
    if arc_left and arc_right and arc_definition:
        radius = evaluate_drawing_expression(
            arc_definition.get("radius"), values
        )
        bend_angle = evaluate_drawing_expression(
            control.get("bend_angle", 45.0), values
        )
        arc_values = {**values, "radius": radius, "bend_angle": bend_angle}
        left_x = evaluate_drawing_expression(arc_left.get("x_expr"), arc_values)
        right_x = evaluate_drawing_expression(arc_right.get("x_expr"), arc_values)
        arc_y = cap.height_mid_mm - cap.cover_y_mm - dia
        center_x = (left_x + right_x) / 2.0
        right_bottom = _mapping(key_points.get("right_bottom_end"))
        bottom_y = cap.cover_y_mm + dia
        if right_bottom:
            end_x = evaluate_drawing_expression(
                right_bottom.get("x_expr"), arc_values
            )
            bottom_y = _control_point_y(right_bottom, dia=dia, cap=cap)
        else:
            end_x = right_x + max(arc_y - bottom_y, dia)
        left_end_x = left_x - max(arc_y - bottom_y, dia)
        paths = [
            (
                "source",
                (center_x, left_x, right_x, left_end_x, end_x),
            )
        ]
        if str(bar.get("mirror") or "").lower() == "longitudinal":
            paths.append(
                (
                    "mirrored",
                    (-center_x, -right_x, -left_x, -end_x, -left_end_x),
                )
            )
        entities: list[Any] = []
        for side, (arc_center_x, start_x, finish_x, left_end, right_end) in paths:
            entities.extend(
                [
                    LineEntity(
                        entity_id=f"{prefix}-control-{bar_index:03d}-{side}-01",
                        layer="R-EXTRA",
                        start=(left_end, bottom_y),
                        end=(start_x, arc_y),
                        metadata={**metadata, "mirror_side": side},
                    ),
                    ArcEntity(
                        entity_id=f"{prefix}-control-{bar_index:03d}-{side}-02",
                        layer="R-EXTRA",
                        center=(arc_center_x, arc_y),
                        radius=radius,
                        start_angle_deg=0.0,
                        end_angle_deg=180.0,
                        clockwise=False,
                        metadata={**metadata, "mirror_side": side},
                    ),
                    LineEntity(
                        entity_id=f"{prefix}-control-{bar_index:03d}-{side}-03",
                        layer="R-EXTRA",
                        start=(finish_x, arc_y),
                        end=(right_end, bottom_y),
                        metadata={**metadata, "mirror_side": side},
                    ),
                ]
            )
        return entities

    if not left or not right:
        return []

    left_x = evaluate_drawing_expression(left.get("x_expr"), values)
    right_x = evaluate_drawing_expression(right.get("x_expr"), values)
    left_y = _control_point_y(left, dia=dia, cap=cap)
    right_y = _control_point_y(right, dia=dia, cap=cap)
    bottom_y = cap.cover_y_mm + dia
    left_end_x = left_x - max(left_y - bottom_y, 0.0)
    right_end_x = right_x + max(right_y - bottom_y, 0.0)
    limit = cap.length_mm / 2.0 - cap.cover_x_mm
    source_points = [
        (max(left_end_x, -limit), bottom_y),
        (left_x, left_y),
        (right_x, right_y),
        (min(right_end_x, limit), bottom_y),
    ]
    paths = [("source", source_points)]
    if str(bar.get("mirror") or "").lower() == "longitudinal":
        paths.append(("mirrored", [(-x, y) for x, y in source_points]))

    entities: list[LineEntity] = []
    for mirror_side, points in paths:
        for segment_index in range(1, len(points)):
            entities.append(
                LineEntity(
                    entity_id=(
                        f"{prefix}-control-{bar_index:03d}-"
                        f"{mirror_side}-{segment_index:02d}"
                    ),
                    layer="R-EXTRA",
                    start=points[segment_index - 1],
                    end=points[segment_index],
                    metadata={
                        **metadata,
                        "mirror_side": mirror_side,
                    },
                )
            )
    return entities


def _title(
    *,
    prefix: str,
    text: str,
    insertion: tuple[float, float],
) -> TextEntity:
    return TextEntity(
        entity_id=f"{prefix}-title-001",
        layer="A-TITLE",
        insertion=insertion,
        text=text,
        height=250.0,
        metadata={"role": "view_title"},
    )


def build_cap_elevation(source: DrawingGroupSource) -> GeometryView:
    cap = adapt_cap_geometry(source)
    column = adapt_column_geometry(source)
    cap_reinforcement, _ = _reinforcement_parts(source)
    prefix = "cap-elevation"
    entities: list[Any] = _outline_lines(
        prefix=prefix,
        layer="S-BEAM",
        role="cap_outline",
        points=_cap_outline(cap, column),
    )
    context = _cap_context(cap, column)

    for index, bar in enumerate(
        cap_reinforcement.get("longitudinal_bars", []) or [], start=1
    ):
        if not isinstance(bar, Mapping):
            continue
        range_definition = _mapping(bar.get("range_definition"))
        if not range_definition:
            entities.extend(
                _cap_control_segments(
                    prefix=prefix,
                    bar_index=index,
                    bar=bar,
                    cap=cap,
                    context=context,
                )
            )
            continue
        dia = float(bar.get("dia") or 0.0)
        values = {**context, "dia": dia}
        start_x = evaluate_drawing_expression(
            range_definition.get("x_start_expr"), values
        )
        end_x = evaluate_drawing_expression(
            range_definition.get("x_end_expr", 0), values
        )
        if str(bar.get("mirror") or "").lower() == "longitudinal":
            end_x = -start_x if math.isclose(end_x, 0.0, abs_tol=1e-9) else end_x
        entities.append(
            LineEntity(
                entity_id=f"{prefix}-cap-bar-{index:03d}",
                layer="R-MAIN",
                start=(start_x, _bar_y(bar, cap)),
                end=(end_x, _bar_y(bar, cap)),
                metadata={
                    "bar_role": "cap_longitudinal",
                    "bar_mark": str(bar.get("id") or f"N{index}"),
                    "diameter_mm": dia,
                },
            )
        )

    stirrups = _mapping(cap_reinforcement.get("stirrups"))
    stirrup_index = 0
    for zone in stirrups.get("longitudinal_distribution", []) or []:
        if not isinstance(zone, Mapping):
            continue
        start_x = evaluate_drawing_expression(zone.get("x_from"), context)
        end_x = evaluate_drawing_expression(zone.get("x_to"), context)
        spacing = evaluate_drawing_expression(zone.get("spacing"), context)
        for x in _positions(start_x, end_x, spacing):
            stirrup_index += 1
            entities.append(
                LineEntity(
                    entity_id=f"{prefix}-stirrup-{stirrup_index:04d}",
                    layer="R-STIRRUP",
                    start=(x, cap.cover_y_mm),
                    end=(x, cap.height_mid_mm - cap.cover_y_mm),
                    metadata={
                        "bar_role": "cap_stirrup",
                        "zone_name": str(zone.get("name") or "未命名分区"),
                        "spacing_mm": spacing,
                    },
                )
            )

    entities.extend(
        [
            DimensionEntity(
                entity_id=f"{prefix}-dimension-length-001",
                layer="A-DIMS",
                start=(-cap.length_mm / 2.0, 0.0),
                end=(cap.length_mm / 2.0, 0.0),
                offset=-600.0,
                metadata={"dimension_role": "cap_length"},
            ),
            DimensionEntity(
                entity_id=f"{prefix}-dimension-height-001",
                layer="A-DIMS",
                start=(cap.length_mm / 2.0, 0.0),
                end=(cap.length_mm / 2.0, cap.height_mid_mm),
                offset=600.0,
                metadata={"dimension_role": "cap_height"},
            ),
            _title(
                prefix=prefix,
                text="盖梁配筋立面",
                insertion=(-cap.length_mm / 2.0, cap.height_mid_mm + 500.0),
            ),
        ]
    )
    return GeometryView(
        view_id=prefix,
        sections=[SectionGeometry(section_id="CAP-ELEVATION", entities=entities)],
    )


def build_cap_section(source: DrawingGroupSource) -> GeometryView:
    cap = adapt_cap_geometry(source)
    cap_reinforcement, _ = _reinforcement_parts(source)
    prefix = "cap-section"
    half_width = cap.width_mm / 2.0
    outline = [
        (-half_width, 0.0),
        (half_width, 0.0),
        (half_width, cap.height_mid_mm),
        (-half_width, cap.height_mid_mm),
        (-half_width, 0.0),
    ]
    entities: list[Any] = _outline_lines(
        prefix=prefix,
        layer="S-BEAM",
        role="cap_section_outline",
        points=outline,
    )
    patterns = _mapping(cap_reinforcement.get("z_patterns"))
    bar_index = 0
    for bar in cap_reinforcement.get("longitudinal_bars", []) or []:
        if not isinstance(bar, Mapping):
            continue
        dia = float(bar.get("dia") or 0.0)
        try:
            pattern = resolve_z_positions(patterns, bar.get("z_pattern") or "")
        except DrawingExpressionError:
            pattern = None
        count = len(pattern) if isinstance(pattern, list) else 0
        if count <= 0:
            continue
        usable = cap.width_mm - 2.0 * cap.cover_z_mm
        xs = [0.0] if count == 1 else [
            -usable / 2.0 + index * usable / (count - 1)
            for index in range(count)
        ]
        y = _bar_y(bar, cap)
        for x in xs:
            bar_index += 1
            entities.append(
                CircleEntity(
                    entity_id=f"{prefix}-cap-bar-{bar_index:04d}",
                    layer="R-MAIN",
                    center=(x, y),
                    radius=dia / 2.0,
                    metadata={
                        "bar_role": "cap_longitudinal_section",
                        "bar_mark": str(bar.get("id") or ""),
                        "diameter_mm": dia,
                    },
                )
            )
    entities.extend(
        [
            DimensionEntity(
                entity_id=f"{prefix}-dimension-width-001",
                layer="A-DIMS",
                start=(-half_width, 0.0),
                end=(half_width, 0.0),
                offset=-400.0,
                metadata={"dimension_role": "cap_width"},
            ),
            _title(
                prefix=prefix,
                text="盖梁配筋截面",
                insertion=(-half_width, cap.height_mid_mm + 400.0),
            ),
        ]
    )
    return GeometryView(
        view_id=prefix,
        sections=[SectionGeometry(section_id="CAP-SECTION", entities=entities)],
    )


def build_column_elevation(source: DrawingGroupSource) -> GeometryView:
    column = adapt_column_geometry(source)
    _, column_reinforcement = _reinforcement_parts(source)
    prefix = "column-elevation"
    entities: list[Any] = []
    centers = _column_centers(column)
    radius = column.diameter_mm / 2.0
    for column_index, center_x in enumerate(centers, start=1):
        outline = [
            (center_x - radius, 0.0),
            (center_x + radius, 0.0),
            (center_x + radius, column.height_mm),
            (center_x - radius, column.height_mm),
            (center_x - radius, 0.0),
        ]
        lines = _outline_lines(
            prefix=f"{prefix}-column-{column_index:02d}",
            layer="S-COLUMN",
            role="column_outline",
            points=outline,
        )
        entities.extend(lines)

    for bar_index, bar in enumerate(
        column_reinforcement.get("longitudinal_bars", []) or [], start=1
    ):
        if not isinstance(bar, Mapping):
            continue
        dia = float(bar.get("dia") or 0.0)
        context = _column_context(column, dia=dia)
        range_definition = _mapping(bar.get("range_definition"))
        y_start = evaluate_drawing_expression(
            range_definition.get("y_start", 0), context
        )
        y_end = evaluate_drawing_expression(
            range_definition.get("y_end", column.height_mm), context
        )
        offset = column.concrete_cover_mm + dia / 2.0
        for column_index, center_x in enumerate(centers, start=1):
            for side_index, x in enumerate(
                (center_x - radius + offset, center_x + radius - offset), start=1
            ):
                entities.append(
                    LineEntity(
                        entity_id=(
                            f"{prefix}-longitudinal-{bar_index:02d}-"
                            f"{column_index:02d}-{side_index:02d}"
                        ),
                        layer="R-MAIN",
                        start=(x, y_start),
                        end=(x, y_end),
                        metadata={
                            "bar_role": "column_longitudinal",
                            "bar_mark": str(bar.get("id") or ""),
                            "diameter_mm": dia,
                        },
                    )
                )

    # 螺旋箍立面：螺旋绕柱连续上升，正立面投影为一条连续斜向折线（Z 形锯齿）。
    # 折返点即螺旋与左右极限子午线交点，相邻两次到达同侧边缘相差一个 pitch，
    # 故每段斜线竖向跨度 = pitch/2，段与段首尾相接贯通全柱（与 detail_sheets 规则一致）。
    spiral_index = 0
    cursor: dict[int, tuple[float, float] | None] = {
        index: None for index in range(1, len(centers) + 1)
    }
    for spiral in column_reinforcement.get("spiral_stirrups", []) or []:
        if not isinstance(spiral, Mapping):
            continue
        dia = float(spiral.get("dia") or 0.0)
        context = _column_context(column, dia=dia)
        section_definition = _mapping(spiral.get("section_definition"))
        hoop_diameter = evaluate_drawing_expression(
            section_definition.get(
                "hoop_diameter",
                column.diameter_mm - 2.0 * column.concrete_cover_mm + dia,
            ),
            context,
        )
        half_w = hoop_diameter / 2.0
        for zone in spiral.get("vertical_distribution", []) or []:
            if not isinstance(zone, Mapping):
                continue
            y_from = evaluate_drawing_expression(zone.get("y_from"), context)
            y_to = evaluate_drawing_expression(zone.get("y_to"), context)
            pitch = evaluate_drawing_expression(zone.get("pitch"), context)
            half_pitch = pitch / 2.0
            for column_index, center_x in enumerate(centers, start=1):
                state = cursor[column_index]
                if state is None or abs(state[0] - y_from) > 1e-6:
                    # 本柱起始或与前一分区不衔接：从 y_from 以左侧边缘起画
                    x_edge = center_x - half_w
                    y_cursor = y_from
                else:
                    x_edge, y_cursor = state
                while y_cursor < y_to - 1e-6:
                    y_next = y_cursor + half_pitch
                    if y_next > y_to + 1e-6:  # 兜底：分区高度应为 pitch 整数倍
                        y_next = y_to
                    spiral_index += 1
                    entities.append(
                        LineEntity(
                            entity_id=f"{prefix}-spiral-{spiral_index:05d}",
                            layer="R-STIRRUP",
                            start=(x_edge, y_cursor),
                            end=(center_x * 2.0 - x_edge, y_next),
                            metadata={
                                "bar_role": "column_spiral",
                                "bar_mark": str(spiral.get("id") or ""),
                                "zone_name": str(zone.get("name") or "未命名分区"),
                                "pitch_mm": pitch,
                                "continuous_helix": True,
                            },
                        )
                    )
                    x_edge = center_x * 2.0 - x_edge
                    y_cursor = y_next
                cursor[column_index] = (x_edge, y_cursor)

    right_edge = max(centers) + radius
    entities.extend(
        [
            DimensionEntity(
                entity_id=f"{prefix}-dimension-height-001",
                layer="A-DIMS",
                start=(right_edge, 0.0),
                end=(right_edge, column.height_mm),
                offset=600.0,
                metadata={"dimension_role": "column_height"},
            ),
            _title(
                prefix=prefix,
                text="墩柱配筋立面",
                insertion=(min(centers) - radius, column.height_mm + 500.0),
            ),
        ]
    )
    return GeometryView(
        view_id=prefix,
        sections=[SectionGeometry(section_id="COLUMN-ELEVATION", entities=entities)],
    )


def build_column_section(source: DrawingGroupSource) -> GeometryView:
    column = adapt_column_geometry(source)
    _, column_reinforcement = _reinforcement_parts(source)
    prefix = "column-section"
    entities: list[Any] = [
        CircleEntity(
            entity_id=f"{prefix}-outline-001",
            layer="S-COLUMN",
            center=(0.0, 0.0),
            radius=column.diameter_mm / 2.0,
            metadata={"role": "column_section_outline"},
        )
    ]
    bar_index = 0
    for bar in column_reinforcement.get("longitudinal_bars", []) or []:
        if not isinstance(bar, Mapping):
            continue
        dia = float(bar.get("dia") or 0.0)
        count = int(bar.get("count") or 0)
        context = _column_context(column, dia=dia)
        section_definition = _mapping(bar.get("section_definition"))
        centerline_diameter = evaluate_drawing_expression(
            section_definition.get(
                "bar_centerline_diameter",
                column.diameter_mm - 2.0 * column.concrete_cover_mm - dia,
            ),
            context,
        )
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            bar_index += 1
            entities.append(
                CircleEntity(
                    entity_id=f"{prefix}-longitudinal-{bar_index:04d}",
                    layer="R-MAIN",
                    center=(
                        centerline_diameter / 2.0 * math.cos(angle),
                        centerline_diameter / 2.0 * math.sin(angle),
                    ),
                    radius=dia / 2.0,
                    metadata={
                        "bar_role": "column_longitudinal",
                        "bar_mark": str(bar.get("id") or ""),
                        "diameter_mm": dia,
                    },
                )
            )

    for index, spiral in enumerate(
        column_reinforcement.get("spiral_stirrups", []) or [], start=1
    ):
        if not isinstance(spiral, Mapping):
            continue
        dia = float(spiral.get("dia") or 0.0)
        context = _column_context(column, dia=dia)
        section_definition = _mapping(spiral.get("section_definition"))
        hoop_diameter = evaluate_drawing_expression(
            section_definition.get(
                "hoop_diameter",
                column.diameter_mm - 2.0 * column.concrete_cover_mm + dia,
            ),
            context,
        )
        entities.append(
            CircleEntity(
                entity_id=f"{prefix}-spiral-{index:03d}",
                layer="R-STIRRUP",
                center=(0.0, 0.0),
                radius=hoop_diameter / 2.0,
                metadata={
                    "bar_role": "column_spiral_section",
                    "bar_mark": str(spiral.get("id") or ""),
                    "diameter_mm": dia,
                },
            )
        )

    entities.extend(
        [
            DimensionEntity(
                entity_id=f"{prefix}-dimension-diameter-001",
                layer="A-DIMS",
                start=(-column.diameter_mm / 2.0, 0.0),
                end=(column.diameter_mm / 2.0, 0.0),
                offset=-400.0,
                metadata={"dimension_role": "column_diameter"},
            ),
            _title(
                prefix=prefix,
                text="墩柱配筋截面",
                insertion=(-column.diameter_mm / 2.0, column.diameter_mm / 2.0 + 400.0),
            ),
        ]
    )
    return GeometryView(
        view_id=prefix,
        sections=[SectionGeometry(section_id="COLUMN-SECTION", entities=entities)],
    )


def _unsupported_cap_bar_marks(source: DrawingGroupSource) -> list[str]:
    cap, _ = _reinforcement_parts(source)
    unsupported: list[str] = []
    for bar in cap.get("longitudinal_bars", []) or []:
        if not isinstance(bar, Mapping):
            continue
        if isinstance(bar.get("range_definition"), Mapping):
            continue
        control = _mapping(bar.get("control_definition"))
        points = _mapping(control.get("key_points"))
        if points.get("left_bend_point") and points.get("right_bend_point"):
            continue
        if points.get("bend_start_point"):
            continue
        if (
            points.get("arc_left_point")
            and points.get("arc_right_point")
            and control.get("arc_definition")
        ):
            continue
        unsupported.append(str(bar.get("id") or ""))
    return unsupported


def build_reinforcement_drawing(source: DrawingGroupSource) -> DrawingDocument:
    unsupported = [mark for mark in _unsupported_cap_bar_marks(source) if mark]
    return DrawingDocument(
        design_group_id=source.design_group_id,
        member_piers=source.member_piers,
        layers=list(DRAWING_LAYERS),
        views=[
            build_cap_elevation(source),
            build_cap_section(source),
            build_column_elevation(source),
            build_column_section(source),
        ],
        metadata={
            "task_id": source.task_id,
            "check_status": source.check_status,
            "unsupported_cap_bar_marks": unsupported,
            "source_hashes": dict(source.source_hashes),
        },
    )
