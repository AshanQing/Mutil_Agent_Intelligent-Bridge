from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict

from .cap_reinforcement import (
    BarPathSegment,
    compile_cap_bar_paths,
    intersections_at_x,
)
from .column_reinforcement import compile_column_reinforcement
from .config import DrawingConfig
from .models import (
    ArcEntity,
    CircleEntity,
    DimensionEntity,
    DrawingDocument,
    DrawingLayer,
    GeometryView,
    LineEntity,
    SectionGeometry,
    TextEntity,
)
from .reinforcement_adapter import DrawingExpressionError, resolve_z_positions
from .reinforcement_resolution import ResolvedCap, ResolvedPierReinforcement


DETAIL_LAYERS = [
    DrawingLayer(name="S-BEAM", color_rgb=(240, 240, 240), lineweight_mm=0.50),
    DrawingLayer(name="S-COLUMN", color_rgb=(180, 180, 180), lineweight_mm=0.50),
    DrawingLayer(name="R-MAIN", color_rgb=(255, 60, 60), lineweight_mm=0.35),
    DrawingLayer(name="R-EXTRA", color_rgb=(255, 160, 40), lineweight_mm=0.30),
    DrawingLayer(name="R-STIRRUP", color_rgb=(0, 190, 255), lineweight_mm=0.25),
    DrawingLayer(name="R-REVIEW", color_rgb=(255, 0, 255), lineweight_mm=0.40),
    DrawingLayer(name="A-DIMS", color_rgb=(40, 220, 100), lineweight_mm=0.18),
    DrawingLayer(name="A-TEXT", color_rgb=(255, 230, 60), lineweight_mm=0.18),
    DrawingLayer(name="A-TITLE", color_rgb=(255, 255, 255), lineweight_mm=0.25),
]


class DrawingSheetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheet_id: str
    title: str
    document: DrawingDocument


def _lines(
    prefix: str,
    points: list[tuple[float, float]],
    *,
    layer: str,
    metadata: dict[str, Any],
) -> list[LineEntity]:
    return [
        LineEntity(
            entity_id=f"{prefix}-{index:03d}",
            layer=layer,
            start=points[index - 1],
            end=points[index],
            metadata=metadata,
        )
        for index in range(1, len(points))
    ]


def _title(prefix: str, text: str, at: tuple[float, float]) -> TextEntity:
    return TextEntity(
        entity_id=f"{prefix}-title-001",
        layer="A-TITLE",
        insertion=at,
        text=text,
        height=180,
        metadata={"role": "view_title"},
    )


def _document(
    *,
    design_group_id: str,
    member_piers: list[str],
    sheet_id: str,
    views: list[GeometryView],
    diagnostics: list[dict[str, Any]],
) -> DrawingDocument:
    return DrawingDocument(
        design_group_id=design_group_id,
        member_piers=member_piers,
        layers=list(DETAIL_LAYERS),
        views=views,
        metadata={
            "sheet_id": sheet_id,
            "representative_design": True,
            "diagnostics": diagnostics,
        },
    )


def _cap_flat_half(cap: ResolvedCap) -> float:
    """平底段半长 = 盖梁半长 - 悬臂长（用户确认：总长-2×悬臂端长度）。"""
    half = cap.length_mm / 2
    return half - cap.cantilever_mm


def _cap_bottom_y(cap: ResolvedCap, x_mm: float) -> float:
    half = cap.length_mm / 2
    flat_half = _cap_flat_half(cap)
    absolute_x = abs(x_mm)
    if absolute_x <= flat_half:
        return 0.0
    end_bottom = cap.height_mid_mm - cap.height_end_mm
    ratio = min(max((absolute_x - flat_half) / (half - flat_half), 0.0), 1.0)
    return ratio * end_bottom


def _cap_outline_points(cap: ResolvedCap, *, y_offset: float = 0.0):
    half = cap.length_mm / 2
    flat_half = _cap_flat_half(cap)
    end_bottom = cap.height_mid_mm - cap.height_end_mm
    return [
        (-half, y_offset + cap.height_mid_mm),
        (half, y_offset + cap.height_mid_mm),
        (half, y_offset + end_bottom),
        (flat_half, y_offset),
        (-flat_half, y_offset),
        (-half, y_offset + end_bottom),
        (-half, y_offset + cap.height_mid_mm),
    ]


def _general_arrangement(
    resolved: ResolvedPierReinforcement,
) -> GeometryView:
    cap = resolved.cap
    column = resolved.column
    entities: list[Any] = []
    half_column = column.diameter_mm / 2
    centers = (
        [0.0]
        if column.count == 1
        else [
            (index - (column.count - 1) / 2) * column.spacing_mm
            for index in range(column.count)
        ]
    )
    for index, center in enumerate(centers, start=1):
        entities.extend(
            _lines(
                f"pier-ga-column-{index}",
                [
                    (center - half_column, 0),
                    (center + half_column, 0),
                    (center + half_column, column.height_mm),
                    (center - half_column, column.height_mm),
                    (center - half_column, 0),
                ],
                layer="S-COLUMN",
                metadata={"role": "column_outline", "column_index": index},
            )
        )
        bar_radius = (column.longitudinal.centerline_diameter_mm or 0) / 2
        for side, x in (("left", center - bar_radius), ("right", center + bar_radius)):
            entities.append(
                LineEntity(
                    entity_id=f"pier-ga-column-{index}-n1-{side}",
                    layer="R-MAIN",
                    start=(x, 0),
                    end=(x, column.longitudinal.y_to_mm or column.height_mm),
                    metadata={
                        "bar_mark": "N1",
                        "bar_role": "column_longitudinal_extent",
                        "column_index": index,
                    },
                )
            )
    entities.extend(
        _lines(
            "pier-ga-cap",
            _cap_outline_points(cap, y_offset=column.height_mm),
            layer="S-BEAM",
            metadata={"role": "cap_outline"},
        )
    )
    entities.extend(
        [
            DimensionEntity(
                entity_id="pier-ga-dimension-column-height",
                layer="A-DIMS",
                start=(centers[0] - half_column, 0),
                end=(centers[0] - half_column, column.height_mm),
                offset=-900,
                text_override=f"柱高 {column.height_mm:g}",
                metadata={"dimension_role": "column_height"},
            ),
            DimensionEntity(
                entity_id="pier-ga-dimension-column-spacing",
                layer="A-DIMS",
                start=(centers[0], 0),
                end=(centers[-1], 0),
                offset=-700,
                text_override=f"柱间距 {column.spacing_mm:g}",
                metadata={"dimension_role": "column_spacing"},
            ),
            _title(
                "pier-ga",
                "桥墩总体布置（同类墩柱采用一套代表性配筋）",
                (-cap.length_mm / 2, column.height_mm + cap.height_mid_mm + 600),
            ),
            TextEntity(
                entity_id="pier-ga-note-001",
                layer="A-TEXT",
                insertion=(-cap.length_mm / 2, -1300),
                text=(
                    f"墩柱：{column.count}×Φ{column.diameter_mm:g}；"
                    f"N1 每柱 {column.longitudinal.count}Φ"
                    f"{column.longitudinal.diameter_mm:g}，伸入盖梁 "
                    f"{(column.longitudinal.y_to_mm or column.height_mm) - column.height_mm:g}"
                ),
                height=150,
            ),
        ]
    )
    return GeometryView(
        view_id="pier-general-arrangement",
        sections=[SectionGeometry(section_id="PIER-GENERAL", entities=entities)],
    )


def _segment_review_required(cap: ResolvedCap, segment: BarPathSegment) -> bool:
    half = cap.length_mm / 2
    points = [segment.start, segment.end]
    if segment.center is not None and segment.radius_mm is not None:
        points.append((segment.center[0], segment.center[1] - segment.radius_mm))
    return any(
        abs(x) > half + 1e-6
        or y > cap.height_mid_mm + 1e-6
        or y < _cap_bottom_y(cap, x) - 1e-6
        for x, y in points
    )


def _cap_skeleton_view(
    resolved: ResolvedPierReinforcement,
    *,
    skeleton_id: str,
    view_id: str,
) -> GeometryView:
    cap = resolved.cap
    skeleton = cap.skeletons[skeleton_id]
    paths = compile_cap_bar_paths(cap)
    entities: list[Any] = _lines(
        f"{view_id}-outline",
        _cap_outline_points(cap),
        layer="S-BEAM",
        metadata={"role": "cap_outline"},
    )
    for mark in skeleton.bar_marks:
        path = paths[mark]
        for component_index, component in enumerate(path.components, start=1):
            for segment_index, segment in enumerate(component.segments, start=1):
                review = _segment_review_required(cap, segment)
                layer = "R-REVIEW" if review else (
                    "R-MAIN" if mark in {"N1", "N2"} else "R-EXTRA"
                )
                metadata = {
                    "bar_mark": mark,
                    "skeleton_id": skeleton_id,
                    "component_id": component.component_id,
                    "review_required": review,
                }
                entity_id = (
                    f"{view_id}-{mark}-{component_index:02d}-{segment_index:02d}"
                )
                if segment.kind == "arc":
                    # 按实际端点相对圆心的角度表达圆弧：
                    # start=arc_left，end=arc_right；upper 弧经圆心正上方顶点，
                    # 用 clockwise 交换使 AutoCAD 逆时针从右侧点扫到左侧点（过顶点）
                    start_angle = math.degrees(
                        math.atan2(
                            segment.start[1] - segment.center[1],
                            segment.start[0] - segment.center[0],
                        )
                    ) % 360
                    end_angle = math.degrees(
                        math.atan2(
                            segment.end[1] - segment.center[1],
                            segment.end[0] - segment.center[0],
                        )
                    ) % 360
                    entities.append(
                        ArcEntity(
                            entity_id=entity_id,
                            layer=layer,
                            center=segment.center,
                            radius=segment.radius_mm,
                            start_angle_deg=start_angle,
                            end_angle_deg=end_angle,
                            clockwise=segment.arc_bulge == "upper",
                            metadata=metadata,
                        )
                    )
                else:
                    entities.append(
                        LineEntity(
                            entity_id=entity_id,
                            layer=layer,
                            start=segment.start,
                            end=segment.end,
                            metadata=metadata,
                        )
                    )
    entities.extend(
        [
            _title(
                view_id,
                f"{skeleton.name}（{skeleton_id}）",
                (-cap.length_mm / 2, cap.height_mid_mm + 500),
            ),
            TextEntity(
                entity_id=f"{view_id}-note-001",
                layer="A-TEXT",
                insertion=(-cap.length_mm / 2, -450),
                text=(
                    f"钢筋：{', '.join(skeleton.bar_marks)}；"
                    f"横向位置：{', '.join(map(str, skeleton.z_positions))}"
                ),
                height=130,
            ),
        ]
    )
    return GeometryView(
        view_id=view_id,
        sections=[SectionGeometry(section_id=view_id.upper(), entities=entities)],
        metadata={"skeleton_id": skeleton_id},
    )


def _stirrup_positions(start: float, end: float, spacing: float) -> list[float]:
    direction = 1 if end >= start else -1
    length = abs(end - start)
    count = int(math.floor(length / spacing + 1e-9))
    positions = [start + direction * spacing * index for index in range(count + 1)]
    if not math.isclose(positions[-1], end, abs_tol=1e-6):
        positions.append(end)
    return positions


def _cap_stirrup_view(resolved: ResolvedPierReinforcement) -> GeometryView:
    cap = resolved.cap
    mismatch_names = {
        item.field_path.rsplit(".", 1)[-1]
        for item in resolved.diagnostics
        if item.code == "stirrup_segment_length_mismatch"
    }
    entities: list[Any] = [
        _title(
            "cap-stirrup-distribution",
            "盖梁箍筋纵向分区（右半按左半镜像）",
            (-cap.length_mm / 2, 1100),
        )
    ]
    for side, multiplier in (("left", 1.0), ("right", -1.0)):
        for zone_index, zone in enumerate(cap.stirrup_segments, start=1):
            start = multiplier * zone.x_from_mm
            end = multiplier * zone.x_to_mm
            review = zone.name in mismatch_names
            layer = "R-REVIEW" if review else "R-STIRRUP"
            for position_index, x in enumerate(
                _stirrup_positions(start, end, zone.spacing_mm), start=1
            ):
                entities.append(
                    LineEntity(
                        entity_id=(
                            f"cap-stirrup-{side}-{zone_index:02d}-{position_index:03d}"
                        ),
                        layer=layer,
                        # 箍筋为竖直贯穿截面的线，随悬臂段变截面：
                        # 顶 y = 顶面-cover_y；底 y = 底面(x)+cover_y
                        start=(x, _cap_bottom_y(cap, x) + cap.cover_y_mm),
                        end=(x, cap.height_mid_mm - cap.cover_y_mm),
                        metadata={
                            "bar_mark": "CAP-STIRRUP",
                            "zone_name": zone.name,
                            "review_required": review,
                        },
                    )
                )
            midpoint = (start + end) / 2
            entities.append(
                TextEntity(
                    entity_id=f"cap-stirrup-{side}-{zone_index:02d}-label",
                    layer="A-TEXT" if not review else "R-REVIEW",
                    insertion=(midpoint, 700),
                    text=f"{zone.name} @{zone.spacing_mm:g}",
                    height=90,
                    rotation_deg=90,
                    metadata={"review_required": review},
                )
            )
    entities.append(
        DimensionEntity(
            entity_id="cap-stirrup-dimension-length",
            layer="A-DIMS",
            start=(-cap.length_mm / 2, 0),
            end=(cap.length_mm / 2, 0),
            offset=-350,
            metadata={"dimension_role": "cap_length"},
        )
    )
    return GeometryView(
        view_id="cap-stirrup-distribution",
        sections=[
            SectionGeometry(section_id="CAP-STIRRUP-DISTRIBUTION", entities=entities)
        ],
    )


def _z_coordinate(cap: ResolvedCap, position: int) -> float:
    usable_width = cap.width_mm - 2 * cap.cover_z_mm
    total_positions = max(cap.z_patterns.get("full", [1]))
    if total_positions == 1:
        return 0.0
    return -usable_width / 2 + (position - 1) * usable_width / (total_positions - 1)


def _cap_section_view(
    resolved: ResolvedPierReinforcement,
    *,
    view_id: str,
    station_x_mm: float,
    title: str,
) -> GeometryView:
    cap = resolved.cap
    paths = compile_cap_bar_paths(cap)
    bottom = _cap_bottom_y(cap, station_x_mm)
    section_height = cap.height_mid_mm - bottom
    half_width = cap.width_mm / 2
    entities: list[Any] = _lines(
        f"{view_id}-outline",
        [
            (-half_width, 0),
            (half_width, 0),
            (half_width, section_height),
            (-half_width, section_height),
            (-half_width, 0),
        ],
        layer="S-BEAM",
        metadata={"role": "cap_section_outline", "station_x_mm": station_x_mm},
    )
    for skeleton_id, skeleton in cap.skeletons.items():
        for mark in skeleton.bar_marks:
            for y_index, y in enumerate(intersections_at_x(paths[mark], station_x_mm), start=1):
                local_y = y - bottom
                review = local_y < 0 or local_y > section_height
                for position in skeleton.z_positions:
                    entities.append(
                        CircleEntity(
                            entity_id=(
                                f"{view_id}-{skeleton_id}-{mark}-{y_index:02d}-{position:02d}"
                            ),
                            layer="R-REVIEW" if review else (
                                "R-MAIN" if mark in {"N1", "N2"} else "R-EXTRA"
                            ),
                            center=(_z_coordinate(cap, position), local_y),
                            radius=cap.bars[mark].diameter_mm / 2,
                            metadata={
                                "bar_role": "cap_longitudinal_section",
                                "bar_mark": mark,
                                "skeleton_id": skeleton_id,
                                "station_x_mm": station_x_mm,
                                "review_required": review,
                            },
                        )
                    )
    cage_layouts = cap.transverse_cage.get("cage_layouts", {})
    components = cage_layouts.get("components", []) if isinstance(cage_layouts, dict) else []
    for component_index, component in enumerate(components, start=1):
        if not isinstance(component, dict):
            continue
        start_position, end_position = component["enclosed_range"]
        # 箍筋包在最外侧钢筋外侧：从最外侧钢筋中心外扩 主筋半径 + 箍筋半径
        main_bar_radius = max(
            (cap.bars[mark].diameter_mm / 2 for mark in cap.bars), default=0.0
        )
        cage_dia = float(cap.transverse_cage.get("dia") or 12.0)
        cage_inset = main_bar_radius + cage_dia / 2.0
        left = _z_coordinate(cap, int(start_position)) - cage_inset
        right = _z_coordinate(cap, int(end_position)) + cage_inset
        cage_id = str(component.get("id"))
        entities.extend(
            _lines(
                f"{view_id}-cage-{component_index:02d}",
                [
                    (left, cap.cover_y_mm - cage_inset),
                    (right, cap.cover_y_mm - cage_inset),
                    (right, section_height - cap.cover_y_mm + cage_inset),
                    (left, section_height - cap.cover_y_mm + cage_inset),
                    (left, cap.cover_y_mm - cage_inset),
                ],
                layer="R-STIRRUP",
                metadata={"cage_component_id": cage_id, "bar_role": "eight_leg_cage"},
            )
        )
    entities.extend(
        [
            _title(view_id, f"{title} x={station_x_mm:g}", (-half_width, section_height + 350)),
            DimensionEntity(
                entity_id=f"{view_id}-dimension-width",
                layer="A-DIMS",
                start=(-half_width, 0),
                end=(half_width, 0),
                offset=-250,
                metadata={"dimension_role": "cap_width"},
            ),
        ]
    )
    return GeometryView(
        view_id=view_id,
        sections=[SectionGeometry(section_id=view_id.upper(), entities=entities)],
        metadata={"station_x_mm": station_x_mm, "layout_scale": 2.0},
    )


def _schedule_view(
    *,
    view_id: str,
    title: str,
    rows: list[str],
) -> GeometryView:
    entities: list[Any] = [_title(view_id, title, (0, 500))]
    for index, row in enumerate(rows, start=1):
        entities.append(
            TextEntity(
                entity_id=f"{view_id}-row-{index:03d}",
                layer="A-TEXT",
                insertion=(0, 500 - index * 220),
                text=row,
                height=130,
                metadata={"role": "schedule_row"},
            )
        )
    return GeometryView(
        view_id=view_id,
        sections=[SectionGeometry(section_id=view_id.upper(), entities=entities)],
        metadata={"layout_scale": 1.5},
    )


def _cap_sheet_views(resolved: ResolvedPierReinforcement) -> list[GeometryView]:
    cap_rows = []
    for mark, bar in resolved.cap.bars.items():
        try:
            positions = resolve_z_positions(resolved.cap.z_patterns, bar.z_pattern or "full")
        except DrawingExpressionError:
            positions = []
        count = len(positions)
        cap_rows.append(
            f"{mark}  Φ{bar.diameter_mm:g}  {bar.subtype}  数量={count}"
        )
    cap_rows.extend(
        [
            "横向箍筋 Φ12，8肢箍：N10_left/N10_right/N11_left/N11_right",
            "紫色图元表示样例几何或计数存在待复核项",
        ]
    )
    return [
        _cap_skeleton_view(resolved, skeleton_id="skeleton_1", view_id="cap-skeleton-1"),
        _cap_skeleton_view(resolved, skeleton_id="skeleton_2", view_id="cap-skeleton-2"),
        _cap_stirrup_view(resolved),
        _cap_section_view(
            resolved, view_id="cap-section-end", station_x_mm=-5200, title="A-A 端部断面"
        ),
        _cap_section_view(
            resolved, view_id="cap-section-support", station_x_mm=-3500, title="B-B 支点断面"
        ),
        _cap_section_view(
            resolved, view_id="cap-section-mid", station_x_mm=0, title="C-C 跨中断面"
        ),
        _schedule_view(view_id="cap-bar-schedule", title="盖梁钢筋表", rows=cap_rows),
    ]


def _column_elevation(resolved: ResolvedPierReinforcement) -> GeometryView:
    column = resolved.column
    cap = resolved.cap
    detail = compile_column_reinforcement(column)
    half = column.diameter_mm / 2
    entities: list[Any] = _lines(
        "column-elevation-outline",
        [(-half, 0), (half, 0), (half, column.height_mm), (-half, column.height_mm), (-half, 0)],
        layer="S-COLUMN",
        metadata={"role": "column_outline", "representative_column": True},
    )
    entities.extend(
        _lines(
            "column-elevation-cap-embed",
            [
                (-cap.width_mm / 2, column.height_mm),
                (cap.width_mm / 2, column.height_mm),
                (cap.width_mm / 2, column.height_mm + cap.height_mid_mm),
                (-cap.width_mm / 2, column.height_mm + cap.height_mid_mm),
                (-cap.width_mm / 2, column.height_mm),
            ],
            layer="S-BEAM",
            metadata={"role": "cap_embed_outline"},
        )
    )
    bar_radius = (column.longitudinal.centerline_diameter_mm or 0) / 2
    # N1 主筋立面：44 根环向均布，正立面画可见弧面（θ∈[0°,180°] 投影）内的根
    count = column.longitudinal.count or 0
    y_n1_from = column.longitudinal.y_from_mm or 0
    y_n1_to = detail.longitudinal_y_to_mm
    if count > 0:
        for visible_index in range(count // 2 + 1):
            theta = math.radians(180.0 * visible_index / (count // 2))
            x = bar_radius * math.cos(theta)
            entities.append(
                LineEntity(
                    entity_id=f"column-elevation-n1-{visible_index:02d}",
                    layer="R-MAIN",
                    start=(x, y_n1_from),
                    end=(x, y_n1_to),
                    metadata={"bar_mark": "N1", "bar_role": "visible_longitudinal"},
                )
            )
    else:
        for side, x in (("left", -bar_radius), ("right", bar_radius)):
            entities.append(
                LineEntity(
                    entity_id=f"column-elevation-n1-{side}",
                    layer="R-MAIN",
                    start=(x, y_n1_from),
                    end=(x, y_n1_to),
                    metadata={"bar_mark": "N1", "bar_role": "visible_longitudinal"},
                )
            )
    # N2 螺旋箍立面：螺旋绕柱连续上升，正立面投影为一条连续斜向折线（Z 形锯齿）。
    # 折返点即螺旋线与左右极限子午线的交点：真实螺旋相邻两次到达同侧边缘相差
    # 恰好一个 pitch，故每段斜线竖向跨度 = pitch/2（左右交替两段合成一圈一个 pitch），
    # 段与段首尾相接自柱底贯通至柱顶，整条折线连续不断、圈距与分区 pitch 严格对应。
    spiral_w = detail.spiral_centerline_radius_mm
    x_edge = -spiral_w
    y_cursor = detail.spiral_zones[0].y_from_mm
    cont_index = 0
    for zone_index, zone in enumerate(detail.spiral_zones, start=1):
        pitch = zone.pitch_mm if zone.pitch_mm > 0 else 1.0
        half_pitch = pitch / 2.0
        y_cursor = max(y_cursor, zone.y_from_mm)
        while y_cursor < zone.y_to_mm - 1e-6:
            y_next = y_cursor + half_pitch
            if y_next > zone.y_to_mm + 1e-6:  # 兜底：分区高度应为 pitch 整数倍
                y_next = zone.y_to_mm
            cont_index += 1
            entities.append(
                LineEntity(
                    entity_id=f"column-elevation-n2-cont-{cont_index:04d}",
                    layer="R-STIRRUP",
                    start=(x_edge, y_cursor),
                    end=(-x_edge, y_next),
                    metadata={
                        "bar_mark": "N2",
                        "zone_name": zone.name,
                        "pitch_mm": pitch,
                        "continuous_helix": True,
                    },
                )
            )
            x_edge = -x_edge
            y_cursor = y_next
        entities.append(
            TextEntity(
                entity_id=f"column-elevation-n2-zone-{zone_index:02d}-label",
                layer="A-TEXT",
                insertion=(half + 300, (zone.y_from_mm + zone.y_to_mm) / 2),
                text=f"N2 {zone.name} @{pitch:g}",
                height=110,
            )
        )
    if not column.ordinary_hoop.bar.diameter_mm <= 1:
        for index, y in enumerate(detail.ordinary_hoop_positions_mm, start=1):
            entities.append(
                LineEntity(
                    entity_id=f"column-elevation-n3-{index:02d}",
                    layer="R-STIRRUP",
                    start=(-detail.spiral_centerline_radius_mm, y),
                    end=(detail.spiral_centerline_radius_mm, y),
                    metadata={"bar_mark": "N3", "bar_role": "ordinary_hoop"},
                )
            )
    if not column.strengthening.bar.diameter_mm <= 1:
        for index, y in enumerate(detail.strengthening_positions_mm, start=1):
            entities.append(
                LineEntity(
                    entity_id=f"column-elevation-n4-{index:02d}",
                    layer="R-REVIEW",
                    start=(-detail.strengthening_radius_mm, y),
                    end=(detail.strengthening_radius_mm, y),
                    metadata={"bar_mark": "N4", "review_required": True},
                )
            )
    entities.extend(
        [
            _title(
                "column-elevation",
                "代表性单柱配筋立面（同组2柱共用）",
                (-cap.width_mm / 2, column.height_mm + cap.height_mid_mm + 500),
            ),
            TextEntity(
                entity_id="column-elevation-n1-label",
                layer="A-TEXT",
                insertion=(half + 300, column.height_mm + 900),
                text=f"N1 {column.longitudinal.count}Φ{column.longitudinal.diameter_mm:g} 环向均布",
                height=120,
                metadata={"bar_mark": "N1"},
            ),
            TextEntity(
                entity_id="column-elevation-n4-review-label",
                layer="R-REVIEW",
                insertion=(half + 300, 1200),
                text="N4 环向加强筋细部语义待原图复核",
                height=110,
                metadata={"bar_mark": "N4", "review_required": True},
            ),
            DimensionEntity(
                entity_id="column-elevation-height-dimension",
                layer="A-DIMS",
                start=(-half, 0),
                end=(-half, column.height_mm),
                offset=-700,
                text_override=f"柱高 {column.height_mm:g}",
                metadata={"dimension_role": "column_height"},
            ),
        ]
    )
    return GeometryView(
        view_id="column-representative-elevation",
        sections=[SectionGeometry(section_id="COLUMN-ELEVATION", entities=entities)],
    )


def _column_section(
    resolved: ResolvedPierReinforcement,
    *,
    view_id: str,
    embed: bool,
) -> GeometryView:
    column = resolved.column
    detail = compile_column_reinforcement(column)
    radius = column.diameter_mm / 2
    entities: list[Any] = [
        CircleEntity(
            entity_id=f"{view_id}-outline",
            layer="S-COLUMN",
            center=(0, 0),
            radius=radius,
            metadata={"role": "column_section_outline"},
        ),
    ]
    # N1 纵筋：44 根环向均布，钢筋中心距 = 半径中心；外缘被箍筋包裹
    for index, center in enumerate(detail.longitudinal_centers, start=1):
        entities.append(
            CircleEntity(
                entity_id=f"{view_id}-n1-{index:02d}",
                layer="R-MAIN",
                center=center,
                radius=column.longitudinal.diameter_mm / 2,
                metadata={"bar_mark": "N1", "bar_role": "column_longitudinal"},
            )
        )
    if embed:
        # ① 伸入盖梁区截面：N3 普通箍（same_circle_as N2，外圈包裹 N1），无螺旋
        if not column.ordinary_hoop.bar.diameter_mm <= 1:
            entities.append(
                CircleEntity(
                    entity_id=f"{view_id}-n3",
                    layer="R-STIRRUP",
                    center=(0, 0),
                    radius=detail.spiral_centerline_radius_mm,
                    metadata={"bar_mark": "N3", "bar_role": "ordinary_hoop"},
                )
            )
    else:
        # ② 柱身截面：N2 螺旋箍（外圈）+ N4 加强箍（内圈，纵筋内侧）+ N1 纵筋
        entities.append(
            CircleEntity(
                entity_id=f"{view_id}-n2",
                layer="R-STIRRUP",
                center=(0, 0),
                radius=detail.spiral_centerline_radius_mm,
                metadata={"bar_mark": "N2"},
            )
        )
        if not column.strengthening.bar.diameter_mm <= 1:
            entities.append(
                CircleEntity(
                    entity_id=f"{view_id}-n4",
                    layer="R-REVIEW",
                    center=(0, 0),
                    radius=detail.strengthening_radius_mm,
                    metadata={"bar_mark": "N4", "review_required": True},
                )
            )
    entities.extend(
        [
            _title(
                view_id,
                "墩柱嵌入盖梁区断面" if embed else "普通柱身断面",
                (-radius, radius + 350),
            ),
            DimensionEntity(
                entity_id=f"{view_id}-diameter-dimension",
                layer="A-DIMS",
                start=(-radius, 0),
                end=(radius, 0),
                offset=-300,
                metadata={"dimension_role": "column_diameter"},
            ),
        ]
    )
    return GeometryView(
        view_id=view_id,
        sections=[SectionGeometry(section_id=view_id.upper(), entities=entities)],
        metadata={"layout_scale": 3.0},
    )


def _column_sheet_views(resolved: ResolvedPierReinforcement) -> list[GeometryView]:
    column = resolved.column
    spiral_distribution = "；".join(
        f"{zone.y_from_mm:g}-{zone.y_to_mm:g}@{zone.pitch_mm:g}"
        for zone in column.spiral_zones
    )
    rows = [
        (
            f"{column.longitudinal.mark}  Φ{column.longitudinal.diameter_mm:g}  "
            f"每柱{column.longitudinal.count}根  环向均布"
        ),
        f"{column.spiral.mark}  Φ{column.spiral.diameter_mm:g}  螺旋箍：{spiral_distribution}",
        (
            f"{column.ordinary_hoop.bar.mark}  Φ{column.ordinary_hoop.bar.diameter_mm:g}  "
            f"{column.ordinary_hoop.count}道@{column.ordinary_hoop.spacing_mm:g}  "
            "位于盖梁嵌入区"
        ),
        (
            f"{column.strengthening.bar.mark}  Φ{column.strengthening.bar.diameter_mm:g}  "
            f"{column.strengthening.count}道  "
            f"环径{column.strengthening.hoop_diameter_mm:g}（细部语义待复核）"
        ),
    ]
    return [
        _column_elevation(resolved),
        _column_section(resolved, view_id="column-section-body", embed=False),
        _column_section(resolved, view_id="column-section-embed", embed=True),
        _schedule_view(view_id="column-bar-schedule", title="墩柱钢筋表", rows=rows),
    ]


def sheet_layout_config(sheet_id: str) -> DrawingConfig:
    """每个分页图纸的视图排版配置（正式打包与样例渲染共用）。"""
    rows = {
        "pier_general_arrangement": [["pier-general-arrangement"]],
        "cap_reinforcement_detail": [
            ["cap-skeleton-1"],
            ["cap-skeleton-2"],
            ["cap-stirrup-distribution"],
            ["cap-section-end", "cap-section-support", "cap-section-mid"],
            ["cap-bar-schedule"],
        ],
        "column_reinforcement_detail": [
            [
                "column-representative-elevation",
                "column-section-body",
                "column-section-embed",
            ],
            ["column-bar-schedule"],
        ],
    }[sheet_id]
    return DrawingConfig(
        view_rows=rows,
        column_gap_mm=1200,
        row_gap_mm=1200,
        config_source="selected_reinforcement_samples",
    )


def build_reinforcement_sheets(
    resolved: ResolvedPierReinforcement,
    *,
    design_group_id: str,
    member_piers: list[str],
) -> list[DrawingSheetSpec]:
    diagnostics = [item.model_dump(mode="json") for item in resolved.diagnostics]
    # 分页图纸仅保留两张配筋详图：盖梁配筋详图 + 墩柱配筋详图（不再输出总体布置图）。
    specifications = [
        (
            "cap_reinforcement_detail",
            "盖梁配筋详图",
            _cap_sheet_views(resolved),
        ),
        (
            "column_reinforcement_detail",
            "墩柱配筋详图",
            _column_sheet_views(resolved),
        ),
    ]
    return [
        DrawingSheetSpec(
            sheet_id=sheet_id,
            title=title,
            document=_document(
                design_group_id=design_group_id,
                member_piers=member_piers,
                sheet_id=sheet_id,
                views=views,
                diagnostics=diagnostics,
            ),
        )
        for sheet_id, title, views in specifications
    ]
