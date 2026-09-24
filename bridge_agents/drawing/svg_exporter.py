from __future__ import annotations

import html
import math
from collections.abc import Callable

from .layout import Bounds, entity_bounds
from .models import (
    ArcEntity,
    CircleEntity,
    DimensionEntity,
    DrawingDocument,
    LeaderEntity,
    LineEntity,
    Point2D,
    TextEntity,
)


_CANVAS_WIDTH = 1600
_CANVAS_HEIGHT = 1200
_PADDING = {"left": 80.0, "right": 80.0, "top": 110.0, "bottom": 70.0}
_BACKGROUND = "#111827"
_VIEW_LABELS = {
    "cap-elevation": "盖梁立面",
    "cap-section": "盖梁截面",
    "column-elevation": "墩柱立面",
    "column-section": "墩柱截面",
}


def _number(value: float) -> str:
    return f"{value:.3f}"


def _screen_transform(
    bounds: Bounds,
) -> tuple[Callable[[Point2D], Point2D], float]:
    drawing_width = max(bounds.width, 1.0)
    drawing_height = max(bounds.height, 1.0)
    inner_width = _CANVAS_WIDTH - _PADDING["left"] - _PADDING["right"]
    inner_height = _CANVAS_HEIGHT - _PADDING["top"] - _PADDING["bottom"]
    scale = min(inner_width / drawing_width, inner_height / drawing_height)
    used_width = drawing_width * scale
    used_height = drawing_height * scale
    origin_x = _PADDING["left"] + (inner_width - used_width) / 2.0
    origin_y = _PADDING["top"] + (inner_height - used_height) / 2.0

    def screen(point: Point2D) -> Point2D:
        return Point2D(
            x=origin_x + (point.x - bounds.min_x) * scale,
            y=origin_y + (bounds.max_y - point.y) * scale,
        )

    return screen, scale


def _stroke_width(lineweight_mm: float) -> float:
    return min(max(lineweight_mm * 4.0, 1.2), 3.5)


def _line(
    *,
    entity_id: str,
    start: Point2D,
    end: Point2D,
    color: str,
    stroke_width: float,
) -> str:
    return (
        f'<line id="{html.escape(entity_id)}" x1="{_number(start.x)}" '
        f'y1="{_number(start.y)}" x2="{_number(end.x)}" '
        f'y2="{_number(end.y)}" stroke="{color}" '
        f'stroke-width="{_number(stroke_width)}" fill="none" '
        'vector-effect="non-scaling-stroke" />'
    )


def _text(
    *,
    entity_id: str,
    position: Point2D,
    value: str,
    color: str,
    font_size: float,
    anchor: str = "start",
    rotation_deg: float = 0.0,
) -> str:
    rotation = ""
    if not math.isclose(rotation_deg, 0.0):
        rotation = (
            f' transform="rotate({_number(-rotation_deg)} '
            f'{_number(position.x)} {_number(position.y)})"'
        )
    return (
        f'<text id="{html.escape(entity_id)}" x="{_number(position.x)}" '
        f'y="{_number(position.y)}" fill="{color}" stroke="none" '
        f'font-family="Arial, Microsoft YaHei, sans-serif" '
        f'font-size="{_number(font_size)}" text-anchor="{anchor}"'
        f'{rotation}>{html.escape(" ".join(value.splitlines()))}</text>'
    )


def _arc_points(entity: ArcEntity, steps: int = 36) -> list[Point2D]:
    start = math.radians(entity.start_angle_deg)
    end = math.radians(entity.end_angle_deg)
    if entity.clockwise:
        delta = -((start - end) % (2.0 * math.pi))
    else:
        delta = (end - start) % (2.0 * math.pi)
    count = max(4, int(math.ceil(abs(math.degrees(delta)) / (360.0 / steps))))
    return [
        Point2D(
            x=entity.center.x + entity.radius * math.cos(start + delta * index / count),
            y=entity.center.y + entity.radius * math.sin(start + delta * index / count),
        )
        for index in range(count + 1)
    ]


def _dimension_geometry(
    entity: DimensionEntity,
) -> tuple[Point2D, Point2D, Point2D, float]:
    dx = entity.end.x - entity.start.x
    dy = entity.end.y - entity.start.y
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    first = Point2D(
        x=entity.start.x + nx * entity.offset,
        y=entity.start.y + ny * entity.offset,
    )
    second = Point2D(
        x=entity.end.x + nx * entity.offset,
        y=entity.end.y + ny * entity.offset,
    )
    midpoint = Point2D(x=(first.x + second.x) / 2, y=(first.y + second.y) / 2)
    return first, second, midpoint, length


def _svg_entity(
    entity,
    *,
    color: str,
    lineweight_mm: float,
    screen: Callable[[Point2D], Point2D],
    scale: float,
) -> list[str]:
    stroke_width = _stroke_width(lineweight_mm)
    if isinstance(entity, LineEntity):
        return [
            _line(
                entity_id=entity.entity_id,
                start=screen(entity.start),
                end=screen(entity.end),
                color=color,
                stroke_width=stroke_width,
            )
        ]
    if isinstance(entity, CircleEntity):
        center = screen(entity.center)
        return [
            f'<circle id="{html.escape(entity.entity_id)}" '
            f'cx="{_number(center.x)}" cy="{_number(center.y)}" '
            f'r="{_number(entity.radius * scale)}" stroke="{color}" '
            f'stroke-width="{_number(stroke_width)}" fill="none" '
            'vector-effect="non-scaling-stroke" />'
        ]
    if isinstance(entity, ArcEntity):
        points = " ".join(
            f"{_number(point.x)},{_number(point.y)}"
            for point in (screen(item) for item in _arc_points(entity))
        )
        return [
            f'<polyline id="{html.escape(entity.entity_id)}" points="{points}" '
            f'stroke="{color}" stroke-width="{_number(stroke_width)}" '
            'fill="none" vector-effect="non-scaling-stroke" />'
        ]
    if isinstance(entity, TextEntity):
        return [
            _text(
                entity_id=entity.entity_id,
                position=screen(entity.insertion),
                value=entity.text,
                color=color,
                font_size=min(max(entity.height * scale, 13.0), 30.0),
                rotation_deg=entity.rotation_deg,
            )
        ]
    if isinstance(entity, LeaderEntity):
        points = [screen(point) for point in entity.points]
        point_text = " ".join(f"{_number(p.x)},{_number(p.y)}" for p in points)
        label_at = Point2D(x=points[-1].x + 8.0, y=points[-1].y - 8.0)
        return [
            f'<polyline id="{html.escape(entity.entity_id)}" points="{point_text}" '
            f'stroke="{color}" stroke-width="{_number(stroke_width)}" '
            'fill="none" vector-effect="non-scaling-stroke" />',
            _text(
                entity_id=f"{entity.entity_id}-label",
                position=label_at,
                value=entity.text,
                color=color,
                font_size=min(max(entity.text_height * scale, 13.0), 26.0),
            ),
        ]
    if isinstance(entity, DimensionEntity):
        first, second, midpoint, length = _dimension_geometry(entity)
        label = entity.text_override or _number(length)
        return [
            _line(
                entity_id=f"{entity.entity_id}-extension-1",
                start=screen(entity.start),
                end=screen(first),
                color=color,
                stroke_width=stroke_width,
            ),
            _line(
                entity_id=f"{entity.entity_id}-extension-2",
                start=screen(entity.end),
                end=screen(second),
                color=color,
                stroke_width=stroke_width,
            ),
            _line(
                entity_id=entity.entity_id,
                start=screen(first),
                end=screen(second),
                color=color,
                stroke_width=stroke_width,
            ),
            _text(
                entity_id=f"{entity.entity_id}-label",
                position=screen(midpoint),
                value=label,
                color=color,
                font_size=14.0,
                anchor="middle",
            ),
        ]
    raise TypeError(f"不支持的图元类型: {type(entity).__name__}")


def render_svg(document: DrawingDocument) -> str:
    try:
        layout = next(view for view in document.views if view.view_id == "cad-layout")
    except StopIteration as exc:
        raise ValueError("SVG 导出要求 DrawingDocument 包含 cad-layout 视图。") from exc
    entities = [entity for section in layout.sections for entity in section.entities]
    bounds_list = [entity_bounds(entity) for entity in entities]
    bounds = Bounds(
        min(item.min_x for item in bounds_list),
        min(item.min_y for item in bounds_list),
        max(item.max_x for item in bounds_list),
        max(item.max_y for item in bounds_list),
    )
    screen, scale = _screen_transform(bounds)
    styles = {
        layer.name: (
            f"rgb({layer.color_rgb[0]},{layer.color_rgb[1]},{layer.color_rgb[2]})",
            layer.lineweight_mm,
        )
        for layer in document.layers
    }
    body: list[str] = []
    for section in layout.sections:
        source_view_id = str(section.metadata.get("source_view_id") or "")
        has_explicit_title = any(
            isinstance(entity, TextEntity) and "-title-" in entity.entity_id
            for entity in section.entities
        )
        if section.entities and source_view_id and not has_explicit_title:
            section_bounds = [entity_bounds(entity) for entity in section.entities]
            label_at = screen(
                Point2D(
                    x=min(item.min_x for item in section_bounds),
                    y=max(item.max_y for item in section_bounds),
                )
            )
            body.append(
                _text(
                    entity_id=f"{section.section_id}-view-label",
                    position=Point2D(x=label_at.x, y=max(label_at.y - 12.0, 88.0)),
                    value=_VIEW_LABELS.get(source_view_id, source_view_id),
                    color="#E5E7EB",
                    font_size=18.0,
                )
            )
        for entity in section.entities:
            color, lineweight = styles[entity.layer]
            body.extend(
                _svg_entity(
                    entity,
                    color=color,
                    lineweight_mm=lineweight,
                    screen=screen,
                    scale=scale,
                )
            )
    title = html.escape(f"{document.design_group_id} 配筋图")
    subtitle = html.escape(
        f"适用桥墩：{', '.join(document.member_piers)}；单位：{document.units}"
    )
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_CANVAS_WIDTH}" '
            f'height="{_CANVAS_HEIGHT}" viewBox="0 0 {_CANVAS_WIDTH} {_CANVAS_HEIGHT}" '
            'role="img">',
            f'<rect width="100%" height="100%" fill="{_BACKGROUND}" />',
            f'<text x="800" y="38" text-anchor="middle" fill="#F9FAFB" '
            f'font-family="Arial, Microsoft YaHei, sans-serif" font-size="26">{title}</text>',
            f'<text x="800" y="66" text-anchor="middle" fill="#CBD5E1" '
            f'font-family="Arial, Microsoft YaHei, sans-serif" font-size="15">{subtitle}</text>',
            *body,
            "</svg>",
            "",
        ]
    )
