from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .config import DrawingConfig
from .models import (
    ArcEntity,
    CircleEntity,
    DimensionEntity,
    DrawingDocument,
    DrawingEntity,
    GeometryView,
    LeaderEntity,
    LineEntity,
    Point2D,
    Region,
    SectionGeometry,
    TextEntity,
)


@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


def _bounds_from_points(points: Iterable[Point2D]) -> Bounds:
    values = list(points)
    if not values:
        return Bounds(0.0, 0.0, 0.0, 0.0)
    return Bounds(
        min(point.x for point in values),
        min(point.y for point in values),
        max(point.x for point in values),
        max(point.y for point in values),
    )


def entity_bounds(entity: DrawingEntity) -> Bounds:
    if isinstance(entity, LineEntity):
        return _bounds_from_points([entity.start, entity.end])
    if isinstance(entity, (CircleEntity, ArcEntity)):
        return Bounds(
            entity.center.x - entity.radius,
            entity.center.y - entity.radius,
            entity.center.x + entity.radius,
            entity.center.y + entity.radius,
        )
    if isinstance(entity, TextEntity):
        width = max(len(entity.text), 1) * entity.height * 0.65
        return Bounds(
            entity.insertion.x,
            entity.insertion.y - entity.height,
            entity.insertion.x + width,
            entity.insertion.y,
        )
    if isinstance(entity, LeaderEntity):
        return _bounds_from_points(entity.points)
    if isinstance(entity, DimensionEntity):
        return Bounds(
            min(entity.start.x, entity.end.x) - abs(entity.offset),
            min(entity.start.y, entity.end.y) - abs(entity.offset),
            max(entity.start.x, entity.end.x) + abs(entity.offset),
            max(entity.start.y, entity.end.y) + abs(entity.offset),
        )
    raise TypeError(f"不支持的图元类型: {type(entity).__name__}")


def view_bounds(view: GeometryView) -> Bounds:
    bounds = [
        entity_bounds(entity)
        for section in view.sections
        for entity in section.entities
    ]
    return Bounds(
        min(item.min_x for item in bounds),
        min(item.min_y for item in bounds),
        max(item.max_x for item in bounds),
        max(item.max_y for item in bounds),
    )


def _view_scale(view: GeometryView) -> float:
    scale = float(view.metadata.get("layout_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"视图 {view.view_id} 的 layout_scale 必须是正有限数值。")
    return scale


def _scaled_view_bounds(view: GeometryView, scale: float) -> Bounds:
    bounds = view_bounds(view)
    return Bounds(
        min_x=bounds.min_x * scale,
        min_y=bounds.min_y * scale,
        max_x=bounds.max_x * scale,
        max_y=bounds.max_y * scale,
    )


def _point(point: Point2D, dx: float, dy: float, scale: float) -> Point2D:
    return Point2D(x=point.x * scale + dx, y=point.y * scale + dy)


def _translated_entity(
    entity: DrawingEntity,
    *,
    prefix: str,
    dx: float,
    dy: float,
    scale: float,
) -> DrawingEntity:
    update: dict[str, object] = {"entity_id": f"{prefix}-{entity.entity_id}"}
    if isinstance(entity, LineEntity):
        update.update(
            start=_point(entity.start, dx, dy, scale),
            end=_point(entity.end, dx, dy, scale),
        )
    elif isinstance(entity, (CircleEntity, ArcEntity)):
        update["center"] = _point(entity.center, dx, dy, scale)
        update["radius"] = entity.radius * scale
    elif isinstance(entity, TextEntity):
        update["insertion"] = _point(entity.insertion, dx, dy, scale)
        update["height"] = entity.height * scale
    elif isinstance(entity, LeaderEntity):
        update["points"] = [
            _point(point, dx, dy, scale) for point in entity.points
        ]
        update["text_height"] = entity.text_height * scale
    elif isinstance(entity, DimensionEntity):
        update.update(
            start=_point(entity.start, dx, dy, scale),
            end=_point(entity.end, dx, dy, scale),
            offset=entity.offset * scale,
        )
    return entity.model_copy(update=update)


def _translated_region(region: Region, *, prefix: str) -> Region:
    def rename(ref):
        return ref.model_copy(update={"entity_id": f"{prefix}-{ref.entity_id}"})

    return region.model_copy(
        update={
            "region_id": f"{prefix}-{region.region_id}",
            "outer": [rename(ref) for ref in region.outer],
            "holes": [[rename(ref) for ref in hole] for hole in region.holes],
        }
    )


def _ordered_rows(document: DrawingDocument, config: DrawingConfig) -> list[list[str]]:
    available = {view.view_id for view in document.views if view.view_id != "cad-layout"}
    rows: list[list[str]] = []
    used: set[str] = set()
    for configured_row in config.view_rows:
        row = [view_id for view_id in configured_row if view_id in available]
        if row:
            rows.append(row)
            used.update(row)
    remaining = sorted(available - used)
    if remaining:
        rows.append(remaining)
    return rows


def build_combined_layout(
    document: DrawingDocument,
    config: DrawingConfig | None = None,
) -> DrawingDocument:
    active = config or DrawingConfig()
    source_views = {
        view.view_id: view for view in document.views if view.view_id != "cad-layout"
    }
    rows = _ordered_rows(document, active)
    row_specs: list[tuple[list[str], float, float]] = []
    for row in rows:
        bounds = [
            _scaled_view_bounds(
                source_views[view_id], _view_scale(source_views[view_id])
            )
            for view_id in row
        ]
        width = sum(item.width for item in bounds)
        width += active.column_gap_mm * max(len(row) - 1, 0)
        row_specs.append((row, width, max(item.height for item in bounds)))
    sheet_width = max((item[1] for item in row_specs), default=0.0)
    current_y = 0.0
    layout_sections: list[SectionGeometry] = []
    for row, row_width, row_height in row_specs:
        current_x = (sheet_width - row_width) / 2.0
        for view_id in row:
            view = source_views[view_id]
            scale = _view_scale(view)
            bounds = _scaled_view_bounds(view, scale)
            dx = current_x - bounds.min_x
            dy = current_y - bounds.min_y
            prefix = f"layout-{view_id}"
            for section in view.sections:
                layout_sections.append(
                    SectionGeometry(
                        section_id=f"{prefix}-{section.section_id}",
                        entities=[
                            _translated_entity(
                                entity,
                                prefix=prefix,
                                dx=dx,
                                dy=dy,
                                scale=scale,
                            )
                            for entity in section.entities
                        ],
                        regions=[
                            _translated_region(region, prefix=prefix)
                            for region in section.regions
                        ],
                        metadata={
                            **section.metadata,
                            "source_view_id": view_id,
                            "layout_scale": scale,
                        },
                    )
                )
            current_x += bounds.width + active.column_gap_mm
        current_y += row_height + active.row_gap_mm
    if not layout_sections:
        raise ValueError("DrawingDocument 没有可排版视图。")
    layout_view = GeometryView(
        view_id="cad-layout",
        sections=layout_sections,
        metadata={"config_source": active.config_source},
    )
    original = [view for view in document.views if view.view_id != "cad-layout"]
    payload = document.model_dump(mode="python")
    payload["views"] = [*original, layout_view]
    return DrawingDocument.model_validate(payload)
