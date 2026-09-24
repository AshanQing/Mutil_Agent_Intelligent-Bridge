from __future__ import annotations

import math
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .models import ArcEntity, CircleEntity, LineEntity, Point2D, SectionGeometry


class GeometryDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["error", "warning"]
    code: str
    message: str
    region_id: str | None = None


class GeometryValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    diagnostics: list[GeometryDiagnostic] = Field(default_factory=list)
    region_areas: dict[str, float] = Field(default_factory=dict)


def _same(first: Point2D, second: Point2D) -> bool:
    return math.isclose(first.x, second.x, abs_tol=1e-6) and math.isclose(
        first.y, second.y, abs_tol=1e-6
    )


def _oriented_endpoints(entity, direction: str) -> tuple[Point2D, Point2D] | None:
    if isinstance(entity, LineEntity):
        endpoints = entity.start, entity.end
    elif isinstance(entity, ArcEntity):
        angles = entity.start_angle_deg, entity.end_angle_deg
        endpoints = tuple(
            Point2D(
                x=entity.center.x + entity.radius * math.cos(math.radians(angle)),
                y=entity.center.y + entity.radius * math.sin(math.radians(angle)),
            )
            for angle in angles
        )
    elif isinstance(entity, CircleEntity):
        point = Point2D(x=entity.center.x + entity.radius, y=entity.center.y)
        endpoints = point, point
    else:
        return None
    return endpoints if direction == "forward" else tuple(reversed(endpoints))


def _signed_area(points: list[Point2D]) -> float:
    return 0.5 * sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(points, points[1:])
    )


def _orientation(first: Point2D, second: Point2D, third: Point2D) -> float:
    return (second.x - first.x) * (third.y - first.y) - (
        second.y - first.y
    ) * (third.x - first.x)


def _segments_intersect(
    first_start: Point2D,
    first_end: Point2D,
    second_start: Point2D,
    second_end: Point2D,
) -> bool:
    values = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    return values[0] * values[1] < -1e-9 and values[2] * values[3] < -1e-9


def _self_intersects(points: list[Point2D]) -> bool:
    segments = list(zip(points, points[1:]))
    count = len(segments)
    for first_index, first in enumerate(segments):
        for second_index in range(first_index + 1, count):
            if second_index in {first_index + 1}:
                continue
            if first_index == 0 and second_index == count - 1:
                continue
            if _segments_intersect(*first, *segments[second_index]):
                return True
    return False


def _point_in_polygon(point: Point2D, polygon: list[Point2D]) -> bool:
    inside = False
    for first, second in zip(polygon, polygon[1:]):
        crosses = (first.y > point.y) != (second.y > point.y)
        if crosses:
            crossing_x = (second.x - first.x) * (point.y - first.y)
            crossing_x /= second.y - first.y
            crossing_x += first.x
            if point.x < crossing_x:
                inside = not inside
    return inside


def _loop_points(refs, by_id: Mapping[str, Any]) -> tuple[list[Point2D], bool]:
    points: list[Point2D] = []
    continuous = True
    previous_end: Point2D | None = None
    for ref in refs:
        endpoints = _oriented_endpoints(by_id[ref.entity_id], ref.direction)
        if endpoints is None:
            continuous = False
            continue
        start, end = endpoints
        if previous_end is not None and not _same(previous_end, start):
            continuous = False
        if not points:
            points.append(start)
        points.append(end)
        previous_end = end
    if len(points) < 2 or not _same(points[0], points[-1]):
        continuous = False
    return points, continuous


def _distance_to_entity(point: Point2D, entity: Any) -> float:
    if isinstance(entity, (CircleEntity, ArcEntity)):
        radial = math.hypot(point.x - entity.center.x, point.y - entity.center.y)
        return abs(radial - entity.radius)
    if isinstance(entity, LineEntity):
        dx = entity.end.x - entity.start.x
        dy = entity.end.y - entity.start.y
        length_squared = dx * dx + dy * dy
        ratio = (
            (point.x - entity.start.x) * dx + (point.y - entity.start.y) * dy
        ) / length_squared
        ratio = min(max(ratio, 0.0), 1.0)
        nearest_x = entity.start.x + ratio * dx
        nearest_y = entity.start.y + ratio * dy
        return math.hypot(point.x - nearest_x, point.y - nearest_y)
    return math.inf


def validate_geometry(section: SectionGeometry) -> GeometryValidationResult:
    diagnostics: list[GeometryDiagnostic] = []
    areas: dict[str, float] = {}
    by_id = {entity.entity_id: entity for entity in section.entities}
    for region in section.regions:
        points, continuous = _loop_points(region.outer, by_id)
        if not continuous:
            diagnostics.append(
                GeometryDiagnostic(
                    severity="error",
                    code="region_not_closed",
                    message=f"Region {region.region_id} 的外边界不连续或未闭合。",
                    region_id=region.region_id,
                )
            )
            continue
        if _self_intersects(points):
            diagnostics.append(
                GeometryDiagnostic(
                    severity="error",
                    code="region_self_intersection",
                    message=f"Region {region.region_id} 的外边界发生自交。",
                    region_id=region.region_id,
                )
            )
        area = _signed_area(points)
        areas[region.region_id] = area
        if area <= 0:
            diagnostics.append(
                GeometryDiagnostic(
                    severity="error",
                    code="outer_orientation_invalid",
                    message=f"Region {region.region_id} 外边界应为逆时针方向。",
                    region_id=region.region_id,
                )
            )
        for hole_index, hole in enumerate(region.holes, start=1):
            hole_points, hole_continuous = _loop_points(hole, by_id)
            if not hole_continuous:
                code = "hole_not_closed"
            elif _signed_area(hole_points) >= 0:
                code = "hole_orientation_invalid"
            else:
                centroid = Point2D(
                    x=sum(point.x for point in hole_points[:-1]) / (len(hole_points) - 1),
                    y=sum(point.y for point in hole_points[:-1]) / (len(hole_points) - 1),
                )
                code = "" if _point_in_polygon(centroid, points) else "hole_outside_outer"
            if code:
                diagnostics.append(
                    GeometryDiagnostic(
                        severity="error",
                        code=code,
                        message=f"Region {region.region_id} 的孔洞 {hole_index} 无效。",
                        region_id=region.region_id,
                    )
                )

    declarations = section.metadata.get("tangent_points", [])
    if isinstance(declarations, list):
        for declaration in declarations:
            if not isinstance(declaration, Mapping):
                continue
            first = by_id.get(str(declaration.get("entity_a") or ""))
            second = by_id.get(str(declaration.get("entity_b") or ""))
            raw_point = declaration.get("point")
            try:
                point = Point2D.model_validate(raw_point)
            except Exception:
                point = None
            valid = (
                first is not None
                and second is not None
                and point is not None
                and _distance_to_entity(point, first) <= 1e-5
                and _distance_to_entity(point, second) <= 1e-5
            )
            if not valid:
                diagnostics.append(
                    GeometryDiagnostic(
                        severity="error",
                        code="declared_tangency_invalid",
                        message="声明的相切点未同时位于两个指定图元上。",
                    )
                )
    return GeometryValidationResult(
        valid=not any(item.severity == "error" for item in diagnostics),
        diagnostics=diagnostics,
        region_areas=areas,
    )
