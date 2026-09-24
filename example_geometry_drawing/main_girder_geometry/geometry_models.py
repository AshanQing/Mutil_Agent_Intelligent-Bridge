from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal


ROUND_DIGITS = 6
Direction = Literal["forward", "reverse"]


def _rounded(value: float) -> float:
    result = round(float(value), ROUND_DIGITS)
    return 0.0 if result == -0.0 else result


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def scale(self, factor: float) -> Point:
        return Point(self.x * factor, self.y * factor)

    def distance_to(self, other: Point) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_dict(self) -> dict[str, float]:
        return {"x": _rounded(self.x), "y": _rounded(self.y)}


@dataclass(frozen=True)
class Line:
    id: str
    start: Point
    end: Point
    layer: str
    linetype: str = "BYLAYER"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "linetype": self.linetype,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
        }


@dataclass(frozen=True)
class Arc:
    id: str
    center: Point
    radius: float
    start_angle_deg: float
    end_angle_deg: float
    clockwise: bool
    layer: str
    linetype: str = "BYLAYER"

    @property
    def start(self) -> Point:
        angle = math.radians(self.start_angle_deg)
        return Point(
            self.center.x + self.radius * math.cos(angle),
            self.center.y + self.radius * math.sin(angle),
        )

    @property
    def end(self) -> Point:
        angle = math.radians(self.end_angle_deg)
        return Point(
            self.center.x + self.radius * math.cos(angle),
            self.center.y + self.radius * math.sin(angle),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "linetype": self.linetype,
            "center": self.center.to_dict(),
            "radius": _rounded(self.radius),
            "start_angle_deg": _rounded(self.start_angle_deg % 360.0),
            "end_angle_deg": _rounded(self.end_angle_deg % 360.0),
            "clockwise": self.clockwise,
        }


@dataclass(frozen=True)
class EntityRef:
    entity_id: str
    direction: Direction = "forward"

    def to_dict(self) -> dict[str, str]:
        return {"entity_id": self.entity_id, "direction": self.direction}


@dataclass(frozen=True)
class Region:
    id: str
    layer: str
    outer: tuple[EntityRef, ...]
    holes: tuple[tuple[EntityRef, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "outer": [item.to_dict() for item in self.outer],
            "holes": [
                [item.to_dict() for item in loop]
                for loop in self.holes
            ],
        }


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"
    entity_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.entity_id is not None:
            result["entity_id"] = self.entity_id
        return result


@dataclass
class SectionGeometry:
    id: str
    lines: list[Line] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entities": {
                "lines": [item.to_dict() for item in self.lines],
                "arcs": [item.to_dict() for item in self.arcs],
                "regions": [item.to_dict() for item in self.regions],
            },
        }


@dataclass
class GeometryView:
    coordinate_system: dict[str, str]
    sections: list[SectionGeometry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_system": dict(self.coordinate_system),
            "sections": [item.to_dict() for item in self.sections],
        }


@dataclass
class GeometryDocument:
    source: dict[str, Any]
    section: SectionGeometry
    additional_sections: list[SectionGeometry] = field(default_factory=list)
    schema_version: str = "2.0.0"
    units: str = "mm"
    angle_unit: str = "degree"
    view_name: str = "cross_section"
    coordinate_system: dict[str, str] | None = None
    additional_views: dict[str, GeometryView] = field(default_factory=dict)

    @property
    def sections(self) -> list[SectionGeometry]:
        return [self.section, *self.additional_sections]

    def to_dict(self) -> dict[str, Any]:
        cs = self.coordinate_system or {"origin": "(0, 0)", "x_axis": "+X", "y_axis": "+Y"}
        views = {
            self.view_name: GeometryView(
                coordinate_system=cs,
                sections=self.sections,
            ).to_dict()
        }
        views.update(
            {
                name: view.to_dict()
                for name, view in self.additional_views.items()
            }
        )
        return {
            "schema_version": self.schema_version,
            "units": self.units,
            "angle_unit": self.angle_unit,
            "source": self.source,
            "views": views,
        }


@dataclass
class DesignParameters:
    longitudinal: dict[str, Any]
    cross_section: dict[str, Any]
    source: dict[str, Any]
    global_parameters: dict[str, Any] = field(default_factory=dict)
