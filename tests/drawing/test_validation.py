from __future__ import annotations

from bridge_agents.drawing import (
    CircleEntity,
    EntityRef,
    LineEntity,
    Region,
    SectionGeometry,
)
from bridge_agents.drawing.validation import validate_geometry


def _square_section(*, close: bool) -> SectionGeometry:
    end = (0, 0) if close else (50, 100)
    entities = [
        LineEntity(entity_id="e1", layer="S", start=(0, 0), end=(100, 0)),
        LineEntity(entity_id="e2", layer="S", start=(100, 0), end=(100, 100)),
        LineEntity(entity_id="e3", layer="S", start=(100, 100), end=(0, 100)),
        LineEntity(entity_id="e4", layer="S", start=(0, 100), end=end),
    ]
    return SectionGeometry(
        section_id="S1",
        entities=entities,
        regions=[
            Region(
                region_id="R1",
                layer="S",
                outer=[EntityRef(entity_id=f"e{index}") for index in range(1, 5)],
            )
        ],
    )


def test_validation_reports_closed_region_and_signed_area() -> None:
    result = validate_geometry(_square_section(close=True))
    assert result.valid is True
    assert result.region_areas["R1"] == 10000


def test_validation_rejects_discontinuous_region() -> None:
    result = validate_geometry(_square_section(close=False))
    assert result.valid is False
    assert any(item.code == "region_not_closed" for item in result.diagnostics)


def test_validation_rejects_self_intersecting_boundary() -> None:
    section = SectionGeometry(
        section_id="S1",
        entities=[
            LineEntity(entity_id="e1", layer="S", start=(0, 0), end=(100, 100)),
            LineEntity(entity_id="e2", layer="S", start=(100, 100), end=(0, 100)),
            LineEntity(entity_id="e3", layer="S", start=(0, 100), end=(100, 0)),
            LineEntity(entity_id="e4", layer="S", start=(100, 0), end=(0, 0)),
        ],
        regions=[
            Region(
                region_id="R1",
                layer="S",
                outer=[EntityRef(entity_id=f"e{index}") for index in range(1, 5)],
            )
        ],
    )

    result = validate_geometry(section)
    assert any(item.code == "region_self_intersection" for item in result.diagnostics)


def test_validation_checks_declared_tangent_point() -> None:
    section = SectionGeometry(
        section_id="S1",
        entities=[
            CircleEntity(entity_id="c1", layer="S", center=(0, 0), radius=100),
            CircleEntity(entity_id="c2", layer="S", center=(250, 0), radius=100),
        ],
        metadata={
            "tangent_points": [
                {"entity_a": "c1", "entity_b": "c2", "point": [100, 0]}
            ]
        },
    )

    result = validate_geometry(section)
    assert any(item.code == "declared_tangency_invalid" for item in result.diagnostics)
