from __future__ import annotations

from bridge_agents.drawing import ArcEntity, CircleEntity, DimensionEntity, LineEntity
from bridge_agents.drawing.views import build_reinforcement_drawing


def _entities(document, view_id: str):
    view = next(view for view in document.views if view.view_id == view_id)
    return [entity for section in view.sections for entity in section.entities]


def test_builds_four_local_engineering_views(drawing_source) -> None:
    document = build_reinforcement_drawing(drawing_source)

    assert document.units == "mm"
    assert [view.view_id for view in document.views] == [
        "cap-elevation",
        "cap-section",
        "column-elevation",
        "column-section",
    ]
    assert document.design_group_id == "B1-U1-1"
    assert document.member_piers == ["P1", "P2"]
    assert document.metadata["check_status"] == "passed"


def test_column_section_contains_declared_longitudinal_bar_count(
    drawing_source,
) -> None:
    document = build_reinforcement_drawing(drawing_source)
    bars = [
        entity
        for entity in _entities(document, "column-section")
        if isinstance(entity, CircleEntity)
        and entity.metadata.get("bar_role") == "column_longitudinal"
    ]

    assert len(bars) == 16
    assert all(entity.layer == "R-MAIN" for entity in bars)
    centers = {(round(bar.center.x, 3), round(bar.center.y, 3)) for bar in bars}
    assert len(centers) == 16


def test_column_elevation_uses_controlling_height_and_spiral_zones(
    drawing_source,
) -> None:
    document = build_reinforcement_drawing(drawing_source)
    entities = _entities(document, "column-elevation")
    outlines = [
        entity
        for entity in entities
        if isinstance(entity, LineEntity)
        and entity.metadata.get("role") == "column_outline"
    ]
    spiral_lines = [
        entity
        for entity in entities
        if isinstance(entity, LineEntity)
        and entity.metadata.get("bar_role") == "column_spiral"
    ]

    assert max(point.y for line in outlines for point in (line.start, line.end)) == 10500
    assert {line.metadata["zone_name"] for line in spiral_lines} == {
        "柱底加密区",
        "柱身普通区",
    }
    assert any(
        isinstance(entity, DimensionEntity)
        and entity.metadata.get("dimension_role") == "column_height"
        for entity in entities
    )


def test_cap_elevation_uses_declared_length_height_and_rebar_marks(
    drawing_source,
) -> None:
    document = build_reinforcement_drawing(drawing_source)
    entities = _entities(document, "cap-elevation")
    outline_points = [
        point
        for entity in entities
        if isinstance(entity, LineEntity)
        and entity.metadata.get("role") == "cap_outline"
        for point in (entity.start, entity.end)
    ]
    marks = {
        entity.metadata.get("bar_mark")
        for entity in entities
        if entity.metadata.get("bar_role") == "cap_longitudinal"
    }

    assert min(point.x for point in outline_points) == -6000
    assert max(point.x for point in outline_points) == 6000
    assert max(point.y for point in outline_points) == 1800
    assert marks == {"N1", "N2"}


def test_cap_elevation_translates_supported_control_bar_to_mirrored_segments(
    drawing_source,
) -> None:
    document = build_reinforcement_drawing(drawing_source)
    control_segments = [
        entity
        for entity in _entities(document, "cap-elevation")
        if isinstance(entity, LineEntity)
        and entity.metadata.get("bar_role") == "cap_control"
        and entity.metadata.get("bar_mark") == "N3"
    ]

    assert len(control_segments) == 6
    assert {entity.metadata["mirror_side"] for entity in control_segments} == {
        "source",
        "mirrored",
    }
    assert "N3" not in document.metadata["unsupported_cap_bar_marks"]


def test_cap_elevation_translates_single_branch_and_arc_control_bars(
    drawing_source,
) -> None:
    document = build_reinforcement_drawing(drawing_source)
    entities = _entities(document, "cap-elevation")
    n5_segments = [
        entity
        for entity in entities
        if isinstance(entity, LineEntity)
        and entity.metadata.get("bar_mark") == "N5"
    ]
    n6_arcs = [
        entity
        for entity in entities
        if isinstance(entity, ArcEntity)
        and entity.metadata.get("bar_mark") == "N6"
    ]

    assert len(n5_segments) == 2
    assert len(n6_arcs) == 2
    assert not ({"N5", "N6"} & set(document.metadata["unsupported_cap_bar_marks"]))
