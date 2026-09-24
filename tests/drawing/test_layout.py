from __future__ import annotations

from bridge_agents.drawing.layout import build_combined_layout


def test_combined_layout_is_independent_of_input_view_order(
    simple_drawing_document,
) -> None:
    first = build_combined_layout(simple_drawing_document)
    reversed_document = simple_drawing_document.model_copy(deep=True)
    reversed_document.views = list(reversed(reversed_document.views))
    second = build_combined_layout(reversed_document)

    assert first.to_stable_json() == second.to_stable_json()
    layout = next(view for view in first.views if view.view_id == "cad-layout")
    identifiers = [
        entity.entity_id
        for section in layout.sections
        for entity in section.entities
    ]
    assert len(identifiers) == len(set(identifiers))
    assert all(identifier.startswith("layout-") for identifier in identifiers)


def test_combined_layout_applies_declared_per_view_scale(
    simple_drawing_document,
) -> None:
    document = simple_drawing_document.model_copy(deep=True)
    column_section = next(
        view for view in document.views if view.view_id == "column-section"
    )
    column_section.metadata["layout_scale"] = 2.0

    combined = build_combined_layout(document)
    layout = next(view for view in combined.views if view.view_id == "cad-layout")
    circle = next(
        entity
        for section in layout.sections
        for entity in section.entities
        if entity.entity_id.endswith("column-circle-001")
    )

    assert circle.radius == 800
