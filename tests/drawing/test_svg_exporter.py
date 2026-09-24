from __future__ import annotations

import xml.etree.ElementTree as ET

from bridge_agents.drawing.layout import build_combined_layout
from bridge_agents.drawing.svg_exporter import render_svg
from bridge_agents.drawing.views import build_reinforcement_drawing


def test_svg_is_valid_deterministic_preview(
    simple_drawing_document,
) -> None:
    document = build_combined_layout(simple_drawing_document)
    svg = render_svg(document)

    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert root.attrib["viewBox"]
    assert svg == render_svg(document)


def test_svg_scales_drawing_onto_a_readable_fixed_canvas(
    simple_drawing_document,
) -> None:
    document = build_combined_layout(simple_drawing_document)
    svg = render_svg(document)
    root = ET.fromstring(svg)

    assert root.attrib["width"] == "1600"
    assert root.attrib["height"] == "1200"
    assert root.attrib["viewBox"] == "0 0 1600 1200"
    assert '<rect width="100%" height="100%" fill="#111827"' in svg
    assert 'vector-effect="non-scaling-stroke"' in svg
    assert "B1-U1-G1 配筋图" in svg


def test_svg_renders_dimension_value_as_visible_text(simple_drawing_document) -> None:
    document = build_combined_layout(simple_drawing_document)
    svg = render_svg(document)

    assert ">1000.000<" in svg


def test_svg_does_not_duplicate_existing_view_titles(drawing_source) -> None:
    document = build_combined_layout(build_reinforcement_drawing(drawing_source))
    svg = render_svg(document)

    assert svg.count(">盖梁配筋立面<") == 1
    assert ">盖梁立面<" not in svg
