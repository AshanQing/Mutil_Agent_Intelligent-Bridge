from __future__ import annotations

from bridge_agents.drawing.layout import build_combined_layout
from bridge_agents.drawing.scr_exporter import render_autocad_script


def test_scr_uses_international_commands_and_never_saves(
    simple_drawing_document,
) -> None:
    document = build_combined_layout(simple_drawing_document)
    script = render_autocad_script(document)

    assert "_.-LINETYPE" in script
    assert "DASHED" in script
    assert "_.-LAYER" in script
    assert "_.LINE" in script
    assert "_.CIRCLE" in script
    assert script.rstrip().endswith("_.ZOOM\n_E")
    assert "QSAVE" not in script.upper()
    assert "SAVEAS" not in script.upper()
    assert "PLOT" not in script.upper()
    assert "盖梁\n配筋" not in script


def test_same_document_produces_byte_identical_script(simple_drawing_document) -> None:
    document = build_combined_layout(simple_drawing_document)
    assert render_autocad_script(document) == render_autocad_script(document)


def test_scr_uses_prompt_stable_layer_and_truecolor_sequences(
    simple_drawing_document,
) -> None:
    document = build_combined_layout(simple_drawing_document)
    script = render_autocad_script(document)

    assert "\n_C\n_T\n" not in script
    assert "_BYLAYER" not in script
    assert "_.-LAYER\n_M\nS-BEAM\n\n" in script
    assert "_.-LAYER\n_LW\n0.25\nS-BEAM\n\n" in script
    assert "_.CECOLOR\n255,0,0\n_.CIRCLE" in script
