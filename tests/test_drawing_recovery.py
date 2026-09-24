from __future__ import annotations

import json
from pathlib import Path

from bridge_agents.agent import _discover_existing_outputs


def test_output_discovery_restores_drawing_package_paths(tmp_path: Path) -> None:
    group = tmp_path / "deliverables" / "drawings" / "G1"
    group.mkdir(parents=True)
    scr = group / "reinforcement_G1.scr"
    svg = group / "reinforcement_G1.svg"
    scr.write_text("_.ZOOM\n_E\n", encoding="utf-8")
    svg.write_text("<svg />\n", encoding="utf-8")
    index_path = tmp_path / "deliverables" / "drawings" / "drawing_index.json"
    manifest_path = tmp_path / "deliverables" / "design_manifest.json"
    index_path.write_text(
        json.dumps({
            "schema_version": "drawing-index-v1",
            "groups": [{
                "design_group_id": "G1",
                "scr_path": "deliverables/drawings/G1/reinforcement_G1.scr",
                "svg_path": "deliverables/drawings/G1/reinforcement_G1.svg",
            }],
        }),
        encoding="utf-8",
    )
    manifest_path.write_text("{}", encoding="utf-8")

    discovered = _discover_existing_outputs(tmp_path)

    assert discovered["drawing_index_path"] == str(index_path)
    assert discovered["design_manifest_path"] == str(manifest_path)
    assert discovered["cad_script_paths"] == [str(scr)]
    assert discovered["drawing_preview_paths"] == [str(svg)]
    assert discovered["drawing_package_result"]["generated_group_ids"] == ["G1"]


def test_output_discovery_restores_all_paged_drawing_paths(tmp_path: Path) -> None:
    group = tmp_path / "deliverables" / "drawings" / "G1"
    group.mkdir(parents=True)
    scr_paths = [group / f"sheet_{index}.scr" for index in range(3)]
    svg_paths = [group / f"sheet_{index}.svg" for index in range(3)]
    for path in scr_paths:
        path.write_text("_.ZOOM\n_E\n", encoding="gbk")
    for path in svg_paths:
        path.write_text("<svg />\n", encoding="utf-8")
    index_path = tmp_path / "deliverables" / "drawings" / "drawing_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "drawing-index-v1",
                "groups": [
                    {
                        "design_group_id": "G1",
                        "scr_paths": [str(path.relative_to(tmp_path)) for path in scr_paths],
                        "svg_paths": [str(path.relative_to(tmp_path)) for path in svg_paths],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    discovered = _discover_existing_outputs(tmp_path)

    assert discovered["cad_script_paths"] == [str(path) for path in scr_paths]
    assert discovered["drawing_preview_paths"] == [str(path) for path in svg_paths]
