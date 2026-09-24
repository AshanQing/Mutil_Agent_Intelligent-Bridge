from __future__ import annotations

import json

from PIL import Image

from bridge_agents.axial_table_render import (
    ensure_axial_check_views,
    render_axial_check_summary_table,
    write_axial_check_summary_csv,
)


def _sample_summary() -> dict:
    return {
        "schema_version": "axial-check-summary-v1",
        "expected_task_count": 2,
        "verdict_counts": {"passed": 1, "failed": 0, "manual_review": 1, "missing": 0},
        "groups": [
            {
                "task_id": "1-1-1-G1",
                "member_piers": ["1", "2"],
                "column_diameter_mm": 1400.0,
                "controlling_net_height_m": 24.68,
                "effective_length_factor": 1.0,
                "slenderness_ratio": 70.51,
                "slenderness_basis": "l0/i",
                "stability_factor_phi": 0.70,
                "utilization": 0.42,
                "check_ok": True,
                "status": "computed",
                "verdict": "passed",
                "message": "轴压承载力满足要求。",
            },
            {
                "task_id": "1-1-3-G1",
                "member_piers": ["7", "8"],
                "column_diameter_mm": 1400.0,
                "controlling_net_height_m": 11.83,
                "effective_length_factor": 1.0,
                "slenderness_ratio": 33.80,
                "slenderness_basis": "l0/i",
                "stability_factor_phi": 0.98,
                "utilization": None,
                "check_ok": None,
                "status": "manual_review",
                "verdict": "manual_review",
                "message": "需人工复核。",
            },
        ],
    }


def test_render_png_and_csv(tmp_path):
    summary = _sample_summary()
    png = tmp_path / "axial_check_summary.png"
    csv_path = tmp_path / "axial_check_summary.csv"
    assert render_axial_check_summary_table(summary, png) is True
    assert png.is_file() and png.stat().st_size > 0
    with Image.open(png) as img:
        assert img.width > 800 and img.height > 300
    write_axial_check_summary_csv(summary, csv_path)
    content = csv_path.read_text(encoding="utf-8-sig")
    assert "设计组" in content
    assert "1-1-1-G1" in content
    assert "需人工复核" in content


def test_ensure_generates_views_and_is_idempotent(tmp_path):
    summary = _sample_summary()
    json_path = tmp_path / "axial_check_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    png_path, csv_path = ensure_axial_check_views(json_path)
    assert png_path and csv_path
    png_first, _ = ensure_axial_check_views(json_path)
    assert png_first == png_path
