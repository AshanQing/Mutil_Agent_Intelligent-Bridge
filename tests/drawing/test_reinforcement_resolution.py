from __future__ import annotations

from pathlib import Path
from copy import deepcopy


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_selected_examples():
    from bridge_agents.drawing.reinforcement_resolution import (
        load_reinforcement_example,
    )

    cap = load_reinforcement_example(
        REPO_ROOT / "samples" / "reinforcement_design_samples.yaml",
        example_name="示例1",
        expected_drawing_id=12,
    )
    column = load_reinforcement_example(
        REPO_ROOT / "samples" / "reinforcement_column.yaml",
        example_name="示例1",
        expected_drawing_id=14,
    )
    return cap, column


def test_real_examples_are_loaded_by_name_and_drawing_id() -> None:
    cap, column = _load_selected_examples()

    assert cap["输入"]["基本信息"]["drawing_id"] == 12
    assert column["输入"]["基本信息"]["drawing_id"] == 14


def test_real_examples_resolve_units_skeletons_and_member_scoped_marks() -> None:
    from bridge_agents.drawing.reinforcement_resolution import (
        resolve_reinforcement_examples,
    )

    cap, column = _load_selected_examples()
    resolved = resolve_reinforcement_examples(cap, column)

    assert resolved.cap.length_mm == 11700
    assert resolved.cap.width_mm == 2200
    assert resolved.cap.skeletons["skeleton_1"].z_positions == [
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23
    ]
    assert resolved.cap.skeletons["skeleton_2"].z_positions == [
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22
    ]
    assert resolved.bar("cap", "N1").key == "cap:N1"
    assert resolved.bar("column", "N1").key == "column:N1"


def test_column_example_resolves_all_n1_to_n4_reinforcement() -> None:
    from bridge_agents.drawing.reinforcement_resolution import (
        resolve_reinforcement_examples,
    )

    cap, column = _load_selected_examples()
    resolved = resolve_reinforcement_examples(cap, column)

    assert resolved.column.height_mm == 11769
    assert resolved.column.longitudinal.count == 44
    assert resolved.column.longitudinal.centerline_diameter_mm == 1682
    assert resolved.column.longitudinal.y_to_mm == 13369
    assert [zone.pitch_mm for zone in resolved.column.spiral_zones] == [
        100, 150, 69, 100
    ]
    assert [(zone.y_from_mm, zone.y_to_mm) for zone in resolved.column.spiral_zones] == [
        (0, 2700),
        (2700, 9000),
        (9000, 9069),
        (9069, 11769),
    ]
    assert resolved.column.ordinary_hoop.count == 7
    assert resolved.column.ordinary_hoop.y_positions_mm == [
        11969, 12169, 12369, 12569, 12769, 12969, 13169
    ]
    assert resolved.column.strengthening.count == 6
    assert resolved.column.strengthening.hoop_diameter_mm == 1615
    assert resolved.column.strengthening.y_positions_mm == [
        884, 2884.2, 4884.4, 6884.6, 8884.8, 10885
    ]


def test_cap_stirrup_inconsistencies_are_reported_without_changing_source_values() -> None:
    from bridge_agents.drawing.reinforcement_resolution import (
        resolve_reinforcement_examples,
    )

    cap, column = _load_selected_examples()
    resolved = resolve_reinforcement_examples(cap, column)

    mismatches = [
        item
        for item in resolved.diagnostics
        if item.code == "stirrup_segment_length_mismatch"
    ]
    # count 是间距个数：区段端点由 distribution_overview 的 x_start 按 count×spacing 累计
    # -5790 -> -5700(blank 90) -> -4200(加密 15×100) -> -2700(普通1 10×150)
    #      -> -600(加密右 21×100) -> 0(普通2 4×150)
    assert [item.field_path for item in mismatches] == []
    assert [item.name for item in resolved.cap.stirrup_segments] == [
        "left_blank",
        "墩柱左侧加密区",
        "普通区1",
        "墩柱右侧加密区",
        "普通区2",
    ]
    assert [(s.x_from_mm, s.x_to_mm, s.spacing_mm, s.declared_interval_count)
            for s in resolved.cap.stirrup_segments] == [
        (-5790, -5700, 90, 1),
        (-5700, -4200, 100, 15),
        (-4200, -2700, 150, 10),
        (-2700, -600, 100, 21),
        (-600, 0, 150, 4),
    ]


def test_runtime_source_normalizes_redundant_times_1000(drawing_source) -> None:
    from bridge_agents.drawing.reinforcement_resolution import resolve_drawing_source

    payload = deepcopy(drawing_source.reinforcement)
    cap = payload["reinforcement"]["pier_cap"]
    cap["stirrups"]["distribution_overview"]["x_start_expr"] = (
        "-(cap_length * 1000 / 2 - cover_x)"
    )
    column = payload["reinforcement"]["pier_column"]
    column["longitudinal_bars"][0]["range_definition"]["y_end"] = (
        "column_height * 1000 + 1200"
    )
    column["spiral_stirrups"][0]["vertical_distribution"][1]["y_to"] = (
        "column_height * 1000"
    )
    source = drawing_source.model_copy(update={"reinforcement": payload})

    resolved = resolve_drawing_source(source)

    assert resolved.cap.stirrup_segments[0].x_from_mm == -5950.0
    assert resolved.column.longitudinal.y_to_mm == 11700.0
    assert resolved.column.spiral_zones[-1].y_to_mm == 10500.0
    assert any(
        item.code == "redundant_mm_conversion_normalized"
        for item in resolved.diagnostics
    )
