from __future__ import annotations

import pytest

from bridge_agents.tool_actions import (
    _fallback_modeling_feedback_decision,
    generate_modeling_feedback_action,
)
from tools.cap_beam_capacity_envelope_overlay_refined import (
    flexural_capacity_kNm,
    shear_capacity_kN,
    slim_control_row,
)


def _bars(area_mm2: float) -> list[dict]:
    return [{"id": "B1", "count": 1, "area_mm2": area_mm2, "y_mm": -500.0}]


def test_flexural_capacity_uses_registered_xi_b_limit() -> None:
    capacity, detail = flexural_capacity_kNm(
        1000.0,
        2000.0,
        _bars(30000.0),
        compression_side="top",
    )

    expected_x = 30000.0 * 330.0 / (18.4 * 1000.0)
    expected_capacity = 18.4 * 1000.0 * expected_x * (1500.0 - expected_x / 2.0) / 1.0e6
    assert capacity == pytest.approx(expected_capacity)
    assert detail["xi_b"] == 0.53
    assert detail["x_limit_mm"] == pytest.approx(0.53 * 1500.0)
    assert detail["x_used_mm"] == pytest.approx(expected_x)
    assert detail["formula_id"] == "F_3362_CH05_5_2_2_1_BENDING_CAPACITY_TENSION_FLANGE"


def test_compression_zone_limit_controls_revision_direction_without_utilization_threshold() -> None:
    decision = _fallback_modeling_feedback_decision(
        {
            "overall_check": {
                "M_pos_ok": False,
                "M_neg_ok": True,
                "V_ok": True,
                "compression_zone_ok": False,
                "all_ok": False,
            },
            "control_sections": {
                "max_positive_moment_utilization": {
                    "x_m": 0.0,
                    "util_M_pos": 1.01,
                    "compression_zone_limit_exceeded": True,
                }
            },
        }
    )

    assert decision["next_action"] == "revise_dimension"
    assert decision["decision_basis"] == "compression_zone_limit_exceeded"


def test_strength_failure_with_valid_compression_zone_returns_to_reinforcement() -> None:
    decision = _fallback_modeling_feedback_decision(
        {
            "overall_check": {
                "M_pos_ok": False,
                "M_neg_ok": True,
                "V_ok": True,
                "compression_zone_ok": True,
                "all_ok": False,
            },
            "control_sections": {
                "max_positive_moment_utilization": {
                    "x_m": 0.0,
                    "util_M_pos": 2.4,
                    "compression_zone_limit_exceeded": False,
                }
            },
        }
    )

    assert decision["next_action"] == "revise_reinforcement"
    assert decision["decision_basis"] == "capacity_or_reinforcement_failure"


def test_shear_capacity_uses_cap_beam_code_formulas() -> None:
    capacity, detail = shear_capacity_kN(
        1900.0,
        1500.0,
        100.0,
        4,
        16.0,
        h=1600.0,
        calculation_span=6000.0,
        longitudinal_As_mm2=22800.0,
    )

    assert capacity == pytest.approx(min(detail["V_limit_kN"], detail["V_reinforced_kN"]))
    assert detail["P"] == pytest.approx(0.8)
    assert detail["rho_sv"] == pytest.approx(detail["Asv_mm2"] / (1900.0 * 100.0))
    assert detail["formula_ids"] == [
        "F_3362_CH08_8_4_4_CAP_BEAM_SHEAR_CAPACITY",
        "F_3362_CH08_8_4_5_CAP_BEAM_INCLINED_SHEAR",
    ]


def test_control_summary_retains_formula_and_reinforcement_ratio_signals() -> None:
    row = slim_control_row(
        {
            "x_m": 0.0,
            "b_mm": 1900.0,
            "h_mm": 1600.0,
            "top_As_mm2": 10000.0,
            "bottom_As_mm2": 12000.0,
            "pos_flexure_detail": {"over_x_limit": True, "formula_id": "F-pos"},
            "neg_flexure_detail": {"over_x_limit": False, "formula_id": "F-neg"},
            "shear_detail": {"formula_ids": ["F-shear"]},
        }
    )

    assert row["compression_zone_limit_exceeded"] is True
    assert row["longitudinal_reinforcement_ratio_percent"]["top"] == pytest.approx(
        100.0 * 10000.0 / (1900.0 * 1600.0)
    )
    assert row["formula_trace"]["positive_flexure"] == "F-pos"
    assert row["formula_trace"]["shear"] == ["F-shear"]


def test_feedback_action_uses_deterministic_rule_without_calling_llm(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "bridge_agents.tool_actions.get_controller_llm",
        lambda config_path: (_ for _ in ()).throw(AssertionError("返修方向不得调用LLM")),
    )
    result = generate_modeling_feedback_action(
        {
            "output_dir": str(tmp_path),
            "check_result": {
                "overall_check": {
                    "M_pos_ok": False,
                    "M_neg_ok": True,
                    "V_ok": True,
                    "compression_zone_ok": True,
                    "all_ok": False,
                },
                "control_sections": {},
            },
        }
    )

    assert result["error"] is None
    assert result["feedback_decision"]["next_action"] == "revise_reinforcement"
    assert result["feedback_decision"]["decision_source"] == "deterministic_rule"
