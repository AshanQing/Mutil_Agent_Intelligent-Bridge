from __future__ import annotations

import math

import pytest

from bridge_agents.code_formula_registry import (
    FormulaInputError,
    UnknownFormulaError,
    execute_code_formula,
    registered_formula_ids,
)


XI_B_TABLE_ID = "T_3362_CH05_5_2_1_RELATIVE_LIMIT_COMPRESSION_ZONE_HEIGHT"
FLEXURE_ID = "F_3362_CH05_5_2_2_1_BENDING_CAPACITY_TENSION_FLANGE"
SHEAR_LIMIT_ID = "F_3362_CH08_8_4_4_CAP_BEAM_SHEAR_CAPACITY"
SHEAR_REINFORCED_ID = "F_3362_CH08_8_4_5_CAP_BEAM_INCLINED_SHEAR"
STABILITY_COEFFICIENT_ID = "T_3362_CH05_5_3_1_STABILITY_COEFFICIENT"
AXIAL_COMPRESSION_ID = "F_3362_CH05_5_3_1_AXIAL_COMPRESSION_CAPACITY"
SPIRAL_CONFINED_ID = "F_3362_CH05_5_3_2_1_SPIRAL_CONFINED_COMPRESSION_CAPACITY"


def test_registry_exposes_only_fixed_python_executors() -> None:
    assert registered_formula_ids() == (
        FLEXURE_ID,
        AXIAL_COMPRESSION_ID,
        SPIRAL_CONFINED_ID,
        SHEAR_LIMIT_ID,
        SHEAR_REINFORCED_ID,
        XI_B_TABLE_ID,
        STABILITY_COEFFICIENT_ID,
    )

    with pytest.raises(UnknownFormulaError):
        execute_code_formula("__import__('os').system('whoami')", {})


@pytest.mark.parametrize(
    ("steel_grade", "concrete_grade", "expected"),
    [
        ("HPB300", "C40", 0.58),
        ("HRB400", "C50", 0.53),
        ("HRBF400", "C35", 0.53),
        ("RRB400", "C50及以下", 0.53),
        ("HRB500", "C45", 0.49),
    ],
)
def test_xi_b_table_lookup_matches_project_code_table(
    steel_grade: str,
    concrete_grade: str,
    expected: float,
) -> None:
    result = execute_code_formula(
        XI_B_TABLE_ID,
        {"steel_grade": steel_grade, "concrete_strength_grade": concrete_grade},
    )

    assert result.status == "computed"
    assert result.outputs["xi_b"] == expected
    assert result.clause == "5.2.1"
    assert result.evidence_ids == (XI_B_TABLE_ID,)


def test_xi_b_table_sends_out_of_range_material_to_manual_review() -> None:
    result = execute_code_formula(
        XI_B_TABLE_ID,
        {"steel_grade": "HRB400", "concrete_strength_grade": "C55"},
    )

    assert result.status == "manual_review"
    assert result.outputs == {}
    assert "C50" in result.message


def test_flexural_executor_applies_xi_b_without_clipping_compression_zone() -> None:
    result = execute_code_formula(
        FLEXURE_ID,
        {
            "b_mm": 1000.0,
            "h0_mm": 1000.0,
            "As_mm2": 30000.0,
            "fcd_MPa": 18.4,
            "fsd_MPa": 330.0,
            "steel_grade": "HRB400",
            "concrete_strength_grade": "C40",
        },
    )

    expected_x = 30000.0 * 330.0 / (18.4 * 1000.0)
    expected_mu = 18.4 * 1000.0 * expected_x * (1000.0 - expected_x / 2.0) / 1.0e6
    assert result.outputs["x_mm"] == pytest.approx(expected_x)
    assert result.outputs["x_limit_mm"] == pytest.approx(530.0)
    assert result.outputs["Mu_kN_m"] == pytest.approx(expected_mu)
    assert result.checks["compression_zone_limit_ok"] is False
    assert result.status == "manual_review"
    assert set(result.evidence_ids) == {FLEXURE_ID, XI_B_TABLE_ID}


def test_cap_beam_shear_executors_match_registered_equations() -> None:
    limit = execute_code_formula(
        SHEAR_LIMIT_ID,
        {
            "l0_mm": 6000.0,
            "h_mm": 1600.0,
            "fcu_k_MPa": 40.0,
            "b_mm": 1900.0,
            "h0_mm": 1500.0,
        },
    )
    expected_limit = 0.33e-4 * (6000.0 / 1600.0 + 10.3) * math.sqrt(40.0) * 1900.0 * 1500.0
    assert limit.outputs["V_limit_kN"] == pytest.approx(expected_limit)

    reinforced = execute_code_formula(
        SHEAR_REINFORCED_ID,
        {
            "alpha1": 1.0,
            "l_mm": 6000.0,
            "h_mm": 1600.0,
            "b_mm": 1900.0,
            "h0_mm": 1500.0,
            "P": 0.8,
            "fcu_k_MPa": 40.0,
            "rho_sv": 0.002,
            "fsv_MPa": 330.0,
        },
    )
    expected_reinforced = (
        0.5e-4
        * (14.0 - 6000.0 / 1600.0)
        * 1900.0
        * 1500.0
        * math.sqrt((2.0 + 0.6 * 0.8) * 40.0 * 0.002 * 330.0)
    )
    assert reinforced.outputs["V_capacity_kN"] == pytest.approx(expected_reinforced)


def test_formula_executor_rejects_missing_or_nonpositive_inputs() -> None:
    with pytest.raises(FormulaInputError):
        execute_code_formula(FLEXURE_ID, {"b_mm": 1000.0})

    with pytest.raises(FormulaInputError):
        execute_code_formula(
            SHEAR_LIMIT_ID,
            {"l0_mm": 6000.0, "h_mm": 0.0, "fcu_k_MPa": 40.0, "b_mm": 1900.0, "h0_mm": 1500.0},
        )
