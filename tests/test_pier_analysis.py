from __future__ import annotations

import math

import pytest

from bridge_agents.joint_reinforcement import (
    compute_axial_capacity_check,
    compute_column_self_weight,
    compute_controlling_axial_force,
    compute_total_vertical_load,
    compute_vertical_balance,
    extract_longitudinal_rebar_area,
    summarize_pier_axial,
)


class TestComputeColumnSelfWeight:
    def test_single_column(self):
        weight = compute_column_self_weight(
            column_count=1,
            column_diameter_m=1.8,
            net_height_m=10.0,
            concrete_unit_weight=26.0,
        )
        expected = math.pi * (0.9 ** 2) * 10.0 * 26.0
        assert weight == pytest.approx(expected)

    def test_multiple_columns(self):
        single = math.pi * (0.9 ** 2) * 10.0 * 26.0
        weight = compute_column_self_weight(3, 1.8, 10.0, 26.0)
        assert weight == pytest.approx(3 * single)


class TestComputeControllingAxialForce:
    def test_max_reaction_plus_self_weight(self):
        force = compute_controlling_axial_force([100.0, 120.0, 110.0], 50.0)
        assert force == pytest.approx(170.0)

    def test_empty_reactions(self):
        assert compute_controlling_axial_force([], 50.0) == pytest.approx(50.0)


class TestComputeVerticalBalance:
    def test_balanced(self):
        result = compute_vertical_balance([100.0, 100.0], 200.0, tolerance=1.0)
        assert result["balanced"] is True
        assert result["error_kN"] == pytest.approx(0.0)

    def test_unbalanced_beyond_tolerance(self):
        result = compute_vertical_balance([100.0, 90.0], 200.0, tolerance=1.0)
        assert result["balanced"] is False
        assert result["error_kN"] == pytest.approx(10.0)

    def test_within_tolerance(self):
        result = compute_vertical_balance([100.0, 99.6], 200.0, tolerance=0.5)
        assert result["balanced"] is True


class TestSummarizePierAxial:
    def test_combines_self_weight_axial_and_balance(self):
        single = math.pi * (0.9 ** 2) * 10.0 * 26.0
        result = summarize_pier_axial(
            column_reactions_kN=[100.0, 120.0, 110.0],
            column_count=3,
            column_diameter_m=1.8,
            net_height_m=10.0,
            total_vertical_kN=330.0,
        )
        assert result["single_column_self_weight_kN"] == pytest.approx(single)
        assert result["total_column_self_weight_kN"] == pytest.approx(3 * single)
        assert result["controlling_axial_force_kN"] == pytest.approx(120.0 + single)
        assert result["balance"]["balanced"] is True


class TestComputeTotalVerticalLoad:
    def test_sums_cap_and_superstructure_dead_load(self):
        load_design_output = {
            "load_design_output": {
                "load_design_result": {
                    "cap_self_weight": {"total_weight_kN": 100.0},
                    "superstructure_dead_load": {
                        "support_points": [
                            {"reaction_kN": 20.0},
                            {"reaction_kN": 30.0},
                        ]
                    },
                }
            }
        }
        assert compute_total_vertical_load(load_design_output) == pytest.approx(150.0)


class TestExtractLongitudinalRebarArea:
    def test_sums_bar_areas(self):
        pier_column = {
            "longitudinal_bars": [
                {"dia": 28, "count": 44},
            ]
        }
        expected = math.pi * (14.0 ** 2) * 44
        assert extract_longitudinal_rebar_area(pier_column) == pytest.approx(expected)

    def test_empty_bars(self):
        assert extract_longitudinal_rebar_area({}) == pytest.approx(0.0)


class TestComputeAxialCapacityCheck:
    def test_computes_slenderness_phi_and_capacity(self):
        result = compute_axial_capacity_check(
            controlling_axial_force_kN=3000.0,
            column_diameter_mm=1800.0,
            net_height_m=8.0,
            longitudinal_bars=[{"dia": 28, "count": 44}],
        )
        assert result["status"] == "computed"
        assert result["reinforcement_ratio"] > 0.0
        assert result["slenderness_ratio"] == pytest.approx(4000.0 * 8.0 / 1800.0)
        assert result["stability_factor_phi"] > 0.0
        assert result["capacity_kN"] > 0.0
        assert "check_ok" in result

    def test_circular_lookup_resolves_moderate_slenderness(self):
        # d=1.0m、H=10m → λ(l0/i)=40,圆形按表5.3.1 l0/i 列保守取 φ=0.95,
        # 不再因旧实现误用矩形 l0/b 列(上限20)而错判"超界转人工"。
        result = compute_axial_capacity_check(
            controlling_axial_force_kN=3000.0,
            column_diameter_mm=1000.0,
            net_height_m=10.0,
            longitudinal_bars=[{"dia": 28, "count": 20}],
        )
        assert result["status"] == "computed"
        assert result["slenderness_ratio"] == pytest.approx(4.0 * 10.0 * 1000.0 / 1000.0)
        assert result["slenderness_basis"] == "l0/i"
        assert result["stability_factor_phi"] == pytest.approx(0.95)
        assert result["effective_length"]["effective_length_factor"] == 1.0
        assert result["effective_length"]["restraint_data_complete"] is False

    def test_manual_review_when_slenderness_out_of_registered_range(self):
        # d=1.0m、H=44m → λ(l0/i)=4*44*1000/1000=176 > 174(规范表 φ=0.19 对应上限),才转人工。
        result = compute_axial_capacity_check(
            controlling_axial_force_kN=3000.0,
            column_diameter_mm=1000.0,
            net_height_m=44.0,
            longitudinal_bars=[{"dia": 28, "count": 20}],
        )
        assert result["status"] == "manual_review"
        assert result["effective_length"]["effective_length_factor"] == 1.0
        assert result["effective_length"]["restraint_data_complete"] is False
        assert "不等同于承载力已判定不通过" in result["message"]

    def test_explicit_effective_length_factor_changes_slenderness(self):
        result = compute_axial_capacity_check(
            controlling_axial_force_kN=3000.0,
            column_diameter_mm=1800.0,
            net_height_m=17.111,
            longitudinal_bars=[{"dia": 28, "count": 48}],
            effective_length_factor=0.5,
        )

        assert result["status"] == "computed"
        assert result["slenderness_ratio"] == pytest.approx(
            0.5 * 17111.0 / (1800.0 / 4.0)
        )
        assert result["effective_length"]["factor_source"] == "explicit_input"


def test_reinforcement_runner_converts_task_diameter_metres_to_mm() -> None:
    from tools.reinforcement_design_tool import (
        _compute_axial_check,
        _compute_pier_axial_summary,
    )

    task = {
        "桥墩尺寸信息": {
            "墩柱几何信息": {"column_count": 2, "column_diameter": 1.8}
        },
        "墩柱净高信息": {"controlling_net_height_m": 17.111},
    }
    force_payload = {
        "internal_force_output": {
            "dead_case_full_beam": {"column_reactions_kN": [3163.9, 3163.7]}
        }
    }
    load_payload = {
        "load_design_output": {
            "load_design_output": {
                "load_design_result": {
                    "cap_self_weight": {"total_weight_kN": 327.6},
                    "superstructure_dead_load": {
                        "support_points": [
                            {"reaction_kN": 3000.0},
                            {"reaction_kN": 3000.0},
                        ]
                    },
                }
            }
        }
    }
    axial = _compute_pier_axial_summary(task, force_payload, load_payload)
    check = _compute_axial_check(
        task,
        axial,
        {"reinforcement": {"pier_column": {"longitudinal_bars": [{"dia": 28, "count": 48}]}}},
    )

    assert axial is not None
    assert axial["single_column_self_weight_kN"] > 1000.0
    assert check is not None
    assert check["reinforcement_ratio"] == pytest.approx(
        (48 * math.pi * 14.0**2) / (math.pi * 900.0**2)
    )
    assert check["slenderness_ratio"] == pytest.approx(17111.0 / 450.0)
