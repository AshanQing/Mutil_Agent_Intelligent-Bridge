from __future__ import annotations

import pytest

from bridge_agents.code_formula_registry import execute_code_formula


STABILITY_ID = "T_3362_CH05_5_3_1_STABILITY_COEFFICIENT"
AXIAL_ID = "F_3362_CH05_5_3_1_AXIAL_COMPRESSION_CAPACITY"
SPIRAL_ID = "F_3362_CH05_5_3_2_1_SPIRAL_CONFINED_COMPRESSION_CAPACITY"


class TestStabilityFactor:
    # 表5.3.1 按截面形状分列：矩形查 l0/b 列，圆形（墩柱）查 l0/i 列。
    @pytest.mark.parametrize(
        ("slenderness", "expected", "shape"),
        [
            # 矩形截面 l0/b 口径（缺省 shape 保持兼容）
            (8.0, 1.0, None),
            (9.0, 0.98, "rectangular"),
            (11.0, 0.95, "rectangular"),
            (12.0, 0.95, "rectangular"),
            (20.0, 0.75, "rectangular"),
            (22.0, 0.70, "rectangular"),
            (28.0, 0.56, "rectangular"),
            # 圆形截面 l0/i 口径（保守取更大一档长细比）
            (27.0, 1.0, "circular"),
            (28.0, 1.0, "circular"),
            (35.0, 0.98, "circular"),
            (48.0, 0.92, "circular"),
            (62.0, 0.81, "circular"),
            (69.0, 0.75, "circular"),
            (70.5, 0.70, "circular"),  # 本工程高墩 λ≈70.5，规范表 l0/i 列覆盖，可自动取值
            (76.0, 0.70, "circular"),
            (90.0, 0.60, "circular"),
            (97.0, 0.56, "circular"),
            (100.0, 0.52, "circular"),
            (104.0, 0.52, "circular"),
            (118.0, 0.44, "circular"),
            (146.0, 0.29, "circular"),
            (160.0, 0.23, "circular"),
            (174.0, 0.19, "circular"),
        ],
    )
    def test_conservative_lookup(self, slenderness, expected, shape):
        payload = {"slenderness_ratio": slenderness, "rebar_type": "ordinary"}
        if shape is not None:
            payload["section_shape"] = shape
        result = execute_code_formula(
            STABILITY_ID,
            payload,
        )
        assert result.status == "computed"
        assert result.outputs["stability_factor_phi"] == pytest.approx(expected)

    def test_rectangular_out_of_range_sends_to_manual_review(self):
        # 矩形 l0/b 口径超出当前登记上限 28
        result = execute_code_formula(
            STABILITY_ID,
            {"slenderness_ratio": 29.0, "rebar_type": "ordinary", "section_shape": "rectangular"},
        )
        assert result.status == "manual_review"

    def test_circular_out_of_registered_range_sends_to_manual_review(self):
        # 圆形 l0/i 口径超出当前登记上限 174（规范表 φ 到 0.19 为止）
        result = execute_code_formula(
            STABILITY_ID,
            {"slenderness_ratio": 175.0, "rebar_type": "ordinary", "section_shape": "circular"},
        )
        assert result.status == "manual_review"

    def test_circular_lookup_records_table_basis(self):
        result = execute_code_formula(
            STABILITY_ID,
            {"slenderness_ratio": 70.5, "rebar_type": "ordinary", "section_shape": "circular"},
        )
        assert result.outputs["section_shape"] == "circular"
        assert result.outputs["slenderness_basis"] == "l0/i"
        assert result.outputs["matched_slenderness_ratio"] == 76.0


class TestAxialCompression:
    def test_capacity_computes_and_passes(self):
        result = execute_code_formula(
            AXIAL_ID,
            {
                "Nd_kN": 3000.0,
                "phi": 0.98,
                "fcd_MPa": 18.4,
                "A_mm2": 2.54e6,
                "fsd_prime_MPa": 330.0,
                "As_prime_mm2": 12000.0,
                "gamma0": 1.0,
            },
        )
        assert result.status == "computed"
        expected = 0.9 * 0.98 * (18.4 * 2.54e6 + 330.0 * 12000.0) / 1.0 / 1000.0
        assert result.outputs["capacity_kN"] == pytest.approx(expected)
        assert result.outputs["check_ok"] is True

    def test_capacity_reports_failure(self):
        result = execute_code_formula(
            AXIAL_ID,
            {
                "Nd_kN": 999999.0,
                "phi": 0.75,
                "fcd_MPa": 18.4,
                "A_mm2": 2.54e6,
                "fsd_prime_MPa": 330.0,
                "As_prime_mm2": 12000.0,
                "gamma0": 1.0,
            },
        )
        assert result.outputs["check_ok"] is False


class TestSpiralConfined:
    def test_capacity_computes(self):
        result = execute_code_formula(
            SPIRAL_ID,
            {
                "fcd_MPa": 18.4,
                "Acor_mm2": 2.2e6,
                "k": 2.0,
                "fsd_MPa": 330.0,
                "Ass0_mm2": 5000.0,
                "fsd_prime_MPa": 330.0,
                "As_prime_mm2": 12000.0,
                "gamma0": 1.0,
            },
        )
        assert result.status == "computed"
        expected = 0.9 * (18.4 * 2.2e6 + 2.0 * 330.0 * 5000.0 + 330.0 * 12000.0) / 1.0 / 1000.0
        assert result.outputs["capacity_kN"] == pytest.approx(expected)

