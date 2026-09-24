from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Tuple


XI_B_TABLE_ID = "T_3362_CH05_5_2_1_RELATIVE_LIMIT_COMPRESSION_ZONE_HEIGHT"
FLEXURE_ID = "F_3362_CH05_5_2_2_1_BENDING_CAPACITY_TENSION_FLANGE"
SHEAR_LIMIT_ID = "F_3362_CH08_8_4_4_CAP_BEAM_SHEAR_CAPACITY"
SHEAR_REINFORCED_ID = "F_3362_CH08_8_4_5_CAP_BEAM_INCLINED_SHEAR"
STABILITY_COEFFICIENT_ID = "T_3362_CH05_5_3_1_STABILITY_COEFFICIENT"
AXIAL_COMPRESSION_ID = "F_3362_CH05_5_3_1_AXIAL_COMPRESSION_CAPACITY"
SPIRAL_CONFINED_ID = "F_3362_CH05_5_3_2_1_SPIRAL_CONFINED_COMPRESSION_CAPACITY"

# 表5.3.1（钢筋混凝土轴心受压构件稳定系数）同一组 φ 同时给出三种"长细比"表示：
#   矩形截面：l0/b（b 为截面短边）；
#   圆截面：规范另列 l0/(2r)；因圆截面回转半径 i=d/4，回转半径口径 l0/i = 4·l0/d，
#           与 l0/(2r) 列一一对应（规范表两列独立给值、略有取整差异）。
# 项目墩柱验算按回转半径口径计算 λ=l0/i 作为输入，因此圆墩必须查 l0/i 列；
# 矩形构件查 l0/b 列（当前无矩形调用，保留通用性与旧调用语义）。
# 登记范围：矩形 l0/b<=28（φ>=0.56，后段 30~50 待按规范原文补录）；
# 圆 l0/i<=174（φ>=0.19，已按规范原表登记完整）。超出登记范围禁止外推，按 manual_review。
_STABILITY_TABLE_RECTANGULAR_L0B: Dict[float, float] = {
    8.0: 1.00, 10.0: 0.98, 12.0: 0.95, 14.0: 0.92,
    16.0: 0.87, 18.0: 0.81, 20.0: 0.75, 22.0: 0.70,
    24.0: 0.65, 26.0: 0.60, 28.0: 0.56,
}
_STABILITY_TABLE_CIRCULAR_L0I: Dict[float, float] = {
    28.0: 1.00, 35.0: 0.98, 42.0: 0.95, 48.0: 0.92,
    55.0: 0.87, 62.0: 0.81, 69.0: 0.75, 76.0: 0.70,
    83.0: 0.65, 90.0: 0.60, 97.0: 0.56,
    104.0: 0.52, 111.0: 0.48, 118.0: 0.44, 125.0: 0.40,
    132.0: 0.36, 139.0: 0.32, 146.0: 0.29, 153.0: 0.26,
    160.0: 0.23, 167.0: 0.21, 174.0: 0.19,
}
# 截面形状 -> (查表口径, 稳定系数表)。缺省按 rectangular 兼容旧语义。
_STABILITY_SHAPE_COLUMNS: Dict[str, Tuple[str, Dict[float, float]]] = {
    "rectangular": ("l0/b", _STABILITY_TABLE_RECTANGULAR_L0B),
    "circular": ("l0/i", _STABILITY_TABLE_CIRCULAR_L0I),
}


class FormulaRegistryError(ValueError):
    """可执行公式注册表错误。"""


class UnknownFormulaError(FormulaRegistryError):
    """请求的规范公式尚无固定 Python 执行器。"""


class FormulaInputError(FormulaRegistryError):
    """公式输入缺失或不满足基本定义域。"""


@dataclass(frozen=True)
class FormulaExecutionResult:
    formula_id: str
    standard: str
    clause: str
    status: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any] = field(default_factory=dict)
    checks: Mapping[str, bool] = field(default_factory=dict)
    evidence_ids: Tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula_id": self.formula_id,
            "standard": self.standard,
            "clause": self.clause,
            "status": self.status,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "checks": dict(self.checks),
            "evidence_ids": list(self.evidence_ids),
            "message": self.message,
        }


Executor = Callable[[Mapping[str, Any]], FormulaExecutionResult]


def _required_number(inputs: Mapping[str, Any], name: str, *, positive: bool = True) -> float:
    if name not in inputs:
        raise FormulaInputError(f"缺少公式输入: {name}")
    try:
        value = float(inputs[name])
    except (TypeError, ValueError) as exc:
        raise FormulaInputError(f"公式输入 {name} 必须为数值。") from exc
    if not math.isfinite(value):
        raise FormulaInputError(f"公式输入 {name} 必须为有限数值。")
    if positive and value <= 0:
        raise FormulaInputError(f"公式输入 {name} 必须大于 0。")
    return value


def _normalize_steel_grade(value: Any) -> str:
    grade = str(value or "").upper().replace(" ", "")
    aliases = {
        "HPB300": "HPB300",
        "HRB400": "HRB400_GROUP",
        "HRBF400": "HRB400_GROUP",
        "RRB400": "HRB400_GROUP",
        "HRB400/HRBF400/RRB400": "HRB400_GROUP",
        "HRB500": "HRB500",
        "预应力钢筋": "PRESTRESSING",
    }
    return aliases.get(grade, grade)


def _concrete_grade_at_most_c50(value: Any) -> bool:
    text = str(value or "").upper().replace(" ", "")
    if text in {"C50及以下", "C50以下", "<=C50", "≤C50"}:
        return True
    match = re.fullmatch(r"C(\d+(?:\.\d+)?)", text)
    return bool(match and float(match.group(1)) <= 50.0)


def _execute_xi_b(inputs: Mapping[str, Any]) -> FormulaExecutionResult:
    steel_grade = _normalize_steel_grade(inputs.get("steel_grade"))
    concrete_grade = inputs.get("concrete_strength_grade")
    normalized_inputs = {
        "steel_grade": str(inputs.get("steel_grade") or ""),
        "concrete_strength_grade": str(concrete_grade or ""),
    }
    if not steel_grade or not concrete_grade:
        raise FormulaInputError("xi_b 查表需要 steel_grade 和 concrete_strength_grade。")
    if not _concrete_grade_at_most_c50(concrete_grade):
        return FormulaExecutionResult(
            formula_id=XI_B_TABLE_ID,
            standard="JTG 3362-2018",
            clause="5.2.1",
            status="manual_review",
            inputs=normalized_inputs,
            evidence_ids=(XI_B_TABLE_ID,),
            message="表5.2.1当前注册范围为C50及以下，超出范围需人工确定xi_b。",
        )
    values = {
        "HPB300": 0.58,
        "HRB400_GROUP": 0.53,
        "HRB500": 0.49,
        "PRESTRESSING": 0.40,
    }
    if steel_grade not in values:
        return FormulaExecutionResult(
            formula_id=XI_B_TABLE_ID,
            standard="JTG 3362-2018",
            clause="5.2.1",
            status="manual_review",
            inputs=normalized_inputs,
            evidence_ids=(XI_B_TABLE_ID,),
            message="钢筋牌号不在表5.2.1当前注册项内，需人工确定xi_b。",
        )
    return FormulaExecutionResult(
        formula_id=XI_B_TABLE_ID,
        standard="JTG 3362-2018",
        clause="5.2.1",
        status="computed",
        inputs=normalized_inputs,
        outputs={"xi_b": values[steel_grade]},
        checks={"table_match_found": True},
        evidence_ids=(XI_B_TABLE_ID,),
    )


def _execute_flexure(inputs: Mapping[str, Any]) -> FormulaExecutionResult:
    b = _required_number(inputs, "b_mm")
    h0 = _required_number(inputs, "h0_mm")
    area = _required_number(inputs, "As_mm2")
    fcd = _required_number(inputs, "fcd_MPa")
    fsd = _required_number(inputs, "fsd_MPa")
    xi_result = _execute_xi_b(inputs)
    if xi_result.status != "computed":
        return FormulaExecutionResult(
            formula_id=FLEXURE_ID,
            standard="JTG 3362-2018",
            clause="5.2.2",
            status="manual_review",
            inputs=dict(inputs),
            evidence_ids=(FLEXURE_ID, XI_B_TABLE_ID),
            message=xi_result.message,
        )
    xi_b = float(xi_result.outputs["xi_b"])
    x = fsd * area / (fcd * b)
    xi = x / h0
    x_limit = xi_b * h0
    limit_ok = x <= x_limit
    mu = fcd * b * x * (h0 - x / 2.0) / 1.0e6
    return FormulaExecutionResult(
        formula_id=FLEXURE_ID,
        standard="JTG 3362-2018",
        clause="5.2.2",
        status="computed" if limit_ok else "manual_review",
        inputs=dict(inputs),
        outputs={"x_mm": x, "xi": xi, "xi_b": xi_b, "x_limit_mm": x_limit, "Mu_kN_m": mu},
        checks={"force_equilibrium_ok": True, "compression_zone_limit_ok": limit_ok},
        evidence_ids=(FLEXURE_ID, XI_B_TABLE_ID),
        message="" if limit_ok else "受压区高度超过表5.2.1规定的相对界限受压区高度。",
    )


def _execute_shear_limit(inputs: Mapping[str, Any]) -> FormulaExecutionResult:
    l0 = _required_number(inputs, "l0_mm")
    h = _required_number(inputs, "h_mm")
    fcu_k = _required_number(inputs, "fcu_k_MPa")
    b = _required_number(inputs, "b_mm")
    h0 = _required_number(inputs, "h0_mm")
    capacity = 0.33e-4 * (l0 / h + 10.3) * math.sqrt(fcu_k) * b * h0
    return FormulaExecutionResult(
        formula_id=SHEAR_LIMIT_ID,
        standard="JTG 3362-2018",
        clause="8.4.4",
        status="computed",
        inputs=dict(inputs),
        outputs={"V_limit_kN": capacity, "span_depth_ratio": l0 / h},
        checks={"positive_capacity": capacity > 0},
        evidence_ids=(SHEAR_LIMIT_ID,),
    )


def _execute_reinforced_shear(inputs: Mapping[str, Any]) -> FormulaExecutionResult:
    alpha1 = _required_number(inputs, "alpha1")
    length = _required_number(inputs, "l_mm")
    h = _required_number(inputs, "h_mm")
    b = _required_number(inputs, "b_mm")
    h0 = _required_number(inputs, "h0_mm")
    p_ratio = _required_number(inputs, "P", positive=False)
    fcu_k = _required_number(inputs, "fcu_k_MPa")
    rho_sv = _required_number(inputs, "rho_sv")
    fsv = _required_number(inputs, "fsv_MPa")
    span_depth_ratio = length / h
    factor = 14.0 - span_depth_ratio
    if p_ratio < 0 or factor <= 0:
        return FormulaExecutionResult(
            formula_id=SHEAR_REINFORCED_ID,
            standard="JTG 3362-2018",
            clause="8.4.5",
            status="manual_review",
            inputs=dict(inputs),
            outputs={"span_depth_ratio": span_depth_ratio},
            checks={"formula_domain_ok": False},
            evidence_ids=(SHEAR_REINFORCED_ID,),
            message="式(8.4.5)输入超出当前固定执行器定义域。",
        )
    capacity = 0.5e-4 * alpha1 * factor * b * h0 * math.sqrt((2.0 + 0.6 * p_ratio) * fcu_k * rho_sv * fsv)
    return FormulaExecutionResult(
        formula_id=SHEAR_REINFORCED_ID,
        standard="JTG 3362-2018",
        clause="8.4.5",
        status="computed",
        inputs=dict(inputs),
        outputs={"V_capacity_kN": capacity, "span_depth_ratio": span_depth_ratio},
        checks={"formula_domain_ok": True, "positive_capacity": capacity > 0},
        evidence_ids=(SHEAR_REINFORCED_ID,),
    )


def _execute_stability_coefficient(inputs: Mapping[str, Any]) -> FormulaExecutionResult:
    slenderness = _required_number(inputs, "slenderness_ratio")
    shape = str(inputs.get("section_shape") or "rectangular").strip().lower()
    shape_config = _STABILITY_SHAPE_COLUMNS.get(shape)
    if shape_config is None:
        return FormulaExecutionResult(
            formula_id=STABILITY_COEFFICIENT_ID,
            standard="JTG 3362-2018",
            clause="5.3.1",
            status="manual_review",
            inputs=dict(inputs),
            evidence_ids=(STABILITY_COEFFICIENT_ID,),
            message=f"截面形状 {shape} 不在表5.3.1 当前登记范围内（rectangular/circular），需人工复核。",
        )
    column_label, table = shape_config
    upper = max(table)
    if slenderness > upper:
        return FormulaExecutionResult(
            formula_id=STABILITY_COEFFICIENT_ID,
            standard="JTG 3362-2018",
            clause="5.3.1",
            status="manual_review",
            inputs=dict(inputs),
            evidence_ids=(STABILITY_COEFFICIENT_ID,),
            message=(
                f"长细比(口径 {column_label}) {slenderness:g} 超过表5.3.1 当前登记范围"
                f"上限 {upper:g}，稳定系数不能自动取值，需人工复核。"
            ),
        )
    matched = min(table)
    for key in sorted(table):
        if slenderness <= key:
            matched = key
            break
    phi = table[matched]
    return FormulaExecutionResult(
        formula_id=STABILITY_COEFFICIENT_ID,
        standard="JTG 3362-2018",
        clause="5.3.1",
        status="computed",
        inputs=dict(inputs),
        outputs={
            "stability_factor_phi": phi,
            "matched_slenderness_ratio": matched,
            "section_shape": shape,
            "slenderness_basis": column_label,
        },
        checks={"table_match_found": True},
        evidence_ids=(STABILITY_COEFFICIENT_ID,),
    )


def _execute_axial_compression(inputs: Mapping[str, Any]) -> FormulaExecutionResult:
    nd = _required_number(inputs, "Nd_kN")
    phi = _required_number(inputs, "phi")
    fcd = _required_number(inputs, "fcd_MPa")
    area = _required_number(inputs, "A_mm2")
    fsd_prime = _required_number(inputs, "fsd_prime_MPa")
    as_prime = _required_number(inputs, "As_prime_mm2", positive=False)
    gamma0 = _required_number(inputs, "gamma0")
    capacity = 0.9 * phi * (fcd * area + fsd_prime * as_prime) / gamma0 / 1000.0
    check_ok = nd <= capacity
    return FormulaExecutionResult(
        formula_id=AXIAL_COMPRESSION_ID,
        standard="JTG 3362-2018",
        clause="5.3.1",
        status="computed",
        inputs=dict(inputs),
        outputs={
            "capacity_kN": capacity,
            "utilization": nd / capacity,
            "check_ok": check_ok,
        },
        checks={"axial_capacity_ok": check_ok},
        evidence_ids=(AXIAL_COMPRESSION_ID, STABILITY_COEFFICIENT_ID),
    )


def _execute_spiral_confined(inputs: Mapping[str, Any]) -> FormulaExecutionResult:
    fcd = _required_number(inputs, "fcd_MPa")
    acor = _required_number(inputs, "Acor_mm2")
    k = _required_number(inputs, "k")
    fsd = _required_number(inputs, "fsd_MPa")
    ass0 = _required_number(inputs, "Ass0_mm2", positive=False)
    fsd_prime = _required_number(inputs, "fsd_prime_MPa")
    as_prime = _required_number(inputs, "As_prime_mm2", positive=False)
    gamma0 = _required_number(inputs, "gamma0")
    capacity = 0.9 * (fcd * acor + k * fsd * ass0 + fsd_prime * as_prime) / gamma0 / 1000.0
    return FormulaExecutionResult(
        formula_id=SPIRAL_CONFINED_ID,
        standard="JTG 3362-2018",
        clause="5.3.2",
        status="computed",
        inputs=dict(inputs),
        outputs={"capacity_kN": capacity},
        checks={"spiral_capacity_computed": True},
        evidence_ids=(SPIRAL_CONFINED_ID,),
    )


_EXECUTORS: Dict[str, Executor] = {
    FLEXURE_ID: _execute_flexure,
    SHEAR_LIMIT_ID: _execute_shear_limit,
    SHEAR_REINFORCED_ID: _execute_reinforced_shear,
    XI_B_TABLE_ID: _execute_xi_b,
    STABILITY_COEFFICIENT_ID: _execute_stability_coefficient,
    AXIAL_COMPRESSION_ID: _execute_axial_compression,
    SPIRAL_CONFINED_ID: _execute_spiral_confined,
}


def registered_formula_ids() -> Tuple[str, ...]:
    """返回具备固定 Python 实现的规范公式/表格 ID。"""
    return tuple(sorted(_EXECUTORS))


def execute_code_formula(formula_id: str, inputs: Mapping[str, Any]) -> FormulaExecutionResult:
    """执行注册公式；YAML 表达式不会在此处解析或求值。"""
    executor = _EXECUTORS.get(str(formula_id))
    if executor is None:
        raise UnknownFormulaError(f"公式尚未注册固定 Python 执行器: {formula_id}")
    if not isinstance(inputs, Mapping):
        raise FormulaInputError("公式输入必须为映射对象。")
    return executor(inputs)
