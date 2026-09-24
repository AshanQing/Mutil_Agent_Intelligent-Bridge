from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

from .code_formula_registry import (
    AXIAL_COMPRESSION_ID,
    STABILITY_COEFFICIENT_ID,
    execute_code_formula,
)


def inject_column_heights(
    reinforcement_tasks: List[Dict[str, Any]],
    pier_group_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把设计组的墩柱净高注入配筋任务，避免依赖 LLM 转抄。

    匹配键必须同时包含桥梁号、单元号与墩号：不同联的同号墩（如桥2的墩1
    与桥3的墩1）若只按墩号匹配会互相串值，导致净高张冠李戴。匹配不到的墩
    在「墩柱净高信息.missing_pier_groups」中显式记录，不再静默置空。
    """
    groups = (pier_group_result or {}).get("design_groups", []) or []
    pier_to_group: Dict[tuple, Dict[str, Any]] = {}
    for group in groups:
        for pier_no in group.get("member_piers", []) or []:
            pier_to_group[
                (str(group.get("bridge_id")), str(group.get("unit_id")), str(pier_no))
            ] = group

    for task in reinforcement_tasks:
        bridge_id = str(task.get("桥梁编号") or task.get("bridge_id") or "").strip()
        unit_id = str(task.get("单元编号") or task.get("unit_id") or "").strip()
        pier_nos = [str(p) for p in task.get("包含桥墩号列表", []) or []]
        controlling_values: List[float] = []
        member_heights: Dict[str, Optional[float]] = {}
        missing_piers: List[str] = []
        for pier_no in pier_nos:
            group = pier_to_group.get((bridge_id, unit_id, pier_no))
            if not group:
                missing_piers.append(pier_no)
                continue
            member_heights[pier_no] = group.get("member_net_heights_m", {}).get(pier_no)
            controlling = group.get("controlling_net_height_m")
            if controlling is not None:
                controlling_values.append(float(controlling))

        controlling_height = max(controlling_values) if controlling_values else None
        task_input = task.get("桥墩尺寸信息") if isinstance(task.get("桥墩尺寸信息"), dict) else task
        geom = task_input.setdefault("墩柱几何信息", {})
        if isinstance(geom, dict) and controlling_height is not None:
            geom["column_height"] = controlling_height
        net_height_info = {
            "controlling_net_height_m": controlling_height,
            "member_net_heights_m": member_heights,
        }
        if missing_piers:
            net_height_info["missing_pier_groups"] = missing_piers
        task["墩柱净高信息"] = net_height_info
    return reinforcement_tasks


def validate_joint_reinforcement(result: Any) -> Dict[str, Any]:
    """校验联合配筋结果同时包含盖梁与墩柱配筋，且墩柱配筋不是空壳。"""
    if not isinstance(result, dict):
        return {"complete": False, "missing": ["reinforcement"]}
    reinforcement = result.get("reinforcement")
    if not isinstance(reinforcement, dict):
        return {"complete": False, "missing": ["reinforcement"]}
    missing = [key for key in ("pier_cap", "pier_column") if key not in reinforcement]
    column = reinforcement.get("pier_column")
    if isinstance(column, dict) and not any(
        column.get(key)
        for key in ("longitudinal_bars", "spiral_stirrups", "ordinary_hoops", "strengthening_bars")
    ):
        missing.append("pier_column.empty")
    return {"complete": not missing, "missing": missing}


# ---------------------------------------------------------------------------
# 联合分析与控制包络（纯计算）
# ---------------------------------------------------------------------------

def compute_column_self_weight(
    column_count: int,
    column_diameter_m: float,
    net_height_m: float,
    concrete_unit_weight: float = 26.0,
) -> float:
    """墩柱自重（全组），按控制净高计算，单位为 kN。"""
    count = int(column_count)
    if count <= 0 or float(column_diameter_m) <= 0 or float(net_height_m) <= 0:
        return 0.0
    radius = float(column_diameter_m) / 2.0
    single = math.pi * radius * radius * float(net_height_m) * float(concrete_unit_weight)
    return count * single


def compute_controlling_axial_force(
    column_reactions_kN: List[float],
    self_weight_kN: float,
) -> float:
    """控制轴力 = 最大柱顶竖向反力 + 单柱自重（保守取全组控制自重）。"""
    reactions = [float(value) for value in (column_reactions_kN or [])]
    max_reaction = max(reactions) if reactions else 0.0
    return max_reaction + float(self_weight_kN)


def compute_vertical_balance(
    column_reactions_kN: List[float],
    total_vertical_kN: float,
    tolerance: float = 1.0,
) -> Dict[str, Any]:
    """竖向荷载与柱反力平衡校验，返回误差与是否平衡。"""
    reactions_sum = sum(float(value) for value in (column_reactions_kN or []))
    error = abs(reactions_sum - float(total_vertical_kN))
    return {
        "balanced": error <= float(tolerance),
        "error_kN": error,
        "reactions_sum_kN": reactions_sum,
        "total_vertical_kN": float(total_vertical_kN),
        "tolerance_kN": float(tolerance),
    }


def summarize_pier_axial(
    column_reactions_kN: List[float],
    column_count: int,
    column_diameter_m: float,
    net_height_m: float,
    total_vertical_kN: float,
    concrete_unit_weight: float = 26.0,
    balance_tolerance: float = 1.0,
) -> Dict[str, Any]:
    """整合柱反力、墩柱自重、控制轴力与竖向平衡误差。"""
    single_self_weight = compute_column_self_weight(
        1, column_diameter_m, net_height_m, concrete_unit_weight
    )
    total_self_weight = compute_column_self_weight(
        column_count, column_diameter_m, net_height_m, concrete_unit_weight
    )
    controlling_axial = compute_controlling_axial_force(
        column_reactions_kN, single_self_weight
    )
    balance = compute_vertical_balance(
        column_reactions_kN, total_vertical_kN, balance_tolerance
    )
    return {
        "column_reactions_kN": [float(value) for value in column_reactions_kN],
        "single_column_self_weight_kN": single_self_weight,
        "total_column_self_weight_kN": total_self_weight,
        "controlling_axial_force_kN": controlling_axial,
        "balance": balance,
    }


def compute_total_vertical_load(load_design_output: Any) -> float:
    """恒载总竖向荷载（kN）= 盖梁自重 + 上部结构恒载反力合计。"""
    root = (load_design_output or {}).get("load_design_output") or {}
    if not isinstance(root, dict):
        return 0.0
    result = root.get("load_design_result") or {}
    total = float((result.get("cap_self_weight") or {}).get("total_weight_kN") or 0.0)
    dead = result.get("superstructure_dead_load") or {}
    for point in dead.get("support_points") or []:
        total += float(point.get("reaction_kN") or 0.0)
    return total


def extract_longitudinal_rebar_area(pier_column: Any) -> float:
    """从墩柱配筋结果提取纵向主筋总截面面积（mm²）。"""
    area = 0.0
    column = pier_column if isinstance(pier_column, dict) else {}
    for bar in column.get("longitudinal_bars", []) or []:
        if not isinstance(bar, dict):
            continue
        dia = float(bar.get("dia") or 0.0)
        count = int(bar.get("count") or 0)
        if dia > 0 and count > 0:
            area += math.pi * (dia / 2.0) ** 2 * count
    return area


def compute_axial_capacity_check(
    controlling_axial_force_kN: float,
    column_diameter_mm: float,
    net_height_m: float,
    longitudinal_bars: List[Dict[str, Any]],
    gamma0: float = 1.0,
    fcd_MPa: float = 18.4,
    fsd_prime_MPa: float = 330.0,
    effective_length_factor: Optional[float] = None,
    rotational_restraint_coefficient: Optional[float] = None,
    horizontal_restraint_coefficient: Optional[float] = None,
) -> Dict[str, Any]:
    """整合墩柱轴压验算，并显式记录计算长度模型与假定。"""
    diameter = float(column_diameter_mm)
    radius = diameter / 2.0
    area = math.pi * radius * radius
    as_prime = extract_longitudinal_rebar_area({"longitudinal_bars": longitudinal_bars})
    ratio = as_prime / area if area > 0 else 0.0

    if effective_length_factor is not None:
        length_factor = float(effective_length_factor)
        length_factor_source = "explicit_input"
    elif (
        rotational_restraint_coefficient is not None
        and horizontal_restraint_coefficient is not None
    ):
        k0 = float(rotational_restraint_coefficient)
        kf = float(horizontal_restraint_coefficient)
        if k0 < 0 or kf < 0:
            raise ValueError("计算长度约束系数不得为负数。")
        length_factor = 0.5 * math.exp(
            0.35 / (1.0 + 0.6 * k0)
            + 0.7 / (1.0 + 0.01 * kf)
            + 0.35 / ((1.0 + 0.75 * k0) * (1.0 + 1.15 * kf))
        )
        length_factor_source = "JTG_3362_2018_E_0_2_1"
    elif horizontal_restraint_coefficient is not None:
        kf = float(horizontal_restraint_coefficient)
        if kf < 0:
            raise ValueError("水平约束系数不得为负数。")
        length_factor = 2.0 - (1.3 * kf ** 1.5) / (9.5 + kf ** 1.5)
        length_factor_source = "JTG_3362_2018_E_0_3"
    else:
        # 普通运行输入尚未提供墩顶/墩底约束刚度。不能臆造边界条件，
        # 因而采用 k=1 的保守筛查值，并在结果中明确标记为假定。
        length_factor = 1.0
        length_factor_source = "conservative_default_missing_restraint_data"
    if length_factor <= 0:
        raise ValueError("effective_length_factor 必须大于 0。")

    if float(net_height_m) <= 0 or diameter <= 0:
        # 净高非正（布跨墩高小于柱位盖梁高度）或柱径缺失：该组没有有效柱身
        # （典型为桥台或埋入式墩），柱身稳定验算不适用。
        # 这里给出显式的 not_applicable 结论而不是抛异常或含糊的待复核：
        # 抛异常会作废整组配筋成果，含糊待复核则会一直卡住流程收尾。
        return {
            "status": "not_applicable",
            "reinforcement_ratio": ratio,
            "slenderness_ratio": None,
            "reinforcement_ratio_percent": (ratio * 100.0 if ratio else 0.0),
            "effective_length": {
                "member_length_mm": float(net_height_m) * 1000.0,
                "effective_length_factor": length_factor,
                "factor_source": length_factor_source,
                "restraint_data_complete": length_factor_source
                != "conservative_default_missing_restraint_data",
                "evidence_ids": [],
            },
            "message": (
                f"墩柱净高 {float(net_height_m):.3f} m、柱径 {diameter:.0f} mm："
                "该设计组没有有效柱身（净高非正通常表示布跨墩高小于柱位盖梁高度，"
                "即桥台或埋入式墩），按不进行柱身稳定验算处理。"
                "如该墩实际存在外露柱身，请修正墩高/地面线数据后重新设计。"
            ),
            "check_ok": None,
        }

    member_length_mm = float(net_height_m) * 1000.0
    l0_mm = length_factor * member_length_mm
    i_mm = diameter / 4.0
    slenderness = l0_mm / i_mm if i_mm > 0 else 0.0
    length_model = {
        "member_length_mm": member_length_mm,
        "effective_length_factor": length_factor,
        "effective_length_mm": l0_mm,
        "factor_source": length_factor_source,
        "restraint_data_complete": length_factor_source
        != "conservative_default_missing_restraint_data",
        "evidence_ids": ["F_3362_APPE_E_0_1_CALC_LENGTH"],
    }

    phi_result = execute_code_formula(
        STABILITY_COEFFICIENT_ID,
        {
            "slenderness_ratio": slenderness,
            "rebar_type": "ordinary",
            # 墩柱为圆形截面：稳定系数必须按表5.3.1 的回转半径口径 l0/i 列查表，
            # 不能与矩形口径 l0/b 列的档位混淆。
            "section_shape": "circular",
        },
    )
    if phi_result.status == "manual_review":
        return {
            "status": "manual_review",
            "reinforcement_ratio": ratio,
            "slenderness_ratio": slenderness,
            "effective_length": length_model,
            "message": (
                f"按计算长度系数 k={length_factor:g} 得长细比(口径 l0/i) {slenderness:.3f}；"
                f"{phi_result.message}"
                " 这表示稳定系数不能自动取值，不等同于承载力已判定不通过。"
                + (
                    " 当前缺少墩顶/墩底约束刚度，k=1仅作为保守筛查假定。"
                    if not length_model["restraint_data_complete"]
                    else ""
                )
            ),
            "partial_coverage": True,
        }

    phi = float(phi_result.outputs["stability_factor_phi"])
    axial = execute_code_formula(
        AXIAL_COMPRESSION_ID,
        {
            "Nd_kN": float(controlling_axial_force_kN),
            "phi": phi,
            "fcd_MPa": fcd_MPa,
            "A_mm2": area,
            "fsd_prime_MPa": fsd_prime_MPa,
            "As_prime_mm2": as_prime,
            "gamma0": gamma0,
        },
    )
    return {
        "status": "computed",
        "reinforcement_ratio": ratio,
        "slenderness_ratio": slenderness,
        "effective_length": length_model,
        "stability_factor_phi": phi,
        "slenderness_basis": phi_result.outputs.get("slenderness_basis", "l0/i"),
        "matched_slenderness_ratio": phi_result.outputs.get("matched_slenderness_ratio"),
        "capacity_kN": axial.outputs["capacity_kN"],
        "utilization": axial.outputs["utilization"],
        "check_ok": axial.outputs["check_ok"],
        "evidence_ids": list(axial.evidence_ids),
        "partial_coverage": True,
    }


def _extract_raw_reinforcement_rows(reinforcement_design_result: Any) -> List[Dict[str, Any]]:
    if not isinstance(reinforcement_design_result, dict):
        return []
    for key in ("分组原始结果",):
        root = reinforcement_design_result.get("任务3_下部结构配筋设计结果")
        if isinstance(root, dict):
            rows = root.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        nested = reinforcement_design_result.get("reinforcement_design_result")
        if isinstance(nested, dict):
            return _extract_raw_reinforcement_rows(nested)
    return []


def _column_definitely_absent(task: Mapping[str, Any], axial_check: Mapping[str, Any]) -> bool:
    """是否可确定该组没有有效柱身（净高非正或柱径非正 → 桥台/埋入式墩）。

    只在几何信息存在且明确非正时判定为"无柱身"；信息整体缺失时返回 False，
    保持原有 manual_review（数据不足需人工）的语义，避免把历史成果误判为不适用。
    """
    geom = (task.get("桥墩尺寸信息") or {}).get("墩柱几何信息")
    geom = geom if isinstance(geom, Mapping) else {}
    info = task.get("墩柱净高信息") if isinstance(task.get("墩柱净高信息"), Mapping) else {}
    net_height = info.get("controlling_net_height_m")
    diameter = geom.get("column_diameter")
    if net_height is None and diameter is None:
        return False
    try:
        if net_height is not None and float(net_height) <= 0:
            return True
        if diameter is not None and float(diameter) <= 0:
            return True
    except (TypeError, ValueError):
        return False
    return False


def aggregate_axial_check_results(reinforcement_design_result: Any) -> Dict[str, Any]:
    """从配筋结果聚合墩柱轴压验算状态，供确定性返修路由使用。

    "无柱身"判定在读取侧也会兜底：历史成果里由旧版本写入的 manual_review
    （净高非正/柱径缺失导致的无法取值）按"不适用"处理，不再要求人工复核，
    否则复验早已通过的任务也会被这一项永久卡住无法收尾。
    """
    axial_fail: List[str] = []
    slenderness_review: List[str] = []
    not_applicable: List[str] = []
    completed = 0
    for row in _extract_raw_reinforcement_rows(reinforcement_design_result):
        axial_check = row.get("axial_check")
        if not isinstance(axial_check, dict):
            continue
        task = row.get("reinforcement_task") or {}
        task_id = str(task.get("task_id") or row.get("task_id") or "")
        completed += 1
        status = axial_check.get("status")
        if status == "not_applicable" or _column_definitely_absent(task, axial_check):
            not_applicable.append(task_id)
        elif status == "manual_review":
            slenderness_review.append(task_id)
        elif axial_check.get("check_ok") is False:
            axial_fail.append(task_id)
    return {
        "completed_checks": completed,
        "axial_fail_task_ids": axial_fail,
        "slenderness_review_task_ids": slenderness_review,
        "not_applicable_task_ids": not_applicable,
        "has_axial_failure": bool(axial_fail),
        "has_slenderness_review": bool(slenderness_review),
    }


def build_axial_check_summary(reinforcement_design_result: Any) -> Dict[str, Any]:
    """从配筋批次结果构建墩柱轴压验算汇总表（供成果展示与人工复核）。

    不重复任何计算，只把批次内已算好的 axial_check 逐组整理为可读表格，
    并给出整体计数与控制组。verdict 语义：
      passed          轴压验算通过（computed 且 check_ok=True）
      failed          轴压承载力不足（computed 且 check_ok=False）
      manual_review   稳定系数/长细比无法自动取值（如超登记范围），需人工或补数据
      not_applicable  该组没有有效柱身（净高非正/柱径缺失，桥台或埋入式墩），不做柱身稳定验算
      missing         该组没有轴向验算记录
    """
    rows = _extract_raw_reinforcement_rows(reinforcement_design_result)
    per_group: List[Dict[str, Any]] = []
    control_by_utilization: Optional[Dict[str, Any]] = None
    max_slenderness_row: Optional[Dict[str, Any]] = None
    for row in rows:
        task = row.get("reinforcement_task") if isinstance(row.get("reinforcement_task"), dict) else row
        axial = row.get("axial_check") if isinstance(row.get("axial_check"), dict) else {}
        task_id = str(task.get("task_id") or row.get("task_id") or "")
        piers = task.get("包含桥墩号列表") or []
        geom = (task.get("桥墩尺寸信息") or {}).get("墩柱几何信息") if isinstance(task.get("桥墩尺寸信息"), dict) else {}
        if not isinstance(geom, dict):
            geom = {}
        diameter_m = geom.get("column_diameter")
        net_height = (task.get("墩柱净高信息") or {}).get("controlling_net_height_m") if isinstance(
            task.get("墩柱净高信息"), dict
        ) else None
        eff = axial.get("effective_length") if isinstance(axial.get("effective_length"), dict) else {}
        status = axial.get("status")
        check_ok = axial.get("check_ok")
        if not axial:
            verdict = "missing"
        elif status == "not_applicable" or _column_definitely_absent(task, axial):
            verdict = "not_applicable"
        elif status == "manual_review":
            verdict = "manual_review"
        elif check_ok is True:
            verdict = "passed"
        else:
            verdict = "failed"
        item: Dict[str, Any] = {
            "task_id": task_id,
            "member_piers": [str(p) for p in piers],
            "column_diameter_mm": float(diameter_m) * 1000.0 if diameter_m else None,
            "controlling_net_height_m": net_height,
            "effective_length_factor": eff.get("effective_length_factor"),
            "slenderness_ratio": axial.get("slenderness_ratio"),
            "slenderness_basis": axial.get("slenderness_basis", "l0/i"),
            "stability_factor_phi": axial.get("stability_factor_phi"),
            "utilization": axial.get("utilization"),
            "check_ok": check_ok,
            "status": status or "missing",
            "verdict": verdict,
            "message": axial.get("message") or "",
        }
        per_group.append(item)
        util = item["utilization"]
        if verdict == "passed" and isinstance(util, (int, float)) and (
            control_by_utilization is None
            or float(util) > float(control_by_utilization.get("utilization") or 0.0)
        ):
            control_by_utilization = item
        slenderness = item["slenderness_ratio"]
        if isinstance(slenderness, (int, float)) and (
            max_slenderness_row is None
            or float(slenderness) > float(max_slenderness_row.get("slenderness_ratio") or 0.0)
        ):
            max_slenderness_row = item

    count = {"passed": 0, "failed": 0, "manual_review": 0, "not_applicable": 0, "missing": 0}
    for item in per_group:
        count[item["verdict"]] = count.get(item["verdict"], 0) + 1
    return {
        "schema_version": "axial-check-summary-v1",
        "expected_task_count": len(per_group),
        "verdict_counts": count,
        "has_axial_failure": count["failed"] > 0,
        "has_manual_review": count["manual_review"] > 0,
        "control_group": (
            {key: control_by_utilization[key] for key in (
                "task_id", "member_piers", "column_diameter_mm", "controlling_net_height_m",
                "effective_length_factor", "slenderness_ratio", "stability_factor_phi",
                "utilization", "check_ok", "verdict",
            )} if control_by_utilization else None
        ),
        "max_slenderness_group": (
            {key: max_slenderness_row[key] for key in (
                "task_id", "member_piers", "column_diameter_mm", "controlling_net_height_m",
                "effective_length_factor", "slenderness_ratio", "stability_factor_phi",
                "status", "verdict",
            )} if max_slenderness_row else None
        ),
        "groups": per_group,
    }
