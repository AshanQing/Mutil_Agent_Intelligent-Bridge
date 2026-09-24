from __future__ import annotations

"""盖梁配筋前置环节 1：荷载设计工具。

输入：reinforcement_task 或其“桥墩尺寸信息”。
输出：load_design_output 与 analysis_load_input，并可写出 JSON 文件。
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import math

CAP_CONCRETE_UNIT_WEIGHT = 26.0  # kN/m3
LANE_LOAD_QK = 10.5              # kN/m
STANDARD_LANE_WIDTH = 3.75

SYSTEM_TYPE_MAP = {
    "简支": "简支桥面连续",
    "简支桥面连续": "简支桥面连续",
    "连续": "先简支后连续",
    "先简支后连续": "先简支后连续",
}

DEAD_LOAD_RULES = {
    ("简支桥面连续", 20): {"rule_name": "20m_simple_span_deck_continuous_T", "reaction_rule": {"edge_beam": 424.7, "interior_beam": 437.8}},
    ("简支桥面连续", 25): {"rule_name": "25m_simple_span_deck_continuous_T", "reaction_rule": {"edge_beam": 578.9, "interior_beam": 588.0}},
    ("简支桥面连续", 30): {"rule_name": "30m_simple_span_deck_continuous_T", "reaction_rule": {"edge_beam": 733.1, "interior_beam": 737.4}},
    ("简支桥面连续", 40): {"rule_name": "40m_simple_span_deck_continuous_T", "reaction_rule": {"edge_beam": 1105.9, "interior_beam": 1096.1}},
    ("先简支后连续", 30): {"rule_name": "30m_ssc_T", "reaction_rule": {"edge_beam": {"edge_support": 633.6, "interior_support": 1441.3}, "interior_beam": {"edge_support": 607.5, "interior_support": 1406.6}}},
    ("先简支后连续", 40): {"rule_name": "40m_ssc_T", "reaction_rule": {"edge_beam": {"edge_support": 1035.4, "interior_support": 2360.3}, "interior_beam": {"edge_support": 1027.6, "interior_support": 2334.9}}},
}

@dataclass
class Support:
    support_id: str
    x_m: float

@dataclass
class Lane:
    lane_id: str
    center_x_m: float

@dataclass
class Pattern:
    pattern_id: str
    pattern_name: str
    lane_factor: float
    lanes: List[Lane]


def round3(x: float) -> float:
    return round(float(x), 3)


def deep_round(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: deep_round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_round(v) for v in obj]
    if isinstance(obj, float):
        return round3(obj)
    return obj


def save_json(data: dict, filepath: Union[str, Path]) -> str:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deep_round(data), f, ensure_ascii=False, indent=2)
    return str(path)


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in [None, "", "null", "None"]:
        return default
    try:
        return float(value)
    except Exception:
        return default


def normalize_system_type(system_type: str) -> str:
    if system_type not in SYSTEM_TYPE_MAP:
        raise ValueError(f"未知 system_type: {system_type}")
    return SYSTEM_TYPE_MAP[system_type]


def get_support_category(pier_role: str) -> str:
    if pier_role in ["中间墩", "中墩", "联内墩"]:
        return "interior_support"
    return "edge_support"


def flatten_reinforcement_task_input(reinforcement_task: Dict[str, Any]) -> Dict[str, Any]:
    task_input = reinforcement_task.get("桥墩尺寸信息") or reinforcement_task
    basic = task_input.get("基本信息", {}) or {}
    upper = task_input.get("上部布置与支承条件", {}) or {}
    front = upper.get("front_span", {}) or {}
    rear = upper.get("rear_span", {}) or {}
    cap = task_input.get("盖梁几何信息", {}) or {}
    col = task_input.get("墩柱几何信息", {}) or {}
    found = task_input.get("基础信息", {}) or {}

    return {
        "drawing_id": basic.get("drawing_id") or reinforcement_task.get("task_id"),
        "remarks": basic.get("remarks", ""),
        "system_type": basic.get("system_type"),
        "pier_role": basic.get("pier_role"),
        "pier_type": basic.get("pier_type"),
        "foundation_type": basic.get("foundation_type"),
        "span_length": _num(upper.get("span_length")),
        "deck_width": _num(upper.get("deck_width")),
        "next_deck_width": _num(upper.get("next_deck_width"), _num(upper.get("deck_width"))),
        "deck_slope": _num(upper.get("deck_slope")),
        "is_double_support": bool(upper.get("is_double_support")),
        "support_symmetry": upper.get("support_symmetry"),
        "front_span_t_beam_count": front.get("t_beam_count"),
        "front_span_t_beam_spacing": _num(front.get("t_beam_spacing")),
        "support_center_distance_front": _num(front.get("support_center_distance")),
        "front_support_edge_distance": front.get("support_edge_distance") or {},
        "rear_span_t_beam_count": rear.get("t_beam_count"),
        "rear_span_t_beam_spacing": _num(rear.get("t_beam_spacing")),
        "support_center_distance_rear": _num(rear.get("support_center_distance")),
        "rear_support_edge_distance": rear.get("support_edge_distance") or {},
        "cap_length": _num(cap.get("cap_length")),
        "cap_width": _num(cap.get("cap_width")),
        "cap_height_mid": _num(cap.get("cap_height_mid")),
        "cap_height_end": _num(cap.get("cap_height_end")),
        "cantilever_length": _num(cap.get("cantilever_length")),
        "cap_slope": _num(cap.get("cap_slope")),
        "column_count": col.get("column_count"),
        "column_diameter": _num(col.get("column_diameter")),
        "column_spacing": _num(col.get("column_spacing")),
        "pile_diameter": _num(found.get("pile_diameter")),
    }


def validate_input(data: Dict[str, Any]) -> None:
    required = [
        "system_type", "pier_role", "pier_type", "foundation_type", "span_length", "deck_width",
        "next_deck_width", "is_double_support", "front_span_t_beam_count", "front_span_t_beam_spacing",
        "front_support_edge_distance", "cap_length", "cap_width", "cap_height_mid", "cap_height_end",
        "cantilever_length", "column_count", "column_diameter", "column_spacing", "pile_diameter",
    ]
    missing = [f for f in required if data.get(f) in [None, "", {}, []]]
    if missing:
        raise ValueError(f"缺少必需字段: {missing}")


def build_geometry_context(inp: Dict[str, Any]) -> Dict[str, Any]:
    system_type_std = normalize_system_type(inp["system_type"])
    front_edge = inp.get("front_support_edge_distance") or {}
    return {
        "basic_info": {
            "drawing_id": inp.get("drawing_id"), "remarks": inp.get("remarks", ""),
            "system_type": system_type_std, "pier_role": inp["pier_role"],
            "pier_type": inp["pier_type"], "foundation_type": inp["foundation_type"],
        },
        "superstructure_context": {
            "span_length": float(inp["span_length"]), "deck_width": float(inp["deck_width"]),
            "next_deck_width": float(inp["next_deck_width"]),
            "deck_slope": float(inp["deck_slope"]) if inp.get("deck_slope") is not None else None,
            "is_double_support": bool(inp["is_double_support"]), "support_symmetry": inp.get("support_symmetry"),
            "front_span": {
                "t_beam_count": int(inp["front_span_t_beam_count"]),
                "t_beam_spacing": float(inp["front_span_t_beam_spacing"]),
                "support_center_distance": float(inp["support_center_distance_front"]) if inp.get("support_center_distance_front") is not None else None,
                "support_edge_distance": {"left": _num(front_edge.get("left")), "right": _num(front_edge.get("right"))},
            },
        },
        "cap_beam_context": {
            "cap_length": float(inp["cap_length"]), "cap_width": float(inp["cap_width"]),
            "cap_height_mid": float(inp["cap_height_mid"]), "cap_height_end": float(inp["cap_height_end"]),
            "cantilever_length": float(inp["cantilever_length"]),
            "cap_slope": float(inp["cap_slope"]) if inp.get("cap_slope") is not None else None,
        },
        "column_context": {
            "column_count": int(inp["column_count"]), "column_diameter": float(inp["column_diameter"]),
            "column_spacing": float(inp["column_spacing"]),
        },
        "foundation_context": {"pile_diameter": float(inp["pile_diameter"])},
    }


def calc_cap_self_weight(geometry_context: Dict[str, Any]) -> Dict[str, Any]:
    cap = geometry_context["cap_beam_context"]
    cap_length, cap_width = cap["cap_length"], cap["cap_width"]
    h_mid, h_end, cantilever = cap["cap_height_mid"], cap["cap_height_end"], cap["cantilever_length"]
    if abs(h_mid - h_end) < 1e-12:
        q = CAP_CONCRETE_UNIT_WEIGHT * cap_width * h_mid
        return {"geometry_type": "constant_section", "segments": [{"segment_id": "seg_constant", "segment_name": "等截面段", "x_start_expr": "-cap_length/2", "x_end_expr": "cap_length/2", "line_load_kN_m": round3(q)}], "total_weight_kN": round3(CAP_CONCRETE_UNIT_WEIGHT * cap_length * cap_width * h_mid)}
    q_eq = CAP_CONCRETE_UNIT_WEIGHT * cap_width * (h_end + h_mid) / 2.0
    q_mid = CAP_CONCRETE_UNIT_WEIGHT * cap_width * h_mid
    total_weight = CAP_CONCRETE_UNIT_WEIGHT * cap_width * ((cap_length - 2 * cantilever) * h_mid + cantilever * (h_end + h_mid))
    return {"geometry_type": "symmetric_variable_section", "segments": [
        {"segment_id": "seg_left_cantilever", "segment_name": "左悬臂变截面段", "x_start_expr": "-cap_length/2", "x_end_expr": "-cap_length/2 + cantilever_length", "line_load_kN_m": round3(q_eq)},
        {"segment_id": "seg_mid_constant", "segment_name": "中部等截面段", "x_start_expr": "-cap_length/2 + cantilever_length", "x_end_expr": "cap_length/2 - cantilever_length", "line_load_kN_m": round3(q_mid)},
        {"segment_id": "seg_right_cantilever", "segment_name": "右悬臂变截面段", "x_start_expr": "cap_length/2 - cantilever_length", "x_end_expr": "cap_length/2", "line_load_kN_m": round3(q_eq)},
    ], "total_weight_kN": round3(total_weight)}


def get_beam_position_type(i: int, n: int) -> str:
    return "edge_beam" if n <= 2 or i in [1, n] else "interior_beam"


def _lerp_rule_value(lo: Any, hi: Any, t: float) -> Any:
    """递归线性插值：标量直接插值，dict 逐键插值（键集取并集，缺失键沿用单侧值）。"""
    if isinstance(lo, dict) and isinstance(hi, dict):
        merged: Dict[str, Any] = {}
        for k in set(lo) | set(hi):
            if k in lo and k in hi:
                merged[k] = _lerp_rule_value(lo[k], hi[k], t)
            else:
                merged[k] = lo.get(k, hi.get(k))
        return merged
    return float(lo) + (float(hi) - float(lo)) * t


def _resolve_dead_rule(system_type: str, span_length: float) -> Tuple[Dict[str, Any], str, tuple]:
    """按跨径对恒载反力规则做线性插值 / 截断 / 外推，返回 (reaction_rule, 匹配标签, 匹配键)。

    - span 落在两档之间：按比例线性插值；
    - span 低于最小档：取最小档（截断）；
    - span 高于最大档：按最后两档斜率外推（支持大跨 T 梁近似，作为初步流程简化）；
    - 只有一档时：直接使用该档。
    """
    candidates = sorted((k for k in DEAD_LOAD_RULES if k[0] == system_type), key=lambda k: k[1])
    if not candidates:
        raise ValueError(f"未找到结构体系 {system_type} 的恒载规则")
    spans = [k[1] for k in candidates]

    def exact(idx: int) -> Tuple[Dict[str, Any], str, tuple]:
        key = candidates[idx]
        return DEAD_LOAD_RULES[key]["reaction_rule"], DEAD_LOAD_RULES[key]["rule_name"], key

    if len(candidates) == 1 or span_length <= spans[0]:
        return exact(0)
    if span_length >= spans[-1]:
        lo_key, hi_key = candidates[-2], candidates[-1]
        t = (span_length - lo_key[1]) / float(hi_key[1] - lo_key[1])
        rule = _lerp_rule_value(DEAD_LOAD_RULES[lo_key]["reaction_rule"], DEAD_LOAD_RULES[hi_key]["reaction_rule"], t)
        label = f"extrap:{DEAD_LOAD_RULES[lo_key]['rule_name']}~{DEAD_LOAD_RULES[hi_key]['rule_name']}@span={span_length:g}"
        return rule, label, (system_type, span_length)
    for i in range(len(spans) - 1):
        lo_key, hi_key = candidates[i], candidates[i + 1]
        if spans[i] <= span_length <= spans[i + 1]:
            t = (span_length - lo_key[1]) / float(hi_key[1] - lo_key[1])
            rule = _lerp_rule_value(DEAD_LOAD_RULES[lo_key]["reaction_rule"], DEAD_LOAD_RULES[hi_key]["reaction_rule"], t)
            label = f"interp:{DEAD_LOAD_RULES[lo_key]['rule_name']}~{DEAD_LOAD_RULES[hi_key]['rule_name']}@span={span_length:g}"
            return rule, label, (system_type, span_length)
    raise RuntimeError("unreachable")


def calc_superstructure_dead_load(geometry_context: Dict[str, Any]) -> Dict[str, Any]:
    basic, sup = geometry_context["basic_info"], geometry_context["superstructure_context"]
    system_type, span_length, pier_role = basic["system_type"], sup["span_length"], basic["pier_role"]
    reaction_rule, matched_label, matched_key = _resolve_dead_rule(system_type, span_length)
    front = sup["front_span"]
    n, spacing = front["t_beam_count"], front["t_beam_spacing"]
    edge_left = front["support_edge_distance"].get("left")
    if edge_left is None:
        edge_left = (geometry_context["cap_beam_context"]["cap_length"] - (n - 1) * spacing) / 2.0
    support_category = get_support_category(pier_role) if system_type == "先简支后连续" else ""
    points = []
    for i in range(1, n + 1):
        beam_pos = get_beam_position_type(i, n)
        if system_type == "简支桥面连续":
            reaction = reaction_rule[beam_pos] * (2.0 if sup["is_double_support"] else 1.0)
        else:
            reaction = reaction_rule[beam_pos][support_category]
        points.append({"support_id": f"S{i}", "span_side": "front", "x_m": round3(edge_left + (i - 1) * spacing), "beam_position_type": beam_pos, "support_row": "double_row_unsplit" if sup["is_double_support"] else "single_row", "reaction_kN": round3(reaction)})
    return {"matched_rule": matched_label, "matched_rule_key": list(matched_key), "support_category": support_category, "support_points": points}


def calc_lane_count(deck_width: float) -> int:
    # 与原脚本保持近似：按标准车道宽度估算，但上限暂控制为6，避免25m整体式宽度直接中断。
    # 若后续明确单幅/整幅宽度，应在上游传入真实单幅桥面宽度或 lane_count_override。
    return max(1, min(6, int(math.floor(float(deck_width) / STANDARD_LANE_WIDTH))))


def calc_lane_factor(lane_count: int) -> float:
    return {1: 1.2, 2: 1.0, 3: 0.78, 4: 0.67, 5: 0.60, 6: 0.55}.get(lane_count, 0.55)


def calc_Pk(span_length: float) -> float:
    if span_length <= 5:
        return 270.0
    if span_length < 50:
        return 2.0 * (span_length + 130.0)
    return 360.0


def calc_vehicle_ref_load(span_length: float) -> float:
    return span_length * LANE_LOAD_QK + 1.2 * calc_Pk(span_length)


def find_bracket(x: float, supports: List[Support]) -> Tuple[int, int]:
    for i, s in enumerate(supports):
        if abs(x - s.x_m) < 1e-12:
            return i, i
    if x < supports[0].x_m:
        return 0, 1
    if x > supports[-1].x_m:
        return len(supports) - 2, len(supports) - 1
    for i in range(len(supports) - 1):
        if supports[i].x_m <= x <= supports[i + 1].x_m:
            return i, i + 1
    raise RuntimeError("未找到对应区间")


def distribute_lane(P_ref: float, lane_x: float, supports: List[Support]) -> Dict[str, float]:
    result = {s.support_id: 0.0 for s in supports}
    i, j = find_bracket(lane_x, supports)
    if i == j:
        result[supports[i].support_id] = P_ref
    else:
        xi, xj = supports[i].x_m, supports[j].x_m
        result[supports[i].support_id] = (xj - lane_x) / (xj - xi) * P_ref
        result[supports[j].support_id] = (lane_x - xi) / (xj - xi) * P_ref
    return result


def solve_pattern(pattern: Pattern, supports: List[Support], P_ref: float) -> Dict[str, Any]:
    combined = {s.support_id: 0.0 for s in supports}
    for lane in pattern.lanes:
        one = distribute_lane(P_ref, lane.center_x_m, supports)
        for sid, val in one.items():
            combined[sid] += val
    for sid in combined:
        combined[sid] *= pattern.lane_factor
    return {"pattern_id": pattern.pattern_id, "pattern_name": pattern.pattern_name, "lane_factor": round3(pattern.lane_factor), "lane_count": len(pattern.lanes), "support_reactions_kN": {k: round3(v) for k, v in combined.items()}}


def generate_symmetrical_lane_centers(lane_count: int) -> Dict[str, float]:
    total = lane_count * STANDARD_LANE_WIDTH
    start_x = -total / 2.0
    return {f"L{i}": round3(start_x + (i - 0.5) * STANDARD_LANE_WIDTH) for i in range(1, lane_count + 1)}


def generate_live_supports(cap_length: float, beam_count: int, beam_spacing: float, edge_left: float) -> List[Support]:
    cap_center = cap_length / 2.0
    return [Support(f"S{i}", round3(edge_left + (i - 1) * beam_spacing - cap_center)) for i in range(1, beam_count + 1)]


def generate_live_patterns_from_lane_count(lane_centers: Dict[str, float]) -> List[Pattern]:
    lane_ids = list(lane_centers.keys())
    lane_objs = [Lane(lid, lane_centers[lid]) for lid in lane_ids]
    patterns, pid = [], 1
    for loaded_count in range(1, len(lane_objs) + 1):
        factor = calc_lane_factor(loaded_count)
        for start_idx in range(len(lane_objs) - loaded_count + 1):
            group = lane_objs[start_idx:start_idx + loaded_count]
            name = f"{loaded_count}车道-全载" if loaded_count == len(lane_objs) else f"{loaded_count}车道-偏载({group[0].lane_id}~{group[-1].lane_id})"
            patterns.append(Pattern(f"VL{pid}", name, factor, group))
            pid += 1
    return patterns


def calc_vehicle_live_load(geometry_context: Dict[str, Any]) -> Dict[str, Any]:
    sup, cap = geometry_context["superstructure_context"], geometry_context["cap_beam_context"]
    span_length, deck_width = sup["span_length"], sup["deck_width"]
    front = sup["front_span"]
    beam_count, spacing = front["t_beam_count"], front["t_beam_spacing"]
    edge_left = front["support_edge_distance"].get("left")
    if edge_left is None:
        edge_left = (cap["cap_length"] - (beam_count - 1) * spacing) / 2.0
    lane_count = calc_lane_count(deck_width)
    p_ref = calc_vehicle_ref_load(span_length)
    supports = generate_live_supports(cap["cap_length"], beam_count, spacing, edge_left)
    lane_centers = generate_symmetrical_lane_centers(lane_count)
    patterns = generate_live_patterns_from_lane_count(lane_centers)
    return {"input": {"P_ref_kN": round3(p_ref), "supports": [asdict(s) for s in supports], "lane_count": lane_count, "lane_centers_m": lane_centers}, "pattern_results": [solve_pattern(p, supports, p_ref) for p in patterns]}


def generate_load_design_output(inp: Dict[str, Any]) -> Dict[str, Any]:
    validate_input(inp)
    geometry_context = build_geometry_context(inp)
    return {"load_design_output": {"geometry_context": geometry_context, "load_design_result": {"cap_self_weight": calc_cap_self_weight(geometry_context), "superstructure_dead_load": calc_superstructure_dead_load(geometry_context), "vehicle_live_load": calc_vehicle_live_load(geometry_context)}}}


def convert_load_design_output_to_analysis_input(load_design_output: Dict[str, Any]) -> Dict[str, Any]:
    root = load_design_output["load_design_output"]
    geo, res = root["geometry_context"], root["load_design_result"]
    return {"analysis_load_input": {"meta": geo["basic_info"], "model_geometry": {**geo["cap_beam_context"], **geo["column_context"], "span_length": geo["superstructure_context"]["span_length"], "deck_width": geo["superstructure_context"]["deck_width"], "next_deck_width": geo["superstructure_context"]["next_deck_width"], "deck_slope": geo["superstructure_context"]["deck_slope"], "is_double_support": geo["superstructure_context"]["is_double_support"], "support_symmetry": geo["superstructure_context"]["support_symmetry"]}, "dead_load": {"cap_self_weight": res["cap_self_weight"], "superstructure_dead_load": res["superstructure_dead_load"]}, "live_load": {"reference": res["vehicle_live_load"]["input"], "patterns": res["vehicle_live_load"]["pattern_results"]}, "load_combinations": {"ULS_basic": {"dead": 1.2, "live": 1.4}}}}


def cap_load_design_tool(
    reinforcement_task: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    task_id: Optional[str] = None,
    write_files: bool = True,
) -> Dict[str, Any]:
    try:
        flat_input = flatten_reinforcement_task_input(reinforcement_task)
        load_design_output = generate_load_design_output(flat_input)
        analysis_load_input = convert_load_design_output_to_analysis_input(load_design_output)
        files: Dict[str, str] = {}
        if write_files:
            root = Path(output_dir or ".")
            root.mkdir(parents=True, exist_ok=True)
            name = task_id or flat_input.get("drawing_id") or "cap_beam"
            files["load_design_output_path"] = save_json(load_design_output, root / f"load_design_output_{name}.json")
            files["analysis_load_input_path"] = save_json(analysis_load_input, root / f"analysis_load_input_{name}.json")
        return {"success": True, "flat_input": flat_input, "load_design_output": load_design_output, "analysis_load_input": analysis_load_input, "output_files": files}
    except Exception as e:
        return {"success": False, "error": str(e)}
