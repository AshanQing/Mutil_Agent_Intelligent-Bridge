from __future__ import annotations

"""盖梁配筋前置环节 3：内力控制信息提取工具。

将 OpenSeesPy 全梁内力包络输出转换为配筋 Prompt 可用的“内力控制信息”。
支持双柱、三柱及多柱盖梁的控制区概括。
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import yaml

Number = float
Point = Dict[str, Number]


def save_yaml(obj: Dict[str, Any], path: Union[str, Path]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False, default_flow_style=False, width=120)
    return str(path)


def round_float(x: Optional[Number], ndigits: int = 3) -> Optional[Number]:
    if x is None:
        return None
    return round(float(x), ndigits)


def round_range(values: List[Number], ndigits: int = 3) -> List[Number]:
    return [round_float(v, ndigits) for v in values]


def choose_combination(data: Dict[str, Any], combo_name: Optional[str]) -> str:
    combos = data.get("combined_envelopes_full_beam", {})
    if not combos:
        raise KeyError("输入文件缺少 combined_envelopes_full_beam。")
    if combo_name:
        if combo_name not in combos:
            raise KeyError(f"找不到组合 {combo_name}，可用组合: {list(combos.keys())}")
        return combo_name
    return "ULS_basic" if "ULS_basic" in combos else list(combos.keys())[0]


def aggregate_moment_points(moment_elements: List[Dict[str, Any]]) -> List[Point]:
    table = defaultdict(lambda: {"M_max": -1.0e30, "M_min": 1.0e30})
    for ele in moment_elements:
        for end in ("i", "j"):
            x = round(float(ele[f"x_{end}_m"]), 6)
            table[x]["M_max"] = max(table[x]["M_max"], float(ele[f"M_{end}_kN_m_max"]))
            table[x]["M_min"] = min(table[x]["M_min"], float(ele[f"M_{end}_kN_m_min"]))
    return [{"x": x, "M_max": v["M_max"], "M_min": v["M_min"]} for x, v in sorted(table.items())]


def aggregate_shear_points(shear_elements: List[Dict[str, Any]]) -> List[Point]:
    table = defaultdict(lambda: {"V_max": -1.0e30, "V_min": 1.0e30})
    for ele in shear_elements:
        for end in ("i", "j"):
            x = round(float(ele[f"x_{end}_m"]), 6)
            table[x]["V_max"] = max(table[x]["V_max"], float(ele[f"V_{end}_kN_max"]))
            table[x]["V_min"] = min(table[x]["V_min"], float(ele[f"V_{end}_kN_min"]))
    return [{"x": x, "V_max": v["V_max"], "V_min": v["V_min"], "V_abs": max(abs(v["V_max"]), abs(v["V_min"]))} for x, v in sorted(table.items())]


def interpolate_x(x1: Number, y1: Number, x2: Number, y2: Number, target: Number = 0.0) -> Optional[Number]:
    if abs(y2 - y1) < 1.0e-12:
        return None
    return x1 + (target - y1) * (x2 - x1) / (y2 - y1)


def find_zero_points(points: List[Point], key: str, eps: Number = 1.0e-8) -> List[Number]:
    zeros: List[Number] = []
    for p1, p2 in zip(points[:-1], points[1:]):
        x1, y1 = p1["x"], p1[key]
        x2, y2 = p2["x"], p2[key]
        if abs(y1) < eps:
            zeros.append(x1)
        if y1 * y2 < 0:
            x0 = interpolate_x(x1, y1, x2, y2, 0.0)
            if x0 is not None:
                zeros.append(x0)
    if points and abs(points[-1][key]) < eps:
        zeros.append(points[-1]["x"])
    cleaned: List[Number] = []
    for x in zeros:
        if not cleaned or abs(x - cleaned[-1]) > 1.0e-4:
            cleaned.append(x)
    return cleaned


def peak_in_range(points: List[Point], key: str, x_range: List[Number], mode: str) -> Optional[Point]:
    x0, x1 = x_range
    candidates = [p for p in points if x0 - 1.0e-9 <= p["x"] <= x1 + 1.0e-9]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p[key]) if mode == "max" else min(candidates, key=lambda p: p[key])


def classify_intensity(value: Number, reference_value: Number) -> str:
    ref = abs(reference_value)
    if ref < 1.0e-12:
        return "low"
    ratio = abs(value) / ref
    if ratio >= 0.70:
        return "high"
    if ratio >= 0.40:
        return "medium"
    return "low"


def detect_symmetry(values: List[Number], tolerance: Number = 0.08) -> bool:
    if len(values) < 2:
        return True
    abs_values = [abs(v) for v in values]
    ref = max(abs_values + [1.0e-12])
    return (max(abs_values) - min(abs_values)) / ref <= tolerance


def make_column_centers(column_count: int, column_spacing: float) -> List[float]:
    start_x = - (column_count - 1) * column_spacing / 2.0
    return [start_x + i * column_spacing for i in range(column_count)]


def make_column_control_ranges(centers: List[float], half_len: float) -> List[List[float]]:
    ranges = []
    for i, c in enumerate(centers):
        left = -half_len if i == 0 else 0.5 * (centers[i - 1] + c)
        right = half_len if i == len(centers) - 1 else 0.5 * (c + centers[i + 1])
        ranges.append([left, right])
    return ranges


def find_positive_moment_regions(moment_points: List[Point], half_len: float) -> List[Dict[str, Any]]:
    zeros = [x for x in find_zero_points(moment_points, "M_max") if abs(x) < half_len - 1.0e-6]
    boundaries = [-half_len] + zeros + [half_len]
    regions: List[Dict[str, Any]] = []
    global_peak = max(moment_points, key=lambda p: p["M_max"])
    for i in range(len(boundaries) - 1):
        xr = [boundaries[i], boundaries[i + 1]]
        peak = peak_in_range(moment_points, "M_max", xr, "max")
        if peak and peak["M_max"] > 1.0e-6:
            regions.append({
                "name": f"正弯矩控制区_{len(regions) + 1}",
                "side": "middle" if abs(0.5 * (xr[0] + xr[1])) < 0.1 else ("left" if 0.5 * (xr[0] + xr[1]) < 0 else "right"),
                "x_peak_m": round_float(peak["x"]),
                "M_peak_kN_m": round_float(peak["M_max"], 1),
                "x_range_m": round_range(xr),
                "intensity_level": classify_intensity(peak["M_max"], global_peak["M_max"]),
            })
    if not regions:
        regions.append({"name": "跨中正弯矩区", "side": "middle", "x_peak_m": round_float(global_peak["x"]), "M_peak_kN_m": round_float(global_peak["M_max"], 1), "x_range_m": round_range([-half_len, half_len]), "intensity_level": "high"})
    return regions


def build_force_control_info(
    data: Dict[str, Any],
    combo_name: Optional[str] = None,
    shear_ratio: Number = 0.75,
    symmetry_tolerance: Number = 0.08,
) -> Dict[str, Any]:
    model = data["model_summary"]
    combo = choose_combination(data, combo_name)
    envelopes = data["combined_envelopes_full_beam"][combo]
    moment_points = aggregate_moment_points(envelopes["moment_envelope"]["element_forces_envelope"])
    shear_points = aggregate_shear_points(envelopes["shear_envelope"]["element_forces_envelope"])

    cap_length = float(model["cap_length"])
    column_spacing = float(model["column_spacing"])
    column_diameter = float(model["column_diameter"])
    column_count = int(model.get("column_count", 2))
    half_len = cap_length / 2.0
    centers = make_column_centers(column_count, column_spacing)
    col_edge_ranges = [[c - column_diameter / 2.0, c + column_diameter / 2.0] for c in centers]
    col_control_ranges = make_column_control_ranges(centers, half_len)

    positive_peak = max(moment_points, key=lambda p: p["M_max"])
    negative_global_peak = min(moment_points, key=lambda p: p["M_min"])
    shear_global_peak = max(shear_points, key=lambda p: p["V_abs"])

    positive_regions = find_positive_moment_regions(moment_points, half_len)

    negative_regions: List[Dict[str, Any]] = []
    negative_peaks: List[float] = []
    for idx, (c, xr) in enumerate(zip(centers, col_control_ranges), start=1):
        peak = peak_in_range(moment_points, "M_min", xr, "min") or negative_global_peak
        negative_peaks.append(float(peak["M_min"]))
        negative_regions.append({
            "name": f"P{idx}柱顶负弯矩区",
            "column_index": idx,
            "side": "middle" if abs(c) < 0.1 else ("left" if c < 0 else "right"),
            "x_peak_m": round_float(peak["x"]),
            "M_peak_kN_m": round_float(peak["M_min"], 1),
            "related_column_centerline_x_m": round_float(c),
            "related_column_edge_range_m": round_range(col_edge_ranges[idx - 1]),
            "x_range_m": round_range(xr),
            "intensity_level": classify_intensity(peak["M_min"], negative_global_peak["M_min"]),
        })

    shear_threshold = shear_ratio * float(shear_global_peak["V_abs"])
    high_shear_regions: List[Dict[str, Any]] = []
    shear_peaks: List[float] = []
    for idx, (c, xr) in enumerate(zip(centers, col_control_ranges), start=1):
        local_points = [p for p in shear_points if xr[0] - 1.0e-9 <= p["x"] <= xr[1] + 1.0e-9]
        if not local_points:
            continue
        peak = max(local_points, key=lambda p: p["V_abs"])
        shear_peaks.append(float(peak["V_abs"]))
        # 用局部阈值附近点形成简化区间；若无点满足，则退化为峰值点。
        candidates = [p["x"] for p in local_points if p["V_abs"] >= shear_threshold]
        x_range = [min(candidates), max(candidates)] if candidates else [peak["x"], peak["x"]]
        high_shear_regions.append({
            "name": f"P{idx}柱顶高剪区",
            "column_index": idx,
            "side": "middle" if abs(c) < 0.1 else ("left" if c < 0 else "right"),
            "x_peak_m": round_float(peak["x"]),
            "V_abs_peak_kN": round_float(peak["V_abs"], 1),
            "related_column_centerline_x_m": round_float(c),
            "related_column_edge_range_m": round_range(col_edge_ranges[idx - 1]),
            "x_range_m": round_range(x_range),
            "intensity_level": classify_intensity(peak["V_abs"], shear_global_peak["V_abs"]),
        })

    combination_rule = model.get("load_combinations", {}).get(combo, None)
    moment_sign = data.get("note_moment_sign", "跨中正弯矩为正，支点负弯矩为负")
    symmetry = bool(detect_symmetry(negative_peaks, symmetry_tolerance) and detect_symmetry(shear_peaks, symmetry_tolerance))

    return {"内力控制信息": {
        "数据来源": {
            "load_combination_name": combo,
            "load_combination_type": "承载能力极限状态基本组合" if "ULS" in combo.upper() else "未指定",
            "load_combination_rule": combination_rule,
            "moment_sign_convention": moment_sign,
        },
        "几何参考": {
            "cap_length_m": round_float(cap_length),
            "column_count": column_count,
            "column_spacing_m": round_float(column_spacing),
            "column_diameter_m": round_float(column_diameter),
            "pier_centerline_x_m": [round_float(c) for c in centers],
            "column_edge_ranges_m": [round_range(r) for r in col_edge_ranges],
            "overall_symmetry": symmetry,
        },
        "弯矩包络": {
            "positive_moment": {
                "global_peak": {"x_m": round_float(positive_peak["x"]), "M_kN_m": round_float(positive_peak["M_max"], 1)},
                "zero_moment_points_m": [round_float(x) for x in find_zero_points(moment_points, "M_max") if abs(x) < half_len - 1.0e-6],
                "dominant_regions": positive_regions,
            },
            "negative_moment": {
                "global_peak": {"x_m": round_float(negative_global_peak["x"]), "M_kN_m": round_float(negative_global_peak["M_min"], 1)},
                "zero_moment_points_m": [round_float(x) for x in find_zero_points(moment_points, "M_min") if abs(x) < half_len - 1.0e-6],
                "dominant_regions": negative_regions,
            },
        },
        "剪力包络": {
            "absolute_shear": {
                "global_peak": {"x_m": round_float(shear_global_peak["x"]), "V_abs_kN": round_float(shear_global_peak["V_abs"], 1)},
                "high_shear_threshold_ratio": shear_ratio,
                "high_shear_regions": high_shear_regions,
            }
        },
        "设计语义摘要": {
            "force_pattern": f"{column_count}柱盖梁_柱顶负弯矩_跨中正弯矩_柱顶高剪共同控制",
            "reinforcement_implication": "内力信息用于识别顶部负弯矩加强区、底部正弯矩加强区、弯起筋跨区连接路径和箍筋加密区；不在本步骤执行精确承载力验算。",
        },
    }}


def cap_force_control_info_tool(
    internal_force_output: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    task_id: str = "cap_beam",
    combo_name: Optional[str] = None,
    shear_ratio: float = 0.75,
    symmetry_tolerance: float = 0.08,
    write_files: bool = True,
) -> Dict[str, Any]:
    try:
        info = build_force_control_info(internal_force_output, combo_name=combo_name, shear_ratio=shear_ratio, symmetry_tolerance=symmetry_tolerance)
        files: Dict[str, str] = {}
        if write_files:
            root = Path(output_dir or ".")
            root.mkdir(parents=True, exist_ok=True)
            files["force_control_info_path"] = save_yaml(info, root / f"force_control_info_{task_id}.yaml")
        return {"success": True, "force_control_info": info, "output_files": files}
    except Exception as e:
        return {"success": False, "error": str(e)}
