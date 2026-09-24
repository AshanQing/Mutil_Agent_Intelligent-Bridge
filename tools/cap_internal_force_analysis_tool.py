# -*- coding: utf-8 -*-
"""
基于 analysis_load_input.json 的盖梁 OpenSeesPy 建模、全梁内力提取、
内力图绘制与组合包络输出脚本（精简标注版）

修复与优化内容：
1. 确保弯矩图：跨中正弯矩画在下方且 Y 轴刻度为正，支点负弯矩画在上方且 Y 轴刻度为负，完美符合工程习惯。
2. 将原有的固定2柱支撑改为“动态支撑”算法，完美支持三柱及任意多柱体系。
3. 尺寸与排版：修改了绘图尺寸与Y轴留白比例，使得最终输出的内力图视觉上更“扁”；图例移至上方。
4. [最新] 精简输出：移除了所有单工况的绘图，仅输出 ULS_basic 的弯矩和剪力包络图。
5. [最新] 极值标注：在包络图上自动寻找并标注所有波峰、波谷及支点突变极值的具体数值，带有防重叠机制。
6. [修复] 修复了旧版本 Python (低于3.9) 中不支持字典 `|` 运算导致的报错问题。
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt
ops = None

def _ensure_opensees():
    global ops
    if ops is None:
        import openseespy.opensees as _ops
        ops = _ops
    return ops


# =========================
# 用户可调整参数
# =========================
LOAD_COMBINATIONS = {
    "ULS_basic": {"dead": 1.2, "live": 1.4}
}

MAX_ELE_LEN = 0.5
MODEL_E_CONCRETE = 3.25e7   # kN/m^2，对应约 3.25e4 MPa
SUPPORT_STIFFNESS = 1.0e9   # zeroLength 支承刚度

# 【绘图调整参数】
FIG_SIZE = (16, 2.5)        # 画板尺寸（宽，高），数字越小越扁
Y_MARGIN_RATIO = 0.4        # Y轴上下留白比例，留出足够的空间用于标注文字


# =========================
# 基础读写工具
# =========================
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def x_left_to_center(x_left: float, cap_length: float) -> float:
    return x_left - cap_length / 2.0


def safe_eval_expr(expr: str, cap_length: float, cantilever_length: float) -> float:
    return eval(expr, {}, {"cap_length": cap_length, "cantilever_length": cantilever_length})


# =========================
# 输入解析
# =========================
def parse_input(data: dict) -> dict:
    root = data["analysis_load_input"]
    geo = root["model_geometry"]
    dead = root["dead_load"]
    live = root["live_load"]

    return {
        "cap_length": float(geo["cap_length"]),
        "cap_width": float(geo["cap_width"]),
        "cap_height_mid": float(geo["cap_height_mid"]),
        "cap_height_end": float(geo["cap_height_end"]),
        "cantilever_length": float(geo["cantilever_length"]),
        "cap_slope": float(geo["cap_slope"]) if geo.get("cap_slope") is not None else None,
        "column_spacing": float(geo["column_spacing"]),
        "column_count": int(geo["column_count"]),
        "column_diameter": float(geo["column_diameter"]),
        "span_length": float(geo["span_length"]),
        "deck_width": float(geo["deck_width"]),
        "next_deck_width": float(geo["next_deck_width"]),
        "deck_slope": float(geo["deck_slope"]) if geo.get("deck_slope") is not None else None,
        "is_double_support": bool(geo["is_double_support"]),
        "support_symmetry": geo.get("support_symmetry"),
        "cap_self_weight": dead["cap_self_weight"],
        "superstructure_dead_load": dead["superstructure_dead_load"],
        "vehicle_live_load": {
            "input": live["reference"],
            "pattern_results": live["patterns"],
        },
        "load_combinations": root.get("load_combinations", {}),
    }

# =========================
# 建模辅助
# =========================
def build_key_positions(cap_length, cantilever_length, column_spacing, column_count, dead_support_points, live_supports) -> List[float]:
    xs = set()
    xs.add(-cap_length / 2.0)
    xs.add(cap_length / 2.0)
    xs.add(-cap_length / 2.0 + cantilever_length)
    xs.add(cap_length / 2.0 - cantilever_length)

    start_x = - (column_count - 1) * column_spacing / 2.0
    for i in range(column_count):
        xs.add(start_x + i * column_spacing)

    for p in dead_support_points:
        xs.add(x_left_to_center(float(p["x_m"]), cap_length))
    for s in live_supports:
        xs.add(float(s["x_m"]))
    return sorted(xs)

def refine_positions(xs: List[float], max_len: float = 0.5) -> List[float]:
    out = [xs[0]]
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        L = x2 - x1
        if L <= max_len:
            out.append(x2)
        else:
            n = math.ceil(L / max_len)
            for j in range(1, n + 1):
                out.append(x1 + L * j / n)
    refined = []
    for x in out:
        if not refined or abs(x - refined[-1]) > 1e-10:
            refined.append(x)
    return refined

def find_node_by_x(node_x_map: Dict[int, float], x_target: float, tol: float = 1e-6) -> int:
    for nid, x in node_x_map.items():
        if abs(x - x_target) <= tol:
            return nid
    raise ValueError(f"未找到 x={x_target:.6f} 对应节点")

def build_model(model_data: dict):
    _ensure_opensees()
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    cap_length = model_data["cap_length"]
    cap_width = model_data["cap_width"]
    cap_height_mid = model_data["cap_height_mid"]
    cantilever_length = model_data["cantilever_length"]
    column_spacing = model_data["column_spacing"]
    column_count = model_data["column_count"]
    dead_support_points = model_data["superstructure_dead_load"]["support_points"]
    live_supports = model_data["vehicle_live_load"]["input"]["supports"]

    A = cap_width * cap_height_mid
    Iz = cap_width * cap_height_mid ** 3 / 12.0
    E = MODEL_E_CONCRETE

    ops.geomTransf("Linear", 1)

    xs = build_key_positions(cap_length, cantilever_length, column_spacing, column_count, dead_support_points, live_supports)
    xs = refine_positions(xs, max_len=MAX_ELE_LEN)

    node_x_map = {}
    node_tag = 1
    for x in xs:
        ops.node(node_tag, x, 0.0)
        node_x_map[node_tag] = x
        node_tag += 1

    ele_info = []
    sorted_nodes = sorted(node_x_map.keys(), key=lambda n: node_x_map[n])
    ele_tag = 1
    for i in range(len(sorted_nodes) - 1):
        ni, nj = sorted_nodes[i], sorted_nodes[i + 1]
        ops.element("elasticBeamColumn", ele_tag, ni, nj, A, E, Iz, 1)
        ele_info.append((ele_tag, ni, nj, node_x_map[ni], node_x_map[nj]))
        ele_tag += 1

    ops.uniaxialMaterial("Elastic", 1, SUPPORT_STIFFNESS)
    start_x = - (column_count - 1) * column_spacing / 2.0
    column_x_coords = [start_x + i * column_spacing for i in range(column_count)]
    column_supports = []
    
    for i, col_x in enumerate(column_x_coords):
        pier_node = find_node_by_x(node_x_map, col_x)
        ground_node = node_tag
        node_tag += 1
        ops.node(ground_node, col_x, 0.0)
        ops.fix(ground_node, 1, 1, 1) 
        ops.element("zeroLength", ele_tag, pier_node, ground_node, "-mat", 1, "-dir", 2)
        ele_tag += 1
        if i == 0:
            ops.fix(pier_node, 1, 0, 0)
        column_supports.append({
            "column_index": i,
            "x_m": col_x,
            "pier_node": pier_node,
            "ground_node": ground_node,
        })

    return node_x_map, ele_info, column_supports

def get_elements_in_range(ele_info, x1, x2) -> List[int]:
    return [etag for etag, ni, nj, xi, xj in ele_info if xi >= x1 - 1e-8 and xj <= x2 + 1e-8]

# =========================
# 荷载施加与分析
# =========================
def apply_cap_self_weight(model_data, ele_info):
    cap_length = model_data["cap_length"]
    cantilever_length = model_data["cantilever_length"]
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for seg in model_data["cap_self_weight"]["segments"]:
        x1 = safe_eval_expr(seg["x_start_expr"], cap_length, cantilever_length)
        x2 = safe_eval_expr(seg["x_end_expr"], cap_length, cantilever_length)
        q = float(seg["line_load_kN_m"])
        for etag in get_elements_in_range(ele_info, x1, x2):
            ops.eleLoad("-ele", etag, "-type", "-beamUniform", -q)

def apply_superstructure_dead_load(model_data, node_x_map):
    cap_length = model_data["cap_length"]
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    for p in model_data["superstructure_dead_load"]["support_points"]:
        x = x_left_to_center(float(p["x_m"]), cap_length)
        nid = find_node_by_x(node_x_map, x)
        ops.load(nid, 0.0, -float(p["reaction_kN"]), 0.0)

def apply_vehicle_live_pattern(model_data, node_x_map, pattern_obj, ts_tag, pattern_tag):
    out = {s["support_id"]: find_node_by_x(node_x_map, float(s["x_m"])) for s in model_data["vehicle_live_load"]["input"]["supports"]}
    ops.timeSeries("Linear", ts_tag)
    ops.pattern("Plain", pattern_tag, ts_tag)
    for sid, force_kN in pattern_obj["support_reactions_kN"].items():
        if abs(float(force_kN)) > 1e-12:
            ops.load(out[sid], 0.0, -float(force_kN), 0.0)

def setup_analysis():
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")

def run_analysis():
    if ops.analyze(1) != 0: raise RuntimeError("静力分析失败")

def extract_full_beam_force_data(ele_info) -> dict:
    rows = []
    for etag, ni, nj, xi, xj in ele_info:
        lf = list(ops.eleForce(etag))
        rows.append({
            "element": etag, "node_i": ni, "node_j": nj, "x_i_m": xi, "x_j_m": xj,
            "N_i_kN": float(lf[0]), "V_i_kN": float(lf[1]), "M_i_kN_m": -float(lf[2]),
            "N_j_kN": -float(lf[3]), "V_j_kN": -float(lf[4]), "M_j_kN_m": float(lf[5]),
        })
    return {"element_forces": rows}


def extract_column_reactions(column_supports) -> List[float]:
    """提取各柱底竖向反力（kN），向上为正。"""
    # OpenSees 在 analyze() 后不会自动组装 nodeReaction() 查询所需的反力向量。
    # 若省略该调用，梁单元内力正常但固定节点反力会被读成 0。
    ops.reactions()
    reactions = []
    for support in column_supports:
        reaction = ops.nodeReaction(support["ground_node"])
        reactions.append(float(reaction[1]))
    return reactions


# =========================
# 内力处理与绘图逻辑
# =========================
def get_jump_nodes(model_data: dict) -> Set[float]:
    cap_length = model_data["cap_length"]
    xs = set()
    for p in model_data["superstructure_dead_load"]["support_points"]:
        xs.add(round(x_left_to_center(float(p["x_m"]), cap_length), 8))
    for s in model_data["vehicle_live_load"]["input"]["supports"]:
        xs.add(round(float(s["x_m"]), 8))
    cs = model_data["column_spacing"]
    count = model_data["column_count"]
    start_x = - (count - 1) * cs / 2.0
    for i in range(count):
        xs.add(round(start_x + i * cs, 8))
    return xs

def build_section_series(full_force_data, key_i, key_j, jump_nodes=None, tol=1e-8):
    if jump_nodes is None: jump_nodes = set()
    rows = full_force_data["element_forces"]
    if not rows: return [], []
    xs, ys = [rows[0]["x_i_m"]], [rows[0][key_i]]
    for idx in range(len(rows) - 1):
        cur, nxt = rows[idx], rows[idx + 1]
        x, left_val, right_val = cur["x_j_m"], cur[key_j], nxt[key_i]
        if any(abs(x - xn) <= tol for xn in jump_nodes):
            xs.extend([x, x])
            ys.extend([left_val, right_val])
        else:
            xs.append(x)
            ys.append(0.5 * (left_val + right_val))
    xs.append(rows[-1]["x_j_m"])
    ys.append(rows[-1][key_j])
    return xs, ys

def envelope_full_force_data(cases, key_i, key_j) -> dict:
    case_names = list(cases.keys())
    first_rows = cases[case_names[0]]["element_forces"]
    out_rows = []
    for idx in range(len(first_rows)):
        base = first_rows[idx]
        row = {"element": base["element"], "x_i_m": base["x_i_m"], "x_j_m": base["x_j_m"]}
        for end_key in [key_i, key_j]:
            vals = [(name, cases[name]["element_forces"][idx][end_key]) for name in case_names]
            row[f"{end_key}_max"] = max(vals, key=lambda t: t[1])[1]
            row[f"{end_key}_min"] = min(vals, key=lambda t: t[1])[1]
        out_rows.append(row)
    return {"element_forces_envelope": out_rows}

def envelope_to_force_data(env_data, key_i, key_j, mode) -> dict:
    return {"element_forces": [
        {"element": r["element"], "x_i_m": r["x_i_m"], "x_j_m": r["x_j_m"], 
         key_i: r[f"{key_i}_{mode}"], key_j: r[f"{key_j}_{mode}"]}
        for r in env_data["element_forces_envelope"]
    ]}

def combine_full_force_data(dead_data, live_data, gamma_dead, gamma_live) -> dict:
    return {"element_forces": [{
        "element": rd["element"], "x_i_m": rd["x_i_m"], "x_j_m": rd["x_j_m"],
        "V_i_kN": gamma_dead * rd["V_i_kN"] + gamma_live * rl["V_i_kN"],
        "V_j_kN": gamma_dead * rd["V_j_kN"] + gamma_live * rl["V_j_kN"],
        "M_i_kN_m": gamma_dead * rd["M_i_kN_m"] + gamma_live * rl["M_i_kN_m"],
        "M_j_kN_m": gamma_dead * rd["M_j_kN_m"] + gamma_live * rl["M_j_kN_m"]
    } for rd, rl in zip(dead_data["element_forces"], live_data["element_forces"])]}


def plot_envelope_diagram_with_extrema(
    env_data: dict, key_i: str, key_j: str, title: str, ylabel: str,
    filepath: str, jump_nodes: Set[float], invert_for_display: bool
):
    """绘制包络图，并在图上自动寻找和标注极值点数值"""
    xs_max, ys_max = build_section_series(envelope_to_force_data(env_data, key_i, key_j, "max"), key_i, key_j, jump_nodes)
    xs_min, ys_min = build_section_series(envelope_to_force_data(env_data, key_i, key_j, "min"), key_i, key_j, jump_nodes)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(xs_max, ys_max, label="Max Envelope", linewidth=1.5, color="tab:red")
    ax.plot(xs_min, ys_min, label="Min Envelope", linewidth=1.5, color="tab:blue")
    ax.axhline(0.0, linewidth=1.0, color='black', linestyle='--')

    # 【核心】智能提取极值点并标注
    def annotate_extrema(xs, ys, color, invert):
        if not ys: return
        annotated_pts = []
        x_range = (max(xs) - min(xs)) if (max(xs) - min(xs)) > 1e-3 else 1
        y_max_abs = max(abs(max(ys)), abs(min(ys)))
        y_range = y_max_abs if y_max_abs > 1e-3 else 1

        def check_overlap(nx, ny):
            for (ox, oy) in annotated_pts:
                # 若坐标过于接近，认定为同一处的极值避免文字重叠堆积
                if abs(nx - ox) < x_range * 0.04 and abs(ny - oy) < y_range * 0.08:
                    return True
            return False

        for i in range(len(ys)):
            if abs(ys[i]) < 1.0: # 过滤零点附近的噪音
                continue
            
            # 判断当前点是否为局部极大/极小值或首尾端点
            is_peak = False
            if i == 0 or i == len(ys) - 1:
                is_peak = True
            else:
                is_max = (ys[i] > ys[i-1] + 1e-5 and ys[i] >= ys[i+1]) or (ys[i] >= ys[i-1] and ys[i] > ys[i+1] + 1e-5)
                is_min = (ys[i] < ys[i-1] - 1e-5 and ys[i] <= ys[i+1]) or (ys[i] <= ys[i-1] and ys[i] < ys[i+1] - 1e-5)
                if is_max or is_min:
                    is_peak = True

            if is_peak and not check_overlap(xs[i], ys[i]):
                # 判定“视觉上方”：如果常规系y>0则在上；反转系(invert)则y<0在上
                is_visual_up = (ys[i] > 0) if not invert else (ys[i] < 0)
                
                offset_y = 6 if is_visual_up else -6
                va_align = 'bottom' if is_visual_up else 'top'
                
                ax.annotate(
                    f"{ys[i]:.1f}",
                    xy=(xs[i], ys[i]),
                    xytext=(0, offset_y), textcoords="offset points",
                    ha='center', va=va_align, fontsize=9, color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)
                )
                annotated_pts.append((xs[i], ys[i]))

    annotate_extrema(xs_max, ys_max, "darkred", invert_for_display)
    annotate_extrema(xs_min, ys_min, "darkblue", invert_for_display)

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    if y_range > 1e-6:
        ax.set_ylim(ymin - Y_MARGIN_RATIO * y_range, ymax + Y_MARGIN_RATIO * y_range)

    if invert_for_display:
        ax.invert_yaxis()
        
    ax.set_xlabel("x along cap beam centerline (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=35) # 加大pad避免标题跟顶部的图例重叠
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def cap_internal_force_analysis_tool(
    analysis_load_input: dict,
    output_dir: str = ".",
    task_id: str = "cap_beam",
    save_figures: bool = True,
) -> dict:
    """盖梁配筋前置环节 2：OpenSeesPy 盖梁内力计算。

    输入 analysis_load_input 对象，输出全梁内力包络 JSON 与 ULS_basic 弯矩/剪力图。
    """
    try:
        _ensure_opensees()
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        fig_dir = root / f"beam_force_figures_{task_id}"
        fig_dir.mkdir(parents=True, exist_ok=True)

        model_data = parse_input(analysis_load_input)
        jump_nodes = get_jump_nodes(model_data)

        dead_case = run_dead_case(model_data)
        live_cases = run_live_cases(model_data)

        combined_results = {}
        combined_envelopes = {}
        figure_files = {}

        for comb_name, factors in model_data.get("load_combinations", LOAD_COMBINATIONS).items():
            cases_for_env = {}
            for pid, item in live_cases.items():
                combined = combine_full_force_data(dead_case, item["full_beam_forces"], factors["dead"], factors["live"])
                cases_for_env[f"{comb_name}+{pid}"] = combined
                combined_results[f"{comb_name}+{pid}"] = combined

            env_V = envelope_full_force_data(cases_for_env, "V_i_kN", "V_j_kN")
            env_M = envelope_full_force_data(cases_for_env, "M_i_kN_m", "M_j_kN_m")
            combined_envelopes[comb_name] = {"shear_envelope": env_V, "moment_envelope": env_M}

            if save_figures and comb_name == "ULS_basic":
                shear_path = fig_dir / f"{comb_name}_shear_envelope.png"
                moment_path = fig_dir / f"{comb_name}_moment_envelope.png"
                plot_envelope_diagram_with_extrema(
                    env_V, "V_i_kN", "V_j_kN", f"{comb_name} Shear Envelope of Full Cap Beam",
                    "Shear force (kN)", str(shear_path), jump_nodes=jump_nodes, invert_for_display=False
                )
                plot_envelope_diagram_with_extrema(
                    env_M, "M_i_kN_m", "M_j_kN_m", f"{comb_name} Moment Envelope of Full Cap Beam",
                    "Bending moment (kN·m)", str(moment_path), jump_nodes=set(), invert_for_display=True
                )
                figure_files["shear_envelope_figure_path"] = str(shear_path)
                figure_files["moment_envelope_figure_path"] = str(moment_path)

        output = {
            "model_summary": model_data,
            "dead_case_full_beam": dead_case,
            "combined_envelopes_full_beam": combined_envelopes,
            "note_moment_sign": "输出弯矩数值已统一为工程习惯：跨中正弯矩（下部受拉）为正值，支点负弯矩为负值；图像利用坐标轴反转实现了下正上负的受拉侧绘制。",
        }
        output_json = root / f"internal_force_output_full_beam_{task_id}.json"
        save_json(output, str(output_json))
        return {"success": True, "internal_force_output": output, "output_files": {"internal_force_output_path": str(output_json), **figure_files}}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =========================
# 工况运行与主函数
# =========================
def run_dead_case(model_data):
    node_x_map, ele_info, column_supports = build_model(model_data)
    apply_cap_self_weight(model_data, ele_info)
    apply_superstructure_dead_load(model_data, node_x_map)
    setup_analysis()
    run_analysis()
    full_beam = extract_full_beam_force_data(ele_info)
    full_beam["column_reactions_kN"] = extract_column_reactions(column_supports)
    return full_beam

def run_live_cases(model_data):
    results = {}
    for i, pat in enumerate(model_data["vehicle_live_load"]["pattern_results"], start=1):
        node_x_map, ele_info, column_supports = build_model(model_data)
        apply_vehicle_live_pattern(model_data, node_x_map, pat, 100 + i, 100 + i)
        setup_analysis()
        run_analysis()
        full_beam = extract_full_beam_force_data(ele_info)
        full_beam["column_reactions_kN"] = extract_column_reactions(column_supports)
        # 兼容低于 Python 3.9 的字典合并写法
        results[pat["pattern_id"]] = {**pat, "full_beam_forces": full_beam}
    return results

def main():
    base_dir = Path(".")
    json_path = base_dir / "analysis_load_input_test01.json"
    output_json = base_dir / "internal_force_output_full_beam_test01.json"
    fig_dir = base_dir / "beam_force_figures_final_test01"
    fig_dir.mkdir(exist_ok=True)

    data = load_json(str(json_path))
    model_data = parse_input(data)
    jump_nodes = get_jump_nodes(model_data)
    
    # 运行结构分析
    dead_case = run_dead_case(model_data)
    live_cases = run_live_cases(model_data)

    combined_results = {}
    combined_envelopes = {}

    for comb_name, factors in model_data.get("load_combinations", LOAD_COMBINATIONS).items():
        cases_for_env = {}
        for pid, item in live_cases.items():
            combined = combine_full_force_data(dead_case, item["full_beam_forces"], factors["dead"], factors["live"])
            cases_for_env[f"{comb_name}+{pid}"] = combined
            combined_results[f"{comb_name}+{pid}"] = combined

        env_V = envelope_full_force_data(cases_for_env, "V_i_kN", "V_j_kN")
        env_M = envelope_full_force_data(cases_for_env, "M_i_kN_m", "M_j_kN_m")
        combined_envelopes[comb_name] = {"shear_envelope": env_V, "moment_envelope": env_M}

        # 【重点需求】只绘制最关键的 ULS_basic 的两张图，并带极值标注
        if comb_name == "ULS_basic":
            plot_envelope_diagram_with_extrema(
                env_V, "V_i_kN", "V_j_kN",
                f"{comb_name} Shear Envelope of Full Cap Beam",
                "Shear force (kN)",
                str(fig_dir / f"{comb_name}_shear_envelope.png"),
                jump_nodes=jump_nodes, invert_for_display=False
            )
            plot_envelope_diagram_with_extrema(
                env_M, "M_i_kN_m", "M_j_kN_m",
                f"{comb_name} Moment Envelope of Full Cap Beam",
                "Bending moment (kN·m)",
                str(fig_dir / f"{comb_name}_moment_envelope.png"),
                jump_nodes=set(), invert_for_display=True
            )
            print(f"已生成关键包络图件：{comb_name}_shear_envelope.png 及 {comb_name}_moment_envelope.png")

    # JSON 结果完整输出
    output = {
        "model_summary": model_data,
        "dead_case_full_beam": dead_case,
        "combined_envelopes_full_beam": combined_envelopes,
        "note_moment_sign": "输出弯矩数值已统一为工程习惯：跨中正弯矩（下部受拉）为正值，支点负弯矩为负值；图像利用坐标轴反转实现了下正上负的受拉侧绘制。"
    }
    save_json(output, str(output_json))
    print(f"已输出精简版全梁内力 JSON: {output_json}")

if __name__ == "__main__":
    main()
