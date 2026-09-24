# -*- coding: utf-8 -*-
"""
盖梁配筋承载能力包络计算与内力需求叠加绘图脚本（改进验算版）

本版相较初版的主要改进：
1. 端部保护层退让区不参与抗弯、抗剪强度验算，绘图采用 NaN 断开，避免承载力从 0 突跳；
2. y_offset 按“向截面内部的层级退让量”解释，避免顶部加强筋被错误解释到混凝土截面外；
3. 弯起钢筋斜段不直接作为正截面抗弯纵筋计入，只有水平段参与顶部/底部抗弯承载力统计；
4. 墩柱支承宽度范围内不作为抗剪控制截面，抗剪控制截面转移至柱边及柱边外侧；
5. 承载力利用率图对无效验算区段自动断开，并在图中浅色标出柱身支承区。

功能：
1. 读取 LLM 生成的盖梁配筋 YAML（默认：Output test01.yaml）；
2. 读取 OpenSees 内力包络结果 JSON（默认：internal_force_output_full_beam_test01.json）；
3. 根据盖梁几何、纵筋路径、箍筋分区，逐截面计算：
   - 正弯矩抗弯承载力 Mud_pos(x)
   - 负弯矩抗弯承载力 Mud_neg(x)
   - 正负剪力抗剪承载力 Vud(x)
4. 将承载能力包络与 OpenSees 作用效应包络叠加绘图；
5. 输出承载能力数据、验算利用率和控制截面摘要。

重要说明：
- OpenSees 脚本负责计算作用效应 Md(x)、Vd(x)；
- 本脚本负责根据配筋结果计算抗力 Mud(x)、Vud(x)；
- 正截面抗弯执行 JTG 3362-2018 第5.2.2条固定 Python 公式，并按表5.2.1校核 xi_b；
- 盖梁抗剪执行第8.4.4条和第8.4.5条固定 Python 公式；
- 裂缝、挠度、疲劳、抗震及完整构造审查仍属于当前概念设计未覆盖项。

运行方式：
1. 将本脚本与以下文件放在同一目录：
   - Output test01.yaml
   - internal_force_output_full_beam_test01.json
   - opensees_cap_beam_full_force_diagrams_fixed 改边界_多柱_压缩_变截面_精简.py（可选，自动运行用）
2. 若 internal_force_output_full_beam_test01.json 尚未生成，脚本会尝试自动运行 OpenSees 脚本。
3. 在 VS Code 中直接运行本文件即可。
"""

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    import yaml
except ImportError as e:
    raise RuntimeError("缺少 PyYAML，请先安装：pip install pyyaml") from e

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from bridge_agents.code_formula_registry import (
    FLEXURE_ID,
    SHEAR_LIMIT_ID,
    SHEAR_REINFORCED_ID,
    execute_code_formula,
)


# ============================================================
# 0. 用户可调整参数
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

REINFORCEMENT_YAML = BASE_DIR / "Output test01.yaml"
OPENSEES_FORCE_JSON = BASE_DIR / "internal_force_output_full_beam_test01.json"
OPENSEES_SCRIPT = BASE_DIR / "opensees_cap_beam_full_force_diagrams_fixed 改边界_多柱_压缩_变截面_精简.py"

OUTPUT_DIR = BASE_DIR / "capacity_envelope_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 若 OpenSees 输出 JSON 不存在，是否自动运行 OpenSees 内力脚本
AUTO_RUN_OPENSEES_IF_MISSING = True

# 需要读取的组合名称，应与 OpenSees 输出中的 combined_envelopes_full_beam 键一致
COMBINATION_NAME = "ULS_basic"

# 结构重要性系数 γ0：如果 OpenSees 输出已经包含 γ0，可改为 1.0
GAMMA_0 = 1.10
APPLY_GAMMA0_TO_DEMAND = True

# 截面扫描间距，单位 m；越小图越平滑，但计算量略增
CAPACITY_SCAN_STEP_M = 0.05

# 在配筋/箍筋突变点两侧增加微小偏移点，便于画出承载力突变
EPS_M = 0.001

# 验算合理性控制：
# 1) 端部保护层退让区不参与强度验算，绘图用 NaN 断开；
# 2) 弯起钢筋斜段不直接作为抗弯纵筋计入，避免承载力曲线锯齿化；
# 3) 柱身支承宽度范围内不作为抗剪控制截面，抗剪控制转移到柱边及柱边外侧；
# 4) y_offset 作为“向截面内部退让的层级距离”解释，更贴合配筋语义。
EXCLUDE_END_BLANK_ZONE = True
FLEXURE_ONLY_HORIZONTAL_SEGMENTS = True
EXCLUDE_COLUMN_CORE_FOR_SHEAR_CHECK = True
INTERPRET_Y_OFFSET_AS_INWARD_LAYER = True
INVALID_STIRRUP_ZONE_KEYWORDS = ("blank", "空白", "保护层退让", "端部保护层")

# 图形参数
FIG_SIZE = (16, 4.2)
DPI = 300

# 默认保护层与材料设计参数，单位：mm、MPa
DEFAULTS = {
    "cover_x": 50.0,
    "cover_y": 50.0,
    "cover_z": 60.0,
    # 当前概念设计材料输入。每次验算均在摘要中显式输出，后续应由项目材料参数直接传入。
    "fcd": 18.4,       # 混凝土轴心抗压强度设计值 MPa
    "ftd": 1.65,       # 混凝土轴心抗拉强度设计值 MPa
    "fsd": 330.0,      # HRB400 普通钢筋抗拉强度设计值 MPa
    "fsvd": 330.0,     # 箍筋抗拉强度设计值 MPa
    "steel_grade": "HRB400",
    "concrete_strength_grade": "C40",
    "fcu_k": 40.0,
    "alpha1_for_cap_shear": 1.0,
}


# ============================================================
# 1. 基础工具
# ============================================================
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def area_from_dia_mm2(dia_mm: float) -> float:
    return math.pi * dia_mm * dia_mm / 4.0


def tand(deg: float) -> float:
    return math.tan(math.radians(deg))


def safe_eval(expr: Any, context: Dict[str, Any]) -> float:
    """安全计算 YAML 中的简单表达式。"""
    if isinstance(expr, (int, float)):
        return float(expr)
    if expr is None:
        raise ValueError("表达式为空，无法计算。")
    s = str(expr).strip()
    if s.startswith("="):
        s = s[1:].strip()
    allowed = {
        "math": math,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "radians": math.radians,
        "abs": abs,
        "max": max,
        "min": min,
        "pi": math.pi,
        "tand": tand,
    }
    allowed.update(context)
    try:
        return float(eval(s, {"__builtins__": {}}, allowed))
    except Exception as e:
        raise ValueError(f"表达式计算失败：{expr}，当前上下文={context}") from e


def interpolate(x: float, xs: List[float], ys: List[float]) -> float:
    """一维线性插值；xs 需基本单调递增。"""
    if not xs:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x1, x2 = xs[lo], xs[hi]
    y1, y2 = ys[lo], ys[hi]
    if abs(x2 - x1) < 1e-12:
        return y2
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


# ============================================================
# 2. 从 OpenSees 输出读取几何与内力需求
# ============================================================
def maybe_run_opensees_script() -> None:
    if OPENSEES_FORCE_JSON.exists():
        return
    if not AUTO_RUN_OPENSEES_IF_MISSING:
        raise FileNotFoundError(f"未找到 OpenSees 输出文件：{OPENSEES_FORCE_JSON}")
    if not OPENSEES_SCRIPT.exists():
        raise FileNotFoundError(
            f"未找到 OpenSees 输出文件：{OPENSEES_FORCE_JSON}\n"
            f"且无法自动运行，因为未找到脚本：{OPENSEES_SCRIPT}"
        )
    print(f"未找到 {OPENSEES_FORCE_JSON.name}，开始自动运行 OpenSees 内力脚本...")
    subprocess.check_call([sys.executable, str(OPENSEES_SCRIPT)], cwd=str(BASE_DIR))
    if not OPENSEES_FORCE_JSON.exists():
        raise RuntimeError("OpenSees 脚本已运行，但仍未生成内力输出 JSON，请检查原脚本中的输出文件名。")


def get_force_envelopes(force_json: dict, comb_name: str) -> Tuple[dict, dict]:
    env_root = force_json.get("combined_envelopes_full_beam", {})
    if comb_name not in env_root:
        raise KeyError(f"OpenSees 输出中未找到组合 {comb_name}，可用组合：{list(env_root.keys())}")
    item = env_root[comb_name]
    return item["moment_envelope"], item["shear_envelope"]


def envelope_to_xy(env_data: dict, key_i: str, key_j: str, mode: str, gamma: float = 1.0) -> Tuple[List[float], List[float]]:
    """将 OpenSees 包络 JSON 转为绘图用折线数据。"""
    rows = env_data.get("element_forces_envelope", [])
    xs, ys = [], []
    suffix_i = f"{key_i}_{mode}"
    suffix_j = f"{key_j}_{mode}"
    for r in rows:
        xi, xj = float(r["x_i_m"]), float(r["x_j_m"])
        yi, yj = gamma * float(r[suffix_i]), gamma * float(r[suffix_j])
        if not xs:
            xs.append(xi); ys.append(yi)
        else:
            # 保留相同 x 处的突变，尤其对剪力图有意义
            if abs(xs[-1] - xi) > 1e-12 or abs(ys[-1] - yi) > 1e-9:
                xs.append(xi); ys.append(yi)
        xs.append(xj); ys.append(yj)
    return xs, ys


def extract_all_demand_x(force_json: dict, comb_name: str) -> List[float]:
    m_env, v_env = get_force_envelopes(force_json, comb_name)
    xs = set()
    for env in [m_env, v_env]:
        for r in env.get("element_forces_envelope", []):
            xs.add(float(r["x_i_m"]))
            xs.add(float(r["x_j_m"]))
    return sorted(xs)


# ============================================================
# 3. 几何、配筋路径与箍筋分区解析
# ============================================================
@dataclass
class Geometry:
    cap_length_m: float
    cap_width_m: float
    cap_height_mid_m: float
    cap_height_end_m: float
    cantilever_length_m: float
    column_spacing_m: float
    column_count: int
    column_diameter_m: float

    @property
    def L(self) -> float:
        return self.cap_length_m * 1000.0

    @property
    def b(self) -> float:
        return self.cap_width_m * 1000.0

    @property
    def h_mid(self) -> float:
        return self.cap_height_mid_m * 1000.0

    @property
    def h_end(self) -> float:
        return self.cap_height_end_m * 1000.0

    @property
    def cantilever(self) -> float:
        return self.cantilever_length_m * 1000.0

    @property
    def column_spacing(self) -> float:
        return self.column_spacing_m * 1000.0

    @property
    def column_diameter(self) -> float:
        return self.column_diameter_m * 1000.0


def geometry_from_force_json(force_json: dict) -> Geometry:
    ms = force_json["model_summary"]
    return Geometry(
        cap_length_m=float(ms["cap_length"]),
        cap_width_m=float(ms["cap_width"]),
        cap_height_mid_m=float(ms["cap_height_mid"]),
        cap_height_end_m=float(ms["cap_height_end"]),
        cantilever_length_m=float(ms["cantilever_length"]),
        column_spacing_m=float(ms["column_spacing"]),
        column_count=int(ms["column_count"]),
        column_diameter_m=float(ms["column_diameter"]),
    )


def section_height_mm(x_m: float, geo: Geometry) -> float:
    """按对称变截面盖梁计算当前截面高度，单位 mm。"""
    x = abs(x_m * 1000.0)
    half = geo.L / 2.0
    constant_limit = half - geo.cantilever
    if x <= constant_limit:
        return geo.h_mid
    if x >= half:
        return geo.h_end
    t = (x - constant_limit) / max(geo.cantilever, 1e-9)
    return geo.h_mid + t * (geo.h_end - geo.h_mid)


def column_centers_m(geo: Geometry) -> List[float]:
    start = - (geo.column_count - 1) * geo.column_spacing_m / 2.0
    return [start + i * geo.column_spacing_m for i in range(geo.column_count)]


def left_reference_pier_center_mm(geo: Geometry) -> float:
    """当前配筋 YAML 采用左半盖梁定义并镜像，pier_centerline_x 取左侧柱中心线。"""
    return - (geo.column_count - 1) * geo.column_spacing / 2.0


def y_from_ref(h: float, y_ref: str, y_offset: Any, ctx: Dict[str, Any]) -> float:
    """
    根据 y_ref 与 y_offset 得到钢筋 y 坐标。

    说明：
    - LLM 输出中 y_offset 通常表示“第几层钢筋”的层级退让量；
    - 对 top_cover，正的 y_offset 应向截面内部，即向下；
    - 对 bottom_cover，正的 y_offset 应向截面内部，即向上；
    - 为兼容少量负号写法，默认取 abs(offset) 作为层级距离，避免钢筋被解释到混凝土边界外。
    """
    off_raw = safe_eval(y_offset, ctx) if isinstance(y_offset, str) else float(y_offset or 0.0)
    cover_y = float(ctx["cover_y"])
    off = abs(off_raw) if INTERPRET_Y_OFFSET_AS_INWARD_LAYER else off_raw
    if y_ref == "top_cover":
        y = h / 2.0 - cover_y - off if INTERPRET_Y_OFFSET_AS_INWARD_LAYER else h / 2.0 - cover_y + off_raw
    elif y_ref == "bottom_cover":
        y = -h / 2.0 + cover_y + off if INTERPRET_Y_OFFSET_AS_INWARD_LAYER else -h / 2.0 + cover_y + off_raw
    elif y_ref == "mid_depth":
        y = off_raw
    else:
        raise ValueError(f"未知 y_ref：{y_ref}")

    # 防止异常表达式把钢筋推出混凝土截面；保守夹回保护层控制范围。
    y_max = h / 2.0 - cover_y
    y_min = -h / 2.0 + cover_y
    return max(min(y, y_max), y_min)


@dataclass
class Segment:
    x1: float
    x2: float
    y1: float
    y2: float
    kind: str

    def contains(self, x: float, tol: float = 1e-7) -> bool:
        return min(self.x1, self.x2) - tol <= x <= max(self.x1, self.x2) + tol

    def y_at(self, x: float) -> float:
        if abs(self.x2 - self.x1) < 1e-12:
            return self.y2
        t = (x - self.x1) / (self.x2 - self.x1)
        return self.y1 + t * (self.y2 - self.y1)


def z_count(bar_def: dict, pier_cap_cfg: dict) -> int:
    name = bar_def.get("z_pattern", "full")
    z_patterns = pier_cap_cfg.get("z_patterns", {})
    if isinstance(name, list):
        return len(name)
    return len(z_patterns.get(name, []))


def base_context(geo: Geometry, dia: Optional[float] = None) -> Dict[str, Any]:
    ctx = dict(DEFAULTS)
    ctx.update({
        "cap_length": geo.L,
        "cap_width": geo.b,
        "cap_height_mid": geo.h_mid,
        "cap_height_end": geo.h_end,
        "cantilever_length": geo.cantilever,
        "column_spacing": geo.column_spacing,
        "column_diameter": geo.column_diameter,
        "pier_centerline_x": left_reference_pier_center_mm(geo),
        "cap_centerline_x": 0.0,
        "dia": float(dia or 0.0),
    })
    return ctx


def point_xy(point_def: dict, h: float, ctx: Dict[str, Any]) -> Tuple[float, float]:
    x = safe_eval(point_def["x_expr"], ctx) if "x_expr" in point_def else safe_eval(point_def.get("x", 0.0), ctx)
    y = y_from_ref(h, point_def["y_ref"], point_def.get("y_offset", 0.0), ctx)
    return x, y


def build_constant_segments(bar_def: dict, h: float, ctx: Dict[str, Any]) -> List[Segment]:
    rd = bar_def["range_definition"]
    ld = bar_def["layer_definition"]
    x1 = safe_eval(rd["x_start_expr"], ctx)
    x2 = safe_eval(rd["x_end_expr"], ctx)
    y = y_from_ref(h, ld["y_ref"], ld.get("y_offset", 0.0), ctx)
    return [Segment(x1, x2, y, y, "constant")]


def add_slope_from_point(
    segs: List[Segment],
    x0: float,
    y0: float,
    target_def: dict,
    h: float,
    ctx: Dict[str, Any],
    direction: str,
    angle_deg: float,
) -> Tuple[float, float]:
    """根据方向和目标 y_ref 生成 45° 斜段，返回终点。"""
    y_target = y_from_ref(h, target_def["y_ref"], target_def.get("y_offset", 0.0), ctx)
    dy = abs(y0 - y_target)
    dx = dy / max(tand(angle_deg), 1e-9)

    if "left" in direction:
        x1 = x0 - dx
    elif "right" in direction:
        x1 = x0 + dx
    else:
        raise ValueError(f"无法识别斜段方向：{direction}")

    segs.append(Segment(x0, x1, y0, y_target, "ramp"))
    return x1, y_target


def _x_expr_value(value: Any, default: float = 0.0) -> Any:
    """兼容位置槽位的两种写法：对象 {x: 表达式}/{x_expr: 表达式} 或裸表达式字符串。"""
    if isinstance(value, Mapping):
        return value.get("x_expr", value.get("x", default))
    if value is None:
        return default
    return value


def build_control_segments(bar_def: dict, h: float, ctx: Dict[str, Any]) -> List[Segment]:
    """
    将当前 YAML 中的 control_definition 转换为分段钢筋路径。
    支持本次配筋结果中的 N3/N4 结构：left_branch、middle_branch、right_branch(step_1/step_2)。
    """
    cd = bar_def["control_definition"]
    kp_defs = cd.get("key_points", {})
    br = cd.get("branch_rules", {})
    angle = float(cd.get("bend_angle", 45.0))
    local_ctx = dict(ctx)
    local_ctx["bend_angle"] = angle
    arc_radius = cd.get("arc_definition", {}).get("radius")
    if arc_radius is not None:
        local_ctx["radius"] = safe_eval(arc_radius, local_ctx)

    # 预计算关键点
    kps = {}
    for name, pdef in kp_defs.items():
        if "y_ref" in pdef:
            kps[name] = point_xy(pdef, h, local_ctx)
        else:
            kps[name] = (safe_eval(_x_expr_value(pdef), local_ctx), None)

    segs: List[Segment] = []

    # middle_branch：顶部水平段
    if "middle_branch" in br:
        mb = br["middle_branch"]
        p1 = kps[mb["start_from"]]
        p2 = kps[mb["end_at"]]
        y = y_from_ref(h, mb["y_ref"], mb.get("y_offset", 0.0), local_ctx)
        segs.append(Segment(p1[0], p2[0], y, y, "constant"))

    # left_branch：从左弯点向左下弯至底部
    if "left_branch" in br:
        lb = br["left_branch"]
        p0 = kps[lb["start_from"]]
        if "until" in lb and p0[1] is not None:
            add_slope_from_point(segs, p0[0], p0[1], lb["until"], h, local_ctx, lb["direction"], angle)

    # right_branch：可包含 step_1 斜段与 step_2 水平段
    if "right_branch" in br:
        rb = br["right_branch"]
        if "step_1" in rb:
            st1 = rb["step_1"]
            p0 = kps[st1["start_from"]]
            if p0[1] is None:
                return segs
            x_end, y_end = add_slope_from_point(segs, p0[0], p0[1], st1["until"], h, local_ctx, st1["direction"], angle)
            if "step_2" in rb:
                st2 = rb["step_2"]
                x_to = safe_eval(_x_expr_value(st2.get("end_at")), local_ctx)
                y2 = y_from_ref(h, st2["y_ref"], st2.get("y_offset", 0.0), local_ctx)
                segs.append(Segment(x_end, x_to, y2, y2, "constant"))
        elif "until" in rb:
            # 兼容无 step 写法
            p0 = kps[rb["start_from"]]
            if p0[1] is not None:
                add_slope_from_point(segs, p0[0], p0[1], rb["until"], h, local_ctx, rb["direction"], angle)

    return segs


def build_bar_segments(bar_def: dict, h: float, geo: Geometry) -> List[Segment]:
    dia = float(bar_def.get("dia", 0.0))
    ctx = base_context(geo, dia=dia)
    if "range_definition" in bar_def and "layer_definition" in bar_def:
        return build_constant_segments(bar_def, h, ctx)
    if "control_definition" in bar_def:
        return build_control_segments(bar_def, h, ctx)
    return []


def eval_bar_segment_at_x_m(bar_def: dict, x_m: float, h: float, geo: Geometry) -> Optional[Tuple[float, Segment]]:
    """计算某根钢筋组在 x 截面处的 y 坐标及所在路径段；不存在则返回 None。"""
    x_mm = x_m * 1000.0
    # 当前输出采用左半定义 + 纵向镜像；用 -abs(x) 映射到左半
    if bar_def.get("mirror") == "longitudinal":
        query_x = -abs(x_mm)
    else:
        query_x = x_mm

    segs = build_bar_segments(bar_def, h, geo)
    for seg in segs:
        if seg.contains(query_x):
            return seg.y_at(query_x), seg
    return None


def eval_bar_y_at_x_m(bar_def: dict, x_m: float, h: float, geo: Geometry) -> Optional[float]:
    res = eval_bar_segment_at_x_m(bar_def, x_m, h, geo)
    return None if res is None else res[0]


def get_longitudinal_bars_at_section(x_m: float, h: float, geo: Geometry, pier_cap_cfg: dict) -> List[dict]:
    bars = []
    for bar_def in pier_cap_cfg.get("longitudinal_bars", []):
        res = eval_bar_segment_at_x_m(bar_def, x_m, h, geo)
        if res is None:
            continue
        y, seg = res
        dia = float(bar_def.get("dia", 0.0))
        n = z_count(bar_def, pier_cap_cfg)
        if n <= 0 or dia <= 0:
            continue
        bars.append({
            "id": bar_def.get("id", "unknown"),
            "category": bar_def.get("category", ""),
            "subtype": bar_def.get("subtype", ""),
            "dia": dia,
            "count": n,
            "area_mm2": n * area_from_dia_mm2(dia),
            "y_mm": y,
            "segment_kind": seg.kind,
            "is_horizontal_segment": seg.kind == "constant",
        })
    return bars


def parse_stirrup_legs(system_text: str) -> int:
    if not system_text:
        return 0
    m = re.search(r"(\d+)\s*肢", str(system_text))
    if m:
        return int(m.group(1))
    return 0


def is_invalid_stirrup_zone(zone: dict) -> bool:
    name = str(zone.get("name", ""))
    reason = str(zone.get("raw", {}).get("control_reason", ""))
    text = name + " " + reason
    return any(k in text for k in INVALID_STIRRUP_ZONE_KEYWORDS)


def stirrup_zone_at_x_m(x_m: float, geo: Geometry, pier_cap_cfg: dict) -> dict:
    stir = pier_cap_cfg.get("stirrups", {})
    zones = stir.get("longitudinal_distribution", [])
    x_left = -abs(x_m * 1000.0)
    ctx = base_context(geo)
    for z in zones:
        try:
            x1 = safe_eval(z["x_from"], ctx)
            x2 = safe_eval(z["x_to"], ctx)
        except Exception:
            continue
        if min(x1, x2) - 1e-7 <= x_left <= max(x1, x2) + 1e-7:
            spacing = safe_eval(z.get("spacing", 1e12), ctx)
            out = {"name": z.get("name", ""), "spacing_mm": spacing, "raw": z}
            out["is_invalid"] = is_invalid_stirrup_zone(out)
            return out
    return {"name": "未定义箍筋区", "spacing_mm": 1e12, "raw": {}, "is_invalid": True}


def is_inside_column_core(x_m: float, geo: Geometry, tol: float = 1e-9) -> bool:
    """柱身支承宽度内不作为抗剪控制截面。"""
    for cx in column_centers_m(geo):
        if cx - geo.column_diameter_m / 2.0 + tol < x_m < cx + geo.column_diameter_m / 2.0 - tol:
            return True
    return False


def is_end_blank_section(x_m: float, geo: Geometry, pier_cap_cfg: dict) -> bool:
    zone = stirrup_zone_at_x_m(x_m, geo, pier_cap_cfg)
    return bool(zone.get("is_invalid"))


def stirrup_basic_info(pier_cap_cfg: dict) -> Tuple[int, float]:
    stir = pier_cap_cfg.get("stirrups", {})
    cage = stir.get("transverse_cage_definition", {})
    legs = parse_stirrup_legs(cage.get("stirrup_system", ""))
    if legs <= 0:
        # 兜底：若有 4 个子箍，按 8 肢估算
        comps = cage.get("cage_layouts", {}).get("components", [])
        legs = max(2 * len(comps), 2)
    dia = float(cage.get("dia", 12.0))
    return legs, dia


def collect_capacity_keypoints_m(geo: Geometry, force_json: dict, pier_cap_cfg: dict) -> List[float]:
    keys = set()
    half = geo.cap_length_m / 2.0
    # 几何关键点
    keys.update([-half, half, 0.0])
    keys.update([-(half - geo.cantilever_length_m), (half - geo.cantilever_length_m)])
    for cx in column_centers_m(geo):
        keys.add(cx)
        keys.add(cx - geo.column_diameter_m / 2.0)
        keys.add(cx + geo.column_diameter_m / 2.0)

    # 支座与集中荷载位置
    ms = force_json.get("model_summary", {})
    cap_length = float(ms.get("cap_length", geo.cap_length_m))
    for p in ms.get("superstructure_dead_load", {}).get("support_points", []):
        # 原 OpenSees 脚本中 support_points 的 x_m 为从左端计，需转为中心坐标
        keys.add(float(p["x_m"]) - cap_length / 2.0)
    for s in ms.get("vehicle_live_load", {}).get("input", {}).get("supports", []):
        keys.add(float(s["x_m"]))

    # 纵筋路径端点
    for bar in pier_cap_cfg.get("longitudinal_bars", []):
        for probe_h in [geo.h_mid, geo.h_end]:
            try:
                segs = build_bar_segments(bar, probe_h, geo)
            except Exception:
                continue
            for seg in segs:
                for x in [seg.x1, seg.x2]:
                    xm = x / 1000.0
                    keys.add(xm)
                    if bar.get("mirror") == "longitudinal":
                        keys.add(-xm)

    # 箍筋分区端点
    ctx = base_context(geo)
    for z in pier_cap_cfg.get("stirrups", {}).get("longitudinal_distribution", []):
        for k in ["x_from", "x_to"]:
            try:
                xm = safe_eval(z[k], ctx) / 1000.0
                keys.add(xm)
                keys.add(-xm)
            except Exception:
                pass

    # 限制在盖梁范围内，并在突变点两侧加微小偏移
    all_keys = set()
    for x in keys:
        if -half - 1e-9 <= x <= half + 1e-9:
            all_keys.add(round(x, 6))
            if -half <= x - EPS_M <= half:
                all_keys.add(round(x - EPS_M, 6))
            if -half <= x + EPS_M <= half:
                all_keys.add(round(x + EPS_M, 6))
    return sorted(all_keys)


def build_capacity_scan_xs(geo: Geometry, force_json: dict, pier_cap_cfg: dict) -> List[float]:
    half = geo.cap_length_m / 2.0
    xs = set(extract_all_demand_x(force_json, COMBINATION_NAME))
    xs.update(collect_capacity_keypoints_m(geo, force_json, pier_cap_cfg))
    n = int(math.ceil(geo.cap_length_m / CAPACITY_SCAN_STEP_M))
    for i in range(n + 1):
        x = -half + i * geo.cap_length_m / n
        xs.add(round(x, 6))
    return sorted(x for x in xs if -half - 1e-8 <= x <= half + 1e-8)


# ============================================================
# 4. 截面承载能力计算
# ============================================================
def flexural_capacity_kNm(b: float, h: float, tension_bars: List[dict], compression_side: str) -> Tuple[float, dict]:
    """
    单筋矩形截面简化抗弯承载力。
    b,h 单位 mm；强度单位 MPa；返回 kN·m。

    compression_side:
    - "top"：正弯矩，顶部受压、底部受拉；
    - "bottom"：负弯矩，底部受压、顶部受拉。
    """
    As = sum(bi["area_mm2"] for bi in tension_bars)
    if As <= 1e-9:
        return 0.0, {"As_mm2": 0.0, "d_mm": 0.0, "x_c_mm": 0.0, "steel_ids": []}

    y_bar = sum(bi["area_mm2"] * bi["y_mm"] for bi in tension_bars) / As
    if compression_side == "top":
        d = h / 2.0 - y_bar
    elif compression_side == "bottom":
        d = y_bar + h / 2.0
    else:
        raise ValueError("compression_side 只能为 top 或 bottom")

    fcd = DEFAULTS["fcd"]
    fsd = DEFAULTS["fsd"]
    formula_result = execute_code_formula(FLEXURE_ID, {
        "b_mm": b,
        "h0_mm": d,
        "As_mm2": As,
        "fcd_MPa": fcd,
        "fsd_MPa": fsd,
        "steel_grade": DEFAULTS["steel_grade"],
        "concrete_strength_grade": DEFAULTS["concrete_strength_grade"],
    })
    outputs = formula_result.outputs
    x_c = float(outputs["x_mm"])
    x_limit = float(outputs["x_limit_mm"])
    over_limit = not formula_result.checks["compression_zone_limit_ok"]
    M_kNm = float(outputs["Mu_kN_m"])
    return M_kNm, {
        "As_mm2": As,
        "d_mm": d,
        "x_c_mm": x_c,
        "x_used_mm": x_c,
        "xi": outputs["xi"],
        "xi_b": outputs["xi_b"],
        "x_limit_mm": x_limit,
        "over_x_limit": over_limit,
        "formula_id": formula_result.formula_id,
        "formula_status": formula_result.status,
        "formula_evidence_ids": list(formula_result.evidence_ids),
        "steel_ids": [f"{bi['id']}×{bi['count']}" for bi in tension_bars],
    }


def shear_capacity_kN(
    b: float,
    h0: float,
    stirrup_spacing: float,
    stirrup_legs: int,
    stirrup_dia: float,
    *,
    h: float,
    calculation_span: float,
    longitudinal_As_mm2: float,
) -> Tuple[float, dict]:
    """
    按 JTG 3362-2018 式(8.4.4)、式(8.4.5)计算盖梁斜截面抗剪上限与配筋承载力。
    b,h0,h、calculation_span 单位 mm；强度单位 MPa；返回 kN。
    """
    fsvd = DEFAULTS["fsvd"]
    if stirrup_spacing <= 0 or stirrup_spacing > 1.0e8 or stirrup_legs <= 0:
        Asv = 0.0
        rho_sv = 0.0
        return 0.0, {
            "Asv_mm2": Asv,
            "rho_sv": rho_sv,
            "spacing_mm": stirrup_spacing,
            "stirrup_legs": stirrup_legs,
            "stirrup_dia_mm": stirrup_dia,
            "formula_ids": [SHEAR_LIMIT_ID, SHEAR_REINFORCED_ID],
            "formula_status": "manual_review",
            "formula_message": "当前截面无有效箍筋，式(8.4.5)不能形成正的配筋抗剪承载力。",
        }
    else:
        Asv = stirrup_legs * area_from_dia_mm2(stirrup_dia)
        rho_sv = Asv / (b * stirrup_spacing)
    p_ratio = 100.0 * longitudinal_As_mm2 / (b * h0)
    limit_result = execute_code_formula(SHEAR_LIMIT_ID, {
        "l0_mm": calculation_span,
        "h_mm": h,
        "fcu_k_MPa": DEFAULTS["fcu_k"],
        "b_mm": b,
        "h0_mm": h0,
    })
    reinforced_result = execute_code_formula(SHEAR_REINFORCED_ID, {
        "alpha1": DEFAULTS["alpha1_for_cap_shear"],
        "l_mm": calculation_span,
        "h_mm": h,
        "b_mm": b,
        "h0_mm": h0,
        "P": p_ratio,
        "fcu_k_MPa": DEFAULTS["fcu_k"],
        "rho_sv": rho_sv,
        "fsv_MPa": fsvd,
    })
    if reinforced_result.status != "computed":
        return 0.0, {
            "Asv_mm2": Asv,
            "rho_sv": rho_sv,
            "P": p_ratio,
            "spacing_mm": stirrup_spacing,
            "stirrup_legs": stirrup_legs,
            "stirrup_dia_mm": stirrup_dia,
            "formula_ids": [SHEAR_LIMIT_ID, SHEAR_REINFORCED_ID],
            "formula_status": "manual_review",
            "formula_message": reinforced_result.message,
        }
    v_limit = float(limit_result.outputs["V_limit_kN"])
    v_reinforced = float(reinforced_result.outputs["V_capacity_kN"])
    return min(v_limit, v_reinforced), {
        "V_limit_kN": v_limit,
        "V_reinforced_kN": v_reinforced,
        "Asv_mm2": Asv,
        "rho_sv": rho_sv,
        "P": p_ratio,
        "P_definition": "100*As/(b*h0)",
        "spacing_mm": stirrup_spacing,
        "stirrup_legs": stirrup_legs,
        "stirrup_dia_mm": stirrup_dia,
        "formula_ids": [SHEAR_LIMIT_ID, SHEAR_REINFORCED_ID],
        "formula_status": "computed",
        "formula_evidence_ids": list(dict.fromkeys(limit_result.evidence_ids + reinforced_result.evidence_ids)),
        "alpha1_source": "concept_design_default",
    }


def capacity_at_section(x_m: float, geo: Geometry, pier_cap_cfg: dict) -> dict:
    h = section_height_mm(x_m, geo)
    b = geo.b
    stir_zone = stirrup_zone_at_x_m(x_m, geo, pier_cap_cfg)
    in_end_blank = EXCLUDE_END_BLANK_ZONE and bool(stir_zone.get("is_invalid"))
    in_column_core = EXCLUDE_COLUMN_CORE_FOR_SHEAR_CHECK and is_inside_column_core(x_m, geo)

    bars = get_longitudinal_bars_at_section(x_m, h, geo, pier_cap_cfg)

    # 抗弯有效纵筋：仅计入水平段；斜向弯起段不机械计入正截面抗弯，避免承载力包络锯齿化。
    if FLEXURE_ONLY_HORIZONTAL_SEGMENTS:
        flexure_bars = [bi for bi in bars if bi.get("is_horizontal_segment")]
    else:
        flexure_bars = bars

    # 正弯矩：底部受拉；负弯矩：顶部受拉
    bottom_tension = [bi for bi in flexure_bars if bi["y_mm"] < -1e-6]
    top_tension = [bi for bi in flexure_bars if bi["y_mm"] > 1e-6]

    if in_end_blank:
        Mud_pos = float("nan")
        Mud_neg_abs = float("nan")
        pos_detail = {"As_mm2": 0.0, "d_mm": 0.0, "x_c_mm": 0.0, "steel_ids": [], "excluded_reason": "端部保护层退让区不参与抗弯验算"}
        neg_detail = {"As_mm2": 0.0, "d_mm": 0.0, "x_c_mm": 0.0, "steel_ids": [], "excluded_reason": "端部保护层退让区不参与抗弯验算"}
    else:
        Mud_pos, pos_detail = flexural_capacity_kNm(b, h, bottom_tension, compression_side="top")
        Mud_neg_abs, neg_detail = flexural_capacity_kNm(b, h, top_tension, compression_side="bottom")

    # 剪切有效高度取正负弯矩有效高度的较小值，避免偏不安全；若无钢筋则退化为 h-cover。
    d_candidates = [d for d in [pos_detail.get("d_mm", 0.0), neg_detail.get("d_mm", 0.0)] if d and d > 1e-6]
    h0_shear = min(d_candidates) if d_candidates else h - DEFAULTS["cover_y"]

    if in_end_blank:
        Vud = float("nan")
        shear_detail = {
            "Vc_kN": None, "Vs_kN": None, "Asv_mm2": None,
            "spacing_mm": stir_zone.get("spacing_mm"),
            "stirrup_legs": None, "stirrup_dia_mm": None,
            "excluded_reason": "端部保护层退让区不参与抗剪验算",
        }
    else:
        legs, stir_dia = stirrup_basic_info(pier_cap_cfg)
        longitudinal_as = max(
            sum(bi["area_mm2"] for bi in bottom_tension),
            sum(bi["area_mm2"] for bi in top_tension),
        )
        Vud, shear_detail = shear_capacity_kN(
            b,
            h0_shear,
            stir_zone["spacing_mm"],
            legs,
            stir_dia,
            h=h,
            calculation_span=geo.column_spacing_m * 1000.0,
            longitudinal_As_mm2=longitudinal_as,
        )

    return {
        "x_m": x_m,
        "h_mm": h,
        "b_mm": b,
        "Mud_pos_kN_m": Mud_pos,
        "Mud_neg_kN_m": -Mud_neg_abs if math.isfinite(Mud_neg_abs) else float("nan"),
        "Mud_neg_abs_kN_m": Mud_neg_abs,
        "Vud_pos_kN": Vud,
        "Vud_neg_kN": -Vud if math.isfinite(Vud) else float("nan"),
        "Vud_abs_kN": Vud,
        "top_As_mm2": neg_detail.get("As_mm2", 0.0),
        "bottom_As_mm2": pos_detail.get("As_mm2", 0.0),
        "all_bar_ids_at_section": [f"{bi['id']}×{bi['count']}({bi['segment_kind']})" for bi in bars],
        "top_steel_ids": neg_detail.get("steel_ids", []),
        "bottom_steel_ids": pos_detail.get("steel_ids", []),
        "pos_flexure_detail": pos_detail,
        "neg_flexure_detail": neg_detail,
        "compression_zone_limit_exceeded": bool(
            pos_detail.get("over_x_limit") or neg_detail.get("over_x_limit")
        ),
        "stirrup_zone": stir_zone["name"],
        "stirrup_spacing_mm": stir_zone["spacing_mm"],
        "shear_detail": shear_detail,
        "is_end_blank_section": in_end_blank,
        "is_inside_column_core": in_column_core,
        "valid_flexure_check_section": not in_end_blank,
        "valid_shear_check_section": (not in_end_blank) and (not in_column_core),
    }


def compute_capacity_envelope(xs: List[float], geo: Geometry, pier_cap_cfg: dict) -> List[dict]:
    return [capacity_at_section(x, geo, pier_cap_cfg) for x in xs]


# ============================================================
# 5. 需求-能力对比、绘图与摘要
# ============================================================
def demand_series(force_json: dict, comb_name: str) -> dict:
    moment_env, shear_env = get_force_envelopes(force_json, comb_name)
    gamma = GAMMA_0 if APPLY_GAMMA0_TO_DEMAND else 1.0
    Mmax_x, Mmax_y = envelope_to_xy(moment_env, "M_i_kN_m", "M_j_kN_m", "max", gamma)
    Mmin_x, Mmin_y = envelope_to_xy(moment_env, "M_i_kN_m", "M_j_kN_m", "min", gamma)
    Vmax_x, Vmax_y = envelope_to_xy(shear_env, "V_i_kN", "V_j_kN", "max", gamma)
    Vmin_x, Vmin_y = envelope_to_xy(shear_env, "V_i_kN", "V_j_kN", "min", gamma)
    return {
        "Mmax": (Mmax_x, Mmax_y),
        "Mmin": (Mmin_x, Mmin_y),
        "Vmax": (Vmax_x, Vmax_y),
        "Vmin": (Vmin_x, Vmin_y),
    }


def _finite_positive(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 1e-9


def add_demand_capacity_ratios(cap_rows: List[dict], demand: dict) -> List[dict]:
    Mmax_x, Mmax_y = demand["Mmax"]
    Mmin_x, Mmin_y = demand["Mmin"]
    Vmax_x, Vmax_y = demand["Vmax"]
    Vmin_x, Vmin_y = demand["Vmin"]

    out = []
    for row in cap_rows:
        x = row["x_m"]
        md_max = interpolate(x, Mmax_x, Mmax_y)
        md_min = interpolate(x, Mmin_x, Mmin_y)
        vd_max = interpolate(x, Vmax_x, Vmax_y)
        vd_min = interpolate(x, Vmin_x, Vmin_y)

        util_m_pos = None
        util_m_neg = None
        util_v = None

        if row.get("valid_flexure_check_section") and _finite_positive(row.get("Mud_pos_kN_m")):
            util_m_pos = max(md_max, 0.0) / row["Mud_pos_kN_m"]
        if row.get("valid_flexure_check_section") and _finite_positive(abs(row.get("Mud_neg_kN_m", float("nan")))):
            util_m_neg = abs(min(md_min, 0.0)) / abs(row["Mud_neg_kN_m"])
        if row.get("valid_shear_check_section") and _finite_positive(row.get("Vud_abs_kN")):
            util_v = max(abs(vd_max), abs(vd_min)) / row["Vud_abs_kN"]

        r = dict(row)
        r.update({
            "demand_Mmax_kN_m": md_max,
            "demand_Mmin_kN_m": md_min,
            "demand_Vmax_kN": vd_max,
            "demand_Vmin_kN": vd_min,
            "util_M_pos": util_m_pos,
            "util_M_neg": util_m_neg,
            "util_V": util_v,
            "is_M_pos_ok": None if util_m_pos is None else util_m_pos <= 1.0,
            "is_M_neg_ok": None if util_m_neg is None else util_m_neg <= 1.0,
            "is_V_ok": None if util_v is None else util_v <= 1.0,
        })
        out.append(r)
    return out


def _rows_with_metric(rows: List[dict], metric: str) -> List[dict]:
    return [r for r in rows if isinstance(r.get(metric), (int, float)) and math.isfinite(float(r[metric]))]


def find_control_rows(rows: List[dict]) -> dict:
    if not rows:
        return {}
    out = {}
    mpos = _rows_with_metric(rows, "util_M_pos")
    mneg = _rows_with_metric(rows, "util_M_neg")
    shear = _rows_with_metric(rows, "util_V")
    if mpos:
        out["max_positive_moment_utilization"] = max(mpos, key=lambda r: r["util_M_pos"])
    if mneg:
        out["max_negative_moment_utilization"] = max(mneg, key=lambda r: r["util_M_neg"])
    if shear:
        out["max_shear_utilization"] = max(shear, key=lambda r: r["util_V"])
    return out


def slim_control_row(row: dict) -> dict:
    keys = [
        "x_m", "h_mm", "Mud_pos_kN_m", "Mud_neg_kN_m", "Vud_abs_kN",
        "demand_Mmax_kN_m", "demand_Mmin_kN_m", "demand_Vmax_kN", "demand_Vmin_kN",
        "util_M_pos", "util_M_neg", "util_V",
        "top_As_mm2", "bottom_As_mm2", "stirrup_zone", "stirrup_spacing_mm",
        "top_steel_ids", "bottom_steel_ids"
    ]
    result = {k: row.get(k) for k in keys}
    pos_detail = row.get("pos_flexure_detail") if isinstance(row.get("pos_flexure_detail"), dict) else {}
    neg_detail = row.get("neg_flexure_detail") if isinstance(row.get("neg_flexure_detail"), dict) else {}
    shear_detail = row.get("shear_detail") if isinstance(row.get("shear_detail"), dict) else {}
    b = float(row.get("b_mm") or 0.0)
    h = float(row.get("h_mm") or 0.0)
    gross_area = b * h
    result["compression_zone_limit_exceeded"] = bool(
        row.get("compression_zone_limit_exceeded")
        or pos_detail.get("over_x_limit")
        or neg_detail.get("over_x_limit")
    )
    result["longitudinal_reinforcement_ratio_percent"] = {
        "top": 100.0 * float(row.get("top_As_mm2") or 0.0) / gross_area if gross_area > 0 else None,
        "bottom": 100.0 * float(row.get("bottom_As_mm2") or 0.0) / gross_area if gross_area > 0 else None,
        "definition": "As/(b*h)×100%，作为概念设计诊断量，不代替构造条文合规判定。",
    }
    result["formula_trace"] = {
        "positive_flexure": pos_detail.get("formula_id"),
        "negative_flexure": neg_detail.get("formula_id"),
        "shear": shear_detail.get("formula_ids") or [],
    }
    return result


def add_column_core_shading(ax, geo: Geometry) -> None:
    for cx in column_centers_m(geo):
        x1 = cx - geo.column_diameter_m / 2.0
        x2 = cx + geo.column_diameter_m / 2.0
        ax.axvspan(x1, x2, alpha=0.08, color="gray", linewidth=0)


def _plot_nan_safe(ax, xs: List[float], ys: List[float], *args, **kwargs) -> None:
    ax.plot(xs, ys, *args, **kwargs)


def plot_moment_overlay(cap_rows: List[dict], demand: dict, out_path: Path, geo: Geometry) -> None:
    xs = [r["x_m"] for r in cap_rows]
    Mud_pos = [r["Mud_pos_kN_m"] for r in cap_rows]
    Mud_neg = [r["Mud_neg_kN_m"] for r in cap_rows]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(*demand["Mmax"], label="Demand Mmax (OpenSees ULS)", linewidth=1.4)
    ax.plot(*demand["Mmin"], label="Demand Mmin (OpenSees ULS)", linewidth=1.4)
    _plot_nan_safe(ax, xs, Mud_pos, label="Capacity +Mud(x)", linewidth=2.0, linestyle="--")
    _plot_nan_safe(ax, xs, Mud_neg, label="Capacity -Mud(x)", linewidth=2.0, linestyle="--")
    ax.axhline(0.0, linewidth=1.0, color="black")
    add_column_core_shading(ax, geo)

    ax.set_title("盖梁抗弯承载能力包络与 OpenSees 作用效应包络对比")
    ax.set_xlabel("x along cap beam centerline (m)")
    ax.set_ylabel("Bending moment / capacity (kN·m)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    # 与原 OpenSees 内力图保持一致：跨中正弯矩视觉向下，支点负弯矩视觉向上
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_shear_overlay(cap_rows: List[dict], demand: dict, out_path: Path, geo: Geometry) -> None:
    xs = [r["x_m"] for r in cap_rows]
    Vud_pos = [r["Vud_pos_kN"] for r in cap_rows]
    Vud_neg = [r["Vud_neg_kN"] for r in cap_rows]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(*demand["Vmax"], label="Demand Vmax (OpenSees ULS)", linewidth=1.4)
    ax.plot(*demand["Vmin"], label="Demand Vmin (OpenSees ULS)", linewidth=1.4)
    _plot_nan_safe(ax, xs, Vud_pos, label="Capacity +Vud(x)", linewidth=2.0, linestyle="--")
    _plot_nan_safe(ax, xs, Vud_neg, label="Capacity -Vud(x)", linewidth=2.0, linestyle="--")
    ax.axhline(0.0, linewidth=1.0, color="black")
    add_column_core_shading(ax, geo)

    ax.set_title("盖梁抗剪承载能力包络与 OpenSees 作用效应包络对比")
    ax.set_xlabel("x along cap beam centerline (m)")
    ax.set_ylabel("Shear force / capacity (kN)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_utilization(cap_rows: List[dict], out_path: Path, geo: Geometry) -> None:
    xs = [r["x_m"] for r in cap_rows]
    def ys(key: str) -> List[float]:
        out = []
        for r in cap_rows:
            v = r.get(key)
            out.append(float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else float("nan"))
        return out
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(xs, ys("util_M_pos"), label="M+ demand/capacity", linewidth=1.6)
    ax.plot(xs, ys("util_M_neg"), label="M- demand/capacity", linewidth=1.6)
    ax.plot(xs, ys("util_V"), label="V demand/capacity", linewidth=1.6)
    ax.axhline(1.0, linewidth=1.2, color="black", linestyle="--", label="limit = 1.0")
    add_column_core_shading(ax, geo)
    ax.set_title("盖梁承载能力利用率沿程分布")
    ax.set_xlabel("x along cap beam centerline (m)")
    ax.set_ylabel("Demand / Capacity")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def sanitize_for_json(obj: Any) -> Any:
    """将 NaN/Inf 转为 None，保证输出 JSON 合法。"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _all_valid_ok(rows: List[dict], key: str) -> bool:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return all(v is True for v in vals) if vals else True


def _all_compression_zone_limits_ok(rows: List[dict]) -> bool:
    checked = [
        row
        for row in rows
        if row.get("valid_flexure_check_section")
        and (
            row.get("pos_flexure_detail", {}).get("As_mm2", 0.0) > 0
            or row.get("neg_flexure_detail", {}).get("As_mm2", 0.0) > 0
        )
    ]
    return bool(checked) and not any(row.get("compression_zone_limit_exceeded") for row in checked)


# ============================================================
# 6. 主程序
# ============================================================
def main() -> None:
    if not REINFORCEMENT_YAML.exists():
        raise FileNotFoundError(f"未找到配筋 YAML：{REINFORCEMENT_YAML}")

    maybe_run_opensees_script()

    reinf = load_yaml(REINFORCEMENT_YAML)
    pier_cap_cfg = reinf["reinforcement"]["pier_cap"]
    force_json = load_json(OPENSEES_FORCE_JSON)
    geo = geometry_from_force_json(force_json)

    xs = build_capacity_scan_xs(geo, force_json, pier_cap_cfg)
    cap_rows = compute_capacity_envelope(xs, geo, pier_cap_cfg)
    demand = demand_series(force_json, COMBINATION_NAME)
    cap_rows = add_demand_capacity_ratios(cap_rows, demand)

    controls = find_control_rows(cap_rows)
    summary = {
        "combination_name": COMBINATION_NAME,
        "gamma0_applied_to_demand": GAMMA_0 if APPLY_GAMMA0_TO_DEMAND else 1.0,
        "material_design_values": {
            "fcd_MPa": DEFAULTS["fcd"],
            "ftd_MPa": DEFAULTS["ftd"],
            "fsd_MPa": DEFAULTS["fsd"],
            "fsvd_MPa": DEFAULTS["fsvd"],
            "steel_grade": DEFAULTS["steel_grade"],
            "concrete_strength_grade": DEFAULTS["concrete_strength_grade"],
            "fcu_k_MPa": DEFAULTS["fcu_k"],
            "alpha1_for_cap_shear": DEFAULTS["alpha1_for_cap_shear"],
            "alpha1_source": "concept_design_default",
        },
        "geometry": {
            "cap_length_m": geo.cap_length_m,
            "cap_width_m": geo.cap_width_m,
            "cap_height_mid_m": geo.cap_height_mid_m,
            "cap_height_end_m": geo.cap_height_end_m,
            "cantilever_length_m": geo.cantilever_length_m,
            "column_spacing_m": geo.column_spacing_m,
            "column_count": geo.column_count,
            "column_diameter_m": geo.column_diameter_m,
        },
        "control_sections": {name: slim_control_row(row) for name, row in controls.items()},
        "overall_check": {
            "M_pos_ok": _all_valid_ok(cap_rows, "is_M_pos_ok"),
            "M_neg_ok": _all_valid_ok(cap_rows, "is_M_neg_ok"),
            "V_ok": _all_valid_ok(cap_rows, "is_V_ok"),
            "compression_zone_ok": _all_compression_zone_limits_ok(cap_rows),
            "all_ok": (
                _all_valid_ok(cap_rows, "is_M_pos_ok")
                and _all_valid_ok(cap_rows, "is_M_neg_ok")
                and _all_valid_ok(cap_rows, "is_V_ok")
                and _all_compression_zone_limits_ok(cap_rows)
            ),
            "ignored_sections_rule": {
                "flexure": "端部保护层退让区不参与正截面抗弯验算",
                "shear": "端部保护层退让区与墩柱支承宽度范围内不作为抗剪控制截面",
            },
        },
        "formula_coverage": {
            "executed": [
                FLEXURE_ID,
                SHEAR_LIMIT_ID,
                SHEAR_REINFORCED_ID,
            ],
            "diagnostic_only": ["纵向钢筋毛截面配筋率 As/(b*h)"],
            "not_covered": ["裂缝", "挠度", "疲劳", "抗震", "普通钢筋混凝土构件构造配筋完整审查"],
        },
        "note": "概念设计验算：端部退让区用 NaN 断开；斜向弯起段不直接计入正截面抗弯；柱身支承宽度内不作为抗剪控制截面。正截面抗弯及盖梁斜截面抗剪采用固定 Python 公式执行器，并保留规范公式 ID。",
    }

    # 输出数据
    save_json(sanitize_for_json({"capacity_sections": cap_rows}), OUTPUT_DIR / "capacity_envelope_sections.json")
    save_json(sanitize_for_json(summary), OUTPUT_DIR / "capacity_check_summary.json")

    # 绘图
    plot_moment_overlay(cap_rows, demand, OUTPUT_DIR / "moment_demand_vs_capacity_overlay.png", geo)
    plot_shear_overlay(cap_rows, demand, OUTPUT_DIR / "shear_demand_vs_capacity_overlay.png", geo)
    plot_utilization(cap_rows, OUTPUT_DIR / "demand_capacity_utilization.png", geo)

    print("\n========== 盖梁承载能力包络计算完成 ==========")
    print(f"输出目录：{OUTPUT_DIR}")
    print("已生成：")
    print("  1) moment_demand_vs_capacity_overlay.png")
    print("  2) shear_demand_vs_capacity_overlay.png")
    print("  3) demand_capacity_utilization.png")
    print("  4) capacity_envelope_sections.json")
    print("  5) capacity_check_summary.json")
    print("\n控制截面摘要：")
    for name, row in summary["control_sections"].items():
        def fmt(v):
            return "-" if v is None else f"{v:.3f}"
        print(f"- {name}: x={row['x_m']:.3f} m, util_M+={fmt(row['util_M_pos'])}, util_M-={fmt(row['util_M_neg'])}, util_V={fmt(row['util_V'])}")
    print(f"\n总体是否通过：{summary['overall_check']['all_ok']}")


if __name__ == "__main__":
    main()
