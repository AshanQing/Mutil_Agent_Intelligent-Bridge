"""确定性初设可视化器（initial_visualizer）。

职责：
- render_plan_overlay：以 pred 地形栅格 + PGW 地理参考为底图，叠加平曲线
  路线（K_plane.json 各段起终点 XY）与各桥位墩台序列，输出布跨平面叠加示意图；
- render_profile：路线纵断数据（设计线/地形线高程序列，来自布跨输入的
  input_data.json K[Z].纵断面结构）可用时，按 zdm_draw_k12.360_k13.940 Final.py
  的逻辑绘制真实纵断面——设计线含竖曲线恢复、地面线、各桥位墩台与桥面线、
  跨径标注；无纵断数据时退化为墩高柱示意并注明；
- build_catalog_views：初始方案（最新 design_run 布跨结果）与最终修正方案
  （final_layout_result.json）分别渲染到 catalog_views/{initial,final}/，
  互不覆盖。

取舍：
- 平面叠加不做障碍物几何叠加（障碍可视化由 obstacle_semantic 的 merged_vis
  现成图承担）；坐标基准（PGW）缺失或与平曲线不一致时不强行叠加。

纯 matplotlib（Agg）输出 PNG，不依赖本地 Cairo。
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .result_catalog import _first, _latest, _load_optional_json

# 蓝图风格配色（与 GUI 视觉一致）
COLOR_BG_DARK = "#0B1F3A"
COLOR_ROUTE = "#00E5FF"  # 青色路线
COLOR_DESIGN = "#FFD54A"  # 黄色设计信息
COLOR_CONFLICT = "#FF5252"  # 红色冲突（桥台端部）
COLOR_PASS = "#69F0AE"  # 绿色通过状态
COLOR_AXIS = "#9FB6D4"
COLOR_TEXT = "#E8EEF6"

_KM = 1000.0


def parse_k(value: Any) -> float:
    """解析桩号文本为米：'K41+080.000' -> 41080.0；支持数字直接返回。"""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        clean = str(value).strip().upper().replace("K", "").replace(" ", "")
        parts = clean.split("+")
        if len(parts) == 2:
            return float(parts[0]) * _KM + float(parts[1])
        number = float(clean)
        if number < _KM:
            return number * _KM
        return number
    except Exception:
        return 0.0


def format_k(meters: float) -> str:
    km = int(meters // _KM)
    remainder = meters - km * _KM
    return f"K{km}+{remainder:07.3f}".rstrip("0").rstrip(".")


def plane_segments(plane_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 K_plane.json 平曲线结构转为有序段（起终点含 XY 与桩号）。"""
    curves = plane_json.get("平曲线结构") if isinstance(plane_json, dict) else None
    segments: List[Dict[str, Any]] = []
    if not isinstance(curves, list):
        return segments
    for item in curves:
        if not isinstance(item, dict):
            continue
        start, end = item.get("起点"), item.get("终点")
        if not isinstance(start, dict) or not isinstance(end, dict):
            continue
        try:
            segments.append(
                {
                    "k0": parse_k(start.get("桩号")),
                    "k1": parse_k(end.get("桩号")),
                    "x0": float(start["X"]),
                    "y0": float(start["Y"]),
                    "x1": float(end["X"]),
                    "y1": float(end["Y"]),
                    "name": str(item.get("曲线类型") or ""),
                }
            )
        except Exception:
            continue
    segments.sort(key=lambda s: s["k0"])
    return segments


def station_to_xy(plane_json: Dict[str, Any], station_m: float) -> Optional[Tuple[float, float]]:
    """桩号 -> 平面 XY：在平曲线段内按桩号线性插值（示意图精度）。"""
    segments = plane_segments(plane_json)
    if not segments:
        return None
    for seg in segments:
        if seg["k0"] <= station_m <= seg["k1"]:
            span = seg["k1"] - seg["k0"]
            if span <= 0:
                continue
            t = (station_m - seg["k0"]) / span
            x = seg["x0"] + t * (seg["x1"] - seg["x0"])
            y = seg["y0"] + t * (seg["y1"] - seg["y0"])
            return (x, y)
    # 越界：取首末段端点方向延伸（示意图允许轻微外延）
    if station_m < segments[0]["k0"]:
        seg = segments[0]
    elif station_m > segments[-1]["k1"]:
        seg = segments[-1]
    else:
        return None
    span = seg["k1"] - seg["k0"]
    if span <= 0:
        return None
    t = (station_m - seg["k0"]) / span
    return (seg["x0"] + t * (seg["x1"] - seg["x0"]), seg["y0"] + t * (seg["y1"] - seg["y0"]))


def _extract_elevation(note: Any) -> Dict[str, Optional[float]]:
    text = str(note or "")
    result: Dict[str, Optional[float]] = {"design": None, "ground": None}
    match = re.search(r"设计高约?\s*([\d.]+)\s*m", text)
    if match:
        result["design"] = float(match.group(1))
    match = re.search(r"地面高约?\s*([\d.]+)\s*m", text)
    if match:
        result["ground"] = float(match.group(1))
    return result


def pier_rows(layout_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从布跨方案 JSON 提取各桥位墩位（桩号/类型/墩高/端部高程）。"""
    bridges: List[Dict[str, Any]] = []
    bridge_list = layout_json.get("桥位列表") if isinstance(layout_json, dict) else None
    if not isinstance(bridge_list, list):
        return bridges
    for index, bridge in enumerate(bridge_list):
        if not isinstance(bridge, dict):
            continue
        scheme = bridge.get("统一布跨方案")
        if not isinstance(scheme, dict):
            scheme = bridge.get("分幅布跨方案列表")
            if isinstance(scheme, list) and scheme:
                scheme = scheme[0]
            if not isinstance(scheme, dict):
                continue
        piers: List[Dict[str, Any]] = []
        for row in scheme.get("墩位与墩高") or []:
            if not isinstance(row, dict):
                continue
            elev = _extract_elevation(row.get("墩位校核说明"))
            piers.append(
                {
                    "no": str(row.get("墩号") or ""),
                    "station": parse_k(row.get("桩号")),
                    "type": str(row.get("墩位类型") or ""),
                    "height_l": _as_float(row.get("左幅墩高(m)")),
                    "height_r": _as_float(row.get("右幅墩高(m)")),
                    "design_h": elev["design"],
                    "ground_h": elev["ground"],
                }
            )
        piers.sort(key=lambda p: p["station"])
        info = scheme.get("设桥信息")
        bridges.append(
            {
                "no": str(bridge.get("桥位编号") or (index + 1)),
                "span_text": str(info.get("跨径组合") or "") if isinstance(info, dict) else "",
                "piers": piers,
            }
        )
    return bridges


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
        return number if number == number else None  # noqa: PLR0124
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# PGW 世界文件
# ---------------------------------------------------------------------------


def read_pgw(pgw_path: Any) -> Optional[Tuple[float, float, float, float, float, float]]:
    """读取 ESRI world file（六参数 A B C D E F 布局）。
    返回 (A, D, B, E, C, F) 即仿射 X=A*x+B*y+C; Y=D*x+E*y+F。
    """
    try:
        values = [float(line.strip()) for line in open(pgw_path, encoding="utf-8") if line.strip()]
        if len(values) < 6:
            return None
        a, d, b, e, c, f = values[:6]
        return (a, d, b, e, c, f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# matplotlib 渲染
# ---------------------------------------------------------------------------


def _setup_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def render_plan_overlay(
    *,
    terrain_png: str,
    pgw_path: str,
    plane_json_path: str,
    layout_json_path: str,
    out_png: str,
    variant: str = "initial",
) -> Optional[str]:
    """渲染平面叠加示意 PNG 到 out_png；数据不足或失败返回 None。"""
    if not all(os.path.isfile(p) for p in (terrain_png, pgw_path, plane_json_path, layout_json_path)):
        return None
    import json

    try:
        with open(plane_json_path, encoding="utf-8") as f:
            plane_json = json.load(f)
        with open(layout_json_path, encoding="utf-8") as f:
            layout_json = json.load(f)
        geo = read_pgw(pgw_path)
        if not geo:
            return None
        segments = plane_segments(plane_json)
        bridges = pier_rows(layout_json)
        if not segments or not bridges:
            return None

        _setup_style()
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        for font in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "simsun.ttc"):
            try:
                font_manager.findfont(font_manager.FontProperties(fname=font), fallback_to_default=False)
                break
            except Exception:
                continue
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]

        from PIL import Image

        with Image.open(terrain_png) as terrain:
            width, height = terrain.size
            terrain_rgb = terrain.convert("L")

        a, _d, _b, e, c, f = geo
        img_left, img_right = c, c + a * width
        img_bottom, img_top = f + e * height, f

        # 数据 bbox：路线端点 + 所有墩台坐标（视图以数据为准，墩台/路线必在图内）
        data_xs: List[float] = []
        data_ys: List[float] = []
        for seg in segments:
            data_xs.extend([seg["x0"], seg["x1"]])
            data_ys.extend([seg["y0"], seg["y1"]])
        pier_pts: List[Tuple[float, float]] = []
        for bridge in bridges:
            for pier in bridge["piers"]:
                xy = station_to_xy(plane_json, pier["station"])
                if xy:
                    pier_pts.append(xy)
                    data_xs.append(xy[0])
                    data_ys.append(xy[1])
        if not data_xs or not data_ys:
            return None
        x_min, x_max = min(data_xs), max(data_xs)
        y_min, y_max = min(data_ys), max(data_ys)
        span_x = max(x_max - x_min, 1.0)
        span_y = max(y_max - y_min, 1.0)
        view = (
            x_min - 0.06 * span_x,
            x_max + 0.06 * span_x,
            y_min - 0.08 * span_y,
            y_max + 0.10 * span_y,
        )

        import matplotlib.patheffects as path_effects

        fig, ax = plt.subplots(figsize=(10.4, 6.4))
        fig.patch.set_facecolor(COLOR_BG_DARK)
        ax.set_facecolor(COLOR_BG_DARK)

        # 地形底图仅在其基准覆盖大部分数据区时叠加；否则如实提示不强行叠加
        overlap_w = max(0.0, min(view[1], img_right) - max(view[0], img_left))
        overlap_h = max(0.0, min(view[3], img_top) - max(view[2], img_bottom))
        view_area = (view[1] - view[0]) * (view[3] - view[2])
        overlap_area = overlap_w * overlap_h
        show_terrain = overlap_area >= 0.6 * view_area
        if show_terrain:
            ax.imshow(
                terrain_rgb,
                extent=(img_left, img_right, img_bottom, img_top),
                cmap="gray",
                alpha=0.4,
                aspect="auto",
            )
        ax.set_xlim(view[0], view[1])
        ax.set_ylim(view[2], view[3])

        # 平曲线路线（各段端点折线）：白描边 + 亮青主线
        xs = [seg["x0"] for seg in segments] + [segments[-1]["x1"]]
        ys = [seg["y0"] for seg in segments] + [segments[-1]["y1"]]
        ax.plot(xs, ys, color="#FFFFFF", linewidth=4.6, alpha=0.9, solid_capstyle="round", zorder=3)
        ax.plot(xs, ys, color=COLOR_ROUTE, linewidth=2.4, solid_capstyle="round", zorder=4, label="路线中线")

        # 各桥位墩台
        for bridge in bridges:
            piers = bridge["piers"]
            for pier in piers:
                xy = station_to_xy(plane_json, pier["station"])
                if not xy:
                    continue
                is_abutment = "台" in pier["type"] or "0" in pier["no"]
                color = COLOR_CONFLICT if is_abutment else COLOR_DESIGN
                ax.plot([xy[0]], [xy[1]], marker="^" if is_abutment else "o", markersize=15,
                        color=color, markeredgecolor="#FFFFFF", markeredgewidth=1.2, zorder=5)
                offset = 14 if is_abutment else 12
                ax.annotate(
                    pier["no"],
                    xy,
                    textcoords="offset points",
                    xytext=(0, offset),
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color=COLOR_TEXT,
                    zorder=6,
                    path_effects=[path_effects.withStroke(linewidth=2.2, foreground="#0B1F3A")],
                )
            stations = [p["station"] for p in piers]
            if len(stations) >= 2:
                start_xy = station_to_xy(plane_json, min(stations))
                end_xy = station_to_xy(plane_json, max(stations))
                if start_xy and end_xy:
                    ax.plot(
                        [start_xy[0], end_xy[0]],
                        [start_xy[1], end_xy[1]],
                        color=COLOR_PASS,
                        linewidth=2.0,
                        linestyle="--",
                        alpha=0.95,
                        zorder=4,
                    )

        ax.set_title(f"布跨平面叠加示意（{variant}）", color=COLOR_TEXT, fontsize=12)
        ax.tick_params(colors=COLOR_AXIS, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLOR_AXIS)
        if not show_terrain:
            ax.text(
                0.5, -0.06,
                "地形栅格基准与布跨坐标不一致，未叠加底图（仅显示路线与墩台）",
                transform=ax.transAxes, ha="center", fontsize=8,
                color=COLOR_AXIS,
            )
        ax.legend(loc="upper right", fontsize=8, facecolor=COLOR_BG_DARK, edgecolor=COLOR_AXIS,
                  labelcolor=COLOR_TEXT)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_png
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 纵断面真实渲染：按 zdm_draw_k12.360_k13.940 Final.py 的逻辑移植
# （设计线竖曲线恢复 + 地面线 + 桥位/墩台叠绘）。数据来自路线纵断
# （input_data.json 的 K[Z].纵断面结构），非墩位推测示意。
# ---------------------------------------------------------------------------


def build_vertical_profile(design_list):
    """按设计线高程序列恢复纵断设计线函数（移植自 zdm_draw 脚本）。

    - 控制点按桩号升序，相邻点构成直坡段；
    - 内部点视为变坡点 PVI，若带有效竖曲线半径 R 且前后坡度不同，则按
      对称抛物线插入竖曲线（长度 L=|R·Δg|，起终点不超过相邻控制点）；
    - 两端点仅作边界控制点，不展开竖曲线。

    返回 callable(x_query: 桩号数组/标量) -> 高程（numpy 数组）。
    """
    import numpy as np

    points = sorted(design_list, key=lambda d: float(d["桩号"]))
    xs = np.array([float(p["桩号"]) for p in points], dtype=float)
    ys = np.array([float(p["高程"]) for p in points], dtype=float)
    radii = np.array([float(p.get("R") or 0.0) for p in points], dtype=float)
    if len(xs) < 2:
        raise ValueError("设计线高程序列至少需要 2 个点。")

    grades = np.diff(ys) / np.diff(xs)
    vc_list: List[Dict[str, Any]] = []
    for index in range(1, len(xs) - 1):
        x_pvi, y_pvi = xs[index], ys[index]
        g1, g2 = float(grades[index - 1]), float(grades[index])
        radius = float(radii[index])
        delta = g2 - g1
        if radius <= 0 or abs(delta) < 1e-12:
            continue
        curve_length = abs(radius * delta)
        x_vc1 = max(x_pvi - curve_length / 2.0, float(xs[index - 1]))
        x_vc2 = min(x_pvi + curve_length / 2.0, float(xs[index + 1]))
        if x_vc2 - x_vc1 <= 1e-8:
            continue
        y_vc1 = y_pvi - g1 * (x_pvi - x_vc1)
        vc_list.append(
            {
                "x_vc1": x_vc1,
                "x_vc2": x_vc2,
                "y_vc1": y_vc1,
                "g1": g1,
                "g2": g2,
                "delta": delta,
                "length": x_vc2 - x_vc1,
            }
        )

    def profile_func(x_query):
        raw = np.asarray(x_query, dtype=float)
        scalar_input = raw.ndim == 0
        xq = np.atleast_1d(raw)
        output = np.zeros_like(xq, dtype=float)
        for j, x in enumerate(xq):
            handled = False
            for vc in vc_list:
                if vc["x_vc1"] <= x <= vc["x_vc2"]:
                    dx = x - vc["x_vc1"]
                    output[j] = vc["y_vc1"] + vc["g1"] * dx + (
                        vc["delta"] / (2.0 * vc["length"])
                    ) * dx * dx
                    handled = True
                    break
            if handled:
                continue
            if x <= xs[0]:
                output[j] = ys[0] + grades[0] * (x - xs[0])
            elif x >= xs[-1]:
                output[j] = ys[-1] + grades[-1] * (x - xs[-1])
            else:
                for i in range(len(xs) - 1):
                    if xs[i] <= x <= xs[i + 1]:
                        output[j] = ys[i] + grades[i] * (x - xs[i])
                        break
        return float(output[0]) if scalar_input else output

    return profile_func


def _ground_interpolator(ground_list):
    """地面线线性插值（桩号/高程点列），越界按端点坡度外推（同 zdm 语义）。"""
    import numpy as np

    points = sorted(ground_list, key=lambda d: float(d["桩号"]))
    xs = np.array([float(p["桩号"]) for p in points], dtype=float)
    ys = np.array([float(p["高程"]) for p in points], dtype=float)
    if len(xs) < 2:
        raise ValueError("地形线高程序列至少需要 2 个点。")

    def func(x_query):
        xq = np.asarray(x_query, dtype=float)
        left = ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xq - xs[0])
        right = ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (xq - xs[-1])
        inner = np.interp(xq, xs, ys)
        return np.where(xq < xs[0], left, np.where(xq > xs[-1], right, inner))

    return func


def _render_zdm_profile(
    *, layout_json_path: str, vertical: Dict[str, Any], out_png: str, variant: str
) -> Optional[str]:
    """按 zdm_draw_k12.360_k13.940 Final.py 主流程绘制真实纵断面：
    地面线 + 设计线（含竖曲线）+ 各桥位墩台与桥面线 + 跨径标注。
    """
    import json

    try:
        with open(layout_json_path, encoding="utf-8") as f:
            layout_json = json.load(f)
    except Exception:
        return None
    design_list = vertical.get("设计线高程序列") or []
    ground_list = vertical.get("地形线高程序列") or []
    if len(design_list) < 2 or len(ground_list) < 2:
        return None

    import numpy as np

    try:
        f_design = build_vertical_profile(design_list)
        f_ground = _ground_interpolator(ground_list)
    except Exception:
        return None

    raw_bridges = [b for b in pier_rows(layout_json) if len(b["piers"]) >= 2]
    if not raw_bridges:
        return None
    bridge_colors = ["#C0392B", "#1F5C99", "#B8860B"]
    bridges = []
    for index, bridge in enumerate(raw_bridges):
        stations = [p["station"] for p in bridge["piers"]]
        bridges.append(
            {
                "id": bridge["no"],
                "desc": bridge["span_text"],
                "start": min(stations),
                "end": max(stations),
                "color": bridge_colors[index % len(bridge_colors)],
                "piers": bridge["piers"],
            }
        )

    try:
        _setup_style()
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # 横轴范围以地形线（地面线）序列覆盖区间为基准并保证包住各桥位；
        # 设计线序列可能含桥位外的竖曲线控制点（如 K39+900），不应拉宽横轴。
        bridge_min_k = min(bridge["start"] for bridge in bridges)
        bridge_max_k = max(bridge["end"] for bridge in bridges)
        start_k = min(float(ground_list[0]["桩号"]), bridge_min_k)
        end_k = max(float(ground_list[-1]["桩号"]), bridge_max_k)
        x_dense = np.linspace(start_k, end_k, 2400)
        y_ground = f_ground(x_dense)
        y_design = f_design(x_dense)
        y_min = float(np.min(np.concatenate([y_ground, y_design]))) - 8
        y_max = float(np.max(np.concatenate([y_ground, y_design]))) + 12

        fig, ax = plt.subplots(figsize=(12.4, 5.4), dpi=150)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        ax.fill_between(x_dense, y_min, y_ground, color="#E7E3D7", alpha=0.6,
                        linewidth=0, zorder=1)
        ax.plot(x_dense, y_ground, color="#6F7F70", linewidth=1.4, zorder=2,
                label="地面线")
        ax.plot(x_dense, y_design, color="#333333", linewidth=1.2, zorder=3,
                linestyle="--", label="设计线（含竖曲线）")

        # 墩台与桥面线
        for bridge in bridges:
            color = bridge["color"]
            for pier in bridge["piers"]:
                station = pier["station"]
                y_top = float(f_design(np.array([station]))[0])
                y_bot = float(f_ground(np.array([station]))[0])
                if y_bot > y_top - 0.5:
                    y_bot = y_top - 0.5
                ax.vlines([station], y_bot, y_top, colors=color, linewidth=1.1,
                          zorder=5, capstyle="butt")
            deck_x = np.linspace(bridge["start"], bridge["end"], 600)
            deck_y = f_design(deck_x)
            ax.plot(deck_x, deck_y, color=color, linewidth=2.6, zorder=6,
                    solid_capstyle="butt", solid_joinstyle="miter")

        for bridge in bridges:
            mid_k = (bridge["start"] + bridge["end"]) / 2
            mid_y = float(f_design(np.array([mid_k]))[0])
            label = bridge["desc"] if bridge["desc"] else f"桥位 {bridge['id']}"
            ax.annotate(
                label,
                xy=(mid_k, mid_y + 0.6),
                xytext=(mid_k, mid_y + max(6, (y_max - y_min) * 0.045)),
                textcoords="data",
                ha="center", va="bottom",
                fontsize=8.5, color=bridge["color"],
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=bridge["color"],
                                lw=0.7, shrinkA=4, shrinkB=4),
                zorder=10,
            )

        ax.set_xlim(start_k, end_k)
        ax.set_ylim(y_min, y_max)
        xticks = np.arange(float(np.floor(start_k / 200) * 200), end_k + 1, 200)
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_k(float(x)) for x in xticks], fontsize=8)
        ax.tick_params(colors="#3a3a3c", labelsize=8)
        ax.set_xlabel("桩号", fontsize=10)
        ax.set_ylabel("高程 (m)", fontsize=10)
        ax.set_title(f"桥位与跨径布置纵断面（{variant}方案）", fontsize=12, pad=10)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#d1d1d6")

        legend_handles = [
            Line2D([0], [0], color="#6F7F70", lw=1.5, label="地面线"),
            Line2D([0], [0], color="#333333", lw=1.2, linestyle="--", label="设计线（含竖曲线）"),
        ]
        for index, bridge in enumerate(bridges[:3]):
            legend_handles.append(
                Line2D([0], [0], color=bridge["color"], lw=2.6,
                       label=f"桥位 {bridge['id']} 布跨")
            )
        ax.legend(handles=legend_handles, loc="upper left", frameon=False,
                  fontsize=9)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        return out_png
    except Exception:
        return None


def render_profile(
    *, layout_json_path: str, out_png: str, variant: str = "final",
    vertical: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """渲染各桥位纵断面 PNG。

    vertical（含"设计线高程序列"/"地形线高程序列"，来自路线纵断数据）存在时，
    按 zdm_draw 脚本逻辑绘制真实纵断面（设计线含竖曲线 + 地面线 + 桥位叠绘）；
    无纵断数据时退化为墩高柱示意并注明"示意"。
    """
    if vertical:
        return _render_zdm_profile(
            layout_json_path=layout_json_path,
            vertical=vertical,
            out_png=out_png,
            variant=variant,
        )
    if not os.path.isfile(layout_json_path):
        return None
    import json

    try:
        with open(layout_json_path, encoding="utf-8") as f:
            layout_json = json.load(f)
        bridges = pier_rows(layout_json)
        bridges = [b for b in bridges if len(b["piers"]) >= 2]
        if not bridges:
            return None

        _setup_style()
        import matplotlib.pyplot as plt

        n = len(bridges)
        fig, axes = plt.subplots(n, 1, figsize=(10.4, 3.1 * n), squeeze=False)
        fig.patch.set_facecolor(COLOR_BG_DARK)
        for row, bridge in enumerate(bridges):
            ax = axes[row][0]
            ax.set_facecolor(COLOR_BG_DARK)
            piers = bridge["piers"]
            stations = [p["station"] for p in piers]
            heights = [p["height_l"] or p["height_r"] or 0.0 for p in piers]
            kinds = [p["type"] for p in piers]

            # 墩高柱
            bar_colors = [COLOR_CONFLICT if "台" in t else COLOR_ROUTE for t in kinds]
            ax.bar(stations, heights, width=(max(stations) - min(stations)) / len(stations) * 0.6,
                   color=bar_colors, alpha=0.55, label="墩高 (m)")
            import matplotlib.patheffects as path_effects

            for p, h in zip(piers, heights):
                if h > 0:
                    ax.annotate(
                        f"{h:.1f}", (p["station"], h), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=9, color=COLOR_TEXT,
                        path_effects=[path_effects.withStroke(linewidth=2.0, foreground="#0B1F3A")],
                    )

            # 示意高程线：两端桥台含设计/地面高时内插
            design_pts = [(p["station"], p["design_h"]) for p in piers if p["design_h"] is not None]
            ground_pts = [(p["station"], p["ground_h"]) for p in piers if p["ground_h"] is not None]
            if len(design_pts) >= 2:
                xs, ys = zip(*design_pts)
                ax.plot(xs, ys, color=COLOR_DESIGN, linewidth=1.4, label="设计高（示意）")
                # 反推地面线：设计高（内插） - 墩高
                gx: List[float] = []
                gy: List[float] = []
                for p, h in zip(piers, heights):
                    interp = _linear_interp(dict(zip(xs, ys)), p["station"])
                    if interp is not None and h > 0:
                        gx.append(p["station"])
                        gy.append(interp - h)
                if len(gx) >= 2:
                    ax.plot(gx, gy, color="#B0BEC5", linewidth=1.2, linestyle="--", label="反推地面（示意）")

            for p, kind in zip(piers, kinds):
                marker = "^" if "台" in kind else "v"
                ax.plot([p["station"]], [0], marker=marker, color=COLOR_TEXT, markersize=8,
                        markeredgecolor="#0B1F3A", markeredgewidth=0.8)
                ax.annotate(
                    p["no"], (p["station"], 0), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color=COLOR_TEXT,
                    path_effects=[path_effects.withStroke(linewidth=2.0, foreground="#0B1F3A")],
                )

            ax.set_title(
                f"桥位 {bridge['no']} 布跨纵断示意（{variant}）"
                + (f" · {bridge['span_text']}" if bridge["span_text"] else ""),
                color=COLOR_TEXT, fontsize=11,
            )
            ax.set_ylabel("墩高 (m)", color=COLOR_AXIS, fontsize=9)
            ax.tick_params(colors=COLOR_AXIS, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(COLOR_AXIS)
            ax.legend(loc="upper left", fontsize=8, facecolor=COLOR_BG_DARK, edgecolor=COLOR_AXIS,
                      labelcolor=COLOR_TEXT)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_png
    except Exception:
        return None


def _linear_interp(points: Dict[float, float], station: float) -> Optional[float]:
    keys = sorted(points)
    if not keys:
        return None
    if station <= keys[0]:
        return points[keys[0]]
    if station >= keys[-1]:
        return points[keys[-1]]
    for a, b in zip(keys, keys[1:]):
        if a <= station <= b:
            if b == a:
                return points[a]
            return points[a] + (points[b] - points[a]) * (station - a) / (b - a)
    return None


# ---------------------------------------------------------------------------
# 顶层：初始/最终视图批量生成
# ---------------------------------------------------------------------------


def _locate_artifacts(output_dir: str) -> Dict[str, Optional[str]]:
    root = str(output_dir)
    pgw = _first(os.path.join(root, "pred_*.pgw"), os.path.join(root, "*.pgw"))
    terrain = _first(os.path.join(root, "pred_*.png"), os.path.join(root, "pred_*.jpg"),
                     os.path.join(root, "*.png"))
    plane = _first(os.path.join(root, "plane_from_loader", "*_plane.json"),
                   os.path.join(root, "preprocess", "plane", "*_plane.json"),
                   _latest(os.path.join(root, "**", "*_plane.json")))
    final_layout = _first(os.path.join(root, "layout_revision", "final_layout_result.json"),
                          _latest(os.path.join(root, "revision_results", "revision_design_round_*.json")))
    # 初始方案：取最新一轮 design_run 布跨结果（若存在）
    design_runs = sorted(
        glob.glob(os.path.join(root, "design_run_*", "design_result.json")),
        key=os.path.getmtime,
    )
    initial_layout = design_runs[-1] if design_runs else None
    # 路线纵断（设计线/地形线高程序列）：最新一轮布跨输入 input_data.json
    # （结构 {K|Z: {纵断面结构: {...}}}），用于真实纵断面渲染
    input_runs = sorted(
        glob.glob(os.path.join(root, "design_run_*", "input_data.json")),
        key=os.path.getmtime,
    )
    vertical = None
    for input_path in reversed(input_runs):
        payload = _load_optional_json(input_path)
        if not payload:
            continue
        for carriage in ("K", "Z"):
            section = payload.get(carriage)
            if isinstance(section, dict) and isinstance(section.get("纵断面结构"), dict):
                section = section["纵断面结构"]
                if section.get("设计线高程序列") and section.get("地形线高程序列"):
                    vertical = section
                    break
        if vertical:
            break
    return {
        "pgw": pgw,
        "terrain": terrain,
        "plane": plane,
        "final_layout": final_layout,
        "initial_layout": initial_layout,
        "vertical": vertical,
    }


def build_catalog_views(output_dir: str | Path) -> Dict[str, Optional[str]]:
    """为 output_dir 生成初始/最终两套目录视图（plan_overlay + profile）。

    返回 {initial_plan, initial_profile, final_plan, final_profile}，失败项为 None。
    纵断数据（设计线/地形线高程序列）可用时 profile 按 zdm 逻辑绘制真实纵断面，
    否则退化为墩高示意。输出目录 catalog_views/{initial,final}/ 只写新增 PNG，
    不触碰既有运行产物。
    """
    root = str(output_dir)
    located = _locate_artifacts(root)
    result: Dict[str, Optional[str]] = {
        "initial_plan": None,
        "initial_profile": None,
        "final_plan": None,
        "final_profile": None,
    }
    if not located["final_layout"] and not located["initial_layout"]:
        return result

    for variant, layout in (("final", located["final_layout"]), ("initial", located["initial_layout"])):
        if not layout:
            continue
        views_dir = os.path.join(root, "catalog_views", variant)
        if located["pgw"] and located["terrain"] and located["plane"]:
            result[f"{variant}_plan"] = render_plan_overlay(
                terrain_png=located["terrain"],
                pgw_path=located["pgw"],
                plane_json_path=located["plane"],
                layout_json_path=layout,
                out_png=os.path.join(views_dir, "plan_overlay.png"),
                variant=variant,
            )
        result[f"{variant}_profile"] = render_profile(
            layout_json_path=layout,
            out_png=os.path.join(views_dir, "profile.png"),
            variant=variant,
            vertical=located["vertical"],
        )
    return result
