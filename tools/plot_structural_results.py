from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("缺少 matplotlib/numpy，请先在当前环境安装后再运行绘图脚本。") from exc
    return plt, np


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须为对象: {path}")
    return data


def first_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.replace("mm", "").replace("m", "").strip()
        try:
            return float(text)
        except ValueError:
            return default
    return default


def safe_eval_expr(expr: Any, context: Dict[str, float], default: Optional[float] = None) -> Optional[float]:
    if isinstance(expr, (int, float)):
        return float(expr)
    if expr is None:
        return default
    text = str(expr).strip()
    if not text:
        return default
    allowed = {
        "abs": abs,
        "max": max,
        "min": min,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
    }
    allowed.update(context)
    try:
        return float(eval(text, {"__builtins__": {}}, allowed))
    except Exception:
        return default


def resolve_run_dir(run_dir: Optional[str]) -> Path:
    return Path(run_dir) if run_dir else Path("output/layout_agent_run_example_K1_000-K2_600")


def reinforcement_summary_path(run_dir: Path) -> Path:
    return run_dir / "structural_design" / "reinforcement_design" / "reinforcement_design_result.json"


def iter_reinforcement_tasks(summary: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    root = next(iter(summary.values()))
    if not isinstance(root, dict):
        return []
    for value in root.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "reinforcement_task" in value[0]:
            return value
    return []


def get_dimension_info(task: Dict[str, Any]) -> Dict[str, Any]:
    data = task.get("桥墩尺寸信息") or {}
    return {
        "basic": data.get("基本信息") or {},
        "layout": data.get("上部布置与支承条件") or {},
        "cap": data.get("盖梁几何信息") or {},
        "column": data.get("墩柱几何信息") or {},
        "foundation": data.get("基础信息") or {},
        "cover": data.get("保护层与控制参数") or {},
    }


def geometry_context(info: Dict[str, Any]) -> Dict[str, float]:
    cap = info["cap"]
    column = info["column"]
    cover = info["cover"]
    cap_length = 1000.0 * (first_number(cap.get("cap_length"), 11.5) or 11.5)
    cap_height_mid = 1000.0 * (first_number(cap.get("cap_height_mid"), 1.8) or 1.8)
    cap_height_end = 1000.0 * (first_number(cap.get("cap_height_end"), 0.95) or 0.95)
    column_spacing = 1000.0 * (first_number(column.get("column_spacing"), 0.0) or 0.0)
    column_dia = 1000.0 * (first_number(column.get("column_diameter"), 0.0) or 0.0)

    half_len = cap_length / 2.0
    if column_spacing > 0 and column_dia > 0:
        flat_half = column_spacing / 2.0 + column_dia / 2.0 - 100.0
    else:
        flat_half = half_len - max(1000.0, cap_height_end * 1.4)
    flat_half = max(min(flat_half, half_len - 100.0), half_len * 0.45)

    slope_run = max(half_len - flat_half, 1.0)
    slope_k = cap_height_end / slope_run
    cos_theta = 1.0 / math.sqrt(1.0 + slope_k**2)

    return {
        "cap_length": cap_length,
        "cap_half": half_len,
        "cap_width": 1000.0 * (first_number(cap.get("cap_width"), 2.4) or 2.4),
        "cap_height_mid": cap_height_mid,
        "cap_height_end": cap_height_end,
        "flat_half": flat_half,
        "slope_k": slope_k,
        "cos_theta": cos_theta,
        "cover_x": first_number(cover.get("cover_x"), 50.0) or 50.0,
        "top_cover": first_number(cover.get("top_cover"), 50.0) or 50.0,
        "bottom_cover": first_number(cover.get("bottom_cover"), 50.0) or 50.0,
        "side_cover": first_number(cover.get("side_cover"), 50.0) or 50.0,
        "column_spacing": column_spacing,
        "column_diameter": column_dia,
        "pier_centerline_x": -column_spacing / 2.0 if column_spacing > 0 else 0.0,
        "cap_centerline_x": 0.0,
    }


def y_from_ref(ref: Dict[str, Any], dia: float, ctx: Dict[str, float], default_ref: str = "top_cover") -> float:
    y_ref = str(ref.get("y_ref") or default_ref)
    offset = safe_eval_expr(ref.get("y_offset", 0), {**ctx, "dia": dia}, 0.0) or 0.0
    if "bottom" in y_ref or "底" in y_ref:
        return ctx["bottom_cover"] + offset
    return ctx["cap_height_mid"] - ctx["top_cover"] - offset


def x_expr(expr: Any, ctx: Dict[str, float], default: float = 0.0) -> float:
    value = safe_eval_expr(expr, ctx, default)
    return default if value is None else value


def bottom_offset_polyline(offset: float, ctx: Dict[str, float]) -> Tuple[List[float], List[float]]:
    half = ctx["cap_half"]
    flat = ctx["flat_half"]
    k = ctx["slope_k"]
    cos_theta = ctx["cos_theta"]
    dy_sloped = offset / cos_theta
    kink = flat - (dy_sloped - offset) / max(k, 1e-9)
    y_end = k * (half - flat) + dy_sloped
    return [-half, -kink, kink, half], [y_end, offset, offset, y_end]


def bottom_y_at_x(x: float, offset: float, ctx: Dict[str, float]) -> float:
    flat = ctx["flat_half"]
    k = ctx["slope_k"]
    cos_theta = ctx["cos_theta"]
    ax = abs(x)
    if ax <= flat:
        return offset
    return k * (ax - flat) + offset / cos_theta


def drop_to_bottom(x0: float, y0: float, direction: str, bottom_offset: float, ctx: Dict[str, float]) -> Tuple[float, float]:
    flat = ctx["flat_half"]
    k = ctx["slope_k"]
    cos_theta = ctx["cos_theta"]
    dy_sloped = bottom_offset / cos_theta
    m = 1.0 if direction == "left_down" else -1.0

    x_flat = (bottom_offset - y0) / m + x0
    if -flat <= x_flat <= 0:
        return x_flat, bottom_offset

    if x_flat < -flat:
        x_slope = (m * x0 - y0 - k * flat + dy_sloped) / (m + k)
        y_slope = m * (x_slope - x0) + y0
        return x_slope, y_slope

    return x_flat, bottom_offset


def mirrored(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    if xs and abs(xs[-1]) < 1e-6:
        return xs + [-x for x in reversed(xs[:-1])], ys + [y for y in reversed(ys[:-1])]
    return xs + [-x for x in reversed(xs)], ys + [y for y in reversed(ys)]


def plot_poly(ax: Any, xs: List[float], ys: List[float], color: str, label: Optional[str], mirror: bool = False, lw: float = 2.3) -> None:
    if mirror:
        xs, ys = mirrored(xs, ys)
    ax.plot(xs, ys, color=color, linewidth=lw, label=label)


def draw_outline_and_axes(ax: Any, ctx: Dict[str, float], np: Any) -> None:
    half = ctx["cap_half"]
    flat = ctx["flat_half"]
    h_mid = ctx["cap_height_mid"]
    h_end = ctx["cap_height_end"]
    outline = np.array([
        [-half, h_mid],
        [half, h_mid],
        [half, h_end],
        [flat, 0],
        [-flat, 0],
        [-half, h_end],
        [-half, h_mid],
    ])
    ax.plot(outline[:, 0], outline[:, 1], color="#333333", linewidth=2.0, label="盖梁轮廓")
    spacing = ctx["column_spacing"]
    dia = ctx["column_diameter"]
    if spacing > 0 and dia > 0:
        for c in [-spacing / 2.0, spacing / 2.0]:
            ax.axvline(c, color="gray", linestyle="-.", alpha=0.55, linewidth=1.1, label="墩柱中心线" if c < 0 else None)
            ax.plot([c - dia / 2.0, c - dia / 2.0], [0, -1200], "k--", alpha=0.45, linewidth=1.0)
            ax.plot([c + dia / 2.0, c + dia / 2.0], [0, -1200], "k--", alpha=0.45, linewidth=1.0)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.55, linewidth=1.1)


def draw_constant_bar(ax: Any, bar: Dict[str, Any], ctx: Dict[str, float], color: str, label: str) -> None:
    rd = bar.get("range_definition") or {}
    layer = bar.get("layer_definition") or {}
    dia = first_number(bar.get("dia"), 20.0) or 20.0
    x0 = x_expr(rd.get("x_start_expr"), ctx, -ctx["cap_half"] + ctx["cover_x"])
    x1 = x_expr(rd.get("x_end_expr"), ctx, 0.0)
    mirror_bar = str(bar.get("mirror") or "").lower() == "longitudinal"
    y_ref = str(layer.get("y_ref") or "")
    offset = safe_eval_expr(layer.get("y_offset", 0), {**ctx, "dia": dia}, 0.0) or 0.0

    if "bottom" in y_ref or "底" in y_ref:
        bottom_offset = ctx["bottom_cover"] + offset
        xs, ys = bottom_offset_polyline(bottom_offset, ctx)
        plot_poly(ax, xs, ys, color, label, mirror=False, lw=2.5)
    else:
        y = y_from_ref(layer, dia, ctx, "top_cover")
        if mirror_bar:
            ax.plot([x0, -x0], [y, y], color=color, linewidth=2.5, label=label)
        else:
            ax.plot([x0, x1], [y, y], color=color, linewidth=2.5, label=label)


def draw_control_bar(ax: Any, bar: Dict[str, Any], ctx: Dict[str, float], color: str, label: str) -> None:
    cd = bar.get("control_definition") or {}
    points = cd.get("key_points") or {}
    left = points.get("left_bend_point") or {}
    right = points.get("right_bend_point") or {}
    dia = first_number(bar.get("dia"), 20.0) or 20.0
    bottom_offset = ctx["bottom_cover"] + dia

    lx = x_expr(left.get("x_expr"), ctx, -ctx["flat_half"])
    ly = y_from_ref(left, dia, ctx, "top_cover")
    rx = x_expr(right.get("x_expr"), ctx, -ctx["flat_half"] / 2.0)
    ry = y_from_ref(right, dia, ctx, "top_cover")
    left_end = drop_to_bottom(lx, ly, "left_down", bottom_offset, ctx)
    right_end = drop_to_bottom(rx, ry, "right_down", bottom_offset, ctx)

    xs = [left_end[0], lx, rx, right_end[0]]
    ys = [left_end[1], ly, ry, right_end[1]]

    branch_rules = cd.get("branch_rules") or {}
    right_branch = branch_rules.get("right_branch") or {}
    step_2 = right_branch.get("step_2") if isinstance(right_branch, dict) else None
    if isinstance(step_2, dict):
        end_at = step_2.get("end_at") or {}
        end_x = end_at.get("x") if isinstance(end_at, dict) else end_at
        if str(end_x or "") == "cap_centerline_x":
            xs.append(0.0)
            ys.append(bottom_y_at_x(0.0, bottom_offset, ctx))

    mirror_bar = str(bar.get("mirror") or "").lower() == "longitudinal"
    plot_poly(ax, xs, ys, color, label, mirror=mirror_bar, lw=2.3)


def draw_rebar_skeleton(ax: Any, task_item: Dict[str, Any], np: Any) -> None:
    task = task_item.get("reinforcement_task") or {}
    info = get_dimension_info(task)
    ctx = geometry_context(info)
    reinf = task_item.get("reinforcement_result") or {}
    pier_cap = ((reinf.get("reinforcement") or {}).get("pier_cap") or {})

    draw_outline_and_axes(ax, ctx, np)

    colors = [
        "#d62728",
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
    ]
    bars = [b for b in (pier_cap.get("longitudinal_bars") or []) if isinstance(b, dict)]
    for idx, bar in enumerate(bars):
        bar_id = str(bar.get("id") or f"N{idx + 1}")
        subtype = str(bar.get("subtype") or bar.get("category") or "钢筋")
        dia = first_number(bar.get("dia"), 0.0) or 0.0
        label = f"{bar_id}: {subtype} D{dia:g}"
        color = colors[idx % len(colors)]
        if isinstance(bar.get("range_definition"), dict):
            draw_constant_bar(ax, bar, ctx, color, label)
        elif isinstance(bar.get("control_definition"), dict):
            draw_control_bar(ax, bar, ctx, color, label)

    task_id = task.get("task_id") or ""
    role = task.get("墩位角色") or ""
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{task_id} {role}：盖梁配筋骨架推理图", fontsize=15, pad=18)
    ax.set_xlabel("顺桥向坐标 X (mm)", fontsize=12)
    ax.set_ylabel("竖向高度 Y (mm)", fontsize=12)
    ax.set_xlim(-ctx["cap_half"] - 450, ctx["cap_half"] + 450)
    ax.set_ylim(-500, ctx["cap_height_mid"] + 450)
    ax.grid(True, linestyle=":", alpha=0.55)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered = []
    for h, label in zip(handles, labels):
        if label not in seen:
            filtered.append((h, label))
            seen.add(label)
    if filtered:
        ax.legend(
            [h for h, _ in filtered],
            [label for _, label in filtered],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            fontsize=9,
            frameon=True,
        )


def plot_task(task_item: Dict[str, Any], output_dir: Path, formats: List[str]) -> Dict[str, str]:
    plt, np = _import_matplotlib()
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    task = task_item.get("reinforcement_task") or {}
    task_id = str(task.get("task_id") or f"task_{task_item.get('task_index', 0)}")
    fig, ax = plt.subplots(figsize=(16, 6), dpi=180)
    draw_rebar_skeleton(ax, task_item, np)
    fig.tight_layout()

    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    for fmt in formats:
        out = task_dir / f"cap_rebar_skeleton_{task_id}.{fmt}"
        fig.savefig(out, bbox_inches="tight")
        files[fmt] = str(out)
    plt.close(fig)
    return files


def write_summary(tasks: List[Dict[str, Any]], output_dir: Path) -> Path:
    rows = []
    for item in tasks:
        task = item.get("reinforcement_task") or {}
        info = get_dimension_info(task)
        cap = info["cap"]
        column = info["column"]
        rows.append({
            "task_id": task.get("task_id"),
            "bridge_id": task.get("桥梁编号"),
            "unit_id": task.get("单元编号"),
            "group_id": task.get("分组编号"),
            "pier_role": task.get("墩位角色"),
            "cap_length_m": cap.get("cap_length"),
            "cap_height_mid_m": cap.get("cap_height_mid"),
            "cap_height_end_m": cap.get("cap_height_end"),
            "column_count": column.get("column_count"),
            "column_spacing_m": column.get("column_spacing"),
        })
    path = output_dir / "cap_rebar_skeleton_summary.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="按配筋推理坐标系绘制盖梁尺寸与钢筋骨架图。")
    parser.add_argument("--run-dir", default=None, help="设计运行输出目录。")
    parser.add_argument("--task-id", default=None, help="只绘制指定配筋任务，例如 1-1-1-G2。")
    parser.add_argument("--max-items", type=int, default=12, help="最多绘制多少个分组。")
    parser.add_argument("--output-dir", default=None, help="图片输出目录，默认 run-dir/structural_design/plots。")
    parser.add_argument("--formats", default="png,svg", help="输出格式，逗号分隔，例如 png,svg。")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    summary_file = reinforcement_summary_path(run_dir)
    if not summary_file.exists():
        raise SystemExit(f"未找到配筋汇总结果: {summary_file}")

    summary = load_json(summary_file)
    tasks = list(iter_reinforcement_tasks(summary))
    if args.task_id:
        tasks = [item for item in tasks if (item.get("reinforcement_task") or {}).get("task_id") == args.task_id]
    if args.max_items and args.max_items > 0:
        tasks = tasks[: args.max_items]
    if not tasks:
        raise SystemExit("未找到可绘制的配筋任务。")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "structural_design" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]

    produced = {}
    for item in tasks:
        task = item.get("reinforcement_task") or {}
        task_id = task.get("task_id") or f"task_{item.get('task_index', 0)}"
        produced[str(task_id)] = plot_task(item, output_dir, formats)
    summary_path = write_summary(tasks, output_dir)

    print(json.dumps({
        "success": True,
        "task_count": len(tasks),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "files": produced,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
