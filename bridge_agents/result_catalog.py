"""全阶段成果目录服务（ResultArtifact / ResultCatalog）。

把 output_dir 下五个阶段的工程成果统一扫描为可浏览条目，供桌面成果中心
（run_graph_v2_gui.py）与后续 DesignReviewAgent 共用。模块不依赖 Tkinter，
扫描辅助函数（_latest_file/_first_existing/_all_files/_load_optional_json）
镜像自 bridge_agents/agent.py 中同名逻辑，避免引入 agent.py 的重依赖与循环导入。

黄金样例：output/layout_agent_run_example_K1_000-K2_600（2026-08-28 盘点）。
"""

from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .gui_controller import STAGE_ORDER, load_drawing_catalog

# 阶段别名（Treeview 与界面文案共用，保持单源）
STAGE_TITLES: Dict[str, str] = {
    "initial_design": "初步设计",
    "layout_revision": "布跨修正",
    "structural_design": "结构设计",
    "modeling_check": "建模验算",
    "final_output": "最终交付",
}


@dataclass
class ResultArtifact:
    """一条成果记录。

    stage: 阶段 id（见 STAGE_ORDER）
    kind:  阶段内类型 id（如 terrain / collision_view / check_plot）
    title: 人类可读标题
    file_path: 主文件（数据或图件）
    preview_path: 窗口内可预览的 PNG/JPG；为空则只支持信息面板/外部打开
    status: ok（可直接使用）或 empty（数据缺件，仅信息面板展示）
    external_only: True 表示仅外部打开（SVG/SCR/表格等）
    design_group: 关联设计组（如 1-1-1-G3），可为空
    diagnostics: 展示用键值摘要（全部转字符串，读取失败自动留空）
    layer: 同地理图层组（terrain/mask/obstacle/layout/vertical），用于图层切换
    source: 相对 output_dir 的路径，供信息面板展示
    """

    stage: str
    kind: str
    title: str
    file_path: str
    preview_path: str = ""
    status: str = "ok"
    external_only: bool = False
    design_group: str = ""
    diagnostics: Dict[str, str] = field(default_factory=dict)
    layer: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        self.diagnostics = {str(k): str(v) for k, v in (self.diagnostics or {}).items()}


# ---------------------------------------------------------------------------
# 文件扫描小工具（镜像 agent.py，纯标准库）
# ---------------------------------------------------------------------------


def _is_file(path: Any) -> bool:
    return bool(path) and os.path.isfile(str(path))


def _latest(pattern: str) -> Optional[str]:
    files = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    if not files:
        return None
    return max(files, key=lambda p: os.path.getmtime(p))


def _all(pattern: str) -> List[str]:
    """按修改时间升序返回匹配文件（早期轮次在前，便于按轮浏览）。"""
    files = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    return sorted(files, key=lambda p: os.path.getmtime(p))


def _first(*paths: Any) -> Optional[str]:
    for path in paths:
        if not path:
            continue
        path = str(path)
        if _is_file(path):
            return path
        if any(ch in path for ch in "*?["):
            latest = _latest(path)
            if latest:
                return latest
    return None


def _load_optional_json(path: Any) -> Optional[Dict[str, Any]]:
    if not _is_file(path):
        return None
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _fmt_diag_num(value: Any, digits: int) -> str:
    """把可能是数值的诊断字段格式化为字符串，非数值返回 '-'. """
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.{digits}f}"


def _rel(root: str, path: Any) -> str:
    try:
        return os.path.relpath(str(path), root).replace("\\", "/")
    except Exception:
        return str(path)


def _basename(path: Any) -> str:
    return os.path.basename(str(path))


# ---------------------------------------------------------------------------
# 各阶段扫描
# ---------------------------------------------------------------------------


def _scan_initial_design(root: str) -> List[ResultArtifact]:
    artifacts: List[ResultArtifact] = []
    r = root

    terrain_png = _first(os.path.join(r, "pred_*.png"), os.path.join(r, "*.png"))
    terrain_jpg = _first(os.path.join(r, "pred_*.jpg"), os.path.join(r, "*.jpg"))
    terrain = _first(terrain_png, terrain_jpg)
    if terrain:
        artifacts.append(
            ResultArtifact(
                stage="initial_design",
                kind="terrain",
                title="原始地形栅格",
                file_path=terrain,
                preview_path=terrain,
                layer="terrain",
                source=_rel(r, terrain),
            )
        )

    mask = _first(os.path.join(r, "mask", "obstacle_mask_*.png"))
    if mask:
        artifacts.append(
            ResultArtifact(
                stage="initial_design",
                kind="mask",
                title="分割 Mask",
                file_path=mask,
                preview_path=mask,
                layer="mask",
                source=_rel(r, mask),
            )
        )

    obstacle_json = _first(
        os.path.join(r, "obstacle_semantic", "complete_obstacles_grouped_*.json")
    )
    obstacle_views = _all(os.path.join(r, "obstacle_semantic", "*.jpg"))
    if obstacle_json or obstacle_views:
        title = "障碍语义图"
        main_file = obstacle_json or (obstacle_views[0] if obstacle_views else "")
        preview = ""
        for view in obstacle_views:
            if _basename(view).startswith("merged_vis"):
                preview = view
                break
        if not preview and obstacle_views:
            preview = obstacle_views[0]
        artifacts.append(
            ResultArtifact(
                stage="initial_design",
                kind="obstacle",
                title=title,
                file_path=str(main_file or ""),
                preview_path=preview,
                layer="obstacle",
                source=_rel(r, obstacle_json or main_file) if (obstacle_json or main_file) else "",
            )
        )

    plane = _first(
        os.path.join(r, "plane_from_loader", "*_plane.json"),
        os.path.join(r, "preprocess", "plane", "*_plane.json"),
        os.path.join(r, "**", "*_plane.json"),
    )
    if plane:
        payload = _load_optional_json(plane) or {}
        diag: Dict[str, Any] = {}
        curves = payload.get("平曲线结构")
        diag["平曲线段数"] = len(curves) if isinstance(curves, list) else "-"
        artifacts.append(
            ResultArtifact(
                stage="initial_design",
                kind="initial_plan",
                title="初始路线平面（平曲线结构）",
                file_path=plane,
                layer="layout",
                diagnostics=diag,
                source=_rel(r, plane),
            )
        )

    artifacts.extend(
        _collect_views(r, "initial_design", "catalog_views", "initial", "初始方案视图")
    )
    return artifacts


def _scan_layout_revision(root: str) -> List[ResultArtifact]:
    artifacts: List[ResultArtifact] = []
    r = root

    collision_views = _all(os.path.join(r, "collision_detection", "collision_vis_*.jpg"))
    for index, view in enumerate(collision_views, start=1):
        stem = _basename(view).replace("collision_vis_", "").rsplit(".", 1)[0]
        metrics = _first(os.path.join(r, "collision_detection", f"collision_metrics_{stem}.json"))
        diag: Dict[str, Any] = {}
        payload = _load_optional_json(metrics)
        if payload:
            diag["冲突柱数"] = payload.get("conflict_column_count", "-")
            diag["冲突率"] = (
                f"{payload.get('conflict_column_rate', 0) * 100:.1f}%"
                if isinstance(payload.get("conflict_column_rate"), (int, float))
                else "-"
            )
        artifacts.append(
            ResultArtifact(
                stage="layout_revision",
                kind="collision_view",
                title=f"碰撞可视化 第 {index} 轮",
                file_path=view,
                preview_path=view,
                diagnostics=diag,
                source=_rel(r, view),
            )
        )

    final_layout = _first(
        os.path.join(r, "layout_revision", "final_layout_result.json"),
        _latest(os.path.join(r, "revision_results", "revision_design_round_*.json")),
    )
    if final_layout:
        payload = _load_optional_json(final_layout) or {}
        diag: Dict[str, Any] = {}
        overview = payload.get("设桥总览") if isinstance(payload.get("设桥总览"), dict) else {}
        for key in ("是否设桥", "桥位数量", "方案组织类型", "总体布跨说明"):
            if key in overview:
                value = str(overview.get(key, "")).strip()
                if len(value) > 60:
                    value = value[:60] + "…"
                diag[key] = value
        artifacts.append(
            ResultArtifact(
                stage="layout_revision",
                kind="final_layout",
                title="最终修正布跨方案",
                file_path=final_layout,
                layer="layout",
                diagnostics=diag,
                source=_rel(r, final_layout),
            )
        )

    artifacts.extend(
        _collect_views(r, "layout_revision", "catalog_views", "final", "最终方案视图")
    )
    return artifacts


def _scan_structural_design(root: str) -> List[ResultArtifact]:
    artifacts: List[ResultArtifact] = []
    r = root

    dimension = _first(
        os.path.join(r, "structural_design", "dimension_design", "dimension_design_result.json")
    )
    if dimension:
        payload = _load_optional_json(dimension) or {}
        diag: Dict[str, Any] = {}
        inner = payload.get("任务2_下部结构尺寸设计结果")
        units = (
            inner.get("单元原始结果")
            if isinstance(inner, dict) and isinstance(inner.get("单元原始结果"), list)
            else None
        )
        diag["单元结果数"] = len(units) if units else "-"
        artifacts.append(
            ResultArtifact(
                stage="structural_design",
                kind="dimension_summary",
                title="尺寸设计结果汇总",
                file_path=dimension,
                diagnostics=diag,
                source=_rel(r, dimension),
            )
        )

    reinforcement = _first(
        os.path.join(r, "structural_design", "reinforcement_design", "reinforcement_design_result.json"),
        os.path.join(r, "structural_design", "reinforcement_design", "reinforcement_design_result.yaml"),
    )
    if reinforcement:
        artifacts.append(
            ResultArtifact(
                stage="structural_design",
                kind="reinforcement_summary",
                title="配筋设计结果汇总",
                file_path=reinforcement,
                source=_rel(r, reinforcement),
            )
        )

    axial_summary = _first(
        os.path.join(r, "structural_design", "reinforcement_design", "axial_check_summary.json")
    )
    if axial_summary:
        try:
            from .axial_table_render import ensure_axial_check_views  # PIL 延迟导入

            axial_png, _axial_csv = ensure_axial_check_views(axial_summary)
        except Exception:
            axial_png = ""
        payload = _load_optional_json(axial_summary) or {}
        counts = payload.get("verdict_counts") or {}
        diag: Dict[str, str] = {
            "组数": str(payload.get("expected_task_count") or "-"),
            "通过/不满足": f"{counts.get('passed', 0)} / {counts.get('failed', 0)}",
            "需人工复核/缺失": f"{counts.get('manual_review', 0)} / {counts.get('missing', 0)}",
            "不适用(无柱身)": str(counts.get("not_applicable", 0)),
        }
        max_group = payload.get("max_slenderness_group") or {}
        if max_group.get("task_id"):
            diag["最大长细比组"] = (
                f"{max_group.get('task_id')} λ={_fmt_diag_num(max_group.get('slenderness_ratio'), 1)}"
            )
        artifacts.append(
            ResultArtifact(
                stage="structural_design",
                kind="axial_check_summary",
                title="墩柱轴压验算汇总",
                file_path=axial_summary,
                preview_path=axial_png,
                diagnostics=diag,
                source=_rel(r, axial_summary),
            )
        )

    # 每设计组的骨架图 / 尺寸图（PNG 预览，SVG 由外部入口处理）
    groups: List[str] = []
    for pattern in (
        os.path.join(r, "structural_design", "plots", "*"),
        os.path.join(r, "structural_design", "reinforcement_design", "*"),
    ):
        for path in sorted(glob.glob(pattern)):
            if os.path.isdir(path):
                name = os.path.basename(path)
                if name not in groups and not name.startswith("."):
                    groups.append(name)
    groups.sort()

    for group in groups:
        skeleton = _first(
            os.path.join(r, "structural_design", "plots", group, "cap_rebar_skeleton_*.png")
        )
        if skeleton:
            artifacts.append(
                ResultArtifact(
                    stage="structural_design",
                    kind="structural_plot",
                    title=f"设计组 {group} · 配筋骨架",
                    file_path=skeleton,
                    preview_path=skeleton,
                    design_group=group,
                    source=_rel(r, skeleton),
                )
            )
        dimension_view = _first(
            os.path.join(r, "structural_design", "plots", group, "structural_dimension_rebar_*.png")
        )
        if dimension_view:
            artifacts.append(
                ResultArtifact(
                    stage="structural_design",
                    kind="structural_plot",
                    title=f"设计组 {group} · 结构尺寸配筋",
                    file_path=dimension_view,
                    preview_path=dimension_view,
                    design_group=group,
                    source=_rel(r, dimension_view),
                )
            )
        moment = _first(
            os.path.join(
                r,
                "structural_design",
                "reinforcement_design",
                group,
                "beam_force_figures_*",
                "ULS_basic_moment_envelope.png",
            )
        )
        if moment:
            artifacts.append(
                ResultArtifact(
                    stage="structural_design",
                    kind="envelope_plot",
                    title=f"设计组 {group} · 弯矩包络（ULS）",
                    file_path=moment,
                    preview_path=moment,
                    design_group=group,
                    source=_rel(r, moment),
                )
            )
        shear = _first(
            os.path.join(
                r,
                "structural_design",
                "reinforcement_design",
                group,
                "beam_force_figures_*",
                "ULS_basic_shear_envelope.png",
            )
        )
        if shear:
            artifacts.append(
                ResultArtifact(
                    stage="structural_design",
                    kind="envelope_plot",
                    title=f"设计组 {group} · 剪力包络（ULS）",
                    file_path=shear,
                    preview_path=shear,
                    design_group=group,
                    source=_rel(r, shear),
                )
            )
        reinforcement_yaml = _first(
            os.path.join(
                r, "structural_design", "reinforcement_design", group, "reinforcement_result_*.yaml"
            )
        )
        if reinforcement_yaml:
            artifacts.append(
                ResultArtifact(
                    stage="structural_design",
                    kind="reinforcement_detail",
                    title=f"设计组 {group} · 配筋结果",
                    file_path=reinforcement_yaml,
                    design_group=group,
                    source=_rel(r, reinforcement_yaml),
                )
            )
    return artifacts


def _scan_modeling_check(root: str) -> List[ResultArtifact]:
    artifacts: List[ResultArtifact] = []
    r = root

    summary = _first(
        os.path.join(r, "capacity_check", "capacity_check_batch_summary.json"),
        os.path.join(r, "capacity_check", "capacity_check_summary.json"),
        _latest(os.path.join(r, "capacity_check", "**", "capacity_check_summary.json")),
    )
    if summary:
        payload = _load_optional_json(summary) or {}
        diag: Dict[str, Any] = {}
        diag["组合"] = payload.get("combination_name", "-")
        sections = payload.get("control_sections")
        if isinstance(sections, dict):
            # 控制截面利用率可能是 float，也可能是带 util_* 的控制截面 dict
            for label, key, fallback in (
                ("最大正弯利用率", "max_positive_moment_utilization", "util_M_pos"),
                ("最大负弯利用率", "max_negative_moment_utilization", "util_M_neg"),
                ("最大抗剪利用率", "max_shear_utilization", "util_V"),
            ):
                value = sections.get(key)
                if isinstance(value, dict):
                    value = value.get(fallback)
                if isinstance(value, (int, float)):
                    diag[label] = f"{float(value):.2f}"
                else:
                    diag[label] = "-"
        overall = payload.get("overall_check")
        if isinstance(overall, dict):
            diag["验算结论"] = "通过" if overall.get("all_ok") else "未通过"
        artifacts.append(
            ResultArtifact(
                stage="modeling_check",
                kind="check_summary",
                title="承载力验算汇总",
                file_path=summary,
                diagnostics=diag,
                source=_rel(r, summary),
            )
        )

    plot_titles = {
        "demand_capacity_utilization": "需求-能力利用率图",
        "moment_demand_vs_capacity_overlay": "弯矩需求 vs 能力包络图",
        "shear_demand_vs_capacity_overlay": "剪力需求 vs 能力包络图",
    }
    for pattern, plot_title in plot_titles.items():
        # 验算图按设计组存放在 capacity_check/<group>/ 下，逐个设计组递归收录
        for view in sorted(
            glob.glob(
                os.path.join(r, "capacity_check", "**", f"{pattern}.png"),
                recursive=True,
            )
        ):
            group = os.path.basename(os.path.dirname(view))
            has_group = bool(group) and group != "capacity_check"
            artifacts.append(
                ResultArtifact(
                    stage="modeling_check",
                    kind="check_plot",
                    title=f"{plot_title} · {group}" if has_group else plot_title,
                    file_path=view,
                    preview_path=view,
                    source=_rel(r, view),
                    design_group=group if has_group else None,
                )
            )
    return artifacts


def _scan_final_output(root: str) -> List[ResultArtifact]:
    artifacts: List[ResultArtifact] = []
    r = root

    index = _first(os.path.join(r, "deliverables", "drawings", "drawing_index.json"))
    catalog = load_drawing_catalog(r) if index else []
    if index:
        payload = _load_optional_json(index) or {}
        groups = payload.get("groups") if isinstance(payload, dict) else None
        count = len(groups) if isinstance(groups, list) else len(catalog)
        artifacts.append(
            ResultArtifact(
                stage="final_output",
                kind="drawings_package",
                title=f"设计组绘图成果（{count} 组）",
                file_path=index,
                external_only=True,
                diagnostics={"设计组数": count},
                source=_rel(r, index),
            )
        )
    # 展开到单张图纸：SVG 可窗口内渲染预览，SCR 由外部打开按钮处理
    for group in catalog:
        group_id = str(group.get("design_group_id") or "")
        for sheet in group.get("sheets") or []:
            svg_path = str(sheet.get("svg_path") or "")
            if not svg_path or not os.path.isfile(svg_path):
                continue
            title = str(sheet.get("title") or sheet.get("sheet_id") or "图纸")
            artifacts.append(
                ResultArtifact(
                    stage="final_output",
                    kind="drawing_sheet",
                    title=f"{group_id} · {title}",
                    file_path=svg_path,
                    design_group=group_id,
                    layer="sheet",
                    source=_rel(r, svg_path),
                )
            )
    manifest = _first(os.path.join(r, "deliverables", "design_manifest.json"))
    if manifest:
        artifacts.append(
            ResultArtifact(
                stage="final_output",
                kind="design_manifest",
                title="设计成果清单",
                file_path=manifest,
                external_only=True,
                source=_rel(r, manifest),
            )
        )
    return artifacts


def _collect_views(
    root: str, stage: str, views_rel: str, sub: str, title_prefix: str
) -> List[ResultArtifact]:
    """收录可视化器生成的目录图（catalog_views/{sub}/**），PNG/JPG 均可预览。"""
    artifacts: List[ResultArtifact] = []
    view_dir = os.path.join(root, views_rel, sub)
    if not os.path.isdir(view_dir):
        return artifacts
    for view in sorted(glob.glob(os.path.join(view_dir, "**", "*.png"), recursive=True)):
        artifacts.append(
            ResultArtifact(
                stage=stage,
                kind=f"{sub}_view",
                title=f"{title_prefix} · {os.path.splitext(os.path.basename(view))[0]}",
                file_path=view,
                preview_path=view,
                source=_rel(root, view),
            )
        )
    for view in sorted(glob.glob(os.path.join(view_dir, "**", "*.jpg"), recursive=True)):
        artifacts.append(
            ResultArtifact(
                stage=stage,
                kind=f"{sub}_view",
                title=f"{title_prefix} · {os.path.splitext(os.path.basename(view))[0]}",
                file_path=view,
                preview_path=view,
                source=_rel(root, view),
            )
        )
    return artifacts


# ---------------------------------------------------------------------------
# 目录入口
# ---------------------------------------------------------------------------

_STAGE_SCANNERS = {
    "initial_design": _scan_initial_design,
    "layout_revision": _scan_layout_revision,
    "structural_design": _scan_structural_design,
    "modeling_check": _scan_modeling_check,
    "final_output": _scan_final_output,
}


def scan(output_dir: str | Path) -> List[ResultArtifact]:
    """扫描 output_dir，按 STAGE_ORDER 顺序返回全部成果条目。

    阶段始终完整返回（即使为空列表），缺失目录不报错。
    """
    root = str(output_dir or "")
    if not root or not os.path.isdir(root):
        return []
    artifacts: List[ResultArtifact] = []
    for stage in STAGE_ORDER:
        try:
            artifacts.extend(_STAGE_SCANNERS[stage](root))
        except Exception:
            # 单阶段扫描失败不阻断其它阶段
            continue
    return artifacts


def group_by_stage(artifacts: List[ResultArtifact]) -> "OrderedDict[str, List[ResultArtifact]]":
    grouped: "OrderedDict[str, List[ResultArtifact]]" = OrderedDict()
    for stage in STAGE_ORDER:
        grouped[stage] = [a for a in artifacts if a.stage == stage]
    return grouped


def display_action(artifact: ResultArtifact) -> str:
    """界面处理动作路由：preview=窗口预览 / external=外部打开 / info=信息面板 / missing=文件缺失。

    决策顺序：有可预览的位图 -> 窗口内预览；受控 SVG（本仓库绘图导出）由
    轻量渲染器转 PNG 后窗口内预览；否则显式标记外部打开的类型 -> 外部打开；
    文件存在 -> 信息面板；文件缺失 -> missing。纯函数，供 GUI 页签与测试共用。
    """
    if artifact is None:
        return "missing"
    if artifact.preview_path:
        from .preview_images import supports_preview  # 延迟导入避免顶层 PIL 依赖

        if supports_preview(artifact.preview_path) and os.path.isfile(artifact.preview_path):
            return "preview"
    if artifact.file_path and os.path.isfile(artifact.file_path):
        if artifact.external_only:
            return "external"
        if artifact.file_path.lower().endswith(".svg"):
            return "preview"  # 由 svg_preview 转 PNG 后窗口内预览
        return "info"
    return "missing"
