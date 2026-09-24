"""result_catalog 目录服务测试。

Fixture 复刻黄金样例 output/layout_agent_run_example_K1_000-K2_600 的
目录结构与文件名约定（2026-08-28 盘点），确保扫描规则与真实产物一致。
"""

from __future__ import annotations

import json
import os

from bridge_agents.gui_controller import STAGE_ORDER
from bridge_agents.result_catalog import (
    ResultArtifact,
    display_action,
    group_by_stage,
    scan,
)


def _touch(path, content=b"", mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_bytes(content.encode("utf-8"))
    else:
        path.write_bytes(content)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _build_zhangguan_like(tmp_path):
    """按 示例项目 实际命名构造最小工程目录，返回根路径。"""
    root = tmp_path / "run"
    t0 = 1_700_000_000

    _touch(root / "pred_K41_K42.pgw")
    _touch(root / "pred_K41_K42.png")
    _touch(root / "mask" / "obstacle_mask_K41_K42.png")
    _touch(
        root / "obstacle_semantic" / "complete_obstacles_grouped_section.json",
        json.dumps({"桥位": [{"name": "1"}]}, ensure_ascii=False),
    )
    _touch(root / "obstacle_semantic" / "merged_vis_section.jpg")
    _touch(root / "plane_from_loader" / "K_plane.json", '{"平曲线结构": [{}, {}]}')

    # 两轮碰撞：r1 旧、r2 新（mtime 控制）
    _touch(
        root / "collision_detection" / "collision_metrics_design_result_r1.json",
        json.dumps({"conflict_column_count": 3, "conflict_column_rate": 0.25}),
        mtime=t0 + 100,
    )
    _touch(
        root / "collision_detection" / "collision_vis_design_result_r1.jpg",
        mtime=t0 + 100,
    )
    _touch(
        root / "collision_detection" / "collision_metrics_design_result_r2.json",
        json.dumps({"conflict_column_count": 5, "conflict_column_rate": 0.42}),
        mtime=t0 + 200,
    )
    _touch(root / "collision_detection" / "collision_vis_design_result_r2.jpg", mtime=t0 + 200)

    _touch(
        root / "layout_revision" / "final_layout_result.json",
        json.dumps(
            {
                "设桥总览": {
                    "是否设桥": "是",
                    "桥位数量": 3,
                    "方案组织类型": "多桥位分幅",
                    "总体布跨说明": "各桥位独立分幅",
                },
                "桥位列表": [],
            },
            ensure_ascii=False,
        ),
    )

    plots = root / "structural_design" / "plots"
    _touch(plots / "G1" / "cap_rebar_skeleton_G1.png")
    _touch(plots / "G1" / "structural_dimension_rebar_G1.png")
    _touch(plots / "G2" / "cap_rebar_skeleton_G2.png")
    forces = root / "structural_design" / "reinforcement_design" / "G1" / "beam_force_figures_G1"
    _touch(forces / "ULS_basic_moment_envelope.png")
    _touch(forces / "ULS_basic_shear_envelope.png")
    _touch(
        root / "structural_design" / "reinforcement_design" / "G2" / "reinforcement_result_G2.yaml"
    )
    _touch(
        root / "structural_design" / "reinforcement_design" / "reinforcement_design_result.yaml"
    )
    _touch(
        root / "structural_design" / "dimension_design" / "dimension_design_result.json",
        json.dumps(
            {"任务2_下部结构尺寸设计结果": {"单元原始结果": [{}, {}]}}, ensure_ascii=False
        ),
    )

    _touch(
        root / "capacity_check" / "capacity_check_summary.json",
        json.dumps(
            {
                "combination_name": "组合I",
                "control_sections": {
                    "max_positive_moment_utilization": 0.82,
                    "max_negative_moment_utilization": 0.75,
                    "max_shear_utilization": 0.66,
                },
                "overall_check": {"all_ok": True},
            }
        ),
    )
    _touch(root / "capacity_check" / "demand_capacity_utilization.png")
    return root


def _kinds(artifacts, stage):
    return sorted({a.kind for a in artifacts if a.stage == stage})


def test_zhangguan_like_full_scan(tmp_path):
    root = _build_zhangguan_like(tmp_path)
    artifacts = scan(root)

    stages = [a.stage for a in artifacts]
    assert stages  # 非空

    # 阶段始终按 STAGE_ORDER 顺序分组（含空阶段）
    grouped = group_by_stage(artifacts)
    assert list(grouped.keys()) == list(STAGE_ORDER)

    # 初步设计
    initial = grouped["initial_design"]
    assert {"terrain", "mask", "obstacle", "initial_plan"} <= set(_kinds(initial, "initial_design"))
    terrain = next(a for a in initial if a.kind == "terrain")
    assert terrain.preview_path and terrain.layer == "terrain"
    obstacle = next(a for a in initial if a.kind == "obstacle")
    assert obstacle.preview_path.endswith("merged_vis_section.jpg")

    # 布跨修正：两轮碰撞按时间升序，final_layout 带设桥总览摘要
    layout = grouped["layout_revision"]
    views = [a for a in layout if a.kind == "collision_view"]
    assert len(views) == 2
    assert views[0].title.endswith("第 1 轮")
    assert views[1].title.endswith("第 2 轮")
    assert views[1].diagnostics.get("冲突柱数") == "5"
    final_layout = next(a for a in layout if a.kind == "final_layout")
    assert final_layout.diagnostics.get("桥位数量") == "3"
    assert "是否设桥" in final_layout.diagnostics

    # 结构设计
    structural = grouped["structural_design"]
    kinds = _kinds(structural, "structural_design")
    assert "dimension_summary" in kinds and "reinforcement_summary" in kinds
    plots = [a for a in structural if a.kind == "structural_plot"]
    assert len(plots) == 3  # G1 两张 + G2 一张
    envelopes = [a for a in structural if a.kind == "envelope_plot"]
    assert len(envelopes) == 2 and all(a.design_group == "G1" for a in envelopes)
    details = [a for a in structural if a.kind == "reinforcement_detail"]
    assert len(details) == 1 and details[0].design_group == "G2"

    # 建模验算
    modeling = grouped["modeling_check"]
    assert _kinds(modeling, "modeling_check") == ["check_plot", "check_summary"]
    summary = next(a for a in modeling if a.kind == "check_summary")
    assert summary.diagnostics.get("验算结论") == "通过"
    assert summary.diagnostics.get("最大正弯利用率") == "0.82"

    # 最终交付：示例项目 无 deliverables → 空
    assert grouped["final_output"] == []


def test_collision_views_sorted_by_mtime(tmp_path):
    root = _build_zhangguan_like(tmp_path)
    artifacts = scan(root)
    layout = group_by_stage(artifacts)["layout_revision"]
    views = [a for a in layout if a.kind == "collision_view"]
    assert "r1" in views[0].file_path and "r2" in views[1].file_path


def test_empty_output_dir_returns_stage_order_with_empty_lists(tmp_path):
    artifacts = scan(tmp_path)
    assert artifacts == []
    grouped = group_by_stage(artifacts)
    assert list(grouped.keys()) == list(STAGE_ORDER)
    assert all(len(grouped[s]) == 0 for s in STAGE_ORDER)


def test_missing_stage_folders_do_not_raise(tmp_path):
    root = tmp_path / "run"
    _touch(root / "pred_A.png")
    artifacts = scan(root)
    grouped = group_by_stage(artifacts)
    assert any(a.kind == "terrain" for a in grouped["initial_design"])
    assert grouped["layout_revision"] == []


def test_final_output_with_deliverables(tmp_path):
    root = tmp_path / "run"
    _touch(root / "deliverables" / "drawings" / "drawing_index.json",
           json.dumps({"groups": [{"design_group_id": "A"}, {"design_group_id": "B"}]}))
    _touch(root / "deliverables" / "design_manifest.json", "{}")
    artifacts = scan(root)
    grouped = group_by_stage(artifacts)["final_output"]
    kinds = {a.kind for a in grouped}
    assert kinds == {"design_manifest", "drawings_package"}
    package = next(a for a in grouped if a.kind == "drawings_package")
    assert package.diagnostics.get("设计组数") == "2"
    assert package.external_only


def test_final_output_expands_drawing_sheets_with_svg_preview(tmp_path):
    root = tmp_path / "run"
    group_dir = root / "deliverables" / "drawings" / "EX1-G1"
    _touch(group_dir / "cap.svg", "<svg width='100' height='80'><rect width='100%' height='100%' fill='#111827'/></svg>")
    _touch(root / "deliverables" / "drawings" / "drawing_index.json",
           json.dumps({
               "groups": [{
                   "design_group_id": "EX1-G1",
                   "member_piers": ["P1", "P2"],
                   "sheets": [
                       {"sheet_id": "cap_reinforcement_detail", "title": "盖梁配筋详图",
                        "svg_name": "cap.svg", "scr_name": "cap.scr"},
                   ],
               }]
           }, ensure_ascii=False))
    artifacts = scan(root)
    grouped = group_by_stage(artifacts)["final_output"]
    sheets = [a for a in grouped if a.kind == "drawing_sheet"]
    assert len(sheets) == 1
    sheet = sheets[0]
    assert "盖梁配筋详图" in sheet.title
    assert sheet.file_path.endswith("cap.svg")
    assert display_action(sheet) == "preview"  # 受控 SVG 窗口内渲染预览
    assert sheet.layer == "sheet"


# ---- display_action 路由 ----


def test_display_action_routes(tmp_path):
    png = tmp_path / "view.png"
    png.write_bytes(b"x")

    preview = ResultArtifact(stage="initial_design", kind="terrain", title="t",
                             file_path=str(png), preview_path=str(png))
    assert display_action(preview) == "preview"

    svg = tmp_path / "sheet.svg"
    svg.write_bytes(b"<svg/>")
    external = ResultArtifact(stage="final_output", kind="drawings_package", title="p",
                              file_path=str(svg), external_only=True)
    assert display_action(external) == "external"

    data = tmp_path / "layout.json"
    data.write_text("{}", encoding="utf-8")
    info = ResultArtifact(stage="layout_revision", kind="final_layout", title="f",
                          file_path=str(data))
    assert display_action(info) == "info"

    missing = ResultArtifact(stage="initial_design", kind="mask", title="m",
                             file_path=str(tmp_path / "no.png"), preview_path=str(tmp_path / "no.png"))
    assert display_action(missing) == "missing"

    assert display_action(None) == "missing"
