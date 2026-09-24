from __future__ import annotations

import json
from typing import Any, Dict

from bridge_agents.actions import ActionSpec, get_agent_action_specs, run_action
from bridge_agents.stage_agents import InitialDesignAgent, StructuralDesignAgent
from bridge_agents import tool_actions


def test_run_action_validates_required_inputs() -> None:
    def fake_action(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": True, "error": None}

    spec = ActionSpec(
        name="fake",
        func=fake_action,  # type: ignore[arg-type]
        required_keys=("data_path",),
        required_any=(("layout_result", "existing_layout_result"),),
    )

    result = run_action(spec, {"data_path": "data"})  # type: ignore[arg-type]

    assert result.ok is False
    assert "layout_result or existing_layout_result" in (result.error or "")


def test_run_action_wraps_successful_updates() -> None:
    def fake_action(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"layout_result": {"ok": True}, "empty": None, "error": None}

    spec = ActionSpec(name="fake", func=fake_action, expected_key="layout_result")  # type: ignore[arg-type]

    result = run_action(spec, {})  # type: ignore[arg-type]

    assert result.ok is True
    assert result.update["layout_result"] == {"ok": True}
    assert result.expected_key == "layout_result"
    assert result.produced_keys == ("layout_result",)


def test_stage_plans_resolve_to_registered_action_specs() -> None:
    initial_steps = InitialDesignAgent()._steps_from_stage_plan(
        {"required_steps": ["load_data", "generate_layout_design"]}
    )
    structural_steps = StructuralDesignAgent()._steps_from_stage_plan(
        {"required_steps": ["extract_design_units", "summarize_structural_design_result"]}
    )

    assert [step.name for step in initial_steps] == ["load_data", "generate_layout_design"]
    assert [step.expected_key for step in initial_steps] == ["cropped_data", "layout_result"]
    assert [step.name for step in structural_steps] == ["extract_design_units"]


def test_structural_plan_always_injects_compute_pier_groups_before_reinforcement() -> None:
    # 断点续跑：dimension 已有被跳过，计划只含 reinforcement 时也必须先算 pier_group（墩柱净高）。
    steps = StructuralDesignAgent()._steps_from_stage_plan(
        {"required_steps": ["reinforcement_design", "summarize_structural_design_result"]}
    )
    assert [step.name for step in steps] == ["compute_pier_groups", "reinforcement_design"]

    # 规则7场景：已有 dimension 但缺 design_units，补 extract + reinforcement。
    steps = StructuralDesignAgent()._steps_from_stage_plan(
        {"required_steps": ["extract_design_units", "reinforcement_design"]}
    )
    assert [step.name for step in steps] == [
        "extract_design_units",
        "compute_pier_groups",
        "reinforcement_design",
    ]


def test_structural_plan_compute_after_dimension_when_dimension_planned() -> None:
    # 全流程计划：compute_pier_groups 插在 dimension_design 之后、reinforcement_design 之前。
    steps = StructuralDesignAgent()._steps_from_stage_plan(
        {"required_steps": ["dimension_design", "reinforcement_design"]}
    )
    assert [step.name for step in steps] == [
        "dimension_design",
        "compute_pier_groups",
        "reinforcement_design",
    ]


def test_agent_action_registry_exposes_stage_scoped_specs() -> None:
    layout_specs = get_agent_action_specs("LayoutRevisionAgent")

    assert set(layout_specs) == {
        "run_collision_detection",
        "generate_revision_instruction",
        "build_revision_prompt",
        "generate_revised_layout",
    }


def test_revision_instruction_includes_human_review_feedback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        tool_actions,
        "revision_instruction_tool",
        lambda **kwargs: {
            "success": True,
            "revision_instruction": "自动生成的修正要求",
            "engineering_metrics": {},
        },
    )

    result = tool_actions.generate_revision_instruction_action({
        "layout_result": {"设桥总览": {"桥位列表": []}},
        "collision_metrics": {"conflict_column_rate": 0.1},
        "collision_items": [],
        "iteration_index": 3,
        "output_dir": str(tmp_path),
        "human_review_feedback": "调整桥位5边界",
    })

    assert result["error"] is None
    assert "自动生成的修正要求" in result["revision_instruction"]
    assert "人工复核补充要求" in result["revision_instruction"]
    assert "调整桥位5边界" in result["revision_instruction"]


def test_dimension_action_exposes_evidence_bundles_and_trace(monkeypatch, tmp_path) -> None:
    """Catches retrieved dimension evidence being lost at the Action/state boundary."""
    monkeypatch.setattr(
        tool_actions,
        "dimension_design_tool",
        lambda **_kwargs: {
            "success": True,
            "dimension_design_result": {"ok": True},
            "expected_unit_count": 1,
            "completed_unit_count": 1,
            "failed_unit_count": 0,
            "stage_complete": True,
            "evidence_bundles": {
                "U1": {
                    "stage": "dimension_design",
                    "retrieval_status": "ok",
                    "evidence_ids": ["EVID_DIM"],
                    "query_hash": "dim-hash",
                }
            },
        },
    )

    result = tool_actions.dimension_design_action({
        "design_units": {"single_unit_inputs": [{"单元编号": "U1"}]},
        "output_dir": str(tmp_path),
    })

    assert result["evidence_bundles"]["dimension_design:U1"]["evidence_ids"] == ["EVID_DIM"]
    assert result["code_trace"] == [{
        "stage": "dimension_design",
        "task_id": "U1",
        "retrieval_status": "ok",
        "evidence_ids": ["EVID_DIM"],
        "query_hash": "dim-hash",
    }]


def test_reinforcement_action_preserves_dimension_evidence(monkeypatch, tmp_path) -> None:
    """Catches reinforcement evidence overwriting evidence from the dimension stage."""
    monkeypatch.setattr(
        tool_actions,
        "reinforcement_design_tool",
        lambda **_kwargs: {
            "success": True,
            "reinforcement_design_result": {"ok": True},
            "expected_task_count": 1,
            "completed_task_count": 1,
            "failed_task_count": 0,
            "stage_complete": True,
            "evidence_bundles": {
                "R1": {
                    "stage": "reinforcement_design",
                    "retrieval_status": "ok",
                    "evidence_ids": ["EVID_REBAR"],
                    "query_hash": "rebar-hash",
                }
            },
            "output_files": {},
        },
    )

    result = tool_actions.reinforcement_design_action({
        "design_units": {"ok": True},
        "dimension_design_result": {"ok": True},
        "evidence_bundles": {
            "dimension_design:U1": {"evidence_ids": ["EVID_DIM"]},
        },
        "output_dir": str(tmp_path),
    })

    assert set(result["evidence_bundles"]) == {
        "dimension_design:U1",
        "reinforcement_design:R1",
    }
    assert result["code_trace"][0]["task_id"] == "R1"


def test_dimension_retry_passes_failed_units_and_existing_results(monkeypatch) -> None:
    captured = {}

    def fake_tool(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "dimension_design_result": {"ok": True},
            "expected_unit_count": 1,
            "completed_unit_count": 1,
            "failed_unit_count": 0,
            "stage_complete": True,
        }

    monkeypatch.setattr(tool_actions, "dimension_design_tool", fake_tool)
    existing = {"任务2_下部结构尺寸设计结果": {"单元原始结果": []}}

    tool_actions.dimension_design_action({
        "design_units": {"single_unit_inputs": [{"单元编号": "1-3"}]},
        "dimension_design_result": existing,
        "failed_task_ids": ["1-3"],
        "revision_context": {"target_step": "dimension_design"},
    })

    assert captured["retry_unit_ids"] == ["1-3"]
    assert captured["existing_dimension_design_result"] is existing


def test_dimension_retry_falls_back_to_disk_summary_when_memory_cleared(
    monkeypatch, tmp_path
) -> None:
    """返修失效会清空内存里的尺寸成果，此时合并基准必须回退到磁盘上的上一轮汇总。

    缺了这一步回退，合并基准恒为 None，尺寸成果会坍缩成"只剩被返修的那一联"，
    下游配筋再按坍缩结果重算任务清单，已验算通过的分组就被静默丢弃
    （2026-09-15 示例项目K29：配筋 20 组 → 4 组、最终只出 1 张图）。
    """
    captured = {}

    def fake_tool(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "dimension_design_result": {"ok": True},
            "expected_unit_count": 1,
            "completed_unit_count": 1,
            "failed_unit_count": 0,
            "stage_complete": True,
        }

    monkeypatch.setattr(tool_actions, "dimension_design_tool", fake_tool)
    summary_path = (
        tmp_path / "structural_design" / "dimension_design" / "dimension_design_result.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "任务2_下部结构尺寸设计结果": {
                    "单元原始结果": [
                        {"single_unit_input": {"单元编号": "1-1"}},
                        {"single_unit_input": {"单元编号": "1-3"}},
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tool_actions.dimension_design_action({
        "output_dir": str(tmp_path),
        "design_units": {"single_unit_inputs": [{"单元编号": "1-3"}]},
        "dimension_design_result": None,  # 返修失效已清空
        "failed_task_ids": ["1-3"],
        "revision_context": {"target_step": "dimension_design"},
    })

    merge_base = captured["existing_dimension_design_result"]
    assert merge_base is not None
    rows = merge_base["任务2_下部结构尺寸设计结果"]["单元原始结果"]
    assert [row["single_unit_input"]["单元编号"] for row in rows] == ["1-1", "1-3"]


def test_reinforcement_after_dimension_retry_targets_failed_units(monkeypatch) -> None:
    captured = {}

    def fake_tool(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "reinforcement_design_result": {"ok": True},
            "expected_task_count": 1,
            "completed_task_count": 1,
            "failed_task_count": 0,
            "stage_complete": True,
            "output_files": {},
        }

    monkeypatch.setattr(tool_actions, "reinforcement_design_tool", fake_tool)
    existing = {"任务3_下部结构配筋设计结果": {"分组原始结果": []}}

    tool_actions.reinforcement_design_action({
        "design_units": {"ok": True},
        "dimension_design_result": {"ok": True},
        "reinforcement_design_result": existing,
        "failed_task_ids": ["1-3"],
        "revision_context": {"target_step": "dimension_design"},
    })

    assert captured["retry_unit_ids"] == ["1-3"]
    assert captured.get("retry_task_ids") is None
    assert captured["existing_reinforcement_design_result"] is existing


def test_dimension_retry_prefers_dimension_unit_ids_over_reinforcement_task_ids(monkeypatch) -> None:
    # 两套 ID 命名空间不能混用：尺寸设计只接受单元编号（2-2），
    # 配筋设计任务号（2-2-1-G1）传入会让尺寸工具按 ID 匹配失败。
    captured = {}

    def fake_tool(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "dimension_design_result": {"ok": True},
            "expected_unit_count": 2,
            "completed_unit_count": 2,
            "failed_unit_count": 0,
            "stage_complete": True,
        }

    monkeypatch.setattr(tool_actions, "dimension_design_tool", fake_tool)

    tool_actions.dimension_design_action({
        "design_units": {"single_unit_inputs": [{"单元编号": "2-2"}, {"单元编号": "2-6"}]},
        "dimension_design_result": {"任务2_下部结构尺寸设计结果": {"单元原始结果": []}},
        "failed_task_ids": ["2-2-1-G1"],
        "dimension_batch_status": {"stage_complete": False, "failed_unit_ids": ["2-2", "2-6"]},
        "revision_context": {
            "target_step": "dimension_design",
            "dimension_retry_unit_ids": ["2-2", "2-6"],
            "reinforcement_retry_task_ids": ["2-2-1-G1"],
        },
    })

    assert captured["retry_unit_ids"] == ["2-2", "2-6"]


def test_reinforcement_rework_separates_task_ids_and_dimension_unit_ids(monkeypatch) -> None:
    captured = {}

    def fake_tool(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "reinforcement_design_result": {"ok": True},
            "expected_task_count": 1,
            "completed_task_count": 1,
            "failed_task_count": 0,
            "stage_complete": True,
            "output_files": {},
        }

    monkeypatch.setattr(tool_actions, "reinforcement_design_tool", fake_tool)

    tool_actions.reinforcement_design_action({
        "design_units": {"ok": True},
        "dimension_design_result": {"ok": True},
        "reinforcement_design_result": {"任务3_下部结构配筋设计结果": {"分组原始结果": []}},
        "failed_task_ids": ["2-2-1-G1"],
        "dimension_batch_status": {"stage_complete": False, "failed_unit_ids": ["2-2", "2-6"]},
        "revision_context": {
            "target_step": "dimension_design",
            "dimension_retry_unit_ids": ["2-2", "2-6"],
            "reinforcement_retry_task_ids": ["2-2-1-G1"],
        },
    })

    assert captured["retry_task_ids"] == ["2-2-1-G1"]
    assert captured["retry_unit_ids"] == ["2-2", "2-6"]
