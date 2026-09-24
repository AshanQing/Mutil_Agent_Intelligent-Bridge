"""回归测试：已有最终布跨成果时恢复碰撞指标 → 验算失败 → 配筋返修 → 复验。

对应 2026-09-15 运行暴露的四个缺陷：
1. `_discover_existing_outputs` 在存在 final_layout_result.json 时不再恢复碰撞指标；
2. 协调器把该字段当作 collision_detected_available 的唯一依据，导致结构返修被前置校验挡住；
3. `continue_modeling_revision` 固定返回验算阶段，不遵循 Handoff 指定的结构返修目标；
4. `accept_check_and_finish` 只认内存里的验算结果，而该结果已被返修失效逻辑清空。
另外覆盖：自动返修请求必须写清"只重做失败的那一组配筋"（revision_context 分步骤重试范围）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from bridge_agents.agent import _discover_existing_outputs
from bridge_agents.contracts import StageName
from bridge_agents.design_coordinator import build_coordinator_view
from bridge_agents.graph_v2 import (
    apply_human_review_decision,
    build_graph_v2,
    build_human_review_request,
    route_after_human_review,
)
from bridge_agents.tool_actions import generate_modeling_feedback_action


# --------------------------------------------------------------------------- #
# 1. 恢复：已有最终布跨成果时也要恢复碰撞指标
# --------------------------------------------------------------------------- #
def _write(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_discovery_restores_collision_metrics_when_final_layout_exists(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write(
        output_dir / "layout_revision" / "final_layout_result.json",
        {"桥位列表": [{"桥位编号": 1}]},
    )
    _write(
        output_dir / "collision_detection" / "collision_metrics_design_result_20260915_195700.json",
        {"total_columns": 120, "conflict_column_count": 0, "has_collision": False},
    )
    _write(
        output_dir / "collision_detection" / "collision_report_design_result_20260915_195700.json",
        {"report": [], "collision_items": []},
    )

    discovered = _discover_existing_outputs(output_dir)

    assert discovered["final_layout_result_path"].endswith("final_layout_result.json")
    assert discovered["collision_metrics_json_path"].endswith(".json")
    assert discovered["collision_metrics"]["total_columns"] == 120
    assert discovered["collision_report_json_path"].endswith(".json")


def test_recovered_collision_metrics_unblock_structural_rework(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write(output_dir / "layout_revision" / "final_layout_result.json", {"桥位列表": []})
    _write(
        output_dir / "collision_detection" / "collision_metrics_design_result_20260915_195700.json",
        {"total_columns": 120, "conflict_column_count": 0, "has_collision": False},
    )

    state = {
        **_discover_existing_outputs(output_dir),
        "revision_target": StageName.STRUCTURAL_DESIGN.value,
    }
    view = build_coordinator_view(state)

    assert view.collision_detected_available is True
    assert view.layout_revision_result_available is True


def test_collision_evidence_accepts_report_only_state() -> None:
    # 只恢复出报告路径（没有指标字段）时也不能把结构设计整条链路堵死。
    view = build_coordinator_view(
        {
            "layout_result": {"桥位列表": []},
            "layout_revision_result": {"桥位列表": []},
            "collision_report_json_path": "collision_detection/collision_report_x.json",
        }
    )

    assert view.collision_detected_available is True


# --------------------------------------------------------------------------- #
# 2. 验算失败 → 自动返修请求必须带最小重做范围
# --------------------------------------------------------------------------- #
def _capacity_check_result() -> dict:
    return {
        "check_type": "cap_beam_capacity_envelope_batch",
        "stage_complete": True,
        "expected_task_count": 2,
        "completed_task_count": 2,
        "failed_task_count": 0,
        "failed_task_ids": [],
        "failed_check_task_ids": ["2-2-3-G2"],
        "overall_check": {"M_pos_ok": True, "M_neg_ok": False, "all_ok": False},
        "utilization_summary": {
            "max_utilization": 1.189,
            "control_name": "max_negative_moment_utilization",
            "task_id": "2-2-3-G2",
            "util_type": "util_M_neg",
        },
        "control_sections": {},
        "task_results": [],
    }


def _state_with_units(output_dir: Path) -> dict:
    return {
        "output_dir": str(output_dir),
        "design_units": {
            "任务1_设计单元提取结果": {
                "桥梁列表": [
                    {"设计单元列表": [{"单元编号": "1-1"}, {"单元编号": "2-2"}, {"单元编号": "2-6"}]}
                ]
            }
        },
        "dimension_design_result": {
            "任务2_下部结构尺寸设计结果": {
                "单元原始结果": [
                    {"single_unit_input": {"单元编号": "1-1"}},
                ]
            }
        },
        "reinforcement_design_result": {"任务3_下部结构配筋设计结果": {"分组原始结果": []}},
        "check_result": _capacity_check_result(),
        "capacity_check_result": _capacity_check_result(),
    }


def test_check_failure_rework_context_carries_minimal_retry_scope(tmp_path: Path) -> None:
    state = _state_with_units(tmp_path)

    update = generate_modeling_feedback_action(state)
    context = update["revision_context"]

    assert context["target_step"] == "reinforcement_design"
    assert context["revision_type"] == "revise_reinforcement"
    # 最小返修范围：只重做验算失败的那一组配筋，而不是整批
    assert context["reinforcement_retry_task_ids"] == ["2-2-3-G2"]
    assert context["dimension_retry_unit_ids"] == ["2-2"]

    # 落盘的 revision_context.json 同样要带上重试范围（上一次运行缺的正是这个字段）
    written = json.loads(
        (tmp_path / "modeling_check" / "revision_context.json").read_text(encoding="utf-8")
    )
    assert written["reinforcement_retry_task_ids"] == ["2-2-3-G2"]


def test_rework_scope_limits_reinforcement_rerun_to_failed_group(monkeypatch, tmp_path: Path) -> None:
    from bridge_agents import tool_actions

    captured: dict = {}

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

    state = _state_with_units(tmp_path)
    state.update(generate_modeling_feedback_action(state))
    state["dimension_design_result"]["任务2_下部结构尺寸设计结果"]["单元原始结果"] = [
        {"single_unit_input": {"单元编号": "2-2"}}
    ]
    tool_actions.reinforcement_design_action(state)

    assert captured["retry_task_ids"] == ["2-2-3-G2"]
    # invoke_tool 会丢弃值为 None 的可选参数
    assert captured.get("retry_unit_ids") is None
    assert captured["existing_reinforcement_design_result"] is not None


# --------------------------------------------------------------------------- #
# 3. 人工"继续返修"必须回到 Handoff 指定的结构设计阶段
# --------------------------------------------------------------------------- #
def _modeling_review_state_with_structural_rework() -> dict:
    return {
        "user_request": "请完成全流程设计",
        "user_intent": "full_design",
        "task_status": "modeling_check_revision_required",
        "check_iteration_index": 1,
        "max_check_revision_rounds": 2,
        "max_modeling_check_steps": 6,
        "feedback_decision": {
            "overall_status": "revise_reinforcement",
            "next_action": "revise_reinforcement",
            "target_agent": "StructuralDesignAgent",
            "target_step": "reinforcement_design",
        },
        "revision_context": {
            "source_agent": "ModelingCheckAgent",
            "target_agent": "StructuralDesignAgent",
            "target_step": "reinforcement_design",
            "revision_type": "revise_reinforcement",
            "reinforcement_retry_task_ids": ["2-2-3-G2"],
            "dimension_retry_unit_ids": ["2-2"],
        },
        "revision_target": StageName.STRUCTURAL_DESIGN.value,
        "rework_required_artifacts": ["reinforcement_design_result"],
        "latest_handoff": {
            "stage": StageName.MODELING_CHECK.value,
            "status": "revision_required",
            "message": "建模验算未通过，需返回结构设计阶段做配筋修订。",
            "revision_request": {
                "target_stage": StageName.STRUCTURAL_DESIGN.value,
                "required_artifacts": ["reinforcement_design_result"],
            },
        },
    }


def test_discovery_marks_incomplete_drawing_index_as_not_successful(tmp_path: Path) -> None:
    """只写出部分组的图纸索引不能被当成"图纸已完成"，否则后续运行会假完成。"""
    output_dir = tmp_path / "output"
    _write(output_dir / "layout_revision" / "final_layout_result.json", {"桥位列表": []})
    _write(
        output_dir / "structural_design" / "reinforcement_design" / "reinforcement_design_result.json",
        {
            "任务3_下部结构配筋设计结果": {
                "分组原始结果": [
                    {"reinforcement_task": {"task_id": "1-1-1-G1"}},
                    {"reinforcement_task": {"task_id": "1-1-1-G2"}},
                ]
            }
        },
    )
    _write(
        output_dir / "deliverables" / "drawings" / "drawing_index.json",
        {"groups": [{"design_group_id": "1-1-1-G1"}]},
    )

    partial = _discover_existing_outputs(output_dir)["drawing_package_result"]
    assert partial["success"] is False
    assert partial["missing_group_ids"] == ["1-1-1-G2"]

    _write(
        output_dir / "deliverables" / "drawings" / "drawing_index.json",
        {"groups": [{"design_group_id": "1-1-1-G1"}, {"design_group_id": "1-1-1-G2"}]},
    )
    complete = _discover_existing_outputs(output_dir)["drawing_package_result"]
    assert complete["success"] is True
    assert complete["missing_group_ids"] == []


def test_reinforcement_rework_escalates_to_dimension_after_retry(tmp_path: Path) -> None:
    """配筋返修后承载力仍不通过时，反馈决策必须升级到尺寸设计，避免结构↔验算无限循环。"""
    from bridge_agents import tool_actions

    check = _capacity_check_result()
    check["utilization_summary"]["max_utilization"] = 1.714
    base_state = {
        "output_dir": str(tmp_path),
        "check_result": check,
        "capacity_check_result": check,
        "reinforcement_design_result": {"任务3_下部结构配筋设计结果": {"分组原始结果": []}},
    }

    first = tool_actions.generate_modeling_feedback_action(base_state)
    assert first["feedback_decision"]["next_action"] == "revise_reinforcement"

    retried = tool_actions.generate_modeling_feedback_action(
        {**base_state, "stage_revision_rounds": {"structural_design": 1}}
    )
    assert retried["feedback_decision"]["next_action"] == "revise_dimension"
    assert retried["revision_context"]["target_step"] == "dimension_design"
    assert retried["feedback_decision"]["decision_basis"] == "reinforcement_rework_exhausted"


def test_continue_modeling_revision_grants_one_structural_round_not_inflated() -> None:
    """结构返修路由的预算应按已用结构返修次数+1，而不是用 check_iteration_index 抬。"""
    state = _modeling_review_state_with_structural_rework()
    state.update(
        {
            "check_iteration_index": 5,
            "stage_revision_rounds": {StageName.STRUCTURAL_DESIGN.value: 2},
            "max_check_revision_rounds": 2,
        }
    )

    update = apply_human_review_decision(
        state, {"action": "continue_modeling_revision", "extra_rounds": 1}
    )

    # 2（已用结构返修） + 1，而不是 5（check_iteration_index）+ 1
    assert update["max_check_revision_rounds"] == 3


def test_accept_check_and_finish_clears_rework_invalidation(tmp_path: Path) -> None:
    """人工接受验算风险后必须清空失效标记，否则 final_output 会静默跳过出图。"""
    state = _modeling_review_state_after_invalidation(tmp_path)
    _write(tmp_path / "capacity_check" / "capacity_check_batch_summary.json", _capacity_check_result())
    state["invalidated_artifacts"] = [
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    ]
    state["revision_target"] = StageName.STRUCTURAL_DESIGN.value

    update = apply_human_review_decision(state, {"action": "accept_check_and_finish"})

    assert update["invalidated_artifacts"] == []
    assert update["revision_target"] is None
    assert update["rework_required_artifacts"] == []


def test_final_output_restores_cleared_artifacts_from_disk(tmp_path: Path) -> None:
    """人工接受风险后，final_output 应从磁盘补齐被返修清空的成果字段。"""
    from bridge_agents.graph_v2 import _restore_delivery_inputs

    output_dir = tmp_path / "output"
    _write(
        output_dir / "structural_design" / "reinforcement_design" / "reinforcement_design_result.json",
        {"任务3_下部结构配筋设计结果": {"分组原始结果": []}},
    )
    _write(
        output_dir / "structural_design" / "dimension_design" / "dimension_design_result.json",
        {"任务2_下部结构尺寸设计结果": {}},
    )
    _write(output_dir / "capacity_check" / "capacity_check_batch_summary.json", _capacity_check_result())

    restored = _restore_delivery_inputs(
        {
            "output_dir": str(output_dir),
            "reinforcement_design_result": None,
            "check_result": None,
            "pier_group_result": {"already": True},
        }
    )

    assert restored["reinforcement_design_result"]["任务3_下部结构配筋设计结果"] is not None
    assert restored["check_result"]["overall_check"]["all_ok"] is False
    # 已有值的字段不覆盖
    assert restored["pier_group_result"] == {"already": True}


def test_collision_review_message_lists_failing_thresholds() -> None:
    from bridge_agents.graph_v2 import _collision_threshold_summary

    summary = _collision_threshold_summary(
        {
            "conflict_column_rate": 0.0427,
            "total_intrusion_depth_columns": 27.39,
            "avg_intrusion_depth_columns": 3.91,
            "avg_overlap_ratio_columns": 0.729,
        }
    )

    assert "总侵入深度" in summary and "未达标" in summary
    assert "冲突柱率" in summary and "通过" in summary
    assert "平均重叠率" in summary


def test_review_request_exposes_pending_rework_target() -> None:
    request = build_human_review_request(_modeling_review_state_with_structural_rework())

    assert request["review_type"] == "modeling_check_review"
    assert request["rework_target"] == StageName.STRUCTURAL_DESIGN.value
    assert "结构设计" in request["message"]


def test_continue_modeling_revision_returns_to_pending_structural_rework() -> None:
    state = _modeling_review_state_with_structural_rework()

    update = apply_human_review_decision(state, {"action": "continue_modeling_revision", "extra_rounds": 1})

    assert update["human_review_route"] == StageName.STRUCTURAL_DESIGN.value
    assert route_after_human_review({**state, **update}) == StageName.STRUCTURAL_DESIGN.value
    # 返修上下文（含最小重做范围）必须保留，供结构阶段重新规划 required_steps
    assert update["revision_context"]["target_step"] == "reinforcement_design"
    assert update["revision_context"]["reinforcement_retry_task_ids"] == ["2-2-3-G2"]
    assert update["revision_target"] == StageName.STRUCTURAL_DESIGN.value
    assert update["max_check_revision_rounds"] == 2


def test_continue_modeling_revision_keeps_modeling_route_without_rework_target() -> None:
    state = {
        "user_request": "请完成全流程设计",
        "user_intent": "full_design",
        "check_iteration_index": 3,
        "max_check_revision_rounds": 3,
        "latest_handoff": {"stage": StageName.MODELING_CHECK.value, "status": "manual_review"},
    }

    update = apply_human_review_decision(
        state, {"action": "continue_modeling_revision", "extra_rounds": 2}
    )

    assert update["human_review_route"] == StageName.MODELING_CHECK.value
    assert update["revision_context"]["target_step"] == "modeling_check"
    assert update["max_check_revision_rounds"] == 5


# --------------------------------------------------------------------------- #
# 4. 成果被返修失效后仍可用磁盘证据接受风险并结束
# --------------------------------------------------------------------------- #
def _modeling_review_state_after_invalidation(output_dir: Path) -> dict:
    return {
        "user_request": "请完成全流程设计",
        "user_intent": "full_design",
        "task_status": "modeling_check_revision_required",
        "output_dir": str(output_dir),
        "capacity_check_result": None,
        "check_result": None,
        "capacity_batch_status": {"stage_complete": True},
        "latest_handoff": {
            "stage": StageName.MODELING_CHECK.value,
            "status": "revision_required",
            "produced_artifacts": [
                {
                    "artifact_type": "check_result",
                    "state_key": "check_result",
                    "path": str(output_dir / "capacity_check" / "capacity_check_batch_summary.json"),
                    "valid": True,
                }
            ],
            "message": "建模验算未通过，需返回结构设计阶段做配筋修订。",
        },
    }


def test_accept_check_and_finish_uses_on_disk_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write(output_dir / "capacity_check" / "capacity_check_batch_summary.json", _capacity_check_result())
    state = _modeling_review_state_after_invalidation(output_dir)

    update = apply_human_review_decision(
        state, {"action": "accept_check_and_finish", "feedback": "接受已知负弯矩超限风险"}
    )

    assert update["human_review_route"] == StageName.FINAL_OUTPUT.value
    risk = update["accepted_risks"][0]
    assert risk["scope"] == "modeling_check"
    assert risk["check_summary"]["overall_check"]["all_ok"] is False
    assert risk["evidence_path"].endswith("capacity_check_batch_summary.json")


def test_accept_check_and_finish_still_rejects_when_no_evidence(tmp_path: Path) -> None:
    state = _modeling_review_state_after_invalidation(tmp_path / "empty_output")
    state["latest_handoff"]["produced_artifacts"] = []

    with pytest.raises(ValueError, match="完整承载力验算结果"):
        apply_human_review_decision(state, {"action": "accept_check_and_finish"})


# --------------------------------------------------------------------------- #
# 5. 端到端：恢复 → 返修 → 复验（不再转入人工复核）
# --------------------------------------------------------------------------- #
def test_graph_runs_reinforcement_rework_and_recheck_without_manual_review(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write(output_dir / "layout_revision" / "final_layout_result.json", {"桥位列表": []})
    _write(
        output_dir / "collision_detection" / "collision_metrics_design_result_20260915_195700.json",
        {"total_columns": 120, "conflict_column_count": 0, "has_collision": False},
    )
    recovered = _discover_existing_outputs(output_dir)

    responses = iter(
        [
            # 协调器：执行结构返修（返修目标由 Handoff/state 指定，必须与原目标一致）
            json.dumps(
                {
                    "intent": "full_design",
                    "next_stage": StageName.STRUCTURAL_DESIGN.value,
                    "stage_objective": "按最小范围重做配筋",
                    "reason": "验算失败要求返修配筋",
                    "revision_target": StageName.STRUCTURAL_DESIGN.value,
                    "required_artifacts": ["reinforcement_design_result"],
                    "task_complete": False,
                },
                ensure_ascii=False,
            ),
            # 返修完成后：进入复验
            json.dumps(
                {
                    "intent": "full_design",
                    "next_stage": StageName.MODELING_CHECK.value,
                    "stage_objective": "复验承载力",
                    "reason": "配筋已返修",
                    "revision_target": None,
                    "required_artifacts": [],
                    "task_complete": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "intent": "full_design",
                    "next_stage": None,
                    "stage_objective": "完成任务",
                    "reason": "复验通过",
                    "revision_target": None,
                    "required_artifacts": [],
                    "task_complete": True,
                },
                ensure_ascii=False,
            ),
        ]
    )
    executed: list = []

    def structural_runner(state):
        executed.append(StageName.STRUCTURAL_DESIGN.value)
        # 返修范围必须来自建模阶段生成的上下文，而不是整批重跑
        assert state["revision_context"]["reinforcement_retry_task_ids"] == ["2-2-3-G2"]
        return {
            "reinforcement_design_result": {"status": "reworked"},
            "reinforcement_batch_status": {"stage_complete": True, "failed_task_ids": []},
            "opensees_force_json_path": "structural_design/forces.json",
            "task_status": "structural_design_completed",
        }

    def modeling_runner(_state):
        executed.append(StageName.MODELING_CHECK.value)
        return {
            "check_result": {"overall_check": {"all_ok": True}, "stage_complete": True},
            "capacity_check_result": {"overall_check": {"all_ok": True}, "stage_complete": True},
            "capacity_batch_status": {"stage_complete": True},
            "task_status": "modeling_check_completed",
        }

    stage_runners = {
        StageName.INITIAL_DESIGN.value: lambda state: {"task_status": "initial_design_completed"},
        StageName.LAYOUT_REVISION.value: lambda state: {"task_status": "layout_revision_completed"},
        StageName.STRUCTURAL_DESIGN.value: structural_runner,
        StageName.MODELING_CHECK.value: modeling_runner,
    }
    graph = build_graph_v2(
        llm_invoke=lambda _system, _user: next(responses),
        stage_runners=stage_runners,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            **recovered,
            "user_request": "请完成全流程设计",
            "user_intent": "full_design",
            "task_status": "modeling_check_revision_required",
            "feedback_decision": {"next_action": "revise_reinforcement"},
            "revision_context": {
                "target_step": "reinforcement_design",
                "reinforcement_retry_task_ids": ["2-2-3-G2"],
                "dimension_retry_unit_ids": ["2-2"],
            },
            "revision_target": StageName.STRUCTURAL_DESIGN.value,
            "rework_required_artifacts": ["reinforcement_design_result"],
            "reinforcement_design_result": {"status": "old"},
            "reinforcement_batch_status": {"stage_complete": False, "failed_task_ids": []},
        },
        config={"configurable": {"thread_id": "rework-recheck-1"}},
    )

    # 关键：没有转入人工复核，返修与复验都真实执行了
    assert "__interrupt__" not in result
    assert executed == [StageName.STRUCTURAL_DESIGN.value, StageName.MODELING_CHECK.value]
    assert result["revision_target"] is None


# --------------------------------------------------------------------------- #
# 6. 断点恢复：checkpoint 里缺失、磁盘上已有的成果要补回来（且不覆盖、不复活已失效成果）
# --------------------------------------------------------------------------- #
class _FakeSnapshot:
    def __init__(self, values: dict) -> None:
        self.values = values


class _FakeGraph:
    def __init__(self, values: dict) -> None:
        self._values = values

    def get_state(self, _config) -> _FakeSnapshot:
        return _FakeSnapshot(self._values)


def test_resume_recovery_patch_fills_missing_but_not_invalidated_artifacts(tmp_path: Path) -> None:
    from bridge_agents.agent import _resume_recovery_patch

    output_dir = tmp_path / "output"
    _write(output_dir / "layout_revision" / "final_layout_result.json", {"桥位列表": []})
    _write(
        output_dir / "collision_detection" / "collision_metrics_design_result_20260915_195700.json",
        {"total_columns": 120, "has_collision": False},
    )
    _write(output_dir / "capacity_check" / "capacity_check_batch_summary.json", _capacity_check_result())
    discovered = _discover_existing_outputs(output_dir)

    checkpoint_values = {
        # 9/15 运行的真实状态：碰撞指标为空、验算结果被返修清空
        "collision_metrics": None,
        "collision_metrics_json_path": None,
        "check_result": None,
        "capacity_check_result": None,
        "invalidated_artifacts": ["reinforcement_design_result", "check_result"],
        "layout_result": {"已有": True},  # checkpoint 已有的值不得被覆盖
        "task_status": "modeling_check_revision_required",
    }

    patch = _resume_recovery_patch(_FakeGraph(checkpoint_values), {}, discovered)

    assert patch["collision_metrics"]["total_columns"] == 120
    assert patch["collision_metrics_json_path"].endswith(".json")
    # check_result / capacity_check_result 属于"已失效待重做"的成果，不能补回
    assert "check_result" not in patch
    assert "capacity_check_result" not in patch
    # checkpoint 已有的成果不被覆盖
    assert "layout_result" not in patch
    # 流程状态字段不参与补回
    assert "task_status" not in patch


# --------------------------------------------------------------------------- #
# 7. 返修范围与合并基准：上一轮生成失败 + 本轮验算未通过都要重做，其余组沿用磁盘成果
# --------------------------------------------------------------------------- #
def _write_previous_reinforcement_summary(output_dir: Path) -> Path:
    path = output_dir / "structural_design" / "reinforcement_design" / "reinforcement_design_result.json"
    _write(
        path,
        {
            "任务3_下部结构配筋设计结果": {
                "分组原始结果": [
                    {
                        "task_index": 1,
                        "reinforcement_task": {"task_id": "2-2-3-G1", "单元编号": "2-3"},
                        "reinforcement_result": {"reinforcement": {"pier_cap": {}, "pier_column": {}}},
                        "output_files": {
                            "reinforcement_result_yaml_path": str(output_dir / "y.yaml"),
                            "internal_force_output_path": str(output_dir / "f.json"),
                        },
                    }
                ],
                "失败分组": [
                    {"task_id": "2-2-1-G1", "单元编号": "2-1", "error": "FormulaInputError"},
                ],
                "expected_task_count": 13,
                "completed_task_count": 12,
                "failed_task_count": 1,
                "stage_complete": False,
            }
        },
    )
    return path


def test_rework_scope_unions_previous_and_check_failures(tmp_path: Path) -> None:
    from bridge_agents.tool_actions import _reinforcement_retry_task_ids

    output_dir = tmp_path / "output"
    _write_previous_reinforcement_summary(output_dir)
    _write(output_dir / "capacity_check" / "capacity_check_batch_summary.json", _capacity_check_result())

    state = {
        "output_dir": str(output_dir),
        "latest_handoff": {
            "produced_artifacts": [
                {
                    "artifact_type": "check_result",
                    "path": str(output_dir / "capacity_check" / "capacity_check_batch_summary.json"),
                }
            ]
        },
        "revision_context": {"target_step": "reinforcement_design"},
    }

    assert _reinforcement_retry_task_ids(state, state["revision_context"]) == [
        "2-2-1-G1",
        "2-2-3-G2",
    ]


def test_partial_rework_uses_disk_summary_as_merge_base(monkeypatch, tmp_path: Path) -> None:
    from bridge_agents import tool_actions

    output_dir = tmp_path / "output"
    _write_previous_reinforcement_summary(output_dir)
    captured: dict = {}

    def fake_tool(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "reinforcement_design_result": {"ok": True},
            "expected_task_count": 13,
            "completed_task_count": 13,
            "failed_task_count": 0,
            "stage_complete": True,
            "output_files": {},
        }

    monkeypatch.setattr(tool_actions, "reinforcement_design_tool", fake_tool)

    tool_actions.reinforcement_design_action(
        {
            "output_dir": str(output_dir),
            "design_units": {"ok": True},
            "dimension_design_result": {"任务2_下部结构尺寸设计结果": {"单元原始结果": []}},
            # 返修失效已把内存成果清空
            "reinforcement_design_result": None,
            "revision_context": {"target_step": "reinforcement_design"},
        }
    )

    merge_base = captured["existing_reinforcement_design_result"]
    assert merge_base is not None
    rows = merge_base["任务3_下部结构配筋设计结果"]["分组原始结果"]
    assert [row["reinforcement_task"]["task_id"] for row in rows] == ["2-2-3-G1"]


def test_review_request_recovers_batch_status_from_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_previous_reinforcement_summary(output_dir)
    from bridge_agents.tool_actions import _load_reinforcement_summary

    request = build_human_review_request(
        {
            "user_intent": "full_design",
            "task_status": "manual_review_required",
            "reinforcement_design_result": _load_reinforcement_summary({"output_dir": str(output_dir)}),
            "latest_handoff": {
                "stage": StageName.STRUCTURAL_DESIGN.value,
                "status": "manual_review",
                "message": "配筋批次未完成",
            },
        }
    )

    assert request["review_type"] == "structural_batch_review"
    assert request["failed_task_ids"] == ["2-2-1-G1"]
    assert request["reinforcement_batch_status"]["expected_task_count"] == 13


def test_non_positive_net_height_is_not_applicable_not_failure() -> None:
    from bridge_agents.joint_reinforcement import compute_axial_capacity_check

    check = compute_axial_capacity_check(
        controlling_axial_force_kN=5000.0,
        column_diameter_mm=1400.0,
        net_height_m=-0.91,
        longitudinal_bars=[{"dia": 28, "count": 16}],
    )

    # 净高非正 = 无有效柱身（桥台/埋入式墩）：柱身稳定验算不适用，
    # 既不作废整组成果，也不作为待人工复核项阻塞收尾。
    assert check["status"] == "not_applicable"
    assert check["check_ok"] is None
    assert check["slenderness_ratio"] is None
    assert "净高" in check["message"]


def test_accepted_partial_batches_do_not_mask_modeling_review() -> None:
    """人工已接受的结构/尺寸缺口不应压住承载力侧复核，否则复验通过也无法收尾。"""
    base = {
        "user_intent": "full_design",
        "task_status": "manual_review_required",
        "reinforcement_batch_status": {"stage_complete": True, "human_accepted": True},
        "latest_handoff": {
            "stage": StageName.MODELING_CHECK.value,
            "status": "manual_review",
            "message": "承载力验算通过，但墩柱长细比超出表5.3.1适用范围，需人工复核决定接受风险或返回修订。",
        },
    }

    accepted = build_human_review_request(
        {
            **base,
            "dimension_batch_status": {
                "stage_complete": False,
                "failed_unit_ids": ["2-2", "2-6"],
                "human_accepted": True,
            },
        }
    )
    assert accepted["review_type"] == "modeling_check_review"
    assert "accept_check_and_finish" in accepted["available_actions"]
    assert "continue_modeling_revision" in accepted["available_actions"]

    # 未被人工接受时，仍按结构类复核处理（先补尺寸/配筋再谈收尾）
    blocked = build_human_review_request(
        {
            **base,
            "dimension_batch_status": {"stage_complete": False, "failed_unit_ids": ["2-2", "2-6"]},
        }
    )
    assert blocked["review_type"] == "structural_batch_review"
    assert "retry_failed_tasks" in blocked["available_actions"]
    assert blocked["failed_dimension_unit_ids"] == ["2-2", "2-6"]


def test_graph_still_pauses_when_collision_evidence_missing(tmp_path: Path) -> None:
    """反例：碰撞指标确实缺失时，结构返修必须停在人工复核而不是静默空转。"""
    responses = iter([])
    ran: list = []

    stage_runners = {
        StageName.INITIAL_DESIGN.value: lambda state: {"task_status": "initial_design_completed"},
        StageName.LAYOUT_REVISION.value: lambda state: {"task_status": "layout_revision_completed"},
        StageName.STRUCTURAL_DESIGN.value: lambda state: ran.append(StageName.STRUCTURAL_DESIGN.value)
        or {"task_status": "structural_design_completed"},
        StageName.MODELING_CHECK.value: lambda state: {"task_status": "modeling_check_completed"},
    }
    graph = build_graph_v2(
        llm_invoke=lambda _system, _user: next(responses, "{}"),
        stage_runners=stage_runners,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "user_request": "请完成全流程设计",
            "user_intent": "full_design",
            "task_status": "modeling_check_revision_required",
            "revision_target": StageName.STRUCTURAL_DESIGN.value,
            "rework_required_artifacts": ["reinforcement_design_result"],
            "latest_handoff": {
                "stage": StageName.MODELING_CHECK.value,
                "status": "revision_required",
                "revision_request": {
                    "target_stage": StageName.STRUCTURAL_DESIGN.value,
                    "required_artifacts": ["reinforcement_design_result"],
                },
            },
        },
        config={"configurable": {"thread_id": "missing-collision-1"}},
    )

    assert result["__interrupt__"][0].value["review_type"] in {
        "modeling_check_review",
        "coordinator_review",
        "structural_batch_review",
    }
    assert ran == []
