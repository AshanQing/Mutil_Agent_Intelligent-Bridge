from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from bridge_agents.contracts import StageName
from bridge_agents.design_coordinator import DesignCoordinatorAgent
from bridge_agents.graph_v2 import (
    apply_human_review_decision,
    build_graph_v2,
    build_human_review_request,
    final_output_node,
    route_after_human_review,
)


REVIEW_STATE = {
    "user_request": "请完成全流程设计",
    "user_intent": "full_design",
    "layout_result": {"桥位列表": [{"桥位编号": 5}]},
    "revision_result_path": "revision_results/revision_design_round_3.json",
    "iteration_index": 3,
    "max_revision_rounds": 3,
    "collision_metrics": {
        "conflict_column_rate": 0.0656,
        "total_intrusion_depth_columns": 12.597,
        "avg_intrusion_depth_columns": 1.575,
        "avg_overlap_ratio_columns": 0.638,
    },
    "collision_items": [
        {"bridge_id": 5, "station": "K7+890", "intrusion_depth": 2.1},
        {"bridge_id": 5, "station": "K7+920", "intrusion_depth": 2.0},
    ],
    "latest_handoff": {
        "stage": StageName.LAYOUT_REVISION.value,
        "status": "manual_review",
        "message": "自动修正未收敛",
    },
}


def test_axial_pending_without_handoff_yields_modeling_review_actions() -> None:
    # 成果恢复/断点场景：latest_handoff 为空、无碰撞指标，但墩柱长细比复核未决，
    # 人工复核必须提供“接受风险并完成”，而不是只给 retry/abort。
    state = {
        "user_intent": "full_design",
        "layout_result": {"桥位列表": [{"桥位编号": 1}]},
        "layout_revision_result": {"桥位列表": [{"桥位编号": 1}]},
        "reinforcement_design_result": {
            "任务3_下部结构配筋设计结果": {
                "分组原始结果": [
                    {
                        "reinforcement_task": {"task_id": "B1-U1-G1"},
                        "axial_check": {"status": "manual_review"},
                    }
                ]
            }
        },
        "capacity_check_result": {
            "stage_complete": True,
            "overall_check": {"all_ok": True},
            "task_results": [],
        },
        "check_result": {"overall_check": {"all_ok": True}},
    }
    request = build_human_review_request(state)
    assert request["review_type"] == "modeling_check_review"
    assert "accept_check_and_finish" in request["available_actions"]
    assert "continue_modeling_revision" in request["available_actions"]

    accepted = build_human_review_request(
        {**state, "accepted_risks": [{"scope": "modeling_check", "reason": "人工接受"}]}
    )
    assert "accept_check_and_finish" not in accepted["available_actions"]


def test_human_review_request_exposes_metrics_and_actions() -> None:
    request = build_human_review_request(REVIEW_STATE)

    assert request["review_type"] == "layout_collision_review"
    assert request["metrics"]["conflict_column_rate"] == 0.0656
    assert request["current_round"] == 3
    assert request["remaining_conflict_count"] == 2
    assert request["available_actions"] == [
        "continue_revision",
        "accept_and_continue",
        "abort",
    ]


def test_continue_revision_authorizes_extra_round_and_preserves_metrics() -> None:
    update = apply_human_review_decision(REVIEW_STATE, {
        "action": "continue_revision",
        "extra_rounds": 1,
        "feedback": "调整桥位5边界",
    })

    assert update["max_revision_rounds"] == 4
    assert update["collision_metrics"] == REVIEW_STATE["collision_metrics"]
    assert update["human_review_feedback"] == "调整桥位5边界"
    assert update["task_status"] == "layout_revision_required"
    assert route_after_human_review({**REVIEW_STATE, **update}) == StageName.LAYOUT_REVISION.value


def test_accept_current_records_override_without_changing_failed_metrics() -> None:
    update = apply_human_review_decision(
        REVIEW_STATE,
        {"action": "accept_and_continue", "feedback": "人工接受剩余风险"},
    )

    assert update["human_override"] is True
    assert update["layout_revision_completed"] is True
    assert update["final_layout_result"] == REVIEW_STATE["layout_result"]
    assert update["collision_metrics"] == REVIEW_STATE["collision_metrics"]
    assert update["accepted_risks"][0]["scope"] == "layout_revision"
    assert update["latest_handoff"]["status"] == "completed"
    assert route_after_human_review({**REVIEW_STATE, **update}) == "design_coordinator"


def test_accept_current_can_enter_structural_design_after_layout_round_limit() -> None:
    update = apply_human_review_decision(
        REVIEW_STATE,
        {"action": "accept_and_continue", "feedback": "人工接受剩余风险"},
    )
    accepted_state = {**REVIEW_STATE, **update}
    coordinator = DesignCoordinatorAgent(
        lambda system, user: _decision(StageName.STRUCTURAL_DESIGN.value)
    )

    decision = coordinator.decide(
        accepted_state,
        accepted_state["user_request"],
    )

    assert decision.next_stage == StageName.STRUCTURAL_DESIGN.value


def test_abort_routes_to_controlled_end() -> None:
    update = apply_human_review_decision(REVIEW_STATE, {"action": "abort"})

    assert update["task_status"] == "cancelled"
    assert update["error"] is None
    assert route_after_human_review({**REVIEW_STATE, **update}) == "abort"


def test_invalid_human_review_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="非法人工复核决策"):
        apply_human_review_decision(REVIEW_STATE, {"action": "skip_checks"})


def _decision(next_stage: str) -> str:
    return json.dumps({
        "intent": "full_design",
        "next_stage": next_stage,
        "stage_objective": "人工复核",
        "reason": "测试",
        "revision_target": None,
        "required_artifacts": [],
        "task_complete": False,
    }, ensure_ascii=False)


def _unused_stage_runners():
    return {
        stage.value: lambda state: {"task_status": f"{stage.value}_completed"}
        for stage in (
            StageName.INITIAL_DESIGN,
            StageName.LAYOUT_REVISION,
            StageName.STRUCTURAL_DESIGN,
            StageName.MODELING_CHECK,
        )
    }


def test_graph_interrupts_and_resumes_abort_with_same_thread() -> None:
    graph = build_graph_v2(
        llm_invoke=lambda system, user: _decision(StageName.MANUAL_REVIEW.value),
        stage_runners=_unused_stage_runners(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "review-abort-1"}}

    paused = graph.invoke(REVIEW_STATE, config=config)

    assert paused["__interrupt__"][0].value["review_type"] == "layout_collision_review"

    resumed = graph.invoke(
        Command(resume={"action": "abort"}),
        config=config,
    )

    assert resumed["task_status"] == "cancelled"
    assert resumed["human_review_decision"]["action"] == "abort"


def test_graph_accepts_layout_risk_and_completes_downstream_stages() -> None:
    responses = iter([
        _decision(StageName.MANUAL_REVIEW.value),
        _decision(StageName.STRUCTURAL_DESIGN.value),
        _decision(StageName.MODELING_CHECK.value),
        json.dumps({
            "intent": "full_design",
            "next_stage": None,
            "stage_objective": "完成任务",
            "reason": "后续设计与验算已完成",
            "revision_target": None,
            "required_artifacts": [],
            "task_complete": True,
        }, ensure_ascii=False),
    ])
    executed_stages = []

    def structural_runner(state):
        executed_stages.append(StageName.STRUCTURAL_DESIGN.value)
        return {
            "reinforcement_design_result": {"status": "completed"},
            "opensees_force_json_path": "opensees/forces.json",
            "task_status": "structural_design_completed",
        }

    def modeling_runner(state):
        executed_stages.append(StageName.MODELING_CHECK.value)
        return {
            "check_result": {"overall_check": True},
            "capacity_check_result": {"overall_check": True},
            "task_status": "modeling_check_completed",
        }

    stage_runners = _unused_stage_runners()
    stage_runners.update({
        StageName.STRUCTURAL_DESIGN.value: structural_runner,
        StageName.MODELING_CHECK.value: modeling_runner,
    })
    graph = build_graph_v2(
        llm_invoke=lambda system, user: next(responses),
        stage_runners=stage_runners,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "review-accept-1"}}

    paused = graph.invoke(REVIEW_STATE, config=config)
    assert paused["__interrupt__"][0].value["review_type"] == "layout_collision_review"

    resumed = graph.invoke(
        Command(resume={"action": "accept_and_continue"}),
        config=config,
    )

    assert resumed["task_status"] == "completed_with_accepted_risks"
    assert executed_stages == [
        StageName.STRUCTURAL_DESIGN.value,
        StageName.MODELING_CHECK.value,
    ]
    assert resumed["human_review_decision"]["action"] == "accept_and_continue"
    assert resumed["collision_metrics"] == REVIEW_STATE["collision_metrics"]


def test_restored_manual_review_state_interrupts_without_calling_coordinator_llm() -> None:
    calls = 0

    def unexpected_llm(system: str, user: str) -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("restored manual review must not call coordinator LLM")

    graph = build_graph_v2(
        llm_invoke=unexpected_llm,
        stage_runners=_unused_stage_runners(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "restored-review-1"}}

    paused = graph.invoke(
        {**REVIEW_STATE, "task_status": "manual_review_required"},
        config=config,
    )

    assert paused["__interrupt__"][0].value["current_round"] == 3
    assert calls == 0


STRUCTURAL_REVIEW_STATE = {
    "user_request": "请完成全流程设计",
    "user_intent": "full_design",
    "task_status": "manual_review_required",
    "layout_result": REVIEW_STATE["layout_result"],
    "layout_revision_result": REVIEW_STATE["layout_result"],
    "collision_metrics": REVIEW_STATE["collision_metrics"],
    "reinforcement_design_result": {"partial": True},
    "reinforcement_batch_status": {
        "expected_task_count": 18,
        "completed_task_count": 15,
        "failed_task_count": 3,
        "failed_task_ids": ["T3", "T8", "T17"],
        "stage_complete": False,
        "partial_result_available": True,
    },
    "failed_task_ids": ["T3", "T8", "T17"],
    "latest_handoff": {
        "stage": StageName.STRUCTURAL_DESIGN.value,
        "status": "manual_review",
        "message": "3 个配筋任务失败",
    },
}


def test_structural_review_takes_priority_over_historical_collision_metrics() -> None:
    request = build_human_review_request(STRUCTURAL_REVIEW_STATE)

    assert request["review_type"] == "structural_batch_review"
    assert request["failed_task_ids"] == ["T3", "T8", "T17"]
    assert request["available_actions"] == [
        "retry_failed_tasks",
        "accept_partial_and_continue",
        "abort",
    ]


def test_incomplete_dimension_batch_takes_priority_over_axial_review() -> None:
    state = {
        **STRUCTURAL_REVIEW_STATE,
        "reinforcement_batch_status": {"stage_complete": True},
        "dimension_batch_status": {
            "expected_unit_count": 8,
            "completed_unit_count": 7,
            "failed_unit_count": 1,
            "failed_unit_ids": ["1-3"],
            "stage_complete": False,
        },
        "failed_task_ids": ["1-3"],
        "reinforcement_design_result": {
            "任务3_下部结构配筋设计结果": {
                "分组原始结果": [
                    {
                        "reinforcement_task": {"task_id": "1-1-G1"},
                        "axial_check": {"status": "manual_review"},
                    }
                ]
            }
        },
    }

    request = build_human_review_request(state)

    assert request["review_type"] == "structural_batch_review"
    # 尺寸单元号与配筋任务号分属两套命名空间，必须分开下发。
    assert request["failed_dimension_unit_ids"] == ["1-3"]
    assert request["failed_task_ids"] == []
    assert "accept_check_and_finish" not in request["available_actions"]


def test_retry_failed_structural_tasks_routes_to_structural_design() -> None:
    update = apply_human_review_decision(
        STRUCTURAL_REVIEW_STATE,
        {"action": "retry_failed_tasks", "feedback": "检查三个无效输出"},
    )

    assert update["revision_target"] == StageName.STRUCTURAL_DESIGN.value
    assert update["failed_task_ids"] == ["T3", "T8", "T17"]
    assert update["revision_context"]["failed_task_ids"] == ["T3", "T8", "T17"]
    assert route_after_human_review({**STRUCTURAL_REVIEW_STATE, **update}) == StageName.STRUCTURAL_DESIGN.value


def test_retry_failed_dimension_unit_targets_dimension_design() -> None:
    state = {
        **STRUCTURAL_REVIEW_STATE,
        "dimension_batch_status": {
            "stage_complete": False,
            "failed_unit_ids": ["1-3"],
        },
        "reinforcement_batch_status": {"stage_complete": True},
        "failed_task_ids": ["1-3"],
    }

    update = apply_human_review_decision(state, {"action": "retry_failed_tasks"})

    assert update["revision_context"]["target_step"] == "dimension_design"
    assert update["revision_context"]["dimension_retry_unit_ids"] == ["1-3"]
    # 尺寸单元变了，对应配筋必然失效：返修集要同时覆盖尺寸与配筋两项。
    assert update["rework_required_artifacts"] == [
        "dimension_design_result",
        "reinforcement_design_result",
    ]


def test_retry_failed_tasks_keeps_dimension_and_reinforcement_ids_apart() -> None:
    # 复现真实故障场景：尺寸批次失败单元为 2-2/2-6，配筋批次失败任务为 2-2-1-G1。
    # 旧实现把配筋任务号当尺寸单元号下发，尺寸设计工具按 ID 匹配失败并让整阶段判失败，
    # 随后协调器反复重入结构设计形成死循环。
    state = {
        **STRUCTURAL_REVIEW_STATE,
        "dimension_batch_status": {
            "expected_unit_count": 9,
            "completed_unit_count": 7,
            "failed_unit_count": 2,
            "failed_unit_ids": ["2-2", "2-6"],
            "stage_complete": False,
            "partial_result_available": True,
        },
        "reinforcement_batch_status": {
            "expected_task_count": 13,
            "completed_task_count": 12,
            "failed_task_count": 1,
            "failed_task_ids": ["2-2-1-G1"],
            "stage_complete": False,
            "partial_result_available": True,
        },
        "failed_task_ids": ["2-2-1-G1"],
    }

    request = build_human_review_request(state)
    assert request["failed_dimension_unit_ids"] == ["2-2", "2-6"]
    assert request["failed_task_ids"] == ["2-2-1-G1"]

    update = apply_human_review_decision(state, {"action": "retry_failed_tasks"})
    context = update["revision_context"]

    # 顺序返修：先尺寸设计失败单元，再配筋设计失败任务。
    assert context["target_step"] == "dimension_design"
    assert context["dimension_retry_unit_ids"] == ["2-2", "2-6"]
    assert context["reinforcement_retry_task_ids"] == ["2-2-1-G1"]
    assert update["failed_task_ids"] == ["2-2-1-G1"]
    assert update["failed_dimension_unit_ids"] == ["2-2", "2-6"]
    assert update["rework_required_artifacts"] == [
        "dimension_design_result",
        "reinforcement_design_result",
    ]
    assert route_after_human_review({**state, **update}) == StageName.STRUCTURAL_DESIGN.value


def test_retry_failed_tasks_grants_one_more_revision_round() -> None:
    # 轮次超限后人工授权重试，必须同时放宽一轮预算；
    # 否则协调器会在重入结构设计前再次判超限并立刻转回人工复核（表现为“点了没反应”）。
    state = {
        **STRUCTURAL_REVIEW_STATE,
        "stage_revision_rounds": {StageName.STRUCTURAL_DESIGN.value: 2},
        "max_check_revision_rounds": 2,
    }

    update = apply_human_review_decision(state, {"action": "retry_failed_tasks"})

    assert update["max_check_revision_rounds"] == 3


def test_accept_partial_clears_rework_target() -> None:
    update = apply_human_review_decision(
        {
            **STRUCTURAL_REVIEW_STATE,
            "revision_target": StageName.STRUCTURAL_DESIGN.value,
            "rework_required_artifacts": ["reinforcement_design_result"],
        },
        {"action": "accept_partial_and_continue", "feedback": "接受部分成果"},
    )

    assert update["revision_target"] is None
    assert update["rework_required_artifacts"] == []


def test_accept_partial_structural_result_keeps_risk_and_allows_workflow() -> None:
    update = apply_human_review_decision(
        STRUCTURAL_REVIEW_STATE,
        {"action": "accept_partial_and_continue", "feedback": "仅用于方法流程试验"},
    )

    assert update["reinforcement_batch_status"]["stage_complete"] is False
    assert update["reinforcement_batch_status"]["human_accepted"] is True
    assert update["accepted_risks"][0]["scope"] == "structural_design"
    assert update["unresolved_manual_review"] is False
    assert route_after_human_review({**STRUCTURAL_REVIEW_STATE, **update}) == "design_coordinator"


def test_graph_resumes_structural_retry_and_reaches_completion() -> None:
    responses = iter(
        [
            _decision(StageName.MODELING_CHECK.value),
            json.dumps(
                {
                    "intent": "full_design",
                    "next_stage": None,
                    "stage_objective": "完成任务",
                    "reason": "重试和验算完成",
                    "revision_target": None,
                    "required_artifacts": [],
                    "task_complete": True,
                },
                ensure_ascii=False,
            ),
        ]
    )
    executed = []

    def structural_runner(state):
        executed.append(StageName.STRUCTURAL_DESIGN.value)
        assert state["failed_task_ids"] == ["T3", "T8", "T17"]
        return {
            "reinforcement_design_result": {"status": "completed"},
            "reinforcement_batch_status": {"stage_complete": True},
            "opensees_force_json_path": "forces.json",
            "task_status": "structural_design_completed",
        }

    def modeling_runner(_state):
        executed.append(StageName.MODELING_CHECK.value)
        return {
            "check_result": {"overall_check": True},
            "capacity_check_result": {"overall_check": True},
            "task_status": "modeling_check_completed",
        }

    runners = _unused_stage_runners()
    runners[StageName.STRUCTURAL_DESIGN.value] = structural_runner
    runners[StageName.MODELING_CHECK.value] = modeling_runner
    graph = build_graph_v2(
        llm_invoke=lambda _system, _user: next(responses),
        stage_runners=runners,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "structural-retry-1"}}

    paused = graph.invoke(STRUCTURAL_REVIEW_STATE, config=config)
    assert paused["__interrupt__"][0].value["review_type"] == "structural_batch_review"
    resumed = graph.invoke(
        Command(resume={"action": "retry_failed_tasks"}),
        config=config,
    )

    assert resumed["task_status"] == "completed"
    assert executed == [
        StageName.STRUCTURAL_DESIGN.value,
        StageName.MODELING_CHECK.value,
    ]


def test_rework_stage_failure_goes_back_to_manual_review_instead_of_looping() -> None:
    # 返修阶段再次失败时必须直接回到人工复核：
    # 旧行为是回协调器，而 revision_target 仍等于本阶段，协调器每轮都能合法重入，
    # 于是无限空转（真实运行中空转了 30 分钟、写了 1.1GB checkpoint）。
    executed = []

    def failing_structural_runner(state):
        executed.append(state.get("revision_target"))
        return {
            "task_status": "failed",
            "message": "StructuralDesignAgent 在 dimension_design 阶段失败。",
            "error": "待配筋尺寸设计单元未在当前设计单元中找到: ['2-2-1-G1']",
        }

    def unexpected_llm(_system: str, _user: str) -> str:
        raise AssertionError("failure path must not consult the coordinator LLM")

    runners = _unused_stage_runners()
    runners[StageName.STRUCTURAL_DESIGN.value] = failing_structural_runner
    graph = build_graph_v2(
        llm_invoke=unexpected_llm,
        stage_runners=runners,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "rework-failure-1"}}

    paused = graph.invoke(STRUCTURAL_REVIEW_STATE, config=config)
    assert paused["__interrupt__"][0].value["review_type"] == "structural_batch_review"

    resumed = graph.invoke(
        Command(resume={"action": "retry_failed_tasks"}),
        config=config,
    )

    assert executed == [StageName.STRUCTURAL_DESIGN.value]
    assert resumed["__interrupt__"][0].value["review_type"] == "structural_batch_review"
    assert resumed["rework_failed"] is True
    assert resumed["revision_target"] is None
    assert resumed["stage_revision_rounds"][StageName.STRUCTURAL_DESIGN.value] == 1


def test_structural_revision_round_limit_forces_manual_review() -> None:
    state = {
        **STRUCTURAL_REVIEW_STATE,
        "revision_target": StageName.STRUCTURAL_DESIGN.value,
        "rework_required_artifacts": ["reinforcement_design_result"],
        "stage_revision_rounds": {StageName.STRUCTURAL_DESIGN.value: 2},
        "max_check_revision_rounds": 2,
    }
    rework_decision = json.dumps(
        {
            "intent": "full_design",
            "next_stage": StageName.STRUCTURAL_DESIGN.value,
            "stage_objective": "执行结构返修",
            "reason": "存在返修目标",
            "revision_target": StageName.STRUCTURAL_DESIGN.value,
            "required_artifacts": ["reinforcement_design_result"],
            "task_complete": False,
        },
        ensure_ascii=False,
    )
    coordinator = DesignCoordinatorAgent(lambda _system, _user: rework_decision)

    blocked = coordinator.decide(state, state["user_request"])

    assert blocked.next_stage == StageName.MANUAL_REVIEW.value
    assert "修正轮次超限" in blocked.reason

    # 人工授权重试会放宽一轮预算，此时同一决策应当被接受。
    granted = coordinator.decide({**state, "max_check_revision_rounds": 3}, state["user_request"])

    assert granted.next_stage == StageName.STRUCTURAL_DESIGN.value


MODELING_REVIEW_STATE = {
    "user_request": "请完成全流程设计",
    "user_intent": "full_design",
    "task_status": "manual_review_required",
    "check_iteration_index": 3,
    "max_check_revision_rounds": 3,
    "max_modeling_check_steps": 6,
    "check_result": {
        "overall_check": {"all_ok": False},
        "utilization_summary": {"max_utilization": 1.18},
    },
    "capacity_check_result": {
        "stage_complete": True,
        "overall_check": {"all_ok": False},
        "utilization_summary": {"max_utilization": 1.18},
    },
    "latest_handoff": {
        "stage": StageName.MODELING_CHECK.value,
        "status": "manual_review",
        "message": "达到建模验算返修上限",
    },
}


def test_modeling_review_exposes_continue_accept_and_abort() -> None:
    request = build_human_review_request(MODELING_REVIEW_STATE)

    assert request["review_type"] == "modeling_check_review"
    assert request["current_round"] == 3
    assert request["check_summary"]["overall_check"]["all_ok"] is False
    assert request["available_actions"] == [
        "continue_modeling_revision",
        "accept_check_and_finish",
        "abort",
    ]


def test_continue_modeling_revision_adds_round_and_routes_back() -> None:
    update = apply_human_review_decision(
        MODELING_REVIEW_STATE,
        {"action": "continue_modeling_revision", "extra_rounds": 2},
    )

    assert update["max_check_revision_rounds"] == 5
    assert update["max_modeling_check_steps"] == 8
    assert route_after_human_review({**MODELING_REVIEW_STATE, **update}) == StageName.MODELING_CHECK.value


def test_accept_failed_check_finishes_with_recorded_risk() -> None:
    update = apply_human_review_decision(
        MODELING_REVIEW_STATE,
        {"action": "accept_check_and_finish", "feedback": "仅作为方法可行性展示"},
    )

    assert update["human_override"] is True
    assert update["accepted_risks"][0]["scope"] == "modeling_check"
    assert route_after_human_review({**MODELING_REVIEW_STATE, **update}) == StageName.FINAL_OUTPUT.value
    final = final_output_node({**MODELING_REVIEW_STATE, **update})
    assert final["task_status"] == "completed_with_accepted_risks"
    assert final["final_summary"]["accepted_risks"] == update["accepted_risks"]


def test_accept_check_requires_actual_complete_capacity_result() -> None:
    state = {
        **MODELING_REVIEW_STATE,
        "capacity_check_result": None,
        "capacity_batch_status": None,
        # 内存与磁盘都没有任何验算证据时才必须拒绝
        "check_result": None,
    }

    with pytest.raises(ValueError, match="完整承载力验算结果"):
        apply_human_review_decision(
            state,
            {"action": "accept_check_and_finish", "feedback": "继续"},
        )


def test_accept_check_accepts_complete_check_result_without_batch_status() -> None:
    # check_result 与 capacity_check_result 在实际运行中是同一份批次汇总：
    # 只要其中一份完整存在，就说明验算确实完成过，人工可以据其接受风险。
    update = apply_human_review_decision(
        {
            **MODELING_REVIEW_STATE,
            "capacity_check_result": None,
            "capacity_batch_status": None,
        },
        {"action": "accept_check_and_finish", "feedback": "继续"},
    )

    assert update["human_review_route"] == StageName.FINAL_OUTPUT.value
    assert update["accepted_risks"][0]["check_summary"]["overall_check"]["all_ok"] is False
