from __future__ import annotations

import json

import pytest

from bridge_agents.contracts import StageName
from bridge_agents.design_coordinator import (
    DesignCoordinatorAgent,
    build_coordinator_view,
    _parse_decision,
)


def _empty_state() -> dict:
    return {"user_request": "对某段进行全流程设计"}


def _llm_returns(raw: str):
    def _invoke(system: str, user: str) -> str:
        return raw
    return _invoke


def _llm_raises(*args, **kwargs):
    def _invoke(system: str, user: str) -> str:
        raise RuntimeError("llm down")
    return _invoke


def _valid_decision_raw() -> str:
    return json.dumps(
        {
            "intent": "full_design",
            "next_stage": StageName.INITIAL_DESIGN.value,
            "stage_objective": "生成初步布跨",
            "reason": "从零开始",
            "revision_target": None,
            "required_artifacts": [],
            "task_complete": False,
        },
        ensure_ascii=False,
    )


def test_build_coordinator_view_extracts_flags() -> None:
    state = {
        "layout_result": {"x": 1},
        "reinforcement_design_result": {"y": 1},
        "opensees_force_json_path": "path",
        "check_result": {"overall_check": True},
    }
    view = build_coordinator_view(state)
    assert view.layout_result_available is True
    assert view.reinforcement_design_available is True
    assert view.opensees_force_available is True
    assert view.check_passed is True


def test_parse_decision_plain_json() -> None:
    d = _parse_decision(_valid_decision_raw())
    assert d is not None
    assert d.next_stage == StageName.INITIAL_DESIGN.value


def test_parse_decision_fenced_json() -> None:
    raw = "```json\n" + _valid_decision_raw() + "\n```"
    d = _parse_decision(raw)
    assert d is not None
    assert d.intent == "full_design"


def test_parse_decision_invalid_returns_none() -> None:
    assert _parse_decision("不是 JSON") is None
    assert _parse_decision("") is None


def test_decide_accepts_valid_llm_decision() -> None:
    agent = DesignCoordinatorAgent(_llm_returns(_valid_decision_raw()))
    d = agent.decide(_empty_state(), "全流程设计")
    assert d.next_stage == StageName.INITIAL_DESIGN.value


def test_decide_retries_after_invalid_then_accepts() -> None:
    calls = []

    def invoke(system: str, user: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            return json.dumps({"intent": "full_design", "next_stage": "not_a_stage", "task_complete": False})
        return _valid_decision_raw()

    agent = DesignCoordinatorAgent(invoke)
    d = agent.decide(_empty_state(), "全流程设计")
    assert d.next_stage == StageName.INITIAL_DESIGN.value
    assert len(calls) == 2


def test_retry_prompt_lists_registered_stages_and_previous_validation_errors() -> None:
    payloads = []

    def invoke(system: str, user: str) -> str:
        payloads.append(json.loads(user))
        if len(payloads) == 1:
            return json.dumps({
                "intent": "full_design",
                "next_stage": "layout_design",
                "stage_objective": "生成布跨",
                "reason": "误用了意图枚举",
                "task_complete": False,
            })
        return _valid_decision_raw()

    agent = DesignCoordinatorAgent(invoke)
    decision = agent.decide(_empty_state(), "全流程设计")

    assert decision.next_stage == StageName.INITIAL_DESIGN.value
    assert payloads[0]["允许的 next_stage"] == [
        StageName.INITIAL_DESIGN.value,
        StageName.LAYOUT_REVISION.value,
        StageName.STRUCTURAL_DESIGN.value,
        StageName.MODELING_CHECK.value,
        StageName.MANUAL_REVIEW.value,
    ]
    assert payloads[0]["字段约束"]["intent"] == "任务意图枚举；不得填入 next_stage"
    assert payloads[0]["上次确定性校验错误"] == []
    assert any("未注册的阶段" in error for error in payloads[1]["上次确定性校验错误"])


def test_decide_falls_back_to_manual_review_after_repeated_invalid() -> None:
    agent = DesignCoordinatorAgent(
        _llm_returns(json.dumps({"intent": "full_design", "next_stage": "bad", "task_complete": False}))
    )
    d = agent.decide(_empty_state(), "全流程设计")
    assert d.next_stage == StageName.MANUAL_REVIEW.value


def test_decide_uses_rule_fallback_when_llm_unavailable() -> None:
    agent = DesignCoordinatorAgent(_llm_raises())
    d = agent.decide(_empty_state(), "全流程设计", intent="full_design")
    # 规则兜底：全流程的第一个未完成阶段是 initial_design
    assert d.next_stage == StageName.INITIAL_DESIGN.value


def test_fallback_skips_completed_stages() -> None:
    state = {
        "user_intent": "full_design",
        "layout_result": {"x": 1},
        "layout_revision_result": {"y": 1},
    }
    agent = DesignCoordinatorAgent(_llm_raises())
    d = agent.decide(state, "全流程设计", intent="full_design")
    # layout 和 revision 已完成，下一个是 structural_design
    assert d.next_stage == StageName.STRUCTURAL_DESIGN.value


def test_fallback_marks_complete_when_all_done() -> None:
    state = {
        "user_intent": "full_design",
        "layout_result": {"x": 1},
        "layout_revision_result": {"y": 1},
        "reinforcement_design_result": {"z": 1},
        "capacity_check_result": {"overall_check": True},
    }
    agent = DesignCoordinatorAgent(_llm_raises())
    d = agent.decide(state, "全流程设计", intent="full_design")
    assert d.task_complete is True
    assert d.next_stage is None


def test_incomplete_capacity_batch_does_not_mark_modeling_complete() -> None:
    state = {
        "user_intent": "full_design",
        "layout_result": {"x": 1},
        "layout_revision_result": {"y": 1},
        "reinforcement_design_result": {"z": 1},
        "reinforcement_batch_status": {"stage_complete": True},
        "opensees_force_json_path": "forces.json",
        "capacity_check_result": {
            "check_type": "cap_beam_capacity_envelope_batch",
            "stage_complete": False,
            "overall_check": {"all_ok": False},
        },
        "capacity_batch_status": {
            "expected_task_count": 2,
            "completed_task_count": 1,
            "failed_task_count": 1,
            "stage_complete": False,
        },
    }

    decision = DesignCoordinatorAgent(_llm_raises()).decide(
        state, "全流程设计", intent="full_design"
    )

    assert decision.task_complete is False
    assert decision.next_stage == StageName.MODELING_CHECK.value


def test_invalid_backward_llm_decisions_fall_back_to_completion() -> None:
    backward = json.dumps(
        {
            "intent": "full_design",
            "next_stage": StageName.INITIAL_DESIGN.value,
            "stage_objective": "重新初步设计",
            "reason": "模型错误回跳",
            "revision_target": None,
            "required_artifacts": [],
            "task_complete": False,
        },
        ensure_ascii=False,
    )
    state = {
        "user_intent": "full_design",
        "layout_result": {"x": 1},
        "layout_revision_result": {"y": 1},
        "reinforcement_design_result": {"z": 1},
        "reinforcement_batch_status": {"stage_complete": True},
        "opensees_force_json_path": "forces.json",
        "capacity_check_result": {"overall_check": True},
    }

    decision = DesignCoordinatorAgent(_llm_returns(backward)).decide(
        state, "全流程设计", intent="full_design"
    )

    assert decision.task_complete is True
    assert decision.next_stage is None


def test_completed_workflow_finishes_before_mixed_invalid_llm_errors() -> None:
    backward_without_objective = json.dumps(
        {
            "intent": "full_design",
            "next_stage": StageName.INITIAL_DESIGN.value,
            "stage_objective": "",
            "reason": "模型错误回跳",
            "revision_target": None,
            "required_artifacts": [],
            "task_complete": False,
        },
        ensure_ascii=False,
    )
    state = {
        "user_intent": "full_design",
        "layout_result": {"x": 1},
        "layout_revision_result": {"y": 1},
        "reinforcement_design_result": {"z": 1},
        "reinforcement_batch_status": {"stage_complete": True},
        "opensees_force_json_path": "forces.json",
        "capacity_check_result": {
            "stage_complete": True,
            "overall_check": {"all_ok": True},
        },
        "check_result": {
            "stage_complete": True,
            "overall_check": {"all_ok": True},
        },
        "capacity_batch_status": {"stage_complete": True},
        "iteration_index": 3,
        "check_iteration_index": 3,
        "max_revision_rounds": 3,
    }

    structural_retry_at_limit = json.dumps(
        {
            "intent": "full_design",
            "next_stage": StageName.STRUCTURAL_DESIGN.value,
            "stage_objective": "重复结构设计",
            "reason": "模型再次错误调度",
            "revision_target": None,
            "required_artifacts": [],
            "task_complete": False,
        },
        ensure_ascii=False,
    )
    responses = iter([backward_without_objective, structural_retry_at_limit])

    def mixed_invalid_decisions(system: str, user: str) -> str:
        return next(responses)

    decision = DesignCoordinatorAgent(mixed_invalid_decisions).decide(
        state, "全流程设计", intent="full_design"
    )

    assert decision.task_complete is True
    assert decision.next_stage is None
