from __future__ import annotations

import json

from langgraph.checkpoint.memory import InMemorySaver

from bridge_agents.contracts import StageName
from bridge_agents.graph_v2 import (
    STAGE_AGENT_NAME,
    build_graph_v2,
    route_after_coordinator,
)


def _decision(intent="full_design", next_stage=None, task_complete=False):
    return json.dumps(
        {
            "intent": intent,
            "next_stage": next_stage,
            "stage_objective": f"执行 {next_stage or '完成'}",
            "reason": "测试",
            "revision_target": None,
            "required_artifacts": [],
            "task_complete": task_complete,
        },
        ensure_ascii=False,
    )


class SequenceLLM:
    """按调用次数返回固定决策序列的 fake LLM。"""

    def __init__(self, decisions):
        self._decisions = decisions
        self._i = 0
        self.calls = 0

    def __call__(self, system: str, user: str) -> str:
        self.calls += 1
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d


def _fake_runners():
    return {
        StageName.INITIAL_DESIGN.value: lambda state: {
            "layout_result": {"bridges": []},
            "task_status": "initial_design_completed",
        },
        StageName.LAYOUT_REVISION.value: lambda state: {
            "layout_revision_result": {"ok": True},
            "final_layout_result": {"ok": True},
            "collision_metrics": {"conflict_column_rate": 0.0},
            "task_status": "layout_revision_completed",
        },
        StageName.STRUCTURAL_DESIGN.value: lambda state: {
            "reinforcement_design_result": {"ok": True},
            "opensees_force_json_path": "force.json",
            "task_status": "structural_design_completed",
        },
        StageName.MODELING_CHECK.value: lambda state: {
            "capacity_check_result": {"overall_check": True},
            "check_result": {"overall_check": True},
            "task_status": "modeling_check_passed",
        },
    }


def test_route_after_coordinator_to_stage() -> None:
    state = {
        "coordinator_decision": {
            "task_complete": False,
            "next_stage": StageName.INITIAL_DESIGN.value,
        }
    }
    assert route_after_coordinator(state) == StageName.INITIAL_DESIGN.value


def test_route_after_coordinator_to_final() -> None:
    state = {"coordinator_decision": {"task_complete": True}}
    assert route_after_coordinator(state) == StageName.FINAL_OUTPUT.value


def test_route_after_coordinator_to_manual_review() -> None:
    state = {
        "coordinator_decision": {
            "task_complete": False,
            "next_stage": StageName.MANUAL_REVIEW.value,
        }
    }
    assert route_after_coordinator(state) == StageName.MANUAL_REVIEW.value


def test_route_after_coordinator_to_error_on_unknown() -> None:
    state = {"coordinator_decision": {"task_complete": False, "next_stage": "bogus"}}
    assert route_after_coordinator(state) == StageName.ERROR.value


def test_graph_registers_all_top_level_nodes() -> None:
    graph = build_graph_v2(
        llm_invoke=SequenceLLM([_decision(next_stage=StageName.INITIAL_DESIGN.value)]),
        stage_runners=_fake_runners(),
    )
    nodes = set(graph.get_graph().nodes.keys())
    assert nodes >= {
        "design_coordinator",
        StageName.INITIAL_DESIGN.value,
        StageName.LAYOUT_REVISION.value,
        StageName.STRUCTURAL_DESIGN.value,
        StageName.MODELING_CHECK.value,
        StageName.MANUAL_REVIEW.value,
        StageName.FINAL_OUTPUT.value,
        StageName.ERROR.value,
    }


def test_full_flow_completes_with_fake_llm_and_runners() -> None:
    llm = SequenceLLM(
        [
            _decision(next_stage=StageName.INITIAL_DESIGN.value),
            _decision(next_stage=StageName.LAYOUT_REVISION.value),
            _decision(next_stage=StageName.STRUCTURAL_DESIGN.value),
            _decision(next_stage=StageName.MODELING_CHECK.value),
            _decision(task_complete=True),
        ]
    )
    graph = build_graph_v2(llm_invoke=llm, stage_runners=_fake_runners())
    result = graph.invoke({"user_request": "请进行全流程设计"})
    assert result.get("task_status") == "completed"
    # 四个阶段各调用一次协调器；终态由确定性完成门控直接收敛，不再额外调用 LLM。
    assert llm.calls == 4


def test_repeated_invalid_decision_routes_to_manual_review() -> None:
    llm = SequenceLLM([_decision(next_stage="not_a_stage")])
    graph = build_graph_v2(
        llm_invoke=llm,
        stage_runners=_fake_runners(),
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {"user_request": "请进行全流程设计"},
        config={"configurable": {"thread_id": "invalid-decision"}},
    )
    assert result.get("__interrupt__")
    assert result["__interrupt__"][0].value["review_type"] == "coordinator_review"
