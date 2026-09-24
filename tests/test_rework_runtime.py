from __future__ import annotations

import json

from bridge_agents.contracts import StageName, StageStatus
from bridge_agents.design_coordinator import DesignCoordinatorAgent, build_coordinator_view
from bridge_agents.graph_v2 import build_graph_v2, make_stage_node


def _handoff(stage: str, status: str, **updates):
    handoff = {
        "stage": stage,
        "status": status,
        "produced_artifacts": [],
        "findings": [],
        "revision_request": None,
        "invalidated_artifacts": [],
        "recommended_next_stage": None,
        "unresolved_code_items": [],
        "message": "",
    }
    handoff.update(updates)
    return handoff


def test_modeling_revision_invalidates_target_and_downstream_runtime_fields() -> None:
    revision_handoff = _handoff(
        StageName.MODELING_CHECK.value,
        StageStatus.REVISION_REQUIRED.value,
        revision_request={
            "target_stage": StageName.STRUCTURAL_DESIGN.value,
            "reason": "配筋承载力不足",
            "required_artifacts": ["reinforcement_design_result"],
        },
        recommended_next_stage=StageName.STRUCTURAL_DESIGN.value,
    )
    node = make_stage_node(
        StageName.MODELING_CHECK.value,
        lambda state: {
            "task_status": "modeling_check_revision_required",
            "check_result": {"overall_check": {"all_ok": False}},
            "capacity_check_result": {"overall_check": {"all_ok": False}},
            "latest_handoff": revision_handoff,
        },
    )

    result = node({
        "dimension_design_result": {"dimension": "keep"},
        "reinforcement_design_result": {"bars": 24},
        "reinforcement_yaml_path": "old-reinforcement.yaml",
        "opensees_force_json_path": "old-force.json",
        "analysis_result": {"forces": "old"},
        "check_result": {"overall_check": {"all_ok": False}},
        "capacity_check_result": {"overall_check": {"all_ok": False}},
    })

    assert "dimension_design_result" not in result
    assert result["reinforcement_design_result"] is None
    assert result["reinforcement_yaml_path"] is None
    assert result["opensees_force_json_path"] is None
    assert result["analysis_result"] is None
    assert result["check_result"] is None
    assert result["capacity_check_result"] is None
    assert result["invalidated_artifacts"] == [
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    ]
    assert result["minimal_rework_path"] == result["invalidated_artifacts"]
    assert result["revision_target"] == StageName.STRUCTURAL_DESIGN.value
    assert result["rework_required_artifacts"] == ["reinforcement_design_result"]


def test_structural_rework_restores_reinforcement_and_analysis_artifacts() -> None:
    node = make_stage_node(
        StageName.STRUCTURAL_DESIGN.value,
        lambda state: {
            "task_status": "structural_design_completed",
            "reinforcement_design_result": {"bars": 28},
            "reinforcement_yaml_path": "new-reinforcement.yaml",
            "opensees_force_json_path": "new-force.json",
        },
    )

    result = node({
        "revision_target": StageName.STRUCTURAL_DESIGN.value,
        "rework_required_artifacts": ["reinforcement_design_result"],
        "minimal_rework_path": [
            "reinforcement_design_result",
            "analysis_result",
            "check_result",
        ],
        "invalidated_artifacts": [
            "reinforcement_design_result",
            "analysis_result",
            "check_result",
        ],
    })

    assert result["invalidated_artifacts"] == ["check_result"]
    assert result["minimal_rework_path"] == ["check_result"]
    assert result["revision_target"] is None
    assert result["rework_required_artifacts"] == []


def test_structural_rework_keeps_revision_target_until_all_structural_artifacts_are_restored() -> None:
    node = make_stage_node(
        StageName.STRUCTURAL_DESIGN.value,
        lambda state: {
            "task_status": "structural_design_completed",
            "dimension_design_result": {"dimension": "updated"},
        },
    )

    result = node({
        "revision_target": StageName.STRUCTURAL_DESIGN.value,
        "rework_required_artifacts": ["dimension_design_result"],
        "minimal_rework_path": [
            "dimension_design_result",
            "reinforcement_design_result",
            "analysis_result",
            "check_result",
        ],
        "invalidated_artifacts": [
            "dimension_design_result",
            "reinforcement_design_result",
            "analysis_result",
            "check_result",
        ],
    })

    assert result["invalidated_artifacts"] == [
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
    ]
    assert "revision_target" not in result
    assert "rework_required_artifacts" not in result


def test_modeling_recheck_restores_check_artifact() -> None:
    node = make_stage_node(
        StageName.MODELING_CHECK.value,
        lambda state: {
            "task_status": "modeling_check_passed",
            "check_result": {"overall_check": {"all_ok": True}},
            "capacity_check_result": {"overall_check": {"all_ok": True}},
            "latest_handoff": _handoff(
                StageName.MODELING_CHECK.value,
                StageStatus.COMPLETED.value,
                recommended_next_stage=StageName.FINAL_OUTPUT.value,
            ),
        },
    )

    result = node({
        "invalidated_artifacts": ["check_result"],
        "minimal_rework_path": ["check_result"],
    })

    assert result["invalidated_artifacts"] == []
    assert result["minimal_rework_path"] == []


def test_coordinator_view_treats_invalidated_values_as_unavailable() -> None:
    view = build_coordinator_view({
        "reinforcement_design_result": {"bars": 24},
        "opensees_force_json_path": "old-force.json",
        "check_result": {"overall_check": {"all_ok": True}},
        "invalidated_artifacts": ["reinforcement_design_result", "analysis_result", "check_result"],
    })

    assert view.reinforcement_design_available is False
    assert view.opensees_force_available is False
    assert view.capacity_check_available is False
    assert view.check_passed is None


def test_coordinator_fallback_prioritizes_active_revision_target() -> None:
    def llm_unavailable(system: str, user: str) -> str:
        raise RuntimeError("llm down")

    agent = DesignCoordinatorAgent(llm_unavailable)
    decision = agent.decide({
        "user_intent": "full_design",
        "layout_result": {"bridges": ["B1"]},
        "layout_revision_result": {"bridges": ["B1"]},
        "collision_metrics": {"conflict_column_rate": 0.0},
        "revision_target": StageName.STRUCTURAL_DESIGN.value,
        "rework_required_artifacts": ["reinforcement_design_result"],
        "invalidated_artifacts": [
            "reinforcement_design_result",
            "analysis_result",
            "check_result",
        ],
    }, "全流程设计", intent="full_design")

    assert decision.next_stage == StageName.STRUCTURAL_DESIGN.value
    assert decision.revision_target == StageName.STRUCTURAL_DESIGN.value
    assert decision.required_artifacts == ["reinforcement_design_result"]
    assert decision.task_complete is False


def test_graph_v2_completes_minimal_structural_rework_loop() -> None:
    decisions = iter([
        {
            "intent": "full_design",
            "next_stage": StageName.MODELING_CHECK.value,
            "stage_objective": "执行首次验算",
            "reason": "结构成果已存在",
            "revision_target": None,
            "required_artifacts": [],
            "task_complete": False,
        },
        {
            "intent": "full_design",
            "next_stage": StageName.STRUCTURAL_DESIGN.value,
            "stage_objective": "仅返工配筋",
            "reason": "验算要求修正配筋",
            "revision_target": StageName.STRUCTURAL_DESIGN.value,
            "required_artifacts": ["reinforcement_design_result"],
            "task_complete": False,
        },
        {
            "intent": "full_design",
            "next_stage": StageName.MODELING_CHECK.value,
            "stage_objective": "重新验算",
            "reason": "配筋和分析输入已重建",
            "revision_target": None,
            "required_artifacts": [],
            "task_complete": False,
        },
        {
            "intent": "full_design",
            "next_stage": None,
            "stage_objective": "完成",
            "reason": "复验通过",
            "revision_target": None,
            "required_artifacts": [],
            "task_complete": True,
        },
    ])

    def llm_invoke(system: str, user: str) -> str:
        return json.dumps(next(decisions), ensure_ascii=False)

    modeling_calls = 0

    def modeling_runner(state):
        nonlocal modeling_calls
        modeling_calls += 1
        if modeling_calls == 1:
            return {
                "task_status": "modeling_check_revision_required",
                "check_result": {"overall_check": {"all_ok": False}},
                "capacity_check_result": {"overall_check": {"all_ok": False}},
                "feedback_decision": {
                    "next_action": "revise_reinforcement",
                    "target_step": "reinforcement_design",
                },
                "latest_handoff": _handoff(
                    StageName.MODELING_CHECK.value,
                    StageStatus.REVISION_REQUIRED.value,
                    revision_request={
                        "target_stage": StageName.STRUCTURAL_DESIGN.value,
                        "reason": "配筋承载力不足",
                        "required_artifacts": ["reinforcement_design_result"],
                    },
                    recommended_next_stage=StageName.STRUCTURAL_DESIGN.value,
                ),
            }
        return {
            "task_status": "modeling_check_passed",
            "check_result": {"overall_check": {"all_ok": True}},
            "capacity_check_result": {"overall_check": {"all_ok": True}},
            "latest_handoff": _handoff(
                StageName.MODELING_CHECK.value,
                StageStatus.COMPLETED.value,
                recommended_next_stage=StageName.FINAL_OUTPUT.value,
            ),
        }

    graph = build_graph_v2(
        llm_invoke=llm_invoke,
        stage_runners={
            StageName.INITIAL_DESIGN.value: lambda state: {
                "task_status": "initial_design_completed",
            },
            StageName.LAYOUT_REVISION.value: lambda state: {
                "task_status": "layout_revision_completed",
            },
            StageName.STRUCTURAL_DESIGN.value: lambda state: {
                "task_status": "structural_design_completed",
                "reinforcement_design_result": {"bars": 28},
                "reinforcement_yaml_path": "new-reinforcement.yaml",
                "opensees_force_json_path": "new-force.json",
            },
            StageName.MODELING_CHECK.value: modeling_runner,
        },
    )
    result = graph.invoke({
        "user_request": "完成结构设计验算",
        "user_intent": "full_design",
        "layout_result": {"bridges": ["B1"]},
        "layout_revision_result": {"bridges": ["B1"]},
        "collision_metrics": {"conflict_column_rate": 0.0},
        "dimension_design_result": {"dimension": "keep"},
        "reinforcement_design_result": {"bars": 24},
        "reinforcement_yaml_path": "old-reinforcement.yaml",
        "opensees_force_json_path": "old-force.json",
    })

    assert modeling_calls == 2
    assert result["task_status"] == "completed"
    assert result["reinforcement_design_result"] == {"bars": 28}
    assert result["check_result"]["overall_check"]["all_ok"] is True
    assert result["invalidated_artifacts"] == []
    assert result["revision_target"] is None
