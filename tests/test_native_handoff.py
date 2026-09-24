from __future__ import annotations

from typing import Any, Dict

from bridge_agents.contracts import StageName, StageStatus
from bridge_agents.graph_v2 import make_stage_node
from bridge_agents.stage_agents import LayoutRevisionAgent, ModelingCheckAgent


def test_layout_revision_skip_returns_native_handoff() -> None:
    result = LayoutRevisionAgent().run({
        "layout_revision_completed": True,
        "layout_revision_result": {"bridges": ["B1"]},
        "final_layout_result": {"bridges": ["B1"]},
        "final_layout_result_path": "outputs/layout_revision/final_layout_result.json",
    })

    handoff = result["latest_handoff"]
    assert handoff["stage"] == StageName.LAYOUT_REVISION.value
    assert handoff["status"] == StageStatus.COMPLETED.value
    assert handoff["recommended_next_stage"] == StageName.STRUCTURAL_DESIGN.value
    assert handoff["produced_artifacts"] == [{
        "artifact_type": "final_layout_result",
        "state_key": "final_layout_result",
        "path": "outputs/layout_revision/final_layout_result.json",
        "content_hash": "",
        "producer": "LayoutRevisionAgent",
        "revision": 0,
        "valid": True,
    }]


def test_modeling_check_revision_returns_structural_revision_request(monkeypatch) -> None:
    agent = ModelingCheckAgent()
    monkeypatch.setattr(agent, "_activate_skill", lambda working: "skill")
    monkeypatch.setattr("bridge_agents.stage_agents.get_controller_llm", lambda config_path: object())

    result = agent.run({
        "check_result": {"overall_check": {"all_ok": False}},
        "unresolved_code_items": [{
            "code_item_id": "unit-A:5.2.2:flexure_positive",
            "status": "fail",
            "message": "正弯矩抗弯承载力未通过",
            "source_entity_ids": ["F_3362_CH05_5_2_2_1_BENDING_CAPACITY_TENSION_FLANGE"],
        }],
        "feedback_decision": {
            "overall_status": "revise_reinforcement",
            "next_action": "revise_reinforcement",
            "target_agent": "StructuralDesignAgent",
            "target_step": "reinforcement_design",
            "control_reason": "墩柱纵筋承载力不足",
            "requires_rerun_check": True,
        },
    })

    handoff = result["latest_handoff"]
    assert handoff["stage"] == StageName.MODELING_CHECK.value
    assert handoff["status"] == StageStatus.REVISION_REQUIRED.value
    assert handoff["recommended_next_stage"] == StageName.STRUCTURAL_DESIGN.value
    assert handoff["revision_request"] == {
        "target_stage": StageName.STRUCTURAL_DESIGN.value,
        "reason": "墩柱纵筋承载力不足",
        "required_artifacts": ["reinforcement_design_result"],
    }
    assert handoff["unresolved_code_items"] == [{
        "code_item_id": "unit-A:5.2.2:flexure_positive",
        "status": "fail",
        "message": "正弯矩抗弯承载力未通过",
        "evidence_ids": ["F_3362_CH05_5_2_2_1_BENDING_CAPACITY_TENSION_FLANGE"],
    }]


def test_modeling_check_pass_returns_check_artifact_and_final_recommendation(monkeypatch) -> None:
    agent = ModelingCheckAgent()
    monkeypatch.setattr(agent, "_activate_skill", lambda working: "skill")
    monkeypatch.setattr("bridge_agents.stage_agents.get_controller_llm", lambda config_path: object())

    result = agent.run({
        "check_result": {"overall_check": {"all_ok": True}},
        "capacity_check_summary_path": "outputs/modeling_check/summary.json",
    })

    handoff = result["latest_handoff"]
    assert handoff["stage"] == StageName.MODELING_CHECK.value
    assert handoff["status"] == StageStatus.COMPLETED.value
    assert handoff["recommended_next_stage"] == StageName.FINAL_OUTPUT.value
    assert handoff["revision_request"] is None
    assert handoff["produced_artifacts"][0]["artifact_type"] == "check_result"
    assert handoff["produced_artifacts"][0]["path"] == "outputs/modeling_check/summary.json"


def test_graph_v2_stage_node_preserves_matching_native_handoff() -> None:
    native_handoff: Dict[str, Any] = {
        "stage": StageName.LAYOUT_REVISION.value,
        "status": StageStatus.COMPLETED.value,
        "produced_artifacts": [{"artifact_type": "final_layout_result"}],
        "findings": [],
        "revision_request": None,
        "invalidated_artifacts": [],
        "recommended_next_stage": StageName.STRUCTURAL_DESIGN.value,
        "unresolved_code_items": [],
        "message": "原生交接",
    }
    node = make_stage_node(
        StageName.LAYOUT_REVISION.value,
        lambda state: {
            "task_status": "layout_revision_completed",
            "latest_handoff": native_handoff,
        },
    )

    result = node({})

    assert result["latest_handoff"] == native_handoff


def test_graph_v2_stage_node_keeps_legacy_handoff_fallback() -> None:
    node = make_stage_node(
        StageName.INITIAL_DESIGN.value,
        lambda state: {
            "task_status": "initial_design_completed",
            "message": "旧 runner 完成",
        },
    )

    result = node({})

    assert result["latest_handoff"]["stage"] == StageName.INITIAL_DESIGN.value
    assert result["latest_handoff"]["status"] == StageStatus.COMPLETED.value
    assert result["latest_handoff"]["message"] == "旧 runner 完成"
