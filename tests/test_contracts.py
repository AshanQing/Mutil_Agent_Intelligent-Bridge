from __future__ import annotations

import pytest

from bridge_agents.contracts import (
    AGENT_CONTRACTS,
    AgentContract,
    ArtifactRecord,
    ComplianceResult,
    RevisionRequest,
    StageHandoff,
    StageName,
    StageStatus,
    StageFinding,
    validate_all_contracts,
    validate_contract,
)
from bridge_agents.skill_registry import SkillRegistry


# 与 bridge_agents.actions.AGENT_ACTIONS 对齐的预期动作集合。
# 此处硬编码以避免顶层导入 actions（其依赖 mmseg，导入极慢）。
EXPECTED_ALLOWED_ACTIONS = {
    "InitialDesignAgent": {
        "load_data",
        "drawing_crop_and_mask",
        "obstacle_semantic_extractor",
        "select_samples",
        "generate_layout_design",
    },
    "LayoutRevisionAgent": {
        "run_collision_detection",
        "generate_revision_instruction",
        "build_revision_prompt",
        "generate_revised_layout",
    },
    "StructuralDesignAgent": {
        "extract_design_units",
        "dimension_design",
        "reinforcement_design",
    },
    "ModelingCheckAgent": {
        "run_capacity_check",
        "generate_modeling_feedback",
    },
    "DesignReviewAgent": set(),
}


def test_stage_status_values() -> None:
    assert {s.value for s in StageStatus} == {
        "completed",
        "revision_required",
        "blocked",
        "failed",
        "manual_review",
    }


def test_stage_name_includes_coordinator_and_manual_review() -> None:
    names = {s.value for s in StageName}
    assert "design_coordinator" in names
    assert "manual_review" in names
    assert "final_output" in names


def test_artifact_record_roundtrip() -> None:
    a = ArtifactRecord(
        artifact_type="layout",
        state_key="layout_result",
        path="output/x.json",
        content_hash="abc",
        producer="InitialDesignAgent",
        revision=1,
    )
    d = a.to_dict()
    assert d["state_key"] == "layout_result"
    assert d["revision"] == 1


def test_stage_handoff_defaults() -> None:
    h = StageHandoff(stage=StageName.STRUCTURAL_DESIGN.value, status=StageStatus.COMPLETED.value)
    assert h.is_completed() is True
    assert h.has_unresolved_code_items() is False
    assert h.produced_artifacts == []


def test_stage_handoff_with_unresolved_code_items() -> None:
    h = StageHandoff(
        stage="modeling_check",
        status=StageStatus.MANUAL_REVIEW.value,
        unresolved_code_items=[ComplianceResult(code_item_id="R_X", status="manual_review")],
    )
    assert h.has_unresolved_code_items() is True
    d = h.to_dict()
    assert d["unresolved_code_items"][0]["status"] == "manual_review"


def test_revision_request_roundtrip() -> None:
    r = RevisionRequest(
        target_stage="structural_design",
        reason="配筋不足",
        required_artifacts=["reinforcement_design_result"],
    )
    d = r.to_dict()
    assert d["target_stage"] == "structural_design"
    assert d["required_artifacts"] == ["reinforcement_design_result"]


def test_stage_finding_roundtrip() -> None:
    f = StageFinding(severity="warning", message="证据不足", evidence_ids=["E1"])
    assert f.to_dict()["evidence_ids"] == ["E1"]


def test_validate_contract_passes_for_valid_contract() -> None:
    contract = AgentContract(
        agent_name="X",
        skill_id="initial-design",
        allowed_actions=("load_data",),
        readable_state_keys=("user_request",),
        produced_state_keys=("layout_result",),
    )
    errors = validate_contract(
        contract,
        skill_ids={"initial-design"},
        registered_actions={"load_data"},
    )
    assert errors == []


def test_validate_contract_detects_unknown_skill_action_and_empty_keys() -> None:
    contract = AgentContract(
        agent_name="X",
        skill_id="missing-skill",
        allowed_actions=("not_registered",),
        readable_state_keys=(),
        produced_state_keys=(),
    )
    errors = validate_contract(contract, skill_ids={"ok"}, registered_actions={"ok"})
    assert len(errors) == 4
    joined = "\n".join(errors)
    assert "missing-skill" in joined
    assert "not_registered" in joined
    assert "readable_state_keys" in joined
    assert "produced_state_keys" in joined


def test_agent_contracts_have_expected_entries() -> None:
    assert set(AGENT_CONTRACTS.keys()) == {
        "InitialDesignAgent",
        "LayoutRevisionAgent",
        "StructuralDesignAgent",
        "ModelingCheckAgent",
        "DesignReviewAgent",
    }


def test_contracts_align_with_skills() -> None:
    registry = SkillRegistry("skills")
    skill_ids = set(registry.discover().keys())
    all_actions = set().union(*EXPECTED_ALLOWED_ACTIONS.values())
    errors = validate_all_contracts(skill_ids, all_actions)
    assert errors == [], "\n".join(errors)


def test_contracts_allowed_actions_match_expected() -> None:
    for agent_name, contract in AGENT_CONTRACTS.items():
        assert set(contract.allowed_actions) == EXPECTED_ALLOWED_ACTIONS[agent_name]
