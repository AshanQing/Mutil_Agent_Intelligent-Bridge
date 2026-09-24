from __future__ import annotations

import pytest

from bridge_agents.coordinator import (
    CoordinatorDecision,
    CoordinatorView,
    INTENT_TO_STAGE_SEQUENCE,
    REVISABLE_STAGES,
    validate_and_retry,
    validate_decision,
)
from bridge_agents.contracts import StageName
from bridge_agents.design_coordinator import build_coordinator_view


def _view(**kwargs) -> CoordinatorView:
    return CoordinatorView(**kwargs)


def _decision(**kwargs) -> CoordinatorDecision:
    base = dict(
        intent="full_design",
        next_stage=StageName.STRUCTURAL_DESIGN.value,
        stage_objective="完成下部结构设计",
        reason="布跨已就绪",
    )
    base.update(kwargs)
    return CoordinatorDecision(**base)


def test_decision_roundtrip() -> None:
    d = _decision(revision_target="structural_design", required_artifacts=["layout_result"])
    obj = d.to_dict()
    assert obj["next_stage"] == StageName.STRUCTURAL_DESIGN.value
    assert obj["revision_target"] == "structural_design"
    assert obj["task_complete"] is False


def test_task_complete_allowed_when_clean() -> None:
    view = _view(capacity_check_available=True, check_passed=True)
    d = _decision(next_stage=None, task_complete=True)
    assert validate_decision(d, view) == []


def test_task_complete_blocked_by_manual_review() -> None:
    view = _view(capacity_check_available=True, check_passed=True, unresolved_manual_review=True)
    d = _decision(next_stage=None, task_complete=True)
    errors = validate_decision(d, view)
    assert any("人工复核" in e for e in errors)


def test_task_complete_blocked_by_failed_check() -> None:
    view = _view(capacity_check_available=True, check_passed=False)
    d = _decision(next_stage=None, task_complete=True)
    errors = validate_decision(d, view)
    assert any("失败" in e for e in errors)


def test_unknown_stage_rejected() -> None:
    d = _decision(next_stage="not_a_stage")
    errors = validate_decision(d, _view())
    assert any("未注册" in e for e in errors)


def test_missing_prerequisite_rejected() -> None:
    d = _decision(next_stage=StageName.MODELING_CHECK.value)
    view = _view(reinforcement_design_available=False, opensees_force_available=True)
    errors = validate_decision(d, view)
    assert any("reinforcement_design_available" in e for e in errors)


def test_revision_target_mismatch_rejected() -> None:
    d = _decision(next_stage=StageName.STRUCTURAL_DESIGN.value, revision_target="layout_revision")
    view = _view(layout_result_available=True, revision_target="structural_design")
    errors = validate_decision(d, view)
    assert any("返修目标不一致" in e for e in errors)


def test_layout_revision_round_exceeded_rejected() -> None:
    d = _decision(next_stage=StageName.LAYOUT_REVISION.value)
    view = _view(
        layout_result_available=True,
        layout_revision_round=3,
        max_layout_revision_rounds=3,
    )
    errors = validate_decision(d, view)
    assert any("轮次超限" in e for e in errors)


def test_layout_round_limit_does_not_apply_to_first_structural_design() -> None:
    view = build_coordinator_view({
        "layout_result": {"桥位列表": [{"桥位编号": 5}]},
        "layout_revision_result": {"桥位列表": [{"桥位编号": 5}]},
        "collision_metrics": {"conflict_column_rate": 0.0},
        "iteration_index": 3,
        "max_revision_rounds": 3,
    })

    errors = validate_decision(
        _decision(next_stage=StageName.STRUCTURAL_DESIGN.value),
        view,
    )

    assert errors == []


def test_structural_revision_round_limit_uses_check_iteration() -> None:
    view = build_coordinator_view({
        "layout_result": {"桥位列表": [{"桥位编号": 5}]},
        "check_iteration_index": 2,
        "max_check_revision_rounds": 2,
    })

    errors = validate_decision(
        _decision(next_stage=StageName.STRUCTURAL_DESIGN.value),
        view,
    )

    assert any("轮次超限" in error for error in errors)


def test_mandatory_check_cannot_be_skipped() -> None:
    d = _decision(intent="full_design", next_stage=None, task_complete=True)
    view = _view(capacity_check_available=False, check_passed=None)
    errors = validate_decision(d, view)
    assert any("强制检查" in e for e in errors)


def test_validate_and_retry_passes_when_valid() -> None:
    d = _decision()
    view = _view(
        initial_design_completed=True,
        layout_revision_completed=True,
        layout_result_available=True,
        layout_revision_result_available=True,
        collision_detected_available=True,
    )
    assert validate_and_retry(d, view) is d


def test_structural_rejected_when_layout_revision_not_finished() -> None:
    # full_design 不允许跳过布跨修正直接进入结构设计。
    d = _decision(next_stage=StageName.STRUCTURAL_DESIGN.value)
    view = _view(
        initial_design_completed=True,
        layout_result_available=True,
    )
    errors = validate_decision(d, view)
    assert any("尚未完成" in e and "layout_revision" in e for e in errors)


def test_structural_rejected_without_collision_metrics() -> None:
    # 即便布跨修正结果已产出，也必须至少存在一次碰撞检测结果才能进入结构设计。
    d = _decision(next_stage=StageName.STRUCTURAL_DESIGN.value)
    view = _view(
        initial_design_completed=True,
        layout_result_available=True,
        layout_revision_result_available=True,
        collision_detected_available=False,
    )
    errors = validate_decision(d, view)
    assert any("collision_detected_available" in e for e in errors)


def test_structural_allowed_after_revision_and_collision() -> None:
    d = _decision(next_stage=StageName.STRUCTURAL_DESIGN.value)
    view = _view(
        initial_design_completed=True,
        layout_revision_completed=True,
        layout_result_available=True,
        layout_revision_result_available=True,
        collision_detected_available=True,
    )
    assert validate_decision(d, view) == []


def _manual_review_reinforcement_payload() -> dict:
    return {
        "任务3_下部结构配筋设计结果": {
            "分组原始结果": [
                {
                    "reinforcement_task": {"task_id": "B1-U1-G1"},
                    "axial_check": {
                        "status": "manual_review",
                        "message": "长细比超过表5.3.1上限，需人工复核。",
                    },
                }
            ]
        }
    }


def _modeling_state(**overrides) -> dict:
    state = {
        "user_intent": "full_design",
        "layout_result": {"桥位列表": [{"桥位编号": 1}]},
        "layout_revision_result": {"桥位列表": [{"桥位编号": 1}]},
        "collision_metrics": {"conflict_column_rate": 0.0},
        "reinforcement_design_result": _manual_review_reinforcement_payload(),
        "reinforcement_batch_status": {"stage_complete": True},
        "capacity_check_result": {
            "stage_complete": True,
            "overall_check": {"all_ok": True},
            "task_results": [],
        },
        "check_result": {"overall_check": {"all_ok": True}},
    }
    state.update(overrides)
    return state


def test_slenderness_review_pending_blocks_completion_without_acceptance() -> None:
    # 承载力通过但墩柱长细比复核未决（超表5.3.1）时，不能自动 task_complete。
    view = build_coordinator_view(_modeling_state())
    assert view.modeling_slenderness_review_pending is True
    d = _decision(next_stage=None, task_complete=True)
    errors = validate_decision(d, view)
    assert any("长细比" in error for error in errors)


def test_modeling_risk_acceptance_allows_completion() -> None:
    # 人工接受建模风险（scope=modeling_check）后允许完成并出图。
    view = build_coordinator_view(
        _modeling_state(
            accepted_risks=[{"scope": "modeling_check", "reason": "人工接受长细比复核风险"}],
        )
    )
    assert view.modeling_risk_accepted is True
    assert view.modeling_slenderness_review_pending is False
    d = _decision(next_stage=None, task_complete=True)
    assert validate_decision(d, view) == []


def test_validate_and_retry_raises_when_invalid() -> None:
    d = _decision(next_stage="bad")
    with pytest.raises(ValueError):
        validate_and_retry(d, _view())


def test_intent_to_stage_sequence_covers_supported_intents() -> None:
    supported = {
        "full_design",
        "layout_design",
        "layout_design_check_revision",
        "layout_check_revision",
        "structural_design",
        "layout_to_reinforcement_design",
        "reinforcement_design",
        "reinforcement_design_verification",
        "verification",
    }
    assert set(INTENT_TO_STAGE_SEQUENCE.keys()) == supported


def test_full_design_sequence_ends_with_modeling_check() -> None:
    seq = INTENT_TO_STAGE_SEQUENCE["full_design"]
    assert seq[-1] == StageName.MODELING_CHECK.value
    assert StageName.STRUCTURAL_DESIGN.value in seq


def test_completed_stage_cannot_be_reentered_without_rework_target() -> None:
    decision = _decision(next_stage=StageName.INITIAL_DESIGN.value)
    view = _view(initial_design_completed=True)

    errors = validate_decision(decision, view)

    assert any("已完成阶段" in error for error in errors)


def test_completed_stage_can_be_reentered_for_explicit_rework() -> None:
    decision = _decision(
        next_stage=StageName.INITIAL_DESIGN.value,
        revision_target=StageName.INITIAL_DESIGN.value,
    )
    view = _view(
        initial_design_completed=True,
        revision_target=StageName.INITIAL_DESIGN.value,
    )

    assert validate_decision(decision, view) == []


def test_partial_reinforcement_is_not_available_to_modeling() -> None:
    view = build_coordinator_view(
        {
            "reinforcement_design_result": {"partial": True},
            "reinforcement_batch_status": {
                "stage_complete": False,
                "completed_task_count": 15,
                "expected_task_count": 18,
            },
            "opensees_force_json_path": "force.json",
        }
    )

    assert view.reinforcement_design_available is False


def test_human_accepted_partial_reinforcement_is_available_with_recorded_risk() -> None:
    view = build_coordinator_view(
        {
            "reinforcement_design_result": {"partial": True},
            "reinforcement_batch_status": {
                "stage_complete": False,
                "human_accepted": True,
            },
            "accepted_risks": [{"scope": "structural_design"}],
        }
    )

    assert view.reinforcement_design_available is True
