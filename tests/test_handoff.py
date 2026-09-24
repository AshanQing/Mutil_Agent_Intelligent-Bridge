from __future__ import annotations

import pytest

from bridge_agents.contracts import ArtifactRecord, RevisionRequest, StageStatus
from bridge_agents.handoff import build_handoff, infer_stage_status


def test_infer_error_is_failed() -> None:
    assert infer_stage_status({"error": "boom", "task_status": "x"}) == StageStatus.FAILED.value


def test_infer_failed_task_status() -> None:
    assert infer_stage_status({"task_status": "failed"}) == StageStatus.FAILED.value


def test_infer_manual_review_required() -> None:
    assert infer_stage_status({"task_status": "manual_review_required"}) == StageStatus.MANUAL_REVIEW.value


def test_infer_modeling_revision_required() -> None:
    assert infer_stage_status({"task_status": "modeling_check_revision_required"}) == StageStatus.REVISION_REQUIRED.value


def test_infer_completed_suffixes() -> None:
    for ts in [
        "initial_design_completed",
        "layout_revision_completed",
        "layout_revision_skipped",
        "structural_design_completed",
        "modeling_check_passed",
    ]:
        assert infer_stage_status({"task_status": ts}) == StageStatus.COMPLETED.value, ts


def test_infer_unknown_nonempty_is_manual_review() -> None:
    # 未识别的非空状态保守转人工复核，避免误放行。
    assert infer_stage_status({"task_status": "weird_status"}) == StageStatus.MANUAL_REVIEW.value


def test_infer_empty_is_completed() -> None:
    assert infer_stage_status({}) == StageStatus.COMPLETED.value


def test_build_handoff_assembles_fields() -> None:
    update = {"task_status": "structural_design_completed", "message": "完成"}
    h = build_handoff(
        "structural_design",
        update,
        produced_artifacts=[ArtifactRecord(artifact_type="reinforcement_design_result", state_key="reinforcement_design_result")],
        recommended_next_stage="modeling_check",
    )
    assert h.stage == "structural_design"
    assert h.status == StageStatus.COMPLETED.value
    assert h.message == "完成"
    assert len(h.produced_artifacts) == 1
    assert h.recommended_next_stage == "modeling_check"


def test_build_handoff_with_revision_request() -> None:
    update = {"task_status": "modeling_check_revision_required"}
    h = build_handoff(
        "modeling_check",
        update,
        revision_request=RevisionRequest(target_stage="structural_design", reason="配筋不足"),
    )
    assert h.status == StageStatus.REVISION_REQUIRED.value
    assert h.revision_request is not None
    assert h.revision_request.target_stage == "structural_design"
