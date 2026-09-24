from __future__ import annotations

from typing import Any, Dict, List, Optional

from .contracts import (
    ArtifactRecord,
    ComplianceResult,
    RevisionRequest,
    StageHandoff,
    StageStatus,
)


# 现有阶段 Agent 的 task_status -> 统一 StageStatus 的映射。
_TASK_STATUS_TO_STAGE_STATUS: Dict[str, str] = {
    "failed": StageStatus.FAILED.value,
    "blocked": StageStatus.BLOCKED.value,
    "manual_review_required": StageStatus.MANUAL_REVIEW.value,
    "modeling_check_revision_required": StageStatus.REVISION_REQUIRED.value,
    "layout_revision_required": StageStatus.REVISION_REQUIRED.value,
    "structural_revision_required": StageStatus.REVISION_REQUIRED.value,
}

# 视为"完成"的任务状态后缀。
_COMPLETED_SUFFIXES = (
    "_completed",
    "_passed",
    "_skipped",
)


def infer_stage_status(update: Dict[str, Any]) -> str:
    """从阶段 Agent 的 update dict 推断统一 StageStatus。

    - error 存在 -> failed；
    - task_status 命中已知映射（failed/blocked/manual_review/revision_required）-> 对应状态；
    - 其余 *_completed / *_passed / *_skipped 视为 completed。
    """
    if update.get("error"):
        return StageStatus.FAILED.value
    task_status = str(update.get("task_status") or "")
    if task_status in _TASK_STATUS_TO_STAGE_STATUS:
        return _TASK_STATUS_TO_STAGE_STATUS[task_status]
    if task_status.endswith(_COMPLETED_SUFFIXES):
        return StageStatus.COMPLETED.value
    # 未识别的非空状态保守视为 manual_review，避免误放行。
    if task_status:
        return StageStatus.MANUAL_REVIEW.value
    return StageStatus.COMPLETED.value


def build_handoff(
    stage: str,
    update: Dict[str, Any],
    *,
    produced_artifacts: Optional[List[ArtifactRecord]] = None,
    recommended_next_stage: Optional[str] = None,
    revision_request: Optional[RevisionRequest] = None,
    invalidated_artifacts: Optional[List[str]] = None,
    unresolved_code_items: Optional[List[ComplianceResult]] = None,
) -> StageHandoff:
    """从阶段 Agent 的 update dict 构建统一 StageHandoff。

    stage 使用 StageName 值（如 initial_design / structural_design）。
    """
    if unresolved_code_items is None:
        unresolved_code_items = []
        for item in update.get("unresolved_code_items") or []:
            if isinstance(item, ComplianceResult):
                unresolved_code_items.append(item)
            elif isinstance(item, dict):
                unresolved_code_items.append(
                    ComplianceResult(
                        code_item_id=str(item.get("code_item_id") or "unknown"),
                        status=str(item.get("status") or "manual_review"),
                        message=str(item.get("message") or ""),
                        evidence_ids=list(
                            item.get("evidence_ids")
                            or item.get("source_entity_ids")
                            or []
                        ),
                    )
                )
    return StageHandoff(
        stage=stage,
        status=infer_stage_status(update),
        produced_artifacts=list(produced_artifacts or []),
        revision_request=revision_request,
        invalidated_artifacts=list(invalidated_artifacts or []),
        unresolved_code_items=list(unresolved_code_items),
        recommended_next_stage=recommended_next_stage,
        message=str(update.get("message") or ""),
    )
