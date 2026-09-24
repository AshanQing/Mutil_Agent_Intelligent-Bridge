"""阶段连续失败上限：纯失败（非返修）也必须能停下来，而不是无限重新调度。

背景：`stage_revision_rounds` 只在返修重入时递增，协调器对"阶段未完成"没有任何
刹车。若阶段失败是确定性的（例如设计单元语义校验恒不通过），流程会无上限空转，
既不产出成果也不上报失败。
"""

from __future__ import annotations

from bridge_agents.contracts import StageName, StageStatus
from bridge_agents.graph_v2 import (
    DEFAULT_MAX_STAGE_FAILURES,
    apply_human_review_decision,
    make_stage_node,
    route_after_stage,
)


def _failed_handoff(stage: str, message: str = "阶段执行失败。"):
    return {
        "stage": stage,
        "status": StageStatus.FAILED.value,
        "produced_artifacts": [],
        "findings": [],
        "revision_request": None,
        "invalidated_artifacts": [],
        "recommended_next_stage": None,
        "unresolved_code_items": [],
        "message": message,
    }


def _failed_node(stage: str = StageName.STRUCTURAL_DESIGN.value):
    return make_stage_node(
        stage,
        lambda state: {"latest_handoff": _failed_handoff(stage), "error": "失败"},
    )


def _completed_node(stage: str = StageName.STRUCTURAL_DESIGN.value):
    return make_stage_node(
        stage,
        lambda state: {
            "latest_handoff": {
                **_failed_handoff(stage),
                "status": StageStatus.COMPLETED.value,
                "message": "完成",
            }
        },
    )


def test_failure_count_accumulates_across_rounds() -> None:
    node = _failed_node()
    state: dict = {}

    for expected in (1, 2, 3):
        state = {**state, **node(state)}
        assert state["stage_failure_counts"][StageName.STRUCTURAL_DESIGN.value] == expected


def test_below_limit_still_returns_to_coordinator() -> None:
    node = _failed_node()
    state: dict = {}

    for _ in range(DEFAULT_MAX_STAGE_FAILURES - 1):
        state = {**state, **node(state)}

    assert state["stage_failure_limit_reached"] is False
    assert route_after_stage(state) == "design_coordinator"


def test_reaching_limit_routes_to_manual_review() -> None:
    node = _failed_node()
    state: dict = {}

    for _ in range(DEFAULT_MAX_STAGE_FAILURES):
        state = {**state, **node(state)}

    assert state["stage_failure_limit_reached"] is True
    assert state["unresolved_manual_review"] is True
    assert StageName.STRUCTURAL_DESIGN.value in state["manual_review_reason"]
    assert str(DEFAULT_MAX_STAGE_FAILURES) in state["manual_review_reason"]
    assert route_after_stage(state) == StageName.MANUAL_REVIEW.value


def test_custom_limit_is_honored() -> None:
    node = _failed_node()
    state: dict = {"max_stage_failures": 1}

    state = {**state, **node(state)}

    assert state["stage_failure_limit_reached"] is True


def test_successful_stage_clears_its_failure_count() -> None:
    failed = _failed_node()
    completed = _completed_node()
    state: dict = {}

    state = {**state, **failed(state)}
    state = {**state, **failed(state)}
    assert state["stage_failure_counts"][StageName.STRUCTURAL_DESIGN.value] == 2

    state = {**state, **completed(state)}

    assert state["stage_failure_counts"] == {}
    assert state["stage_failure_limit_reached"] is False


def test_failure_counts_are_tracked_per_stage() -> None:
    structural = _failed_node(StageName.STRUCTURAL_DESIGN.value)
    modeling = _failed_node(StageName.MODELING_CHECK.value)
    state: dict = {}

    state = {**state, **structural(state)}
    state = {**state, **modeling(state)}

    assert state["stage_failure_counts"] == {
        StageName.STRUCTURAL_DESIGN.value: 1,
        StageName.MODELING_CHECK.value: 1,
    }


def test_human_review_decision_renews_failure_budget() -> None:
    """人工决策后必须重置计数，否则"重试"会立刻再次触顶并弹回复核。"""
    node = _failed_node()
    state: dict = {}
    for _ in range(DEFAULT_MAX_STAGE_FAILURES):
        state = {**state, **node(state)}
    assert state["stage_failure_limit_reached"] is True

    update = apply_human_review_decision(
        state,
        {"action": "retry_failed_tasks", "feedback": "已修复校验器，重新执行"},
    )

    assert update["stage_failure_counts"] == {}
    assert update["stage_failure_limit_reached"] is False
