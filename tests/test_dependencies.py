from __future__ import annotations

import pytest

from bridge_agents.contracts import ArtifactRecord
from bridge_agents.dependencies import (
    ARTIFACT_DEPENDENCY_CHAIN,
    ArtifactRegistry,
    compute_invalidated,
    build_revision_state_update,
    compute_revision_invalidated,
    compute_rework_path,
)


def test_route_change_invalidates_everything_downstream() -> None:
    invalidated = compute_invalidated("route_and_obstacles")
    assert invalidated == {
        "layout_result",
        "design_units",
        "dimension_design_result",
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    }


def test_layout_change_invalidates_structure_downstream() -> None:
    invalidated = compute_invalidated("layout_result")
    assert invalidated == {
        "design_units",
        "dimension_design_result",
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    }


def test_dimension_change_only_invalidates_reinforcement_and_after() -> None:
    invalidated = compute_invalidated("dimension_design_result")
    assert invalidated == {
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    }


def test_reinforcement_change_only_invalidates_analysis_and_check() -> None:
    invalidated = compute_invalidated("reinforcement_design_result")
    assert invalidated == {
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    }


def test_revision_invalidation_includes_requested_artifact_and_downstream() -> None:
    assert compute_revision_invalidated(["reinforcement_design_result"]) == {
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    }
    assert compute_revision_invalidated(["dimension_design_result"]) == {
        "dimension_design_result",
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    }


def test_check_result_change_invalidates_drawing_deliverables() -> None:
    assert compute_invalidated("check_result") == {
        "drawing_package",
        "final_deliverables",
    }


def test_non_artifact_change_invalidates_nothing() -> None:
    # 只修改解释、日志或 Prompt 不属于依赖链，不导致工程成果失效。
    assert compute_invalidated("prompt_trace") == set()
    assert compute_invalidated("agent_log") == set()
    assert compute_invalidated("message") == set()


def test_rework_path_for_reinforcement() -> None:
    assert compute_rework_path("reinforcement_design_result") == [
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    ]


def test_chain_order_matches_plan() -> None:
    assert ARTIFACT_DEPENDENCY_CHAIN == (
        "route_and_obstacles",
        "layout_result",
        "design_units",
        "dimension_design_result",
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    )


def test_registry_invalidate_marks_downstream_invalid() -> None:
    registry = ArtifactRegistry()
    for name in ARTIFACT_DEPENDENCY_CHAIN:
        registry.register(ArtifactRecord(artifact_type=name, state_key=name, valid=True))

    invalidated = registry.invalidate("dimension_design_result")
    assert invalidated == {
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    }
    assert registry.is_valid("dimension_design_result") is True  # 上游本身仍有效
    assert registry.is_valid("reinforcement_design_result") is False
    assert registry.is_valid("analysis_result") is False
    assert registry.is_valid("check_result") is False
    assert registry.is_valid("layout_result") is True


def test_registry_minimal_rework_path() -> None:
    registry = ArtifactRegistry()
    assert registry.minimal_rework_path("layout_result") == [
        "design_units",
        "dimension_design_result",
        "reinforcement_design_result",
        "analysis_result",
        "check_result",
        "drawing_package",
        "final_deliverables",
    ]


def test_registry_all_valid() -> None:
    registry = ArtifactRegistry()
    for name in ARTIFACT_DEPENDENCY_CHAIN:
        registry.register(ArtifactRecord(artifact_type=name, state_key=name, valid=True))
    assert registry.all_valid() is True
    registry.invalidate("layout_result")
    assert registry.all_valid() is False


def test_reinforcement_revision_clears_drawing_path_lists() -> None:
    update = build_revision_state_update({
        "revision_request": {
            "target_stage": "structural_design",
            "required_artifacts": ["reinforcement_design_result"],
        },
        "invalidated_artifacts": [],
    })

    assert update["drawing_package_result"] is None
    assert update["cad_script_paths"] == []
    assert update["drawing_preview_paths"] == []
