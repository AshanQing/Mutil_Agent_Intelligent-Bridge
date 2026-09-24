from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from .contracts import ArtifactRecord


# --------------------------------------------------------------------------- #
# 成果依赖链（上游 → 下游）
# --------------------------------------------------------------------------- #
# 依赖关系（对应层级框架计划第 6 节）：
#   路线和障碍资料 -> 布跨方案 -> 设计单元 -> 尺寸方案 -> 配筋方案
#   -> 分析结果 -> 验算结果 -> 绘图包 -> 最终交付成果
ARTIFACT_DEPENDENCY_CHAIN: tuple[str, ...] = (
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

# 每个成果类型对应的 state_key（用于与 AgentState / ArtifactRecord.state_key 对应）。
ARTIFACT_STATE_KEYS: Dict[str, str] = {
    "route_and_obstacles": "obstacle_extractor_result",
    "layout_result": "layout_result",
    "design_units": "design_units",
    "dimension_design_result": "dimension_design_result",
    "reinforcement_design_result": "reinforcement_design_result",
    "analysis_result": "analysis_result",
    "check_result": "check_result",
    "drawing_package": "drawing_package_result",
    "final_deliverables": "design_manifest_path",
}

# 成果失效时必须清空的运行字段。路径字段也需要清空，避免工具继续读取旧文件。
ARTIFACT_RUNTIME_KEYS: Dict[str, tuple[str, ...]] = {
    "route_and_obstacles": (
        "obstacle_extractor_result",
        "obstacle_json_path",
    ),
    "layout_result": (
        "layout_result",
        "layout_revision_result",
        "layout_revision_result_path",
        "final_layout_result",
        "final_layout_result_path",
    ),
    "design_units": ("design_units",),
    "dimension_design_result": ("dimension_design_result",),
    "reinforcement_design_result": (
        "reinforcement_design_result",
        "reinforcement_yaml_path",
        "reinforcement_yaml_paths",
    ),
    "analysis_result": (
        "analysis_result",
        "opensees_force_json_path",
        "internal_force_output_paths",
    ),
    "check_result": (
        "check_result",
        "capacity_check_result",
        "capacity_check_summary_path",
    ),
    "drawing_package": (
        "drawing_package_result",
        "drawing_index_path",
        "cad_script_paths",
        "drawing_preview_paths",
    ),
    "final_deliverables": ("design_manifest_path",),
}

ARTIFACT_EMPTY_VALUES: Dict[str, Any] = {
    "cad_script_paths": [],
    "drawing_preview_paths": [],
}

# 判断一个失效成果是否已被当前阶段的新输出重建。
ARTIFACT_REFRESH_KEYS: Dict[str, tuple[str, ...]] = {
    "route_and_obstacles": ("obstacle_extractor_result", "obstacle_json_path"),
    "layout_result": ("layout_result", "final_layout_result"),
    "design_units": ("design_units",),
    "dimension_design_result": ("dimension_design_result",),
    "reinforcement_design_result": ("reinforcement_design_result",),
    "analysis_result": ("analysis_result", "opensees_force_json_path"),
    "check_result": ("check_result", "capacity_check_result"),
    "drawing_package": ("drawing_package_result", "drawing_index_path"),
    "final_deliverables": ("design_manifest_path",),
}

# 各返修阶段必须负责重建的成果；仍有其中任一项失效时不能清除 revision_target。
STAGE_REWORK_ARTIFACTS: Dict[str, frozenset[str]] = {
    "layout_revision": frozenset({"layout_result"}),
    "structural_design": frozenset({
        "design_units",
        "dimension_design_result",
        "reinforcement_design_result",
        "analysis_result",
    }),
    "modeling_check": frozenset({"check_result"}),
}


def compute_invalidated(changed_artifact: str) -> Set[str]:
    """给定一个发生变更的成果类型，返回所有失效的下游成果类型。

    只修改解释、日志或 Prompt 不属于依赖链，不会导致工程成果失效。
    """
    if changed_artifact not in ARTIFACT_DEPENDENCY_CHAIN:
        return set()
    idx = ARTIFACT_DEPENDENCY_CHAIN.index(changed_artifact)
    return set(ARTIFACT_DEPENDENCY_CHAIN[idx + 1 :])


def compute_rework_path(changed_artifact: str) -> List[str]:
    """返回从 changed_artifact 之后需要重做的最小成果序列（含其直接下游）。"""
    if changed_artifact not in ARTIFACT_DEPENDENCY_CHAIN:
        return []
    idx = ARTIFACT_DEPENDENCY_CHAIN.index(changed_artifact)
    return list(ARTIFACT_DEPENDENCY_CHAIN[idx + 1 :])


def _ordered_artifacts(artifacts: Iterable[str]) -> List[str]:
    names = set(artifacts)
    return [name for name in ARTIFACT_DEPENDENCY_CHAIN if name in names]


def compute_revision_invalidated(required_artifacts: Iterable[str]) -> Set[str]:
    """返回返修请求应失效的成果集合，包含请求成果自身及全部下游。"""
    invalidated: Set[str] = set()
    for artifact in required_artifacts:
        if artifact not in ARTIFACT_DEPENDENCY_CHAIN:
            continue
        invalidated.add(artifact)
        invalidated.update(compute_invalidated(artifact))
    return invalidated


def build_revision_state_update(
    handoff: Dict[str, Any],
    *,
    current_invalidated: Iterable[str] = (),
) -> Dict[str, Any]:
    """把 StageHandoff 中的返修请求转换为可直接合并到 AgentState 的更新。"""
    revision_request = handoff.get("revision_request")
    if not isinstance(revision_request, dict):
        return {}

    required_artifacts = [str(name) for name in revision_request.get("required_artifacts") or []]
    invalidated = set(current_invalidated)
    invalidated.update(str(name) for name in handoff.get("invalidated_artifacts") or [])
    invalidated.update(compute_revision_invalidated(required_artifacts))
    ordered_invalidated = _ordered_artifacts(invalidated)

    update: Dict[str, Any] = {
        "revision_target": revision_request.get("target_stage"),
        "rework_required_artifacts": required_artifacts,
        "invalidated_artifacts": ordered_invalidated,
        "minimal_rework_path": ordered_invalidated,
    }
    for artifact in ordered_invalidated:
        for state_key in ARTIFACT_RUNTIME_KEYS.get(artifact, ()):
            empty_value = ARTIFACT_EMPTY_VALUES.get(state_key)
            update[state_key] = list(empty_value) if isinstance(empty_value, list) else None
    return update


def build_artifact_refresh_update(
    stage_name: str,
    stage_update: Dict[str, Any],
    *,
    current_invalidated: Iterable[str] = (),
    revision_target: Optional[str] = None,
) -> Dict[str, Any]:
    """根据阶段新产物移除已重建的失效项，并在返修阶段完成后清理调度标记。"""
    invalidated = set(current_invalidated)
    if not invalidated and not revision_target:
        return {}

    refreshed = {
        artifact
        for artifact in invalidated
        if any(stage_update.get(key) is not None for key in ARTIFACT_REFRESH_KEYS.get(artifact, ()))
    }
    remaining = _ordered_artifacts(invalidated - refreshed)
    result: Dict[str, Any] = {
        "invalidated_artifacts": remaining,
        "minimal_rework_path": remaining,
    }

    task_status = str(stage_update.get("task_status") or "")
    stage_pending = set(remaining) & STAGE_REWORK_ARTIFACTS.get(stage_name, frozenset())
    if (
        stage_name == revision_target
        and not stage_pending
        and task_status.endswith(("_completed", "_passed"))
    ):
        result["revision_target"] = None
        result["rework_required_artifacts"] = []
    return result


class ArtifactRegistry:
    """成果注册表：登记成果记录，并支持失效传播与最小返工路径计算。"""

    def __init__(self) -> None:
        self._artifacts: Dict[str, ArtifactRecord] = {}

    def register(self, artifact: ArtifactRecord) -> None:
        self._artifacts[artifact.artifact_type] = artifact

    def get(self, artifact_type: str) -> Optional[ArtifactRecord]:
        return self._artifacts.get(artifact_type)

    def is_valid(self, artifact_type: str) -> bool:
        rec = self._artifacts.get(artifact_type)
        return rec is None or rec.valid

    def invalidate(self, changed_artifact: str) -> Set[str]:
        """将 changed_artifact 的下游成果标记为失效，返回被失效的集合。

        changed_artifact 本身假定已被调用方更新为新版本。
        """
        invalidated = compute_invalidated(changed_artifact)
        for name in invalidated:
            rec = self._artifacts.get(name)
            if rec is not None:
                rec.valid = False
        return invalidated

    def minimal_rework_path(self, changed_artifact: str) -> List[str]:
        """返回 changed_artifact 变化后需要重做的最小成果序列。"""
        return compute_rework_path(changed_artifact)

    def all_valid(self) -> bool:
        return all(rec.valid for rec in self._artifacts.values())
