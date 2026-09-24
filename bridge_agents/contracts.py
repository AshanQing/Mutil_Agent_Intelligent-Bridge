from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple


class StageName(str, Enum):
    """顶层图显式注册的阶段节点名。"""

    DESIGN_COORDINATOR = "design_coordinator"
    INITIAL_DESIGN = "initial_design"
    LAYOUT_REVISION = "layout_revision"
    STRUCTURAL_DESIGN = "structural_design"
    MODELING_CHECK = "modeling_check"
    MANUAL_REVIEW = "manual_review"
    FINAL_OUTPUT = "final_output"
    ERROR = "error"


class StageStatus(str, Enum):
    """专业 Agent 阶段交接的统一状态。"""

    COMPLETED = "completed"
    REVISION_REQUIRED = "revision_required"
    BLOCKED = "blocked"
    FAILED = "failed"
    MANUAL_REVIEW = "manual_review"


VALID_STAGE_STATUSES = frozenset(s.value for s in StageStatus)


@dataclass
class ArtifactRecord:
    """一个设计成果的记录，含内容哈希与有效性标记，用于依赖追踪。"""

    artifact_type: str
    state_key: str
    path: Optional[str] = None
    content_hash: str = ""
    producer: str = ""
    revision: int = 0
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "state_key": self.state_key,
            "path": self.path,
            "content_hash": self.content_hash,
            "producer": self.producer,
            "revision": self.revision,
            "valid": self.valid,
        }


@dataclass
class StageFinding:
    """阶段内的一条发现/结论。"""

    severity: str  # info | warning | error
    message: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass
class RevisionRequest:
    """验算/校核失败时发出的定向返修请求。"""

    target_stage: str
    reason: str
    required_artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_stage": self.target_stage,
            "reason": self.reason,
            "required_artifacts": list(self.required_artifacts),
        }


@dataclass
class ComplianceResult:
    """证据门控的合规结论（规范项级）。"""

    code_item_id: str
    status: str  # pass | fail | manual_review | not_applicable | missing_input
    message: str = ""
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_item_id": self.code_item_id,
            "status": self.status,
            "message": self.message,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass
class StageHandoff:
    """所有专业 Agent 统一的阶段交接产物。

    专业 Agent 只能提出 recommended_next_stage 建议，不能直接控制顶层跳转。
    """

    stage: str
    status: str
    produced_artifacts: List[ArtifactRecord] = field(default_factory=list)
    findings: List[StageFinding] = field(default_factory=list)
    revision_request: Optional[RevisionRequest] = None
    invalidated_artifacts: List[str] = field(default_factory=list)
    recommended_next_stage: Optional[str] = None
    unresolved_code_items: List[ComplianceResult] = field(default_factory=list)
    message: str = ""

    def is_completed(self) -> bool:
        return self.status == StageStatus.COMPLETED.value

    def has_unresolved_code_items(self) -> bool:
        return bool(self.unresolved_code_items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "produced_artifacts": [a.to_dict() for a in self.produced_artifacts],
            "findings": [f.to_dict() for f in self.findings],
            "revision_request": (
                self.revision_request.to_dict() if self.revision_request else None
            ),
            "invalidated_artifacts": list(self.invalidated_artifacts),
            "recommended_next_stage": self.recommended_next_stage,
            "unresolved_code_items": [c.to_dict() for c in self.unresolved_code_items],
            "message": self.message,
        }


@dataclass
class AgentContract:
    """一个专业 Agent 的静态契约：技能、动作权限与状态读写边界。"""

    agent_name: str
    skill_id: str
    allowed_actions: Tuple[str, ...]
    readable_state_keys: Tuple[str, ...]
    produced_state_keys: Tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "skill_id": self.skill_id,
            "allowed_actions": list(self.allowed_actions),
            "readable_state_keys": list(self.readable_state_keys),
            "produced_state_keys": list(self.produced_state_keys),
        }


def validate_contract(
    contract: AgentContract,
    skill_ids: Any,
    registered_actions: Any,
) -> List[str]:
    """校验契约与 Skill、Action 注册表一致，返回错误列表（空列表表示通过）。

    - skill_id 必须存在于 skill_ids（SkillRegistry 提供的集合）；
    - allowed_actions 中的每个动作必须在 registered_actions 中注册；
    - readable_state_keys / produced_state_keys 必须非空。
    """
    errors: List[str] = []
    if contract.skill_id not in skill_ids:
        errors.append(f"Skill not found: {contract.skill_id}")
    for action in contract.allowed_actions:
        if action not in registered_actions:
            errors.append(f"Action not registered: {action}")
    if not contract.readable_state_keys:
        errors.append("readable_state_keys must not be empty")
    if not contract.produced_state_keys:
        errors.append("produced_state_keys must not be empty")
    return errors


# --------------------------------------------------------------------------- #
# 契约注册表：与 skills/<name>/SKILL.md 及 bridge_agents.actions.AGENT_ACTIONS 对齐
# --------------------------------------------------------------------------- #
AGENT_CONTRACTS: dict[str, AgentContract] = {
    "InitialDesignAgent": AgentContract(
        agent_name="InitialDesignAgent",
        skill_id="initial-design",
        allowed_actions=(
            "load_data",
            "drawing_crop_and_mask",
            "obstacle_semantic_extractor",
            "select_samples",
            "generate_layout_design",
        ),
        readable_state_keys=(
            "user_intent",
            "user_request",
            "data_path",
            "file_prefix",
            "input_drawing_path",
            "start_station",
            "end_station",
            "layout_result",
            "cropped_data",
            "plane_json_path",
            "mask_path",
            "pgw_path",
            "obstacle_json_path",
            "few_shots",
            "design_input",
        ),
        produced_state_keys=(
            "layout_result",
            "design_input",
            "few_shots",
            "cropped_data",
            "data_loader_result",
            "obstacle_extractor_result",
            "plane_json_path",
            "mask_path",
            "pgw_path",
            "png_path",
            "jpg_path",
            "obstacle_json_path",
        ),
    ),
    "LayoutRevisionAgent": AgentContract(
        agent_name="LayoutRevisionAgent",
        skill_id="layout-revision",
        allowed_actions=(
            "run_collision_detection",
            "generate_revision_instruction",
            "build_revision_prompt",
            "generate_revised_layout",
        ),
        readable_state_keys=(
            "layout_result",
            "collision_result",
            "collision_metrics",
            "collision_items",
            "revision_instruction",
            "revision_prompt",
            "iteration_index",
            "max_revision_rounds",
        ),
        produced_state_keys=(
            "collision_result",
            "collision_metrics",
            "collision_items",
            "revision_instruction",
            "revision_prompt",
            "revision_result_raw",
            "layout_revision_result",
            "final_layout_result",
        ),
    ),
    "StructuralDesignAgent": AgentContract(
        agent_name="StructuralDesignAgent",
        skill_id="structural-design",
        allowed_actions=(
            "extract_design_units",
            "dimension_design",
            "reinforcement_design",
        ),
        readable_state_keys=(
            "layout_result",
            "design_units",
            "dimension_design_result",
            "reinforcement_design_result",
            "structural_design_result",
            "evidence_bundles",
        ),
        produced_state_keys=(
            "design_units",
            "dimension_design_result",
            "reinforcement_design_result",
            "structural_design_result",
            "evidence_bundles",
            "code_trace",
        ),
    ),
    "ModelingCheckAgent": AgentContract(
        agent_name="ModelingCheckAgent",
        skill_id="modeling-check",
        allowed_actions=(
            "run_capacity_check",
            "generate_modeling_feedback",
        ),
        readable_state_keys=(
            "reinforcement_yaml_path",
            "opensees_force_json_path",
            "capacity_check_result",
            "check_result",
            "feedback_decision",
        ),
        produced_state_keys=(
            "capacity_check_result",
            "check_result",
            "feedback_decision",
            "revision_context",
        ),
    ),
    "DesignReviewAgent": AgentContract(
        agent_name="DesignReviewAgent",
        skill_id="design-review",
        allowed_actions=(),
        readable_state_keys=(
            "output_dir",
            "final_summary",
            "layout_result",
            "structural_design_result",
            "capacity_check_result",
            "evidence_bundles",
        ),
        produced_state_keys=(
            "design_assessment",
            "design_review_answer",
            "design_review_audit_path",
        ),
    ),
    "DesignReviewAgent": AgentContract(
        agent_name="DesignReviewAgent",
        skill_id="design-review",
        allowed_actions=(),
        readable_state_keys=(
            "output_dir",
            "final_summary",
            "layout_result",
            "structural_design_result",
            "capacity_check_result",
            "evidence_bundles",
        ),
        produced_state_keys=(
            "design_assessment",
            "design_review_answer",
            "design_review_audit_path",
        ),
    ),
}


def validate_all_contracts(
    skill_ids: Any,
    registered_actions: Any,
) -> List[str]:
    """校验全部 AGENT_CONTRACTS 与 Skill/Action 注册表一致。"""
    errors: List[str] = []
    for agent_name, contract in AGENT_CONTRACTS.items():
        for err in validate_contract(contract, skill_ids, registered_actions):
            errors.append(f"{agent_name}: {err}")
    return errors
