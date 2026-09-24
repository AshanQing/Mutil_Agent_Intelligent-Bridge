from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .contracts import StageName


# --------------------------------------------------------------------------- #
# 协调器决策
# --------------------------------------------------------------------------- #
@dataclass
class CoordinatorDecision:
    """DesignCoordinatorAgent 的一次调度决策。

    LLM 负责提出该决策；确定性策略层负责校验，非法决策不得执行。
    """

    intent: str
    next_stage: Optional[str]  # StageName 值；task_complete=True 时可为 None
    stage_objective: str
    reason: str
    revision_target: Optional[str] = None
    required_artifacts: List[str] = field(default_factory=list)
    task_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "next_stage": self.next_stage,
            "stage_objective": self.stage_objective,
            "reason": self.reason,
            "revision_target": self.revision_target,
            "required_artifacts": list(self.required_artifacts),
            "task_complete": self.task_complete,
        }


# --------------------------------------------------------------------------- #
# 协调器状态视图（与完整 AgentState 解耦，便于确定性校验与单元测试）
# --------------------------------------------------------------------------- #
@dataclass
class CoordinatorView:
    """协调器只读的最小状态视图，用于确定性调度校验。"""

    layout_result_available: bool = False
    layout_revision_result_available: bool = False
    collision_detected_available: bool = False
    design_units_available: bool = False
    dimension_design_available: bool = False
    reinforcement_design_available: bool = False
    opensees_force_available: bool = False
    capacity_check_available: bool = False
    check_passed: Optional[bool] = None
    # 兼容旧调用方的通用轮次字段；新代码按阶段使用下方独立计数。
    revision_round: int = 0
    max_revision_rounds: int = 3
    layout_revision_round: Optional[int] = None
    max_layout_revision_rounds: Optional[int] = None
    structural_revision_round: Optional[int] = None
    max_structural_revision_rounds: Optional[int] = None
    unresolved_manual_review: bool = False
    revision_target: Optional[str] = None
    initial_design_completed: bool = False
    layout_revision_completed: bool = False
    structural_design_completed: bool = False
    modeling_check_completed: bool = False
    modeling_risk_accepted: bool = False
    modeling_slenderness_review_pending: bool = False


# --------------------------------------------------------------------------- #
# 阶段依赖与强制检查
# --------------------------------------------------------------------------- #
# 进入某阶段所需的前置成果（映射到 CoordinatorView 字段）。
# 结构设计必须在“布跨修正完成（final_layout_result）”且“至少完成一次碰撞检测
# （collision_metrics）”之后才可进入——初始布跨结果本身不足以支撑结构设计。
STAGE_DEPENDENCIES: Dict[str, tuple[str, ...]] = {
    StageName.INITIAL_DESIGN.value: (),
    StageName.LAYOUT_REVISION.value: ("layout_result_available",),
    StageName.STRUCTURAL_DESIGN.value: (
        "layout_revision_result_available",
        "collision_detected_available",
    ),
    StageName.MODELING_CHECK.value: (
        "reinforcement_design_available",
        "opensees_force_available",
    ),
}

# 允许"返修"的专业阶段（用于轮次上限校验）。
REVISABLE_STAGES = frozenset(
    {StageName.STRUCTURAL_DESIGN.value, StageName.LAYOUT_REVISION.value}
)

# 需要强制完成验算的意图（full_design 类）。
INTENTS_REQUIRING_CHECK = frozenset(
    {"full_design", "reinforcement_design_verification", "verification"}
)

REGISTERED_STAGES = frozenset(s.value for s in StageName)


# --------------------------------------------------------------------------- #
# 意图 -> 阶段序列（规则兜底，替代旧 INTENT_TO_AGENT_SEQUENCE 的队列语义）
# --------------------------------------------------------------------------- #
INTENT_TO_STAGE_SEQUENCE: Dict[str, List[str]] = {
    "full_design": [
        StageName.INITIAL_DESIGN.value,
        StageName.LAYOUT_REVISION.value,
        StageName.STRUCTURAL_DESIGN.value,
        StageName.MODELING_CHECK.value,
    ],
    "layout_design": [StageName.INITIAL_DESIGN.value],
    "layout_design_check_revision": [
        StageName.INITIAL_DESIGN.value,
        StageName.LAYOUT_REVISION.value,
    ],
    "layout_check_revision": [StageName.LAYOUT_REVISION.value],
    "structural_design": [StageName.STRUCTURAL_DESIGN.value],
    "layout_to_reinforcement_design": [
        StageName.INITIAL_DESIGN.value,
        StageName.LAYOUT_REVISION.value,
        StageName.STRUCTURAL_DESIGN.value,
    ],
    "reinforcement_design": [StageName.STRUCTURAL_DESIGN.value],
    "reinforcement_design_verification": [
        StageName.STRUCTURAL_DESIGN.value,
        StageName.MODELING_CHECK.value,
    ],
    "verification": [StageName.MODELING_CHECK.value],
}


def _dep_satisfied(dep: str, view: CoordinatorView) -> bool:
    return bool(getattr(view, dep, False))


def infer_intent(user_request: str) -> str:
    """规则兜底：从用户请求文本推断 user_intent。

    与旧 TaskAllocationAgent 的规则兜底保持一致，供协调器在 LLM 不可用时使用。
    """
    text = user_request or ""
    if any(x in text for x in ["从头", "原始资料", "完整"]) and any(
        x in text for x in ["配筋", "结构设计", "下部结构"]
    ):
        return "layout_to_reinforcement_design"
    if any(x in text for x in ["全流程", "完整设计", "从头", "全部"]):
        return "full_design"
    if any(x in text for x in ["配筋"]) and any(
        x in text for x in ["验算", "校核", "承载力", "反馈", "OpenSees"]
    ):
        return "reinforcement_design_verification"
    if any(x in text for x in ["验算", "建模", "OpenSees", "分析"]):
        return "verification"
    if any(x in text for x in ["配筋"]):
        return "reinforcement_design"
    if any(x in text for x in ["下部结构", "结构设计", "尺寸设计", "盖梁", "墩柱", "基础"]):
        return "structural_design"
    if any(x in text for x in ["已有", "检测", "校核", "修正", "冲突", "复检"]):
        if any(x in text for x in ["设桥", "布跨", "桥位"]):
            return "layout_design_check_revision"
        return "layout_check_revision"
    if any(x in text for x in ["设桥", "布跨", "桥位"]):
        return "layout_design"
    return "unknown"


def validate_decision(
    decision: CoordinatorDecision,
    view: CoordinatorView,
) -> List[str]:
    """对调度决策做确定性校验，返回错误列表（空列表表示通过）。

    校验规则（对应层级框架计划第 5 节）：
    1. 下一阶段必须已注册；
    2. 前置成果必须有效；
    3. 不得跳过强制检查（验算意图下验算未完成不得直接结束）；
    4. 失败成果不能最终放行；
    5. 返修目标必须与 RevisionRequest 一致；
    6. 修正轮次不得超限；
    7. 存在人工复核项时不能最终放行。
    """
    errors: List[str] = []

    if decision.task_complete:
        if view.unresolved_manual_review:
            errors.append("存在未解决的人工复核项，不能最终放行")
        if view.check_passed is False and not view.modeling_risk_accepted:
            # 人工已经明确"接受未通过的验算结论"时，验算失败不得再阻断最终放行：
            # 否则人工决定只落进审计账本，流程仍被确定性校验挡在门外，
            # 已成功设计出的尺寸/配筋也拿不到图纸（2026-09-16 示例项目K31 接受风险后
            # 重跑仍无法继续，即该规则未认可人工接受所致）。
            errors.append("验算失败成果不能最终放行")
        if view.modeling_slenderness_review_pending and not view.modeling_risk_accepted:
            errors.append("墩柱长细比复核未决（超出表5.3.1适用范围）且未获人工接受，不能最终放行")
        if decision.intent in INTENTS_REQUIRING_CHECK and not view.capacity_check_available:
            errors.append(f"意图 {decision.intent!r} 要求完成验算，不能跳过强制检查直接结束")
        return errors

    stage = decision.next_stage
    if stage not in REGISTERED_STAGES:
        errors.append(f"未注册的阶段: {stage!r}")
        return errors

    completed_by_stage = {
        StageName.INITIAL_DESIGN.value: view.initial_design_completed,
        StageName.LAYOUT_REVISION.value: view.layout_revision_completed,
        StageName.STRUCTURAL_DESIGN.value: view.structural_design_completed,
        StageName.MODELING_CHECK.value: view.modeling_check_completed,
    }
    explicit_rework = (
        decision.revision_target == stage or view.revision_target == stage
    )
    if completed_by_stage.get(stage, False) and not explicit_rework:
        errors.append(f"已完成阶段 {stage!r} 缺少明确返工目标，不允许重复调度")

    # 阶段顺序约束：流程按意图序列逐步推进，不允许跳过序列中尚未完成的阶段
    # （例如 full_design 不允许在布跨修正未完成时直接进入结构设计）。
    sequence = INTENT_TO_STAGE_SEQUENCE.get(decision.intent) or []
    if stage in sequence and not explicit_rework:
        for prior_stage in sequence[: sequence.index(stage)]:
            if not completed_by_stage.get(prior_stage, False):
                errors.append(
                    f"意图 {decision.intent!r} 的阶段 {prior_stage!r} 尚未完成，"
                    f"不能跳过进入 {stage!r}"
                )
                break

    # 前置成果
    for dep in STAGE_DEPENDENCIES.get(stage, ()):
        if not _dep_satisfied(dep, view):
            errors.append(f"阶段 {stage!r} 的前置成果无效: {dep}")

    # 返修目标一致
    if view.revision_target is not None:
        if decision.revision_target != view.revision_target:
            errors.append(
                f"返修目标不一致: 期望 {view.revision_target!r}，实际 {decision.revision_target!r}"
            )

    # 修正轮次按专业阶段独立计数，避免布跨轮次阻断首次结构设计。
    if stage == StageName.LAYOUT_REVISION.value:
        revision_round = (
            view.layout_revision_round
            if view.layout_revision_round is not None
            else view.revision_round
        )
        max_revision_rounds = (
            view.max_layout_revision_rounds
            if view.max_layout_revision_rounds is not None
            else view.max_revision_rounds
        )
    elif stage == StageName.STRUCTURAL_DESIGN.value:
        revision_round = (
            view.structural_revision_round
            if view.structural_revision_round is not None
            else view.revision_round
        )
        max_revision_rounds = (
            view.max_structural_revision_rounds
            if view.max_structural_revision_rounds is not None
            else view.max_revision_rounds
        )
    else:
        revision_round = 0
        max_revision_rounds = view.max_revision_rounds

    if stage in REVISABLE_STAGES and revision_round >= max_revision_rounds:
        errors.append(
            f"修正轮次超限: {revision_round} >= {max_revision_rounds}，应转人工复核"
        )

    # 失败成果不得进入下游
    if stage == StageName.FINAL_OUTPUT.value:
        if view.check_passed is False:
            errors.append("失败成果不能最终放行")

    return errors


def validate_and_retry(
    decision: CoordinatorDecision,
    view: CoordinatorView,
    *,
    max_attempts: int = 2,
) -> CoordinatorDecision:
    """校验决策；非法时抛出 ValueError（由调用方携带校验错误重试）。

    连续两次非法或无法解析由上层转为人工复核。
    """
    errors = validate_decision(decision, view)
    if errors:
        raise ValueError("; ".join(errors))
    return decision
