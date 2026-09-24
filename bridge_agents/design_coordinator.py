from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .contracts import StageName
from .coordinator import (
    CoordinatorDecision,
    CoordinatorView,
    INTENT_TO_STAGE_SEQUENCE,
    infer_intent,
    validate_decision,
)

logger = logging.getLogger(__name__)

# LLM 调用签名：接收 (system_text, user_text) -> 返回文本。
LLMInvoke = Callable[[str, str], str]

COORDINATOR_NEXT_STAGES = [
    StageName.INITIAL_DESIGN.value,
    StageName.LAYOUT_REVISION.value,
    StageName.STRUCTURAL_DESIGN.value,
    StageName.MODELING_CHECK.value,
    StageName.MANUAL_REVIEW.value,
]


def build_coordinator_view(state: Dict[str, Any]) -> CoordinatorView:
    """从 AgentState（dict）抽取协调器只读视图。"""
    invalidated = set(state.get("invalidated_artifacts") or [])
    check_result = state.get("check_result")
    check_passed: Optional[bool] = None
    if "check_result" not in invalidated and isinstance(check_result, dict):
        overall = check_result.get("overall_check")
        if isinstance(overall, bool):
            check_passed = overall
        elif isinstance(overall, dict) and isinstance(overall.get("all_ok"), bool):
            check_passed = overall["all_ok"]

    latest_handoff = state.get("latest_handoff") if isinstance(state.get("latest_handoff"), dict) else {}
    revision_request = latest_handoff.get("revision_request") if isinstance(latest_handoff.get("revision_request"), dict) else {}
    revision_target = state.get("revision_target") or revision_request.get("target_stage")
    reinforcement_batch = (
        state.get("reinforcement_batch_status")
        if isinstance(state.get("reinforcement_batch_status"), dict)
        else {}
    )
    reinforcement_batch_complete = (
        reinforcement_batch.get("stage_complete") is not False
        or reinforcement_batch.get("human_accepted") is True
    )
    capacity_batch = (
        state.get("capacity_batch_status")
        if isinstance(state.get("capacity_batch_status"), dict)
        else {}
    )
    capacity_result = (
        state.get("capacity_check_result")
        if isinstance(state.get("capacity_check_result"), dict)
        else state.get("check_result")
        if isinstance(state.get("check_result"), dict)
        else {}
    )
    capacity_batch_complete = (
        capacity_batch.get("stage_complete") is not False
        and capacity_result.get("stage_complete") is not False
    )
    layout_available = (
        "layout_result" not in invalidated
        and bool(state.get("layout_result") or state.get("existing_layout_result"))
    )
    layout_revision_available = bool(
        "layout_result" not in invalidated
        and (state.get("layout_revision_result") or state.get("final_layout_result"))
    )
    collision_detected_available = bool(
        "layout_result" not in invalidated
        and (
            state.get("collision_metrics")
            or state.get("collision_metrics_json_path")
            # 兼容只恢复出报告/原始结果的入口：任一碰撞检测成果都能证明该阶段已执行过，
            # 缺少指标文件不应把结构设计返修整条链路堵死。
            or state.get("collision_report_json_path")
            or state.get("collision_result")
        )
    )
    reinforcement_available = (
        "reinforcement_design_result" not in invalidated
        and bool(state.get("reinforcement_design_result"))
        and reinforcement_batch_complete
    )
    capacity_available = (
        "check_result" not in invalidated
        and bool(capacity_result)
        and capacity_batch_complete
    )

    accepted_risks = list(state.get("accepted_risks") or [])
    modeling_risk_accepted = any(
        isinstance(risk, dict) and risk.get("scope") == "modeling_check"
        for risk in accepted_risks
    )
    # 结构返修轮次：优先取顶层图按阶段累计的返修重入次数（check_iteration_index
    # 只在建模验算阶段递增，结构阶段返修不会更新它，若只依赖该字段则轮次上限永不触发）。
    stage_rounds = (
        state.get("stage_revision_rounds")
        if isinstance(state.get("stage_revision_rounds"), dict)
        else {}
    )
    structural_revision_round = int(
        stage_rounds.get(StageName.STRUCTURAL_DESIGN.value)
        or state.get("check_iteration_index")
        or 0
    )
    slenderness_review_pending = False
    if not modeling_risk_accepted:
        try:
            from .joint_reinforcement import aggregate_axial_check_results  # 延迟导入避免顶层耦合

            axial = aggregate_axial_check_results(state.get("reinforcement_design_result") or {})
            slenderness_review_pending = bool(axial.get("has_slenderness_review"))
        except Exception:
            slenderness_review_pending = False

    return CoordinatorView(
        layout_result_available=layout_available,
        layout_revision_result_available=layout_revision_available,
        collision_detected_available=collision_detected_available,
        design_units_available="design_units" not in invalidated and bool(state.get("design_units")),
        dimension_design_available=(
            "dimension_design_result" not in invalidated and bool(state.get("dimension_design_result"))
        ),
        reinforcement_design_available=reinforcement_available,
        opensees_force_available=(
            "analysis_result" not in invalidated
            and "reinforcement_design_result" not in invalidated
            and bool(state.get("opensees_force_json_path"))
        ),
        capacity_check_available=capacity_available,
        check_passed=check_passed,
        revision_round=int(state.get("check_iteration_index") or state.get("iteration_index") or 0),
        max_revision_rounds=int(
            state.get("max_check_revision_rounds") or state.get("max_revision_rounds") or 3
        ),
        layout_revision_round=int(state.get("iteration_index") or 0),
        max_layout_revision_rounds=int(state.get("max_revision_rounds") or 3),
        structural_revision_round=structural_revision_round,
        max_structural_revision_rounds=int(state.get("max_check_revision_rounds") or 3),
        unresolved_manual_review=bool(state.get("unresolved_manual_review")),
        revision_target=revision_target,
        initial_design_completed=layout_available,
        layout_revision_completed=layout_revision_available,
        structural_design_completed=reinforcement_available,
        modeling_check_completed=capacity_available,
        modeling_risk_accepted=modeling_risk_accepted,
        modeling_slenderness_review_pending=slenderness_review_pending,
    )


def _parse_decision(raw: str) -> Optional[CoordinatorDecision]:
    """从 LLM 原始文本解析 CoordinatorDecision；失败返回 None。"""
    if not raw:
        return None
    text = raw.strip()
    # 容忍被 ```json ... ``` 包裹的情况
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return CoordinatorDecision(
        intent=str(obj.get("intent") or "unknown"),
        next_stage=obj.get("next_stage"),
        stage_objective=str(obj.get("stage_objective") or ""),
        reason=str(obj.get("reason") or ""),
        revision_target=obj.get("revision_target"),
        required_artifacts=list(obj.get("required_artifacts") or []),
        task_complete=bool(obj.get("task_complete")),
    )


class DesignCoordinatorAgent:
    """总体设计协调智能体：LLM 提出调度决策，确定性规则校验，非法重试后转人工复核。

    - 首次调用：解析用户需求、判断意图、确定流程入口、为首个专业阶段生成目标；
    - 后续调用：读取最新 Handoff 与成果状态，判断完成/返工/下一阶段/人工复核。
    """

    def __init__(
        self,
        llm_invoke: LLMInvoke,
        *,
        max_attempts: int = 2,
        skill_text: str = "",
    ) -> None:
        self.llm_invoke = llm_invoke
        self.max_attempts = max_attempts
        self.skill_text = skill_text

    def decide(
        self,
        state: Dict[str, Any],
        user_request: str,
        *,
        intent: Optional[str] = None,
    ) -> CoordinatorDecision:
        """返回一次经过确定性校验的调度决策。

        LLM 决策非法时携带校验错误重试；连续 max_attempts 次非法、
        LLM 不可用或输出无法解析时，转人工复核。
        """
        view = build_coordinator_view(state)
        last_errors: List[str] = []

        # 首次调度且无意图时，用规则兜底推断意图。
        resolved_intent = intent or state.get("user_intent")
        if not resolved_intent or resolved_intent == "unknown":
            resolved_intent = infer_intent(user_request)

        # 全部必需阶段已经完成时，优先采用确定性完成判定。
        # 这可避免 LLM 在终态继续生成回跳或超限返修决策，造成无意义的人工复核。
        completion_candidate = self._fallback_decide(
            state,
            user_request,
            intent=resolved_intent,
        )
        if completion_candidate.task_complete:
            completion_errors = validate_decision(completion_candidate, view)
            if not completion_errors:
                return completion_candidate

        for _ in range(self.max_attempts):
            decision = self._llm_decide(
                state,
                user_request,
                intent=resolved_intent,
                validation_errors=last_errors,
            )
            if decision is None:
                decision = self._fallback_decide(state, user_request, intent=resolved_intent)
            if not decision.intent or decision.intent == "unknown":
                decision.intent = resolved_intent
            errors = validate_decision(decision, view)
            if not errors:
                return decision
            last_errors = errors

        if last_errors and all("已完成阶段" in error for error in last_errors):
            fallback = self._fallback_decide(
                state,
                user_request,
                intent=resolved_intent,
            )
            fallback_errors = validate_decision(fallback, view)
            if not fallback_errors:
                return fallback
            last_errors.extend(
                error for error in fallback_errors if error not in last_errors
            )

        logger.warning(
            "协调器连续 %s 次非法调度，转人工复核：%s",
            self.max_attempts,
            "; ".join(last_errors),
        )
        return CoordinatorDecision(
            intent=resolved_intent,
            next_stage=StageName.MANUAL_REVIEW.value,
            stage_objective="人工复核",
            reason="调度决策连续非法：" + "; ".join(last_errors),
            task_complete=False,
        )

    # ------------------------------------------------------------------ #
    def _llm_decide(
        self,
        state: Dict[str, Any],
        user_request: str,
        *,
        intent: Optional[str],
        validation_errors: Optional[List[str]] = None,
    ) -> Optional[CoordinatorDecision]:
        payload = {
            "用户自然语言": user_request,
            "用户意图": intent or state.get("user_intent"),
            "允许的 next_stage": list(COORDINATOR_NEXT_STAGES),
            "字段约束": {
                "intent": "任务意图枚举；不得填入 next_stage",
                "next_stage": "只能从允许的 next_stage 选择；task_complete=true 时填 null",
            },
            "上次确定性校验错误": list(validation_errors or []),
            "当前成果状态摘要": {
                "layout_result_available": bool(state.get("layout_result") or state.get("existing_layout_result")),
                "layout_revision_result_available": bool(
                    state.get("layout_revision_result") or state.get("final_layout_result")
                ),
                "design_units_available": bool(state.get("design_units")),
                "dimension_design_available": bool(state.get("dimension_design_result")),
                "reinforcement_design_available": bool(state.get("reinforcement_design_result")),
                "opensees_force_available": bool(state.get("opensees_force_json_path")),
                "capacity_check_available": bool(state.get("capacity_check_result") or state.get("check_result")),
            },
            "最新交接": state.get("latest_handoff"),
            "输出格式": (
                "严格 JSON：{intent, next_stage, stage_objective, reason, "
                "revision_target, required_artifacts, task_complete}"
            ),
        }
        system_text = self.skill_text or "你是桥梁设计总体协调智能体。"
        user_text = json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            raw = self.llm_invoke(system_text, user_text)
        except Exception as e:  # pragma: no cover - 依赖外部 LLM
            logger.warning("协调器 LLM 调用失败：%s", e)
            return None
        return _parse_decision(raw)

    def _fallback_decide(
        self,
        state: Dict[str, Any],
        user_request: str,
        *,
        intent: Optional[str],
    ) -> CoordinatorDecision:
        """规则兜底：按意图阶段序列选第一个未完成阶段。"""
        resolved_intent = intent or state.get("user_intent") or "unknown"
        sequence = INTENT_TO_STAGE_SEQUENCE.get(resolved_intent, [])
        view = build_coordinator_view(state)

        if view.revision_target:
            return CoordinatorDecision(
                intent=resolved_intent,
                next_stage=view.revision_target,
                stage_objective=f"执行最小返工阶段 {view.revision_target}",
                reason="存在活动返修请求，规则兜底优先执行返修目标",
                revision_target=view.revision_target,
                required_artifacts=list(state.get("rework_required_artifacts") or []),
                task_complete=False,
            )

        # 判断每个阶段是否已完成（用 view 字段）。
        completed: Dict[str, bool] = {
            StageName.INITIAL_DESIGN.value: view.layout_result_available,
            StageName.LAYOUT_REVISION.value: view.layout_revision_result_available,
            StageName.STRUCTURAL_DESIGN.value: view.reinforcement_design_available,
            StageName.MODELING_CHECK.value: view.capacity_check_available,
        }
        for stage in sequence:
            if not completed.get(stage, False):
                return CoordinatorDecision(
                    intent=resolved_intent,
                    next_stage=stage,
                    stage_objective=f"执行阶段 {stage}（规则兜底）",
                    reason="LLM 决策不可用，使用规则兜底调度",
                    task_complete=False,
                )
        return CoordinatorDecision(
            intent=resolved_intent,
            next_stage=None,
            stage_objective="全部阶段已完成",
            reason="规则兜底：无剩余未完成阶段",
            task_complete=True,
        )
