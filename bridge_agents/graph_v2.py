from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .contracts import StageName, StageStatus, VALID_STAGE_STATUSES
from .dependencies import build_artifact_refresh_update, build_revision_state_update
from .design_coordinator import DesignCoordinatorAgent, LLMInvoke
from .handoff import build_handoff
from .human_review_store import persist_human_review_decision
from .state import AgentState


# 顶层图单次 invoke 的超级步上限（最后一次保险）。
# 正常全流程（含若干轮返修与人工复核）远低于该值；此处只用于把调度死循环
# 显式暴露成 GraphRecursionError，而不是无限空转消耗 token 与 checkpoint 空间。
DEFAULT_RECURSION_LIMIT = 80


# 同一阶段连续失败多少次后转人工复核。协调器不会因"阶段未完成"而拒绝重新调度，
# 所以纯失败没有轮次上限；该上限保证确定性失败最终会停下来等人处理。
DEFAULT_MAX_STAGE_FAILURES = 3


# 顶层图节点名 -> 阶段 Agent 名。
STAGE_AGENT_NAME: Dict[str, str] = {
    StageName.INITIAL_DESIGN.value: "InitialDesignAgent",
    StageName.LAYOUT_REVISION.value: "LayoutRevisionAgent",
    StageName.STRUCTURAL_DESIGN.value: "StructuralDesignAgent",
    StageName.MODELING_CHECK.value: "ModelingCheckAgent",
}

# 顶层图中显式注册的全部节点。
TOP_LEVEL_NODES = (
    "design_coordinator",
    "render_catalog_views",
    StageName.INITIAL_DESIGN.value,
    StageName.LAYOUT_REVISION.value,
    StageName.STRUCTURAL_DESIGN.value,
    StageName.MODELING_CHECK.value,
    StageName.MANUAL_REVIEW.value,
    StageName.FINAL_OUTPUT.value,
    StageName.ERROR.value,
)


def _default_stage_runners() -> Dict[str, Callable[[AgentState], Dict[str, Any]]]:
    """从 AGENT_REGISTRY 构建默认阶段执行器（惰性导入避免 mmseg 链）。"""
    from .stage_agents import AGENT_REGISTRY  # 惰性导入

    runners: Dict[str, Callable[[AgentState], Dict[str, Any]]] = {}
    for stage_name, agent_name in STAGE_AGENT_NAME.items():
        agent = AGENT_REGISTRY.get(agent_name)
        if agent is not None:
            runners[stage_name] = agent.run
    return runners


def _default_llm_invoke() -> LLMInvoke:
    from .utils import get_controller_llm  # 惰性导入

    def invoke(system: str, user: str) -> str:
        llm = get_controller_llm("config/settings.yaml")
        return llm.invoke([("system", system), ("user", user)]).content

    return invoke


def make_stage_node(stage_name: str, runner: Callable[[AgentState], Dict[str, Any]]):
    """将阶段 Agent 的 run() 包装为顶层图节点，并产出 StageHandoff。"""

    def node(state: AgentState) -> Dict[str, Any]:
        update = runner(state)
        native_handoff = update.get("latest_handoff")
        if (
            isinstance(native_handoff, dict)
            and native_handoff.get("stage") == stage_name
            and native_handoff.get("status") in VALID_STAGE_STATUSES
        ):
            handoff_data = dict(native_handoff)
        else:
            handoff_data = build_handoff(stage_name, update).to_dict()

        if handoff_data.get("status") == "revision_required":
            dependency_update = build_revision_state_update(
                handoff_data,
                current_invalidated=state.get("invalidated_artifacts") or [],
            )
        else:
            dependency_update = build_artifact_refresh_update(
                stage_name,
                update,
                current_invalidated=state.get("invalidated_artifacts") or [],
                revision_target=state.get("revision_target"),
            )
        if "invalidated_artifacts" in dependency_update:
            handoff_data["invalidated_artifacts"] = dependency_update["invalidated_artifacts"]

        result: Dict[str, Any] = {
            **update,
            **dependency_update,
            "latest_handoff": handoff_data,
            "active_agent": STAGE_AGENT_NAME.get(stage_name, stage_name),
        }

        was_revision_target = state.get("revision_target") == stage_name
        if was_revision_target:
            # 按阶段累计返修重入次数，供协调器的轮次上限校验使用。
            stage_rounds = dict(state.get("stage_revision_rounds") or {})
            stage_rounds[stage_name] = int(stage_rounds.get(stage_name) or 0) + 1
            result["stage_revision_rounds"] = stage_rounds

        # 返修阶段再次失败：清掉返修调度标记并直接转人工复核。
        # 否则 revision_target 会一直等于本阶段，协调器每轮都能把它当成"合法返修"重新调度，
        # 形成不产出任何成果、也没有任何用户可见反馈的死循环。
        rework_failed = bool(was_revision_target and handoff_data.get("status") == "failed")
        result["rework_failed"] = rework_failed
        if rework_failed:
            result.update(
                {
                    "revision_target": None,
                    "rework_required_artifacts": [],
                    "unresolved_manual_review": True,
                    "manual_review_reason": (
                        f"{handoff_data.get('message') or '返修阶段执行失败。'}"
                        "请人工决定是否继续返修、接受部分成果或终止任务。"
                    ),
                }
            )

        # 非返修的纯阶段失败没有轮次上限：stage_revision_rounds 只在返修重入时递增，
        # 协调器每轮只看到"阶段未完成"，于是反复重新调度同一阶段。若失败原因是确定性的
        # （如设计单元语义校验恒不通过），重试永远不会成功，形成无成果、无反馈的空转。
        # 这里按阶段累计连续失败次数，达到上限即转人工复核，让失败可被人看到。
        failed = handoff_data.get("status") == "failed"
        failure_counts = dict(state.get("stage_failure_counts") or {})
        if failed:
            failure_counts[stage_name] = int(failure_counts.get(stage_name) or 0) + 1
        elif handoff_data.get("status") == StageStatus.COMPLETED.value:
            failure_counts.pop(stage_name, None)
        result["stage_failure_counts"] = failure_counts

        failure_limit = int(state.get("max_stage_failures") or DEFAULT_MAX_STAGE_FAILURES)
        stage_failure_limit_reached = bool(
            failed and failure_counts.get(stage_name, 0) >= failure_limit
        )
        result["stage_failure_limit_reached"] = stage_failure_limit_reached
        if stage_failure_limit_reached and not rework_failed:
            result.update(
                {
                    "unresolved_manual_review": True,
                    "manual_review_reason": (
                        f"{handoff_data.get('message') or '阶段执行失败。'}"
                        f"阶段 {stage_name!r} 已连续失败 "
                        f"{failure_counts.get(stage_name, 0)} 次仍无进展，"
                        "请人工排查失败原因，再决定是否继续、接受部分成果或终止任务。"
                    ),
                }
            )
        return result

    return node


def route_after_stage(state: AgentState) -> str:
    """阶段节点出口路由：返修失败或连续失败超限直接转人工复核，其余回协调器重新调度。"""
    if state.get("rework_failed") or state.get("stage_failure_limit_reached"):
        return StageName.MANUAL_REVIEW.value
    return "design_coordinator"


def make_coordinator_node(llm_invoke: LLMInvoke):
    """协调器节点：读状态 -> 决策 -> 写 coordinator_decision。"""
    coordinator = DesignCoordinatorAgent(llm_invoke)

    def node(state: AgentState) -> Dict[str, Any]:
        if state.get("task_status") in {"completed", "completed_with_accepted_risks"}:
            return {
                "coordinator_decision": {
                    "intent": str(state.get("user_intent") or "unknown"),
                    "next_stage": None,
                    "stage_objective": "恢复已完成任务",
                    "reason": "工程侧人工复核记录表明该任务已经完成。",
                    "revision_target": None,
                    "required_artifacts": [],
                    "task_complete": True,
                },
                "active_agent": "DesignCoordinatorAgent",
            }
        if state.get("task_status") in {"manual_review", "manual_review_required"}:
            return {
                "coordinator_decision": {
                    "intent": str(
                        state.get("user_intent")
                        or (state.get("coordinator_decision") or {}).get("intent")
                        or "unknown"
                    ),
                    "next_stage": StageName.MANUAL_REVIEW.value,
                    "stage_objective": "恢复待处理的人工复核",
                    "reason": str(state.get("message") or "存在待处理的人工复核决策。"),
                    "revision_target": None,
                    "required_artifacts": [],
                    "task_complete": False,
                },
                "active_agent": "DesignCoordinatorAgent",
            }
        decision = coordinator.decide(dict(state), state.get("user_request") or "")
        return {
            "coordinator_decision": decision.to_dict(),
            "active_agent": "DesignCoordinatorAgent",
        }

    return node


def route_after_coordinator(state: AgentState) -> str:
    decision = state.get("coordinator_decision") or {}
    if decision.get("task_complete"):
        return StageName.FINAL_OUTPUT.value
    next_stage = decision.get("next_stage")
    if next_stage in STAGE_AGENT_NAME:
        return next_stage
    if next_stage == StageName.MANUAL_REVIEW.value:
        return StageName.MANUAL_REVIEW.value
    return StageName.ERROR.value


def _should_render_catalog_views(output_dir: str) -> bool:
    """仅当存在比 catalog_views 更新的布跨结果（或对应 PNG 缺失）时才渲染。"""
    try:
        from .initial_visualizer import _locate_artifacts

        located = _locate_artifacts(output_dir)
        for variant, key in (
            ("initial", "initial_layout"),
            ("final", "final_layout"),
        ):
            layout = located.get(key)
            if not layout:
                continue
            profile_png = os.path.join(output_dir, "catalog_views", variant, "profile.png")
            if not os.path.exists(profile_png) or os.path.getmtime(
                layout
            ) > os.path.getmtime(profile_png):
                return True
    except Exception:
        return False
    return False


def render_catalog_views_node(state: AgentState) -> Dict[str, Any]:
    """初设/最终布跨结果落地后自动渲染目录视图（平面叠加 + 纵断示意）。

    幂等：无新结果时不重画；渲染失败只跳过，不阻断主流程。
    """
    output_dir = str(state.get("output_dir") or "").strip()
    if output_dir and _should_render_catalog_views(output_dir):
        try:
            from .initial_visualizer import build_catalog_views

            build_catalog_views(output_dir)
        except Exception:
            pass
    return {}


def _collision_threshold_summary(metrics: Dict[str, Any]) -> str:
    """把碰撞阈值各项与限值对比，供人工复核请求明确未达标项。"""
    checks = [
        ("总侵入深度(限≤10)", "total_intrusion_depth_columns", 10.0),
        ("平均侵入深度(限≤1.2)", "avg_intrusion_depth_columns", 1.2),
        ("平均重叠率(限≤0.5)", "avg_overlap_ratio_columns", 0.5),
        ("冲突柱率(限≤5%)", "conflict_column_rate", 0.05),
    ]
    parts = []
    for label, key, limit in checks:
        raw = metrics.get(key) if isinstance(metrics, dict) else None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        flag = "通过" if value <= limit else "未达标"
        parts.append(f"{label}={value:.4f}({flag})")
    return ("：" + "、".join(parts)) if parts else ""


def _summarize_collision_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {"value": str(item)}
    keys = (
        "bridge_id",
        "桥位编号",
        "pier_id",
        "墩号",
        "station",
        "桩号",
        "intrusion_depth",
        "侵入深度",
        "overlap_ratio",
        "重叠比例",
    )
    return {key: item[key] for key in keys if key in item}


# 可返修的阶段及其面向用户的名称。
REWORK_TARGET_LABELS: Dict[str, str] = {
    StageName.STRUCTURAL_DESIGN.value: "结构设计阶段（重做尺寸/配筋）",
    StageName.MODELING_CHECK.value: "建模验算阶段",
    StageName.LAYOUT_REVISION.value: "布跨修正阶段",
}


def _handoff_revision_target(handoff: Any) -> str:
    if not isinstance(handoff, dict):
        return ""
    revision_request = handoff.get("revision_request")
    if isinstance(revision_request, dict):
        return str(revision_request.get("target_stage") or "").strip()
    return ""


def resolve_rework_target(state: Dict[str, Any]) -> str:
    """返回当前待执行的返修目标阶段；没有待处理返修时返回空串。

    目标可能只存在于最新 Handoff 的返修请求里（例如建模验算阶段刚生成返修请求、
    revision_target 尚未回写），因此两处都要看。人工复核动作必须遵循该目标，
    否则会出现"验算失败却返回验算阶段"的空转。
    """
    target = str(state.get("revision_target") or "").strip()
    if not target:
        target = _handoff_revision_target(state.get("latest_handoff"))
    return target if target in REWORK_TARGET_LABELS else ""


def _rework_target_label(target: str) -> str:
    return REWORK_TARGET_LABELS.get(target, target or "未指定")


def _rework_step_from_artifacts(required_artifacts: Any) -> str:
    """由返修成果反推结构设计阶段的重做步骤。"""
    artifacts = {str(item) for item in (required_artifacts or [])}
    if "dimension_design_result" in artifacts:
        return "dimension_design"
    return "reinforcement_design"


def _resolve_capacity_evidence(
    state: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """定位"已完成承载力验算"的证据，返回 (结果字典或 None, 证据路径或 None)。

    返修会使 check_result / capacity_check_result 失效并被清空，因此除内存字段外，
    还要看 Handoff 记录的成果路径与 output_dir 下的标准批次汇总文件。
    """
    capacity_result = state.get("capacity_check_result")
    if isinstance(capacity_result, dict) and capacity_result:
        return capacity_result, None
    check_result = state.get("check_result")
    if isinstance(check_result, dict) and check_result:
        return check_result, None

    candidates: list = []
    handoff = state.get("latest_handoff")
    if isinstance(handoff, dict):
        for artifact in handoff.get("produced_artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            if str(artifact.get("artifact_type") or "") != "check_result":
                continue
            if artifact.get("path"):
                candidates.append(str(artifact["path"]))
    if state.get("capacity_check_summary_path"):
        candidates.append(str(state["capacity_check_summary_path"]))
    output_dir = str(state.get("output_dir") or "").strip()
    if output_dir:
        candidates.append(os.path.join(output_dir, "capacity_check", "capacity_check_batch_summary.json"))
        candidates.append(os.path.join(output_dir, "capacity_check", "capacity_check_summary.json"))

    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload:
            return payload, path
    return None, None


def _batch_status_from_result(payload: Any, kind: str) -> Dict[str, Any]:
    """从已恢复的阶段成果里读取批次状态（阶段状态缺失时的回退）。

    断点恢复/成果恢复路径可能没有把批次状态写进图状态，若只读图状态，
    人工复核请求里就会出现"0 个失败任务"这类失真信息。
    """
    if not isinstance(payload, dict):
        return {}
    root = payload.get("任务3_下部结构配筋设计结果") if kind == "reinforcement" else payload.get(
        "任务2_下部结构尺寸设计结果"
    )
    if not isinstance(root, dict):
        root = payload
    if kind == "reinforcement":
        failed_ids = [
            str(item.get("task_id"))
            for item in (root.get("失败分组") or [])
            if isinstance(item, dict) and item.get("task_id")
        ]
        if not failed_ids:
            failed_ids = [str(item) for item in (root.get("failed_task_ids") or [])]
        expected = root.get("expected_task_count")
        completed = root.get("completed_task_count")
    else:
        failed_ids = [
            str(item.get("单元编号"))
            for item in (root.get("失败单元") or [])
            if isinstance(item, dict) and item.get("单元编号")
        ]
        if not failed_ids:
            failed_ids = [str(item) for item in (root.get("failed_unit_ids") or [])]
        expected = root.get("expected_unit_count")
        completed = root.get("completed_unit_count")
    if expected is None and completed is None and not failed_ids:
        return {}
    status = {
        "expected_task_count" if kind == "reinforcement" else "expected_unit_count": expected,
        "completed_task_count" if kind == "reinforcement" else "completed_unit_count": completed,
        "failed_task_ids" if kind == "reinforcement" else "failed_unit_ids": failed_ids,
        "stage_complete": bool(root.get("stage_complete")) if root.get("stage_complete") is not None else None,
    }
    return {key: value for key, value in status.items() if value is not None or key.endswith("_ids")}


def build_human_review_request(state: Dict[str, Any]) -> Dict[str, Any]:
    """构造可由 CLI、Web UI 或 API 直接呈现的审核请求。"""
    latest_handoff = state.get("latest_handoff") if isinstance(state.get("latest_handoff"), dict) else {}
    metrics = state.get("collision_metrics") if isinstance(state.get("collision_metrics"), dict) else {}
    collision_items = list(state.get("collision_items") or [])
    handoff_stage = latest_handoff.get("stage")
    reinforcement_batch = (
        state.get("reinforcement_batch_status")
        if isinstance(state.get("reinforcement_batch_status"), dict)
        else {}
    )
    if not reinforcement_batch:
        reinforcement_batch = _batch_status_from_result(
            state.get("reinforcement_design_result"), "reinforcement"
        )
    dimension_batch = (
        state.get("dimension_batch_status")
        if isinstance(state.get("dimension_batch_status"), dict)
        else {}
    )
    if not dimension_batch:
        dimension_batch = _batch_status_from_result(
            state.get("dimension_design_result"), "dimension"
        )
    # 人工已接受的结构设计/尺寸缺口不再算作"需要人工决策的结构问题"：
    # 否则它会一直压住承载力侧的复核请求——验算已经通过、缺口也已接受，
    # 界面却只给结构类动作，人工无法收尾（2026-09-16 示例项目K29 复验后即为此现象）。
    reinforcement_pending = (
        reinforcement_batch.get("stage_complete") is False
        and reinforcement_batch.get("human_accepted") is not True
    )
    dimension_pending = (
        dimension_batch.get("stage_complete") is False
        and dimension_batch.get("human_accepted") is not True
    )
    is_structural_review = bool(
        handoff_stage == StageName.STRUCTURAL_DESIGN.value
        or reinforcement_pending
        or dimension_pending
    )
    is_modeling_review = handoff_stage == StageName.MODELING_CHECK.value

    # 成果恢复/断点场景下可能没有 latest_handoff；若承载力已齐但墩柱长细比复核
    # 未决且未被人工接受，仍应提供“接受风险并完成”，而不是只给 retry/abort。
    accepted_risks = list(state.get("accepted_risks") or [])
    modeling_risk_accepted = any(
        isinstance(risk, dict) and risk.get("scope") == "modeling_check"
        for risk in accepted_risks
    )
    axial_pending = False
    if not is_modeling_review and not modeling_risk_accepted:
        try:
            from .joint_reinforcement import aggregate_axial_check_results  # noqa: PLC0415

            axial = aggregate_axial_check_results(state.get("reinforcement_design_result") or {})
            axial_pending = bool(axial.get("has_slenderness_review"))
        except Exception:
            axial_pending = False

    check_result = state.get("check_result") if isinstance(state.get("check_result"), dict) else {}
    pending_rework_target = resolve_rework_target(state)

    # 尺寸设计与配筋设计使用两套不同的 ID 命名空间：
    # - 尺寸设计：单元编号（如 2-2），失败清单在 dimension_batch_status.failed_unit_ids；
    # - 配筋设计：设计组任务号（如 2-2-1-G1），失败清单在 reinforcement_batch_status.failed_task_ids。
    # 两者必须分开下发，否则会把配筋任务号当尺寸单元号传给尺寸设计工具而直接报错。
    reinforcement_failed_task_ids = [
        str(item) for item in (reinforcement_batch.get("failed_task_ids") or []) if str(item).strip()
    ]
    dimension_failed_unit_ids = [
        str(item) for item in (dimension_batch.get("failed_unit_ids") or []) if str(item).strip()
    ]
    if not reinforcement_failed_task_ids and not dimension_failed_unit_ids:
        # 兼容历史状态：只有一个 failed_task_ids 字段时按配筋任务号解释
        # （历史实现也是优先取配筋批次失败清单）。
        reinforcement_failed_task_ids = [
            str(item) for item in (state.get("failed_task_ids") or []) if str(item).strip()
        ]
    failed_task_ids = reinforcement_failed_task_ids

    if is_structural_review:
        review_type = "structural_batch_review"
        available_actions = ["retry_failed_tasks", "accept_partial_and_continue", "abort"]
        pending_parts = []
        if dimension_failed_unit_ids:
            pending_parts.append(f"尺寸设计 {len(dimension_failed_unit_ids)} 个单元")
        if reinforcement_failed_task_ids:
            pending_parts.append(f"配筋设计 {len(reinforcement_failed_task_ids)} 个任务")
        if pending_parts:
            message = (
                "结构设计批次仍有" + "、".join(pending_parts) + "未完成，"
                "请选择重新执行结构设计（先补尺寸、再补配筋）、带风险接受部分成果或终止任务。"
            )
        else:
            message = (
                "结构设计批次未完成，请选择重新执行结构设计、"
                "带风险接受部分成果或终止任务。"
            )
        if str(latest_handoff.get("status") or "") == "failed":
            failure_note = str(state.get("manual_review_reason") or "").strip() or str(
                latest_handoff.get("message") or "结构设计返修失败。"
            )
            message = f"{failure_note}\n{message}"
        current_round = int(state.get("check_iteration_index") or 0)
        max_rounds = int(state.get("max_check_revision_rounds") or 3)
    elif is_modeling_review:
        review_type = "modeling_check_review"
        available_actions = ["continue_modeling_revision", "accept_check_and_finish", "abort"]
        message = str(latest_handoff.get("message") or "建模验算自动返修达到上限，请人工决定后续操作。")
        if pending_rework_target:
            message = (
                f"{message}\n当前待处理返修目标：{_rework_target_label(pending_rework_target)}"
                "（选择“继续验算返修”将返回该阶段执行返修）。"
            )
        current_round = int(state.get("check_iteration_index") or 0)
        max_rounds = int(state.get("max_check_revision_rounds") or 3)
    elif axial_pending:
        # 承载力通过但墩柱长细比复核未决（超出表5.3.1适用范围）。
        review_type = "modeling_check_review"
        available_actions = ["continue_modeling_revision", "accept_check_and_finish", "abort"]
        message = "墩柱长细比复核未决（超出表5.3.1适用范围）：请选择接受风险并完成，或返回继续修订。"
        current_round = int(state.get("check_iteration_index") or 0)
        max_rounds = int(state.get("max_check_revision_rounds") or 3)
    elif metrics or handoff_stage == StageName.LAYOUT_REVISION.value:
        review_type = "layout_collision_review"
        available_actions = ["continue_revision", "accept_and_continue", "abort"]
        message = (
            f"第 {int(state.get('iteration_index') or 0)} 轮布跨修正后仍未满足碰撞阈值"
            f"{_collision_threshold_summary(metrics)}，"
            "请选择继续修正、人工接受当前方案或终止任务。"
        )
        current_round = int(state.get("iteration_index") or 0)
        max_rounds = int(state.get("max_revision_rounds") or 3)
    else:
        review_type = "coordinator_review"
        available_actions = ["retry_coordinator", "abort"]
        message = str(latest_handoff.get("message") or "协调器需要人工确认后续操作。")
        current_round = int(state.get("iteration_index") or 0)
        max_rounds = int(state.get("max_revision_rounds") or 3)

    return {
        "review_type": review_type,
        "message": message,
        "current_round": current_round,
        "max_revision_rounds": max_rounds,
        "metrics": metrics,
        "remaining_conflict_count": len(collision_items),
        "remaining_conflicts": [_summarize_collision_item(item) for item in collision_items[:20]],
        "failed_task_ids": failed_task_ids,
        "failed_dimension_unit_ids": dimension_failed_unit_ids,
        "last_error": state.get("error"),
        # 当前待处理的返修目标：GUI/CLI 需要据此说明"继续返修"会回到哪个阶段。
        "rework_target": pending_rework_target or None,
        "rework_target_label": _rework_target_label(pending_rework_target) if pending_rework_target else None,
        "rework_required_artifacts": list(state.get("rework_required_artifacts") or []),
        "dimension_batch_status": dimension_batch,
        "reinforcement_batch_status": reinforcement_batch,
        "check_summary": {
            "overall_check": check_result.get("overall_check"),
            "utilization_summary": check_result.get("utilization_summary"),
        },
        "available_actions": available_actions,
        "latest_handoff": latest_handoff,
    }


def _persist_final_layout_result(state: Dict[str, Any], final_layout: Any) -> Optional[str]:
    """把人工接受的布跨方案落盘为布局修正阶段的标准成果文件。

    背景：自动收敛时由 finish_revision 动作写 layout_revision/final_layout_result.json；
    而"人工接受剩余碰撞风险并继续"这条路径只把结果写入 revision_results/ 与图状态，
    导致按目录文件推断阶段的模块（GUI 运行轨道、成果中心）误判布局修正未完成。
    这里补写标准文件，使文件视图与图状态一致；写失败不阻断决策，返回 None。
    """
    if not isinstance(final_layout, dict) or not final_layout:
        return None
    output_dir = state.get("output_dir")
    if not output_dir:
        return None
    try:
        target_dir = os.path.join(str(output_dir), "layout_revision")
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, "final_layout_result.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(final_layout, handle, ensure_ascii=False, indent=2)
        return path
    except OSError:
        return None


def apply_human_review_decision(
    state: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    """校验并应用人工决策；工程校核指标始终保持原值。"""
    if not isinstance(decision, dict):
        raise ValueError("人工复核决策必须是 JSON 对象。")
    action = str(decision.get("action") or "").strip()
    allowed_actions = {
        "continue_revision",
        "accept_and_continue",
        "abort",
        "retry_coordinator",
        "retry_failed_tasks",
        "accept_partial_and_continue",
        "continue_modeling_revision",
        "accept_check_and_finish",
    }
    if action not in allowed_actions:
        raise ValueError(f"非法人工复核决策: {action or '<empty>'}")

    request = build_human_review_request(state)
    if action not in set(request["available_actions"]):
        raise ValueError(
            f"人工复核类型 {request['review_type']} 不允许决策 {action}"
        )

    normalized_decision = {
        "decision_id": str(decision.get("decision_id") or uuid4()),
        "action": action,
        "feedback": str(decision.get("feedback") or "").strip(),
        "extra_rounds": max(1, int(decision.get("extra_rounds") or 1)),
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    common: Dict[str, Any] = {
        "human_review_request": request,
        "human_review_decision": normalized_decision,
        "human_review_feedback": normalized_decision["feedback"],
        "human_review_scope": request["review_type"],
        "human_review_history": [{
            "review_type": request["review_type"],
            "decision": normalized_decision,
            "metrics": request["metrics"],
            "remaining_conflict_count": request["remaining_conflict_count"],
        }],
        "collision_metrics": state.get("collision_metrics"),
        "collision_items": state.get("collision_items"),
        # 人工决策即视为重新购买失败预算：否则上一轮触顶后，人工选择重试会立刻再次
        # 触顶并弹回人工复核，表现为"点了没反应"（与 max_check_revision_rounds 同理）。
        "stage_failure_counts": {},
        "stage_failure_limit_reached": False,
        "error": None,
    }

    if action == "continue_revision":
        current_round = int(state.get("iteration_index") or 0)
        current_limit = int(state.get("max_revision_rounds") or 3)
        extra_rounds = normalized_decision["extra_rounds"]
        return {
            **common,
            "max_revision_rounds": max(current_limit, current_round + extra_rounds),
            "human_review_route": StageName.LAYOUT_REVISION.value,
            "human_override": False,
            "unresolved_manual_review": False,
            "task_status": "layout_revision_required",
            "message": f"人工授权继续修正，增加 {extra_rounds} 轮。",
            "latest_handoff": {
                "stage": StageName.LAYOUT_REVISION.value,
                "status": "revision_required",
                "produced_artifacts": [],
                "revision_request": {
                    "target_stage": StageName.LAYOUT_REVISION.value,
                    "reason": normalized_decision["feedback"] or "人工授权继续布跨修正",
                    "required_artifacts": ["layout_result"],
                },
                "invalidated_artifacts": [],
                "recommended_next_stage": StageName.LAYOUT_REVISION.value,
                "message": "人工授权继续布跨修正。",
            },
        }

    if action == "retry_failed_tasks":
        dimension_unit_ids = [
            str(item) for item in (request.get("failed_dimension_unit_ids") or []) if str(item).strip()
        ]
        reinforcement_task_ids = [
            str(item) for item in (request.get("failed_task_ids") or []) if str(item).strip()
        ]
        dimension_batch = state.get("dimension_batch_status")
        reinforcement_batch = state.get("reinforcement_batch_status")
        dimension_incomplete = bool(dimension_unit_ids) or (
            isinstance(dimension_batch, dict) and dimension_batch.get("stage_complete") is False
        )
        reinforcement_incomplete = bool(reinforcement_task_ids) or (
            isinstance(reinforcement_batch, dict) and reinforcement_batch.get("stage_complete") is False
        )
        # 顺序返修：先只重跑尺寸设计的失败单元，尺寸恢复后再重跑配筋的失败任务。
        target_step = "dimension_design" if dimension_incomplete else "reinforcement_design"
        # 尺寸结果变更会使配筋失效，因此尺寸返修也必须把配筋列为重做成果。
        rework_artifacts: list = []
        if dimension_incomplete:
            rework_artifacts.append("dimension_design_result")
        if dimension_incomplete or reinforcement_incomplete:
            rework_artifacts.append("reinforcement_design_result")
        revision_context = {
            "target_step": target_step,
            # 兼容旧字段：结构返修期间 state.failed_task_ids 语义是配筋任务号
            "failed_task_ids": reinforcement_task_ids,
            # 分步骤各自的重试范围，避免两个 ID 命名空间互相污染
            "dimension_retry_unit_ids": dimension_unit_ids,
            "reinforcement_retry_task_ids": reinforcement_task_ids,
            "reason": normalized_decision["feedback"] or "人工要求重新执行失败的结构设计任务",
        }
        # 人工授权即视为购买一轮结构返修预算：否则上一次轮次超限后，
        # 协调器会在重入结构设计前就再次判超限并立刻转回人工复核，形成"点了没反应"。
        stage_rounds = (
            state.get("stage_revision_rounds")
            if isinstance(state.get("stage_revision_rounds"), dict)
            else {}
        )
        current_rounds = int(stage_rounds.get(StageName.STRUCTURAL_DESIGN.value) or 0)
        current_limit = int(state.get("max_check_revision_rounds") or 3)
        return {
            **common,
            "human_review_route": StageName.STRUCTURAL_DESIGN.value,
            "human_override": False,
            "unresolved_manual_review": False,
            "task_status": "structural_revision_required",
            "revision_target": StageName.STRUCTURAL_DESIGN.value,
            "rework_required_artifacts": rework_artifacts,
            "failed_task_ids": reinforcement_task_ids,
            "failed_dimension_unit_ids": dimension_unit_ids,
            "max_check_revision_rounds": max(current_limit, current_rounds + 1),
            "revision_context": revision_context,
            "message": (
                f"人工要求先重跑尺寸设计失败单元（{len(dimension_unit_ids)} 个），"
                f"再重跑配筋设计失败任务（{len(reinforcement_task_ids)} 个）。"
                if dimension_incomplete
                else f"人工要求重新执行配筋设计失败任务（{len(reinforcement_task_ids)} 个）。"
            ),
            "latest_handoff": {
                "stage": StageName.STRUCTURAL_DESIGN.value,
                "status": "revision_required",
                "produced_artifacts": [],
                "revision_request": {
                    "target_stage": StageName.STRUCTURAL_DESIGN.value,
                    "reason": revision_context["reason"],
                    "required_artifacts": rework_artifacts,
                },
                "invalidated_artifacts": rework_artifacts,
                "recommended_next_stage": StageName.STRUCTURAL_DESIGN.value,
                "message": "人工要求重试结构设计失败任务。",
            },
        }

    if action == "accept_partial_and_continue":
        batch_status = dict(state.get("reinforcement_batch_status") or {})
        batch_status.update({"human_accepted": True, "accepted_for_workflow": True})
        dimension_status = dict(state.get("dimension_batch_status") or {})
        if dimension_status and dimension_status.get("stage_complete") is False:
            dimension_status.update({"human_accepted": True, "accepted_for_workflow": True})
        risk = {
            "scope": "structural_design",
            "failed_task_ids": list(request.get("failed_task_ids") or []),
            "failed_dimension_unit_ids": list(request.get("failed_dimension_unit_ids") or []),
            "reason": normalized_decision["feedback"] or "人工接受结构设计部分成果",
            "accepted_at": normalized_decision["decided_at"],
        }
        return {
            **common,
            "human_review_route": "design_coordinator",
            "human_override": True,
            "unresolved_manual_review": False,
            "reinforcement_batch_status": batch_status,
            "dimension_batch_status": dimension_status or state.get("dimension_batch_status"),
            "accepted_risks": [*(state.get("accepted_risks") or []), risk],
            "task_status": "structural_design_completed_with_risk",
            # 人工接受即终止本轮结构返修：不清掉返修目标会让协调器继续把流程拉回结构设计。
            "revision_target": None,
            "rework_required_artifacts": [],
            "invalidated_artifacts": [],
            "minimal_rework_path": [],
            "message": "人工接受结构设计部分成果并保留失败任务风险。",
            "latest_handoff": {
                "stage": StageName.STRUCTURAL_DESIGN.value,
                "status": "completed",
                "produced_artifacts": [],
                "revision_request": None,
                "invalidated_artifacts": [],
                "recommended_next_stage": StageName.MODELING_CHECK.value,
                "message": "人工带风险接受结构设计部分成果。",
            },
        }

    if action == "continue_modeling_revision":
        current_round = int(state.get("check_iteration_index") or 0)
        current_limit = int(state.get("max_check_revision_rounds") or 3)
        current_steps = int(state.get("max_modeling_check_steps") or 6)
        extra_rounds = normalized_decision["extra_rounds"]
        # 返修目标以最新 Handoff / state.revision_target 为准：验算失败后自动生成的返修
        # 目标通常是 structural_design（重做配筋或尺寸），此时把流程送回验算阶段毫无意义，
        # 会立刻再次判失败并转回人工复核（2026-09-15 运行即为此现象）。
        rework_target = resolve_rework_target(state)
        route = rework_target or StageName.MODELING_CHECK.value
        if route == StageName.MODELING_CHECK.value:
            revision_context = {
                "target_step": "modeling_check",
                "reason": normalized_decision["feedback"] or "人工授权继续验算返修",
            }
            feedback_decision = None
            task_status = "modeling_check_revision_required"
            new_limit = max(current_limit, current_round + max(1, extra_rounds))
        else:
            # 保留建模验算阶段生成的返修上下文（含分步骤重试范围），只补人工补充要求；
            # 否则结构返修会丢掉 revision_context，重新规划时无法限定重做范围。
            revision_context = dict(state.get("revision_context") or {})
            revision_context.setdefault(
                "target_step", _rework_step_from_artifacts(state.get("rework_required_artifacts"))
            )
            if normalized_decision["feedback"]:
                revision_context["reason"] = normalized_decision["feedback"]
            revision_context.setdefault("reason", "人工授权继续返修")
            feedback_decision = state.get("feedback_decision")
            task_status = "structural_revision_required"
            # 结构返修路由：按"已使用的结构返修次数"购买一轮预算，
            # 不能用 check_iteration_index 去抬（否则一次点击会把上限从 2 直接抬到 4，
            # 导致结构↔验算循环多跑两轮仍不收手）。
            stage_rounds = (
                state.get("stage_revision_rounds")
                if isinstance(state.get("stage_revision_rounds"), dict)
                else {}
            )
            structural_rounds = int(stage_rounds.get(StageName.STRUCTURAL_DESIGN.value) or 0)
            new_limit = max(current_limit, structural_rounds + max(1, extra_rounds))
        return {
            **common,
            "human_review_route": route,
            "human_override": False,
            "unresolved_manual_review": False,
            "max_check_revision_rounds": new_limit,
            "max_modeling_check_steps": current_steps + extra_rounds,
            "revision_context": revision_context,
            # 显式回写返修目标：顶层图据此按"返修"计数并在再次失败时转人工复核，
            # 不能只依赖 Handoff 里的返修请求（断点恢复路径可能没有 state.revision_target）。
            "revision_target": route if route != StageName.MODELING_CHECK.value else state.get("revision_target"),
            "rework_required_artifacts": list(state.get("rework_required_artifacts") or []),
            "feedback_decision": feedback_decision,
            "task_status": task_status,
            "message": (
                f"人工授权继续返修，增加 {extra_rounds} 轮；返回"
                f"{_rework_target_label(route)}执行返修。"
            ),
        }

    if action == "accept_check_and_finish":
        # 返修会把 check_result / capacity_check_result 置为失效并清空内存字段，
        # 因此"是否真的完成过承载力验算"不能只看内存：磁盘批次汇总与 Handoff 记录的
        # 结果路径同样是有效证据（否则人工接受风险的出口会被自身返修逻辑堵死）。
        capacity_result, evidence_path = _resolve_capacity_evidence(state)
        capacity_batch = state.get("capacity_batch_status")
        result_complete = bool(
            isinstance(capacity_result, dict)
            and capacity_result
            and capacity_result.get("stage_complete") is not False
            and capacity_result.get("overall_check") is not None
        )
        batch_complete = (
            not isinstance(capacity_batch, dict)
            or capacity_batch.get("stage_complete") is True
            or capacity_result.get("stage_complete") is not False
        )
        if not result_complete or not batch_complete:
            raise ValueError("缺少完整承载力验算结果，不能接受验算风险并结束。")
        risk = {
            "scope": "modeling_check",
            "check_summary": {
                "overall_check": capacity_result.get("overall_check"),
                "utilization_summary": capacity_result.get("utilization_summary"),
            },
            "evidence_path": evidence_path,
            "reason": normalized_decision["feedback"] or "人工接受未通过的验算结果",
            "accepted_at": normalized_decision["decided_at"],
        }
        return {
            **common,
            "human_review_route": StageName.FINAL_OUTPUT.value,
            "human_override": True,
            "unresolved_manual_review": False,
            "accepted_risks": [*(state.get("accepted_risks") or []), risk],
            "task_status": "completed_with_accepted_risks",
            # 人工接受验算风险即终止返修：清空返修调度与成果失效标记，
            # 否则 final_output 会因"成果已失效/内存被清空"而静默跳过出图
            # （2026-09-16 示例项目K31 接受风险后未出图即为此现象）。
            "revision_target": None,
            "rework_required_artifacts": [],
            "invalidated_artifacts": [],
            "minimal_rework_path": [],
            "message": "人工接受当前验算风险并结束流程。",
        }

    if action == "accept_and_continue":
        final_layout = state.get("layout_result") or state.get("layout_revision_result")
        result_path = (
            state.get("revision_result_path")
            or state.get("latest_revision_result_path")
            or state.get("layout_revision_result_path")
        )
        # 人工接受布跨后补写 layout_revision/final_layout_result.json，
        # 让 GUI 运行轨道/成果中心等"按文件推断阶段"的模块与图状态保持一致。
        persisted_path = _persist_final_layout_result(state, final_layout)
        final_layout_path = persisted_path or result_path
        risk = {
            "scope": "layout_revision",
            "metrics": request.get("metrics") or {},
            "reason": normalized_decision["feedback"] or "人工接受剩余碰撞风险",
            "accepted_at": normalized_decision["decided_at"],
        }
        return {
            **common,
            "human_review_route": "design_coordinator",
            "human_override": True,
            "unresolved_manual_review": False,
            "layout_revision_completed": True,
            "layout_revision_result": final_layout,
            "final_layout_result": final_layout,
            "layout_revision_result_path": final_layout_path,
            "final_layout_result_path": final_layout_path,
            "accepted_risks": [*(state.get("accepted_risks") or []), risk],
            "task_status": "layout_revision_completed",
            "message": "人工接受当前布跨方案，保留未通过指标并继续后续设计。",
            "latest_handoff": {
                "stage": StageName.LAYOUT_REVISION.value,
                "status": "completed",
                "produced_artifacts": [{
                    "artifact_type": "final_layout_result",
                    "state_key": "final_layout_result",
                    "path": final_layout_path,
                    "content_hash": "",
                    "producer": "HumanReview",
                    "revision": int(state.get("iteration_index") or 0),
                    "valid": True,
                }],
                "revision_request": None,
                "invalidated_artifacts": [],
                "recommended_next_stage": StageName.STRUCTURAL_DESIGN.value,
                "message": "人工接受剩余碰撞风险并授权继续。",
            },
        }

    if action == "retry_coordinator":
        return {
            **common,
            "human_review_route": "design_coordinator",
            "unresolved_manual_review": False,
            "task_status": "coordinator_retry_requested",
            "message": "人工要求协调器重新决策。",
        }

    return {
        **common,
        "human_review_route": "abort",
        "unresolved_manual_review": False,
        "task_status": "cancelled",
        "message": "用户终止了当前设计任务，checkpoint 与工程成果已保留。",
    }


def human_review_node(state: AgentState) -> Dict[str, Any]:
    request = build_human_review_request(dict(state))
    decision = interrupt(request)
    update = apply_human_review_decision(dict(state), decision)
    decision_path = persist_human_review_decision(dict(state), update)
    if decision_path:
        update["human_review_decision_path"] = decision_path
    return update


def route_after_human_review(state: AgentState) -> str:
    route = str(state.get("human_review_route") or "")
    if route in {
        StageName.LAYOUT_REVISION.value,
        StageName.STRUCTURAL_DESIGN.value,
        StageName.MODELING_CHECK.value,
        StageName.FINAL_OUTPUT.value,
        "design_coordinator",
        "abort",
    }:
        return route
    return StageName.ERROR.value


def _restore_delivery_inputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """出图前补齐被返修清空的工程成果字段（从 output_dir 磁盘读回）。

    返修会把 reinforcement_design_result / check_result 等失效并清空内存字段；
    人工接受风险后流程走向 final_output，若这些字段仍为空，build_final_deliverables
    会静默走"成果已失效"分支而不出图（2026-09-16 示例项目K31 即为此现象）。
    这里按标准路径从磁盘读回，让出图拿到"被人工接受的当前设计"。
    """
    output_dir = str(state.get("output_dir") or "").strip()
    if not output_dir:
        return state
    result = dict(state)
    candidates = {
        "pier_group_result": os.path.join(output_dir, "structural_design", "pier_group", "pier_group_result.json"),
        "dimension_design_result": os.path.join(output_dir, "structural_design", "dimension_design", "dimension_design_result.json"),
        "reinforcement_design_result": os.path.join(output_dir, "structural_design", "reinforcement_design", "reinforcement_design_result.json"),
        "check_result": os.path.join(output_dir, "capacity_check", "capacity_check_batch_summary.json"),
        "capacity_check_result": os.path.join(output_dir, "capacity_check", "capacity_check_batch_summary.json"),
    }
    for key, path in candidates.items():
        if result.get(key):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload:
            result[key] = payload
    return result


def final_output_node(state: AgentState) -> Dict[str, Any]:
    from .drawing.package import build_final_deliverables

    state = _restore_delivery_inputs(dict(state))
    try:
        drawing_update = build_final_deliverables(state)
    except Exception as exc:
        return {
            "task_status": "failed",
            "message": "最终绘图成果生成失败。",
            "error": str(exc),
        }
    context = {**state, **drawing_update}
    decision = state.get("coordinator_decision") or {}
    accepted_risks = list(state.get("accepted_risks") or [])
    final_status = "completed_with_accepted_risks" if accepted_risks else "completed"
    return {
        "final_summary": {
            "task_status": final_status,
            "completed_agents": state.get("completed_agents") or [],
            "coordinator_decision": decision,
            "latest_handoff": state.get("latest_handoff"),
            "accepted_risks": accepted_risks,
            "drawing_index_path": context.get("drawing_index_path"),
            "design_manifest_path": context.get("design_manifest_path"),
            "cad_script_paths": context.get("cad_script_paths") or [],
            "drawing_preview_paths": context.get("drawing_preview_paths") or [],
        },
        **drawing_update,
        "task_status": final_status,
        "message": "多智能体流程在人工接受已记录风险后完成。" if accepted_risks else "多智能体流程执行完成。",
        "error": None,
    }


def error_node(state: AgentState) -> Dict[str, Any]:
    return {
        "task_status": "failed",
        "message": state.get("message") or "多智能体流程执行失败。",
        "error": state.get("error") or "未知错误。",
    }


def build_graph_v2(
    *,
    llm_invoke: Optional[LLMInvoke] = None,
    stage_runners: Optional[Dict[str, Callable[[AgentState], Dict[str, Any]]]] = None,
    checkpointer: Optional[Any] = None,
):
    """构建显式顶层图（层级框架）。

    - 顶层显式注册 design_coordinator + 四个专业子图节点 + manual_review + final_output + error；
    - 协调器在每个专业节点完成后重新调度（专业节点执行后回到 design_coordinator）；
    - 依赖注入 llm_invoke / stage_runners 便于无 LLM 的结构测试。

    本函数与旧 build_graph() 并存，不替换旧入口，供渐进迁移。
    """
    llm_invoke = llm_invoke or _default_llm_invoke()
    stage_runners = stage_runners or _default_stage_runners()

    builder = StateGraph(AgentState)

    builder.add_node("design_coordinator", make_coordinator_node(llm_invoke))
    builder.add_node("render_catalog_views", render_catalog_views_node)
    for stage_name in STAGE_AGENT_NAME:
        runner = stage_runners.get(stage_name)
        if runner is not None:
            builder.add_node(stage_name, make_stage_node(stage_name, runner))
    builder.add_node(StageName.MANUAL_REVIEW.value, human_review_node)
    builder.add_node(StageName.FINAL_OUTPUT.value, final_output_node)
    builder.add_node(StageName.ERROR.value, error_node)

    builder.set_entry_point("design_coordinator")

    # 协调器每次调度后先尝试渲染初设/最终目录视图（幂等、不阻断、无新结果则跳过），
    # 再按 coordinator_decision 路由到目标节点。
    stage_targets = {
        StageName.INITIAL_DESIGN.value: StageName.INITIAL_DESIGN.value,
        StageName.LAYOUT_REVISION.value: StageName.LAYOUT_REVISION.value,
        StageName.STRUCTURAL_DESIGN.value: StageName.STRUCTURAL_DESIGN.value,
        StageName.MODELING_CHECK.value: StageName.MODELING_CHECK.value,
        StageName.MANUAL_REVIEW.value: StageName.MANUAL_REVIEW.value,
        StageName.FINAL_OUTPUT.value: StageName.FINAL_OUTPUT.value,
        StageName.ERROR.value: StageName.ERROR.value,
    }
    builder.add_edge("design_coordinator", "render_catalog_views")
    builder.add_conditional_edges(
        "render_catalog_views",
        route_after_coordinator,
        stage_targets,
    )

    # 每个专业节点完成后回到协调器重新调度；返修失败则直接转人工复核。
    for stage_name in STAGE_AGENT_NAME:
        if stage_name in stage_runners:
            builder.add_conditional_edges(
                stage_name,
                route_after_stage,
                {
                    "design_coordinator": "design_coordinator",
                    StageName.MANUAL_REVIEW.value: StageName.MANUAL_REVIEW.value,
                },
            )

    builder.add_conditional_edges(
        StageName.MANUAL_REVIEW.value,
        route_after_human_review,
        {
            StageName.LAYOUT_REVISION.value: StageName.LAYOUT_REVISION.value,
            StageName.STRUCTURAL_DESIGN.value: StageName.STRUCTURAL_DESIGN.value,
            StageName.MODELING_CHECK.value: StageName.MODELING_CHECK.value,
            StageName.FINAL_OUTPUT.value: StageName.FINAL_OUTPUT.value,
            "design_coordinator": "design_coordinator",
            "abort": END,
            StageName.ERROR.value: StageName.ERROR.value,
        },
    )
    builder.add_edge(StageName.FINAL_OUTPUT.value, END)
    builder.add_edge(StageName.ERROR.value, END)

    return builder.compile(checkpointer=checkpointer)
