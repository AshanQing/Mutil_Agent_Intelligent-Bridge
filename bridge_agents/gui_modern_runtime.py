from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


AGENT_LABELS = {
    "DesignCoordinatorAgent": "协调器",
    "InitialDesignAgent": "初始布跨设计",
    "LayoutRevisionAgent": "布跨修正",
    "StructuralDesignAgent": "结构设计",
    "ModelingCheckAgent": "建模验算",
}
ACTION_LABELS = {
    "compute_pier_groups": "设计组归并计算",
    "dimension_design": "下部结构尺寸设计",
    "reinforcement_design": "盖梁/墩柱配筋设计",
    "run_capacity_check": "承载力批量验算",
    "generate_initial_layout": "生成初始布跨方案",
    "detect_collision": "碰撞检测",
    "extract_obstacles": "障碍物语义提取",
}


@dataclass(frozen=True)
class OperationResult:
    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    error_type: str = ""


def run_operation(name: str, operation: Callable[[], Mapping[str, Any]]) -> OperationResult:
    try:
        return OperationResult(name=name, payload=dict(operation()))
    except Exception as exc:
        return OperationResult(
            name=name,
            payload={},
            error=str(exc),
            error_type=type(exc).__name__,
        )


def format_run_event(event: Mapping[str, Any]) -> str | None:
    event_type = str(event.get("event_type") or "")
    agent_name = str(event.get("agent") or "")
    action_name = str(event.get("name") or "")
    status = str(event.get("status") or "")
    agent = AGENT_LABELS.get(agent_name, agent_name.replace("Agent", "") or "系统")
    action = ACTION_LABELS.get(action_name, action_name.replace("_", " "))
    if event_type == "stage_plan":
        return f"[阶段规划] {agent}：{action}"
    if event_type == "action_start":
        return f"[开始] {agent} · {action}"
    if event_type == "action_end":
        if status == "completed":
            return f"[完成] {agent} · {action}"
        error = str(event.get("error") or "")[:160]
        return f"[失败] {agent} · {action}：{error or f'状态 {status}'}"
    if event_type == "agent_skill_loaded":
        return f"[准备] {agent} 已加载专业说明"
    return None
