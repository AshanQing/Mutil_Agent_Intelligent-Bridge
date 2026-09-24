from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except TypeError:
        pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return str(obj)


def _short_value(value: Any, max_len: int = 300) -> Any:
    """避免日志里塞入完整 layout_result、完整 prompt、完整样本库。"""
    if value in [None, "", [], {}]:
        return value

    if isinstance(value, str):
        return value if len(value) <= max_len else value[:max_len] + "...<truncated>"

    if isinstance(value, dict):
        summary: Dict[str, Any] = {}
        for k, v in value.items():
            if k in {
                "layout_result",
                "existing_layout_result",
                "design_result",
                "revision_prompt",
                "few_shots",
                "cropped_data",
                "design_input",
            }:
                summary[k] = f"<{type(v).__name__}, omitted>"
            elif isinstance(v, (dict, list)):
                summary[k] = f"<{type(v).__name__}, size={len(v)}>"
            else:
                summary[k] = _short_value(v, max_len=max_len)
        return summary

    if isinstance(value, list):
        return f"<list, size={len(value)}>"

    return value


def summarize_state_for_log(state: Dict[str, Any]) -> Dict[str, Any]:
    """记录关键状态摘要，而不是记录完整 state。"""
    return {
        "user_request": _short_value(state.get("user_request")),
        "task_category": state.get("task_category"),
        "user_intent": state.get("user_intent"),
        "start_station": state.get("start_station"),
        "end_station": state.get("end_station"),
        "active_agent": state.get("active_agent"),
        "current_agent_index": state.get("current_agent_index"),
        "completed_agents": state.get("completed_agents"),
        "layout_result_available": bool(state.get("layout_result") or state.get("existing_layout_result")),
        "design_units_available": bool(state.get("design_units")),
        "dimension_design_available": bool(state.get("dimension_design_result")),
        "reinforcement_design_available": bool(state.get("reinforcement_design_result")),
        "structural_design_result_available": bool(state.get("structural_design_result")),
        "task_status": state.get("task_status"),
        "error": state.get("error"),
    }


def collect_output_files(payload: Dict[str, Any]) -> Dict[str, Any]:
    """从 action 返回值中抽取常见文件路径。"""
    if not isinstance(payload, dict):
        return {}

    result: Dict[str, Any] = {}

    for key, value in payload.items():
        if key.endswith("_path") or key.endswith("_file"):
            result[key] = value

    output_files = payload.get("output_files")
    if isinstance(output_files, dict):
        result.update(output_files)

    return {k: v for k, v in result.items() if v not in [None, "", [], {}]}


class AgentLogger:
    """桥梁多智能体流程结构化日志收集器。

    注意：
    - 内部数据始终保持 JSON 可序列化；
    - 在 AgentState 中保存的是 logger.data，而不是 AgentLogger 对象。
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        if data:
            self.data = data
        else:
            self.data = {
                "run_id": str(uuid.uuid4()),
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "events": [],
                "files": [],
                "summary": {},
            }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "AgentLogger":
        log_data = state.get("agent_log")
        if not isinstance(log_data, dict):
            log_data = None
        return cls(log_data)

    def attach(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.data["updated_at"] = _now_iso()
        state["agent_log"] = self.to_dict()
        return state

    def to_dict(self) -> Dict[str, Any]:
        self.data["updated_at"] = _now_iso()
        return _json_safe(self.data)

    def log_event(
        self,
        event_type: str,
        *,
        agent: Optional[str] = None,
        stage: Optional[str] = None,
        name: Optional[str] = None,
        status: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[Any] = None,
    ) -> None:
        self.data.setdefault("events", []).append({
            "time": _now_iso(),
            "event_type": event_type,
            "agent": agent,
            "stage": stage,
            "name": name,
            "status": status,
            "payload": _short_value(payload or {}),
            "error": _short_value(error),
        })

    def log_task_allocation(
        self,
        *,
        user_intent: str,
        task_category: str,
        agent_sequence: List[Dict[str, Any]],
        extracted_parameters: Dict[str, Any],
    ) -> None:
        self.log_event(
            "task_allocation",
            agent="TaskAllocationAgent",
            stage="task_allocation",
            status="completed",
            payload={
                "task_category": task_category,
                "user_intent": user_intent,
                "agent_sequence": agent_sequence,
                "extracted_parameters": extracted_parameters,
            },
        )

    def log_agent_start(self, agent: str, mode: str, state_summary: Dict[str, Any]) -> None:
        self.log_event(
            "agent_start",
            agent=agent,
            stage=agent,
            status="started",
            payload={
                "mode": mode,
                "state_summary": state_summary,
            },
        )

    def log_agent_end(self, agent: str, status: str, update_summary: Dict[str, Any]) -> None:
        self.log_event(
            "agent_end",
            agent=agent,
            stage=agent,
            status=status,
            payload=update_summary,
            error=update_summary.get("error"),
        )

    def log_action_start(
        self,
        *,
        agent: str,
        action: str,
        input_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_event(
            "action_start",
            agent=agent,
            stage=agent,
            name=action,
            status="started",
            payload=input_summary or {},
        )

    def log_action_end(
        self,
        *,
        agent: str,
        action: str,
        output: Dict[str, Any],
    ) -> None:
        files = collect_output_files(output)
        self.log_event(
            "action_end",
            agent=agent,
            stage=agent,
            name=action,
            status="failed" if output.get("error") else "completed",
            payload={
                "output_summary": _short_value(output),
                "output_files": files,
            },
            error=output.get("error"),
        )
        for file_type, path in files.items():
            self.log_file(file_type=file_type, path=path, producer=f"{agent}.{action}")

    def log_thought(
        self,
        *,
        agent: str,
        thought: str,
        action: Optional[str] = None,
        observation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录显式输出的策略理由，不记录模型隐藏推理。"""
        self.log_event(
            "agent_thought",
            agent=agent,
            stage=agent,
            name=action,
            status="recorded",
            payload={
                "thought": thought,
                "action": action,
                "observation": observation or {},
            },
        )

    def log_file(self, *, file_type: str, path: Any, producer: Optional[str] = None) -> None:
        if path in [None, "", [], {}]:
            return
        self.data.setdefault("files", []).append({
            "time": _now_iso(),
            "file_type": file_type,
            "path": str(path),
            "producer": producer,
        })

    def log_final(self, summary: Dict[str, Any]) -> None:
        self.data["summary"] = _json_safe(summary)
        self.log_event(
            "final_summary",
            stage="final",
            status=summary.get("task_status"),
            payload=summary,
        )

    def save(self, output_dir: str, filename: str = "agent_run_log.json") -> str:
        log_dir = os.path.join(output_dir or "outputs", "logs")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path