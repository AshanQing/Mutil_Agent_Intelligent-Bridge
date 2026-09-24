from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from bridge_agents.prompt_registry import render_prompt
from bridge_agents.prompt_audit import artifact_attempt, record_prompt_audit, reserve_artifact_path


def _read_text_file(path: Optional[str], default: str = "") -> str:
    if not path:
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


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


def _extract_role_block(template_text: str) -> str:
    if not template_text:
        return (
            "你是一名资深桥梁工程设计师，负责在规范约束、线路条件、"
            "障碍物约束和工程可实施性约束下修正设桥布跨方案。"
        )

    marker = "### 设计任务思维链"
    if marker in template_text:
        return template_text.split(marker, 1)[0].strip()

    return template_text[:2500].strip()


def _extract_output_schema_block(template_text: str) -> str:
    if not template_text:
        return "最终仅输出符合设桥布跨结果格式的 JSON 对象，不得输出 Markdown 或解释性文字。"

    for marker in ["**【输出格式强制要求", "【输出格式强制要求"]:
        if marker in template_text:
            return template_text.split(marker, 1)[1].strip()

    return template_text[-5000:].strip()


def _extract_design_payload(design_result: Any) -> Dict[str, Any]:
    if isinstance(design_result, dict) and "设桥总览" in design_result:
        return design_result

    if isinstance(design_result, dict):
        for key in [
            "layout_result",
            "current_layout_result",
            "design_json",
            "design_result",
            "result",
            "data",
        ]:
            value = design_result.get(key)
            if isinstance(value, dict) and "设桥总览" in value:
                return value

    raise ValueError("未找到可用于修正 Prompt 的设桥布跨 JSON。")


def build_compact_collision_context(
    metrics: Dict[str, Any],
    collision_items: list[Any],
) -> Dict[str, Any]:
    """投影并稳定排序碰撞上下文，减少每轮修正 Prompt 的动态 token。"""
    compact_items = []
    for item in collision_items:
        if not isinstance(item, dict):
            continue
        result = item.get("res") if isinstance(item.get("res"), dict) else {}
        compact_items.append({
            "bridge_id": item.get("bridge_id"),
            "side_label": item.get("side_label"),
            "pier_id": item.get("pier_id"),
            "col_name": item.get("col_name"),
            "station": item.get("station_label") or item.get("station"),
            "status": result.get("status") or item.get("status"),
            "intrusion_depth_m": result.get("intrusion_depth_m", item.get("intrusion_depth_m")),
            "overlap_ratio": result.get("overlap_ratio", item.get("overlap_ratio")),
        })

    compact_items.sort(key=lambda item: tuple(
        str(item.get(key) or "")
        for key in ("bridge_id", "side_label", "station", "pier_id", "col_name")
    ))
    stable_metrics = {
        str(key): _json_safe(value)
        for key, value in sorted((metrics or {}).items(), key=lambda pair: str(pair[0]))
    }
    return {
        "collision_metrics": stable_metrics,
        "collision_items": compact_items,
    }


def build_layout_revision_prompt_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    config_path = state.get("config_path") or "config/settings.yaml"
    initial_template_text = (
        render_prompt(
            "tasks.initial_layout_design.v1",
            {"design_input_json": "{}", "few_shots_json": "[]", "standards_text": ""},
            config_path=config_path,
        ).user_content
    )

    standards_text = _read_text_file(
        state.get("standards_path"),
        "未提供独立 standard 文件，请沿用初始设计 Prompt 中的规范约束。",
    )

    verification = state.get("verification_result") or {}
    if not isinstance(verification, dict):
        verification = {}

    feedback_decision = state.get("feedback_decision") or {}
    if not isinstance(feedback_decision, dict):
        feedback_decision = {}

    collision_metrics = (
        verification.get("metrics")
        or state.get("collision_metrics")
        or {}
    )

    collision_items = (
        verification.get("collision_items")
        or state.get("collision_items")
        or []
    )
    compact_collision = build_compact_collision_context(collision_metrics, collision_items)

    previous_layout = _extract_design_payload(
        state.get("layout_result")
        or state.get("design_result")
        or state.get("existing_layout_result")
    )

    design_input = state.get("design_input") or {}

    revision_instruction_text = state.get("revision_instruction") or ""
    if not str(revision_instruction_text).strip():
        revision_instruction_text = (
            "未提供独立修正指令。请根据碰撞指标、冲突墩位和修正决策进行最小必要修正。"
        )

    revision_advice_payload = {
        "iteration_index": int(state.get("iteration_index") or 0),
        "revision_strategy": feedback_decision.get("revision_strategy"),
        "target_bridge_ids": feedback_decision.get("target_bridge_ids", []),
        "target_pier_ids": feedback_decision.get("target_pier_ids", []),
        "code_instruction": feedback_decision.get(
            "code_instruction",
            {
                "preserve": [],
                "modify": [],
                "forbid": [],
            },
        ),
        "revision_advice": feedback_decision.get("revision_advice") or "",
        "collision_metrics": compact_collision["collision_metrics"],
        "collision_items": compact_collision["collision_items"],
    }

    context = {
        "ROLE_BLOCK": _extract_role_block(initial_template_text),
        "STANDARDS_BLOCK": standards_text,
        "DESIGN_INPUT_JSON": json.dumps(
            _json_safe(design_input),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "PREVIOUS_LAYOUT_JSON": json.dumps(
            _json_safe(previous_layout),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "REVISION_ADVICE_JSON": json.dumps(
            _json_safe(revision_advice_payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "REVISION_INSTRUCTION_TEXT": str(revision_instruction_text),
        "OUTPUT_SCHEMA_BLOCK": _extract_output_schema_block(initial_template_text),
    }

    rendered_prompt = render_prompt("tasks.layout_revision_design.v1", context, config_path=config_path)
    prompt_text = rendered_prompt.user_content or rendered_prompt.system_content

    if str(revision_instruction_text).strip() and str(revision_instruction_text).strip() not in prompt_text:
        prompt_text = (
            "# 强制修正指令（由碰撞报告语义转译工具生成）\n\n"
            + str(revision_instruction_text).strip()
            + "\n\n---\n\n"
            + prompt_text
        )

    iteration_index = int(state.get("iteration_index") or 0)
    output_dir = state.get("output_dir") or os.getcwd()

    prompt_dir = os.path.join(output_dir, "revision_prompts")
    os.makedirs(prompt_dir, exist_ok=True)

    prompt_path = reserve_artifact_path(
        os.path.join(prompt_dir, f"revision_prompt_round_{iteration_index + 1}.txt")
    )

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    record_prompt_audit(
        output_dir,
        prompt_id=rendered_prompt.prompt_id,
        version=rendered_prompt.version,
        template_sha256=rendered_prompt.template_sha256,
        stage="layout_revision",
        scope_id=f"round_{iteration_index + 1}",
        attempt=artifact_attempt(prompt_path),
        rendered_path=prompt_path,
        status="rendered",
    )

    return {
        "success": True,
        "revision_prompt": prompt_text,
        "revision_prompt_path": str(prompt_path),
        "prompt_id": rendered_prompt.prompt_id,
        "prompt_version": rendered_prompt.version,
        "template_sha256": rendered_prompt.template_sha256,
        "revision_instruction": str(revision_instruction_text),
        "error": None,
    }


@tool("layout_revision_prompt_builder")
def layout_revision_prompt_builder_tool(design_state_json: str) -> Dict[str, Any]:
    """
    根据当前 AgentState 构造设桥布跨修正 Prompt，并保存为 txt 文件。

    输入:
        design_state_json:
            JSON 字符串格式的 AgentState。

    输出:
        {
            "success": bool,
            "revision_prompt": str,
            "revision_prompt_path": str | None,
            "revision_instruction": str,
            "error": str | None
        }
    """
    try:
        state = json.loads(design_state_json)
        if not isinstance(state, dict):
            raise ValueError("design_state_json 顶层必须是 JSON 对象。")

        return build_layout_revision_prompt_from_state(state)

    except Exception as e:
        return {
            "success": False,
            "revision_prompt": "",
            "revision_prompt_path": None,
            "revision_instruction": "",
            "error": str(e),
        }
