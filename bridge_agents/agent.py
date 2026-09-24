from __future__ import annotations
import os
import json
import glob
import logging
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .stage_agents import AGENT_REGISTRY
from .state import AgentState
from .agent_logger import AgentLogger, summarize_state_for_log
from .prompt_registry import render_prompt
from .utils import extract_json_object, get_controller_llm, json_safe, load_settings, pick, regex_extract_stations

load_dotenv()
logger = logging.getLogger(__name__)


SUPPORTED_INTENTS = {
    "full_design",
    "layout_design",
    "layout_design_check_revision",
    "layout_check_revision",
    "structural_design",
    "layout_to_reinforcement_design",
    "reinforcement_design",
    "reinforcement_design_verification",
    "verification",
}

INTENT_ALIASES = {
    "layout_only": "layout_design",
    "layout_design_verify": "layout_design_check_revision",
    "layout_verify_revise": "layout_check_revision",
    "full_process": "full_design",
}

INTENT_TO_AGENT_SEQUENCE: Dict[str, List[Dict[str, str]]] = {
    "full_design": [
        {"agent": "InitialDesignAgent", "mode": "fixed_pipeline", "purpose": "生成初步设桥布跨方案"},
        {"agent": "LayoutRevisionAgent", "mode": "react_loop", "purpose": "检测并修正布跨方案"},
        {"agent": "StructuralDesignAgent", "mode": "semi_fixed_pipeline", "purpose": "完成下部结构尺寸与配筋设计"},
        {"agent": "ModelingCheckAgent", "mode": "react_loop_with_feedback", "purpose": "建模分析与验算反馈"},
    ],
    "layout_design": [
        {"agent": "InitialDesignAgent", "mode": "fixed_pipeline", "purpose": "生成初步设桥布跨方案"},
    ],
    "layout_design_check_revision": [
        {"agent": "InitialDesignAgent", "mode": "fixed_pipeline", "purpose": "生成初步设桥布跨方案"},
        {"agent": "LayoutRevisionAgent", "mode": "react_loop", "purpose": "检测并修正布跨方案"},
    ],
    "layout_check_revision": [
        {"agent": "LayoutRevisionAgent", "mode": "react_loop", "purpose": "基于已有布跨方案进行检测修正"},
    ],
    "structural_design": [
        {"agent": "StructuralDesignAgent", "mode": "semi_fixed_pipeline", "purpose": "完成下部结构尺寸与配筋设计"},
    ],
    "layout_to_reinforcement_design": [
        {"agent": "InitialDesignAgent", "mode": "fixed_pipeline", "purpose": "生成初步设桥布跨方案"},
        {"agent": "LayoutRevisionAgent", "mode": "react_loop", "purpose": "检测并修正布跨方案"},
        {"agent": "StructuralDesignAgent", "mode": "semi_fixed_pipeline", "purpose": "完成下部结构尺寸与配筋设计"},
    ], 
    "reinforcement_design": [
        {"agent": "StructuralDesignAgent", "mode": "semi_fixed_pipeline", "purpose": "完成或深化配筋设计"},
    ],
    "reinforcement_design_verification": [
        {"agent": "StructuralDesignAgent", "mode": "semi_fixed_pipeline", "purpose": "完成或深化配筋设计"},
        {"agent": "ModelingCheckAgent", "mode": "react_loop_with_feedback", "purpose": "建模分析与验算反馈"},
    ],
    "verification": [
        {"agent": "ModelingCheckAgent", "mode": "react_loop_with_feedback", "purpose": "建模分析与验算反馈"},
    ],
}


def _normalize_user_intent(value: Any) -> str:
    value = str(value or "unknown").strip()
    return INTENT_ALIASES.get(value, value)


def _fallback_allocate(user_input: str) -> Dict[str, Any]:
    text = user_input or ""
    if any(x in text for x in ["从头", "原始资料", "完整"]) and any(x in text for x in ["配筋", "结构设计", "下部结构"]):
        intent = "layout_to_reinforcement_design"
    elif any(x in text for x in ["全流程", "完整设计", "从头", "全部"]):
        intent = "full_design"
    elif any(x in text for x in ["配筋"]) and any(x in text for x in ["验算", "校核", "承载力", "反馈", "OpenSees"]):
        intent = "reinforcement_design_verification"
    elif any(x in text for x in ["验算", "建模", "OpenSees", "分析"]):
        intent = "verification"
    elif any(x in text for x in ["配筋"]):
        intent = "reinforcement_design"
    elif any(x in text for x in ["下部结构", "结构设计", "尺寸设计", "盖梁", "墩柱", "基础"]):
        intent = "structural_design"
    elif any(x in text for x in ["已有", "检测", "校核", "修正", "冲突", "复检"]):
        if any(x in text for x in ["设桥", "布跨", "桥位"]):
            intent = "layout_design_check_revision"
        else:
            intent = "layout_check_revision"
    elif any(x in text for x in ["设桥", "布跨", "桥位"]):
        intent = "layout_design"
    else:
        intent = "unknown"

    return {
        "task_category": "design" if intent != "unknown" else "unknown",
        "user_intent": intent,
        "agent_sequence": INTENT_TO_AGENT_SEQUENCE.get(intent, []),
        "extracted_parameters": {},
    }


def task_allocation_agent_node(state: AgentState) -> Dict[str, Any]:
    user_input = state.get("user_request") or ""
    if not user_input:
        return {"task_category": "unknown", "user_intent": "unknown", "error": "用户输入为空。"}

    payload = {
        "用户自然语言": user_input,
        "用户补充参数": state.get("user_parameters") or {},
        "用户附加文件清单": state.get("user_artifacts") or [],
        "当前已有状态摘要": {
            "layout_result_available": bool(state.get("layout_result") or state.get("existing_layout_result")),
            "design_units_available": bool(state.get("design_units")),
            "dimension_design_available": bool(state.get("dimension_design_result")),
            "reinforcement_design_available": bool(state.get("reinforcement_design_result")),
            "structural_design_result_available": bool(state.get("structural_design_result")),
            "reinforcement_yaml_path_available": bool(state.get("reinforcement_yaml_path")),
            "opensees_force_json_path_available": bool(state.get("opensees_force_json_path")),
            "capacity_check_input_available": bool(
                state.get("reinforcement_yaml_path")
                or state.get("reinforcement_design_result")
            ) and bool(state.get("opensees_force_json_path")),
            "capacity_check_result_available": bool(
                state.get("capacity_check_result")
                or state.get("check_result")
            ),
            "layout_revision_result_available": bool(
                state.get("layout_revision_result")
                or state.get("final_layout_result")
                or state.get("layout_revision_completed")
            ),
        },
    }

    try:
        llm = get_controller_llm(state.get("config_path") or "config/settings.yaml")
        rendered_prompt = render_prompt(
            "agents.task_allocation.v1",
            {},
            config_path=state.get("config_path") or "config/settings.yaml",
        )
        response = llm.invoke([
            ("system", rendered_prompt.system_content),
            ("user", json.dumps(json_safe(payload), ensure_ascii=False, indent=2)),
        ])
        print("\n[TaskAllocationAgent raw LLM response]")
        print(response.content, flush=True)
        parsed = extract_json_object(response.content)
    except Exception as e:
        logger.warning("任务分配 LLM 失败，启用规则兜底：%s", e)
        parsed = _fallback_allocate(user_input)

    task_category = str(parsed.get("task_category") or "unknown").strip()
    user_intent = _normalize_user_intent(parsed.get("user_intent") or parsed.get("intent"))
    if user_intent not in SUPPORTED_INTENTS:
        user_intent = "unknown"

    params = parsed.get("extracted_parameters") if isinstance(parsed.get("extracted_parameters"), dict) else {}
    regex_params = regex_extract_stations(user_input)

    start_station = params.get("start_station") or regex_params.get("start_station") or state.get("start_station")
    end_station = params.get("end_station") or regex_params.get("end_station") or state.get("end_station")

    agent_sequence = parsed.get("agent_sequence") if isinstance(parsed.get("agent_sequence"), list) else []
    if not agent_sequence and user_intent in INTENT_TO_AGENT_SEQUENCE:
        agent_sequence = INTENT_TO_AGENT_SEQUENCE[user_intent]

    # 过滤非法智能体，避免 LLM 编造 agent 名称。
    normalized_sequence: List[Dict[str, Any]] = []
    for item in agent_sequence:
        if not isinstance(item, dict):
            continue
        agent_name = item.get("agent")
        if agent_name in AGENT_REGISTRY:
            normalized_sequence.append({
                "agent": agent_name,
                "mode": item.get("mode") or getattr(AGENT_REGISTRY[agent_name], "mode", "unknown"),
                "purpose": item.get("purpose", ""),
            })

    allocation_update = {
        "task_category": task_category if task_category in ["design", "verification", "unknown"] else "unknown",
        "user_intent": user_intent,
        "extracted_parameters": {
            "start_station": start_station,
            "end_station": end_station,
        },
        "start_station": start_station,
        "end_station": end_station,
        "agent_sequence": normalized_sequence,
        "current_agent_index": 0,
        "completed_agents": [],
        "active_agent": "TaskAllocationAgent",
        "prompt_trace": [rendered_prompt.to_dict()] if "rendered_prompt" in locals() else [],
        "error": None,
    }
    print("\n" + "=" * 80)
    print("[TaskAllocationAgent] 任务分配结果")
    print("-" * 80)
    print(f"user_request  : {user_input}")
    print(f"task_category : {allocation_update.get('task_category')}")
    print(f"user_intent   : {allocation_update.get('user_intent')}")
    print(f"start_station : {allocation_update.get('start_station')}")
    print(f"end_station   : {allocation_update.get('end_station')}")
    print("agent_sequence:")
    for idx, item in enumerate(normalized_sequence, start=1):
        print(
            f"  {idx}. {item.get('agent')} "
            f"| mode={item.get('mode')} "
            f"| purpose={item.get('purpose')}"
        )
    print("=" * 80 + "\n", flush=True)

    run_logger = AgentLogger.from_state(state)
    run_logger.log_task_allocation(
        task_category=allocation_update["task_category"],
        user_intent=user_intent,
        agent_sequence=normalized_sequence,
        extracted_parameters=allocation_update["extracted_parameters"],
    )
    allocation_update["agent_log"] = run_logger.to_dict()
    return allocation_update


def route_after_task_allocation(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    if not state.get("agent_sequence"):
        return "unsupported"
    return "agent_executor"


def _find_agent_index(agent_sequence: List[Dict[str, Any]], agent_name: str) -> Optional[int]:
    for idx, item in enumerate(agent_sequence):
        if isinstance(item, dict) and item.get("agent") == agent_name:
            return idx
    return None


def agent_executor_node(state: AgentState) -> Dict[str, Any]:
    """阶段智能体执行器。

    注意：这里不是工具级 workflow executor。
    它只负责依次启动阶段 Agent；阶段 Agent 内部自行决定固定流程或 ReAct。
    """
    agent_sequence = state.get("agent_sequence") or []
    current_index = int(state.get("current_agent_index") or 0)

    if current_index >= len(agent_sequence):
        return {
            "task_status": "completed",
            "message": "所有阶段智能体已执行完成。",
            "error": None,
        }

    agent_item = agent_sequence[current_index]
    agent_name = agent_item.get("agent") if isinstance(agent_item, dict) else None
    if agent_name not in AGENT_REGISTRY:
        return {"error": f"未注册的阶段智能体: {agent_name}"}

    agent = AGENT_REGISTRY[agent_name]
    logger.info("启动阶段智能体: %s", agent_name)

    print("\n" + "-" * 80)
    print(f"[AgentExecutor] 启动阶段智能体: {agent_name}")
    print(f"current_agent_index: {current_index + 1}/{len(agent_sequence)}")
    print(f"mode: {agent_item.get('mode') if isinstance(agent_item, dict) else None}")
    print(f"purpose: {agent_item.get('purpose') if isinstance(agent_item, dict) else None}")
    print("-" * 80 + "\n", flush=True)

    run_logger = AgentLogger.from_state(state)
    run_logger.log_agent_start(
        agent=agent_name,
        mode=agent_item.get("mode") or getattr(agent, "mode", "unknown"),
        state_summary=summarize_state_for_log(dict(state)),
    )

    state_with_log = dict(state)
    state_with_log["agent_log"] = run_logger.to_dict()

    update = agent.run(state_with_log)

    print("\n" + "-" * 80)
    print(f"[AgentExecutor] 阶段智能体完成: {agent_name}")
    print(f"task_status: {update.get('task_status')}")
    print(f"message: {update.get('message')}")
    print(f"error: {update.get('error')}")
    print("-" * 80 + "\n", flush=True)

    run_logger = AgentLogger.from_state({**state_with_log, **update})
    run_logger.log_agent_end(
        agent=agent_name,
        status="failed" if update.get("error") else "completed",
        update_summary=summarize_state_for_log({**state_with_log, **update}),
    )
    update["agent_log"] = run_logger.to_dict()

    if update.get("error"):
        return update

    completed_agents = list(state.get("completed_agents") or [])
    completed_agents.append(agent_name)

    # ModelingCheckAgent 若判定需要返回结构设计修正，则不直接结束，
    # 而是把执行指针跳回 StructuralDesignAgent。StructuralDesignAgent 完成后，
    # 顶层顺序会自然回到 ModelingCheckAgent 复验。
    feedback_decision = update.get("feedback_decision") if isinstance(update.get("feedback_decision"), dict) else {}
    next_action = feedback_decision.get("next_action")
    if (
        agent_name == "ModelingCheckAgent"
        and update.get("task_status") == "modeling_check_revision_required"
        and next_action in ["revise_reinforcement", "revise_dimension"]
    ):
        target_index = _find_agent_index(agent_sequence, "StructuralDesignAgent")
        if target_index is None:
            expanded_sequence = list(agent_sequence)
            expanded_sequence[current_index + 1:current_index + 1] = [
                {
                    "agent": "StructuralDesignAgent",
                    "mode": "semi_fixed_pipeline",
                    "purpose": "根据建模验算反馈修正结构尺寸或配筋设计",
                },
                {
                    "agent": "ModelingCheckAgent",
                    "mode": "react_loop_with_feedback",
                    "purpose": "对修正后的结构结果进行复验",
                },
            ]
            return {
                **update,
                "completed_agents": completed_agents,
                "active_agent": agent_name,
                "agent_sequence": expanded_sequence,
                "current_agent_index": current_index + 1,
                "message": update.get("message") or "验算反馈要求返回结构设计阶段修正。",
                "error": None,
            }
        return {
            **update,
            "completed_agents": completed_agents,
            "current_agent_index": target_index,
            "active_agent": agent_name,
            "message": update.get("message") or "验算反馈要求返回结构设计阶段修正。",
            "error": None,
        }

    return {
        **update,
        "completed_agents": completed_agents,
        "current_agent_index": current_index + 1,
        "active_agent": agent_name,
        "error": None,
    }


def route_after_agent_executor(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    current_index = int(state.get("current_agent_index") or 0)
    total = len(state.get("agent_sequence") or [])
    if current_index < total:
        return "continue_agents"
    return "final"


def unsupported_intent_node(state: AgentState) -> Dict[str, Any]:
    return {
        "task_status": "unsupported",
        "message": (
            f"当前任务暂不支持自动执行。task_category={state.get('task_category')}, "
            f"user_intent={state.get('user_intent')}。"
        ),
        "error": None,
    }


def final_output_node(state: AgentState) -> Dict[str, Any]:
    from .drawing.package import build_final_deliverables

    try:
        drawing_update = build_final_deliverables(state)
    except Exception as exc:
        return {
            "task_status": "failed",
            "message": "最终绘图成果生成失败。",
            "error": str(exc),
        }
    context = {**state, **drawing_update}
    summary = {
        "task_category": state.get("task_category"),
        "user_intent": state.get("user_intent"),
        "completed_agents": state.get("completed_agents") or [],
        "layout_result_available": bool(state.get("layout_result")),
        "collision_metrics": state.get("collision_metrics"),
        "design_units_available": bool(state.get("design_units")),
        "dimension_design_available": bool(state.get("dimension_design_result")),
        "reinforcement_design_available": bool(state.get("reinforcement_design_result")),
        "structural_design_result_available": bool(state.get("structural_design_result")),
        "structural_design_result_path": state.get("structural_design_result_path"),
        "layout_revision_completed": bool(state.get("layout_revision_completed")),
        "layout_revision_result_available": bool(state.get("layout_revision_result") or state.get("final_layout_result")),
        "capacity_check_result_available": bool(state.get("capacity_check_result")),
        "opensees_force_json_path": state.get("opensees_force_json_path"),
        "reinforcement_yaml_path": state.get("reinforcement_yaml_path"),
        "check_result_available": bool(state.get("check_result")),
        "overall_check": (state.get("check_result") or {}).get("overall_check") if isinstance(state.get("check_result"), dict) else None,
        "feedback_decision": state.get("feedback_decision"),
        "revision_context_available": bool(state.get("revision_context")),
        "drawing_index_path": context.get("drawing_index_path"),
        "design_manifest_path": context.get("design_manifest_path"),
        "cad_script_paths": context.get("cad_script_paths") or [],
        "drawing_preview_paths": context.get("drawing_preview_paths") or [],
        "task_status": state.get("task_status") or "completed",
    }
    run_logger = AgentLogger.from_state(state)
    run_logger.log_final(summary)

    return {
        **drawing_update,
        "final_summary": summary,
        "agent_log": run_logger.to_dict(),
        "task_status": state.get("task_status") or "completed",
        "message": state.get("message") or "多智能体流程执行完成。",
        "error": None,
    }


def error_node(state: AgentState) -> Dict[str, Any]:
    return {
        "task_status": "failed",
        "message": state.get("message") or "多智能体流程执行失败。",
        "error": state.get("error") or "未知错误。",
    }


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("task_allocation", task_allocation_agent_node)
    builder.add_node("agent_executor", agent_executor_node)
    builder.add_node("unsupported_intent", unsupported_intent_node)
    builder.add_node("final_output", final_output_node)
    builder.add_node("error", error_node)

    builder.set_entry_point("task_allocation")

    builder.add_conditional_edges(
        "task_allocation",
        route_after_task_allocation,
        {
            "agent_executor": "agent_executor",
            "unsupported": "unsupported_intent",
            "error": "error",
        },
    )

    builder.add_conditional_edges(
        "agent_executor",
        route_after_agent_executor,
        {
            "continue_agents": "agent_executor",
            "final": "final_output",
            "error": "error",
        },
    )

    builder.add_edge("unsupported_intent", "final_output")
    builder.add_edge("final_output", END)
    builder.add_edge("error", END)

    return builder.compile()

def _load_optional_json(path: Any) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        path = str(path)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _existing_path(path: Any) -> Optional[str]:
    if not path:
        return None
    path = str(path)
    return path if os.path.exists(path) else None


def _first_existing_path(*paths: Any) -> Optional[str]:
    for path in paths:
        existing = _existing_path(path)
        if existing:
            return existing
    return None


def _latest_file(pattern: str) -> Optional[str]:
    files = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    if not files:
        return None
    return max(files, key=lambda p: os.path.getmtime(p))


def _latest_revision_file(output_dir: str) -> Optional[str]:
    pattern = os.path.join(output_dir, "revision_results", "revision_design_round_*.json")
    files = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    if not files:
        return None

    def revision_key(path: str) -> tuple[int, float]:
        match = re.search(r"revision_design_round_(\d+)\.json$", path)
        return (int(match.group(1)) if match else -1, os.path.getmtime(path))

    return max(files, key=revision_key)


def _all_files(pattern: str) -> List[str]:
    return sorted(
        [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)],
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )


def _infer_single_route_prefix(data_path: Any) -> Optional[str]:
    """Infer route file prefix when a route folder contains exactly one PM file."""
    if not data_path:
        return None
    data_path = str(data_path)
    if not os.path.isdir(data_path):
        return None

    candidates: List[str] = []
    for pattern in ["*.pm", "*.PM"]:
        for path in glob.glob(os.path.join(data_path, pattern)):
            if os.path.isfile(path):
                candidates.append(os.path.splitext(os.path.basename(path))[0])

    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None


def _reinforcement_task_ids(payload: Any) -> List[str]:
    """从配筋成果里取出全部设计组任务号（用于判断图纸索引是否覆盖完整）。"""
    if not isinstance(payload, dict):
        return []
    root = payload.get("任务3_下部结构配筋设计结果")
    if not isinstance(root, dict):
        root = payload
    rows = root.get("分组原始结果") if isinstance(root, dict) else None
    ids = {
        str((row.get("reinforcement_task") or {}).get("task_id") or row.get("task_id") or "").strip()
        for row in (rows or [])
        if isinstance(row, dict)
    }
    return sorted(item for item in ids if item)


def _discover_existing_outputs(output_dir: Any) -> Dict[str, Any]:
    """Scan output_dir for completed stage artifacts used by task allocation."""
    if not output_dir:
        return {}

    output_dir = str(output_dir)
    discovered: Dict[str, Any] = {}

    plane_json_path = _first_existing_path(
        os.path.join(output_dir, "plane_from_loader", "K_plane.json"),
        _latest_file(os.path.join(output_dir, "plane_from_loader", "*_plane.json")),
        _latest_file(os.path.join(output_dir, "preprocess", "plane", "*_plane.json")),
        _latest_file(os.path.join(output_dir, "**", "*_plane.json")),
    )
    mask_path = _first_existing_path(
        _latest_file(os.path.join(output_dir, "mask", "obstacle_mask_*.png")),
        _latest_file(os.path.join(output_dir, "**", "mask", "obstacle_mask_*.png")),
        _latest_file(os.path.join(output_dir, "**", "obstacle_mask_*.png")),
    )
    pgw_path = _first_existing_path(
        _latest_file(os.path.join(output_dir, "pred_*.pgw")),
        _latest_file(os.path.join(output_dir, "*.pgw")),
        _latest_file(os.path.join(output_dir, "**", "*.pgw")),
    )
    png_path = _first_existing_path(
        _latest_file(os.path.join(output_dir, "pred_*.png")),
        _latest_file(os.path.join(output_dir, "*.png")),
    )
    jpg_path = _first_existing_path(
        _latest_file(os.path.join(output_dir, "pred_*.jpg")),
        _latest_file(os.path.join(output_dir, "*.jpg")),
        _latest_file(os.path.join(output_dir, "**", "merged_vis*.jpg")),
    )
    obstacle_json_path = _first_existing_path(
        _latest_file(os.path.join(output_dir, "obstacle_semantic", "complete_obstacles_grouped_*.json")),
        _latest_file(os.path.join(output_dir, "**", "complete_obstacles_grouped_*.json")),
    )

    layout_revision_path = _first_existing_path(
        os.path.join(output_dir, "layout_revision", "final_layout_result.json"),
        _latest_file(os.path.join(output_dir, "layout_revision", "**", "final_layout_result.json")),
    )
    layout_result_path = _first_existing_path(
        layout_revision_path,
        _latest_file(os.path.join(output_dir, "design_run_*", "design_result.json")),
        _latest_file(os.path.join(output_dir, "**", "design_result.json")),
    )
    latest_revision_result_path = _latest_revision_file(output_dir)
    latest_collision_metrics_path = _latest_file(
        os.path.join(output_dir, "collision_detection", "collision_metrics_*.json")
    )

    design_units_path = _first_existing_path(
        os.path.join(output_dir, "structural_design", "design_units", "design_units_result.json"),
        _latest_file(os.path.join(output_dir, "structural_design", "**", "design_units_result.json")),
    )
    dimension_path = _first_existing_path(
        os.path.join(output_dir, "structural_design", "dimension_design", "dimension_design_result.json"),
        _latest_file(os.path.join(output_dir, "structural_design", "**", "dimension_design_result.json")),
    )
    reinforcement_path = _first_existing_path(
        os.path.join(output_dir, "structural_design", "reinforcement_design", "reinforcement_design_result.json"),
        _latest_file(os.path.join(output_dir, "structural_design", "**", "reinforcement_design_result.json")),
    )
    pier_group_path = _first_existing_path(
        os.path.join(output_dir, "structural_design", "pier_group", "pier_group_result.json"),
        _latest_file(os.path.join(output_dir, "structural_design", "**", "pier_group_result.json")),
    )
    structural_path = _first_existing_path(
        os.path.join(output_dir, "structural_design", "structural_design_result.json"),
        _latest_file(os.path.join(output_dir, "structural_design", "**", "structural_design_result.json")),
    )

    yaml_paths = _all_files(
        os.path.join(output_dir, "structural_design", "reinforcement_design", "**", "reinforcement_result_*.yaml")
    )
    force_paths = _all_files(
        os.path.join(output_dir, "structural_design", "reinforcement_design", "**", "internal_force_output_full_beam_*.json")
    )
    reinforcement_yaml_path = yaml_paths[0] if yaml_paths else None
    opensees_force_json_path = None

    if reinforcement_yaml_path:
        task_dir = os.path.dirname(reinforcement_yaml_path)
        task_id = os.path.basename(task_dir)
        paired_force = os.path.join(task_dir, f"internal_force_output_full_beam_{task_id}.json")
        opensees_force_json_path = _first_existing_path(paired_force)
    if not opensees_force_json_path and force_paths:
        opensees_force_json_path = force_paths[0]

    capacity_summary_path = _first_existing_path(
        os.path.join(output_dir, "capacity_check", "capacity_check_batch_summary.json"),
        os.path.join(output_dir, "capacity_check", "capacity_check_summary.json"),
        _latest_file(os.path.join(output_dir, "capacity_check", "**", "capacity_check_summary.json")),
    )
    drawing_index_path = _first_existing_path(
        os.path.join(output_dir, "deliverables", "drawings", "drawing_index.json")
    )
    design_manifest_path = _first_existing_path(
        os.path.join(output_dir, "deliverables", "design_manifest.json")
    )
    # 出图时被"按组记录在案"排除的设计组（例如该组验算未通过且无人工接受风险）。
    # 这些组本就不应出图，判断"图纸是否出全"时必须扣除，否则重跑会把一次
    # 合法的部分交付永久判为未完成（2026-09-16 示例项目K31 少数组验算失败即触发）。
    drawing_manifest = _load_optional_json(design_manifest_path) if design_manifest_path else {}
    drawing_excluded_group_ids = sorted(
        {
            str(item.get("design_group_id"))
            for item in (drawing_manifest.get("excluded_groups") or [])
            if isinstance(item, dict) and item.get("design_group_id")
        }
    )

    if layout_result_path:
        discovered["existing_layout_result_path"] = layout_result_path
        discovered["existing_layout_result"] = _load_optional_json(layout_result_path)
    if layout_revision_path:
        discovered["layout_revision_result_path"] = layout_revision_path
        discovered["final_layout_result_path"] = layout_revision_path
        discovered["layout_revision_result"] = _load_optional_json(layout_revision_path)
        discovered["final_layout_result"] = discovered["layout_revision_result"]
        discovered["layout_revision_completed"] = True
    if latest_collision_metrics_path:
        # 碰撞检测成果的恢复不应受"布跨是否仍在修正中"限制：
        # 人工接受布跨/自动收敛后已有 final_layout_result.json，此时若不恢复碰撞指标，
        # 协调器的 collision_detected_available 前置判定会失效，
        # 结构设计返修会被无谓阻断（2026-09-15 运行即因此转入人工复核）。
        discovered["collision_metrics_json_path"] = latest_collision_metrics_path
        discovered["collision_metrics"] = _load_optional_json(latest_collision_metrics_path)
        collision_report_path = _latest_file(
            os.path.join(output_dir, "collision_detection", "collision_report_*.json")
        )
        if collision_report_path:
            discovered["collision_report_json_path"] = collision_report_path
    if latest_revision_result_path:
        # 路径始终记录：人工复核账本里布跨类接受记录指向
        # revision_results/revision_design_round_N.json，恢复时必须能解析到该路径，
        # 否则人工"接受布跨风险"的决定会在重新运行时被静默丢弃。
        discovered["latest_revision_result_path"] = latest_revision_result_path
    if latest_revision_result_path and not layout_revision_path:
        discovered["latest_revision_result"] = _load_optional_json(latest_revision_result_path)
        match = re.search(r"revision_design_round_(\d+)\.json$", latest_revision_result_path)
        if match:
            discovered["iteration_index"] = int(match.group(1))
    if design_units_path:
        discovered["design_units_result_path"] = design_units_path
        discovered["design_units"] = _load_optional_json(design_units_path)
    if dimension_path:
        discovered["dimension_design_result_path"] = dimension_path
        discovered["dimension_design_result"] = _load_optional_json(dimension_path)
    if reinforcement_path:
        discovered["reinforcement_design_result_path"] = reinforcement_path
        discovered["reinforcement_design_result"] = _load_optional_json(reinforcement_path)
    if pier_group_path:
        discovered["pier_group_result_path"] = pier_group_path
        discovered["pier_group_result"] = _load_optional_json(pier_group_path)
    if structural_path:
        discovered["structural_design_result_path"] = structural_path
        discovered["structural_design_result"] = _load_optional_json(structural_path)
    if reinforcement_yaml_path:
        discovered["reinforcement_yaml_path"] = reinforcement_yaml_path
        discovered["reinforcement_yaml_paths"] = yaml_paths
    if opensees_force_json_path:
        discovered["opensees_force_json_path"] = opensees_force_json_path
        discovered["internal_force_output_paths"] = force_paths
    if capacity_summary_path:
        capacity_check_result = _load_optional_json(capacity_summary_path)
        discovered["capacity_check_summary_path"] = capacity_summary_path
        discovered["capacity_check_result"] = capacity_check_result
        discovered["check_result"] = capacity_check_result
        if (
            isinstance(capacity_check_result, dict)
            and capacity_check_result.get("check_type") == "cap_beam_capacity_envelope_batch"
        ):
            discovered["capacity_check_results"] = capacity_check_result.get("task_results") or []
            discovered["capacity_batch_status"] = {
                "expected_task_count": int(capacity_check_result.get("expected_task_count") or 0),
                "completed_task_count": int(capacity_check_result.get("completed_task_count") or 0),
                "failed_task_count": int(capacity_check_result.get("failed_task_count") or 0),
                "failed_task_ids": list(capacity_check_result.get("failed_task_ids") or []),
                "stage_complete": capacity_check_result.get("stage_complete") is True,
            }
    if drawing_index_path:
        drawing_index = _load_optional_json(drawing_index_path) or {}
        groups = drawing_index.get("groups") if isinstance(drawing_index, dict) else []
        groups = groups if isinstance(groups, list) else []

        def resolve_drawing_path(value: Any) -> Optional[str]:
            text = str(value or "").strip()
            if not text:
                return None
            path = os.path.normpath(
                text if os.path.isabs(text) else os.path.join(output_dir, text)
            )
            return path if os.path.isfile(path) else None

        cad_paths = [
            path
            for item in groups
            if isinstance(item, dict)
            for value in ([item.get("scr_path")] if item.get("scr_path") else item.get("scr_paths") or [])
            for path in [resolve_drawing_path(value)]
            if path
        ]
        preview_paths = [
            path
            for item in groups
            if isinstance(item, dict)
            for value in ([item.get("svg_path")] if item.get("svg_path") else item.get("svg_paths") or [])
            for path in [resolve_drawing_path(value)]
            if path
        ]
        generated_ids = [
            str(item.get("design_group_id"))
            for item in groups
            if isinstance(item, dict) and item.get("design_group_id")
        ]
        # 图纸索引只有在覆盖了当前全部"应当出图"的配筋设计组时才算"已完成"：
        # 否则一次失败的出图（只写出部分组）会被下一次运行当成既有成果而复用，
        # 导致"出图未完成却判定任务完成"（2026-09-16 示例项目K29 即为此现象）。
        # 但已按组记录在案、被排除出图的设计组不计入缺失：人工接受风险后
        # 少数组验算不过不影响其余组出图，也不应让流程永远无法收尾。
        expected_group_ids = _reinforcement_task_ids(discovered.get("reinforcement_design_result"))
        missing_group_ids = sorted(
            set(expected_group_ids) - set(generated_ids) - set(drawing_excluded_group_ids)
        )
        drawings_complete = bool(generated_ids) and not missing_group_ids
        discovered.update({
            "drawing_index_path": drawing_index_path,
            "design_manifest_path": design_manifest_path,
            "cad_script_paths": cad_paths,
            "drawing_preview_paths": preview_paths,
            "drawing_package_result": {
                "success": drawings_complete,
                "generated_group_ids": generated_ids,
                "skipped_group_ids": list(drawing_excluded_group_ids),
                "missing_group_ids": missing_group_ids,
                "excluded_group_ids": list(drawing_excluded_group_ids),
                "failed_groups": [],
                "drawing_index_path": drawing_index_path,
                "design_manifest_path": design_manifest_path or "",
                "cad_script_paths": cad_paths,
                "drawing_preview_paths": preview_paths,
            },
        })
    if plane_json_path:
        discovered["plane_json_path"] = plane_json_path
    if mask_path:
        discovered["mask_path"] = mask_path
    if pgw_path:
        discovered["pgw_path"] = pgw_path
    if png_path:
        discovered["png_path"] = png_path
    if jpg_path:
        discovered["jpg_path"] = jpg_path
    if obstacle_json_path:
        discovered["obstacle_json_path"] = obstacle_json_path
        discovered["obstacle_extractor_result"] = _load_optional_json(obstacle_json_path)

    from .human_review_store import load_persisted_human_review_state

    discovered.update(load_persisted_human_review_state(output_dir, discovered))

    return discovered


def _load_initial_state_from_settings(config_path: str) -> Dict[str, Any]:
    settings = load_settings(config_path)
    output_dir = pick(settings, "output_dir", default="outputs")
    discovered = _discover_existing_outputs(output_dir)

    existing_layout_result = discovered.get("existing_layout_result")
    existing_layout_revision_result = discovered.get("layout_revision_result")
    latest_revision_result = discovered.get("latest_revision_result")
    existing_design_units = discovered.get("design_units")
    existing_dimension_design_result = discovered.get("dimension_design_result")
    existing_reinforcement_design_result = discovered.get("reinforcement_design_result")

    layout_revision_result_path = _first_existing_path(
        discovered.get("layout_revision_result_path"),
        discovered.get("final_layout_result_path"),
    )
    layout_result_path = _first_existing_path(discovered.get("existing_layout_result_path"))
    reinforcement_yaml_path = discovered.get("reinforcement_yaml_path")
    opensees_force_json_path = discovered.get("opensees_force_json_path")
    data_path = pick(settings, "data_path", "base_dir")
    file_prefix = pick(settings, "file_prefix") or _infer_single_route_prefix(data_path)
    max_revision_rounds = int(pick(settings, "max_revision_rounds", default=3) or 3)
    restored_revision_round = int(discovered.get("iteration_index") or 0)
    restored_manual_review = bool(
        latest_revision_result
        and discovered.get("collision_metrics")
        and restored_revision_round >= max_revision_rounds
        and not discovered.get("layout_revision_completed")
    )

    return {
        "config_path": config_path,
        "existing_layout_result": existing_layout_result,
        "layout_revision_result": discovered.get("layout_revision_result") or existing_layout_revision_result,
        "final_layout_result": discovered.get("final_layout_result") or existing_layout_revision_result,
        "layout_revision_completed": bool(
            discovered.get("layout_revision_completed") or existing_layout_revision_result
        ),
        "layout_revision_result_path": discovered.get("layout_revision_result_path") or layout_revision_result_path,
        "final_layout_result_path": discovered.get("final_layout_result_path") or layout_revision_result_path,
        "layout_result_path": layout_result_path,
        "latest_revision_result": latest_revision_result,
        "latest_revision_result_path": discovered.get("latest_revision_result_path"),
        "layout_result": existing_layout_revision_result or latest_revision_result or existing_layout_result,
        "iteration_index": discovered.get("iteration_index", 0),
        "collision_metrics": discovered.get("collision_metrics"),
        "collision_metrics_json_path": discovered.get("collision_metrics_json_path"),
        "task_status": (
            "manual_review_required"
            if restored_manual_review
            else discovered.get("task_status")
        ),
        "message": (
            "发现达到自动修正轮次上限的既有方案，等待人工复核。"
            if restored_manual_review
            else None
        ),
        "unresolved_manual_review": restored_manual_review,
        "human_override": discovered.get("human_override"),
        "human_review_history": discovered.get("human_review_history") or [],
        "human_review_decision_path": discovered.get("human_review_decision_path"),
        "accepted_risks": discovered.get("accepted_risks") or [],
        "design_units": existing_design_units,
        "dimension_design_result": existing_dimension_design_result,
        "pier_group_result": discovered.get("pier_group_result"),
        "pier_group_result_path": discovered.get("pier_group_result_path"),
        "reinforcement_design_result": existing_reinforcement_design_result,
        "structural_design_result": discovered.get("structural_design_result"),
        "structural_design_result_path": discovered.get("structural_design_result_path"),
        "capacity_check_result": discovered.get("capacity_check_result"),
        "capacity_check_results": discovered.get("capacity_check_results"),
        "capacity_batch_status": discovered.get("capacity_batch_status"),
        "reinforcement_batch_status": discovered.get("reinforcement_batch_status"),
        "check_result": discovered.get("check_result"),
        "capacity_check_summary_path": discovered.get("capacity_check_summary_path"),
        "drawing_package_result": discovered.get("drawing_package_result"),
        "drawing_index_path": discovered.get("drawing_index_path"),
        "design_manifest_path": discovered.get("design_manifest_path"),
        "cad_script_paths": discovered.get("cad_script_paths") or [],
        "drawing_preview_paths": discovered.get("drawing_preview_paths") or [],
        "agent_log": AgentLogger().to_dict(),
        "data_path": data_path,
        "file_prefix": file_prefix,
        "structures_filename": pick(settings, "structures_filename"),
        "input_drawing_path": pick(settings, "input_drawing_path", "drawing_path"),
        "drawing_path": pick(settings, "drawing_path"),
        "output_dir": output_dir,
        "plane_json_path": discovered.get("plane_json_path"),
        "mask_path": discovered.get("mask_path"),
        "pgw_path": discovered.get("pgw_path"),
        "png_path": discovered.get("png_path"),
        "jpg_path": discovered.get("jpg_path"),
        "obstacle_json_path": discovered.get("obstacle_json_path"),
        "obstacle_extractor_result": discovered.get("obstacle_extractor_result"),
        "config_file": pick(settings, "config_file"),
        "checkpoint_file": pick(settings, "checkpoint_file"),
        "device": pick(settings, "device", default="cuda:0"),
        "converter_path": pick(settings, "converter_path"),
        "dxf_output_dir": pick(settings, "dxf_output_dir", default=os.path.join(output_dir, "dxf")),
        "layer_config_json": pick(settings, "layer_config_json"),
        "swap_xy": pick(settings, "swap_xy", default=True),
        "buffer": pick(settings, "buffer", default=200.0),
        "resolution": pick(settings, "resolution", default=0.5),
        "max_image_pixels": pick(settings, "max_image_pixels", default=20000),
        "expand_inserts": pick(settings, "expand_inserts", default=False),
        "patch_size": pick(settings, "patch_size", default=256),
        "stride": pick(settings, "stride", default=256),
        "background_id": pick(settings, "background_id", default=0),
        "convert_png_to_jpg": pick(settings, "convert_png_to_jpg", default=True),
        "jpg_quality": pick(settings, "jpg_quality", default=95),
        "roadbed_type": pick(settings, "roadbed_type", default="integrated"),
        "section_name": pick(settings, "section_name", default="section"),
        "route_scan_config_json": pick(settings, "route_scan_config_json"),
        "scan_width_integrated": pick(settings, "scan_width_integrated", default=26.5),
        "scan_width_separated": pick(settings, "scan_width_separated", default=13.25),
        "sample_step": pick(settings, "sample_step", default=0.5),
        "step_k": pick(settings, "step_k", default=0.5),
        "clip_to_route_range": pick(settings, "clip_to_route_range", default=True),
        "save_visualization": pick(settings, "save_visualization", default=True),
        "show_visualization": pick(settings, "show_visualization", default=False),
        "preview_wait_ms": pick(settings, "preview_wait_ms", default=0),
        "prompt_manifest_path": pick(settings, "manifest_path", default="prompts/manifest.yaml"),
        "prompt_trace": [],
        "standards_path": pick(settings, "standards_path"),
        "few_shots_dir": pick(settings, "few_shots_dir"),
        "few_shots_k": pick(settings, "few_shots_k", default=3),
        "dimension_few_shots_dir": pick(settings, "dimension_few_shots_dir"),
        "reinforcement_few_shots_dir": pick(settings, "reinforcement_few_shots_dir"),
        "collision_output_dir": pick(
            settings,
            "collision_output_dir",
            default=os.path.join(output_dir, "collision_detection"),
        ),
        "collision_offset_left_m": pick(settings, "collision_offset_left_m", default=7.0),
        "collision_offset_right_m": pick(settings, "collision_offset_right_m", default=7.0),
        "collision_column_half_spacing_m": pick(settings, "collision_column_half_spacing_m", default=3.45),
        "collision_check_radius_m": pick(settings, "collision_check_radius_m", default=1.25),
        "collision_mask_threshold": pick(settings, "collision_mask_threshold", default=127),
        "collision_save_visualization": pick(settings, "collision_save_visualization", default=True),
        "max_revision_rounds": max_revision_rounds,
        "max_react_steps": pick(settings, "max_react_steps", default=16),
        "llm_max_format_repairs": pick(settings, "llm_max_format_repairs", default=1),
        "llm_max_regenerations": pick(settings, "llm_max_regenerations", default=2),
        "reinforcement_yaml_path": reinforcement_yaml_path,
        "reinforcement_yaml_paths": discovered.get("reinforcement_yaml_paths"),
        "opensees_force_json_path": opensees_force_json_path,
        "internal_force_output_paths": discovered.get("internal_force_output_paths"),
        "opensees_script_path": pick(settings, "opensees_script_path"),
        "capacity_check_script_path": pick(settings, "capacity_check_script_path"),
        "capacity_check_output_dir": pick(
            settings,
            "capacity_check_output_dir",
            default=os.path.join(output_dir, "capacity_check"),
        ),
        "capacity_check_combination_name": pick(settings, "capacity_check_combination_name", default="ULS_basic"),
        "capacity_check_gamma_0": pick(settings, "capacity_check_gamma_0", default=1.10),
        "capacity_check_apply_gamma0": pick(settings, "capacity_check_apply_gamma0", default=True),
        "auto_run_opensees_if_missing": pick(settings, "auto_run_opensees_if_missing", default=False),
        "max_modeling_check_steps": pick(settings, "max_modeling_check_steps", default=10),
        "max_check_revision_rounds": pick(settings, "max_check_revision_rounds", default=2),
    }


def _resume_recovery_patch(
    graph: Any,
    config: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """断点恢复时，从磁盘补回 checkpoint 里缺失的工程成果字段。

    背景：恢复运行只从 checkpoint 取值，若某成果在 checkpoint 中为 None（例如历史版本
    漏存、或该成果从未写进图状态），而磁盘上其实存在，就会让协调器的前置成果校验
    失效（2026-09-15 的 示例项目K29 运行即因 collision_metrics 为空而阻断结构返修）。

    规则（保守）：
    - 只补白名单里的工程成果字段；
    - checkpoint 里已有值的一律不覆盖；
    - 属于当前 invalidated_artifacts 的成果不补回（避免把"已失效待重做"的成果
      伪装成可用）。
    """
    try:
        values = dict(getattr(graph.get_state(config), "values", {}) or {})
    except Exception:
        return {}
    if not values:
        return {}

    from .dependencies import ARTIFACT_RUNTIME_KEYS  # 局部导入避免顶层耦合

    blocked: set = set()
    for artifact in values.get("invalidated_artifacts") or []:
        blocked.update(ARTIFACT_RUNTIME_KEYS.get(str(artifact), ()))

    patch: Dict[str, Any] = {}
    for key in _RESUME_RECOVERABLE_KEYS:
        if key in blocked:
            continue
        current = values.get(key)
        if current not in (None, "", [], {}):
            continue
        value = state.get(key)
        if value in (None, "", [], {}):
            continue
        patch[key] = value
    return patch


# 断点恢复允许从磁盘补回的工程成果字段（不含流程状态字段与人工复核记录）。
_RESUME_RECOVERABLE_KEYS = frozenset(
    {
        "existing_layout_result",
        "existing_layout_result_path",
        "layout_result",
        "layout_revision_result",
        "layout_revision_result_path",
        "final_layout_result",
        "final_layout_result_path",
        "layout_revision_completed",
        "latest_revision_result",
        "latest_revision_result_path",
        "collision_metrics",
        "collision_metrics_json_path",
        "collision_report_json_path",
        "design_units",
        "design_units_result_path",
        "dimension_design_result",
        "dimension_design_result_path",
        "reinforcement_design_result",
        "reinforcement_design_result_path",
        "pier_group_result",
        "pier_group_result_path",
        "structural_design_result",
        "structural_design_result_path",
        "reinforcement_yaml_path",
        "reinforcement_yaml_paths",
        "opensees_force_json_path",
        "internal_force_output_paths",
        "capacity_check_result",
        "check_result",
        "capacity_check_summary_path",
        "capacity_check_results",
        "capacity_batch_status",
        "drawing_index_path",
        "design_manifest_path",
    }
)


def run_agent(
    user_request: str,
    config_path: str = "config/settings.yaml",
    initial_state: Optional[Dict[str, Any]] = None,
    use_graph_v2: bool = False,
    thread_id: Optional[str] = None,
    resume: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """运行多智能体桥梁设计流程。

    use_graph_v2=True 时使用显式顶层图（DesignCoordinator 路由），否则使用旧队列式图。
    """
    state: Dict[str, Any] = _load_initial_state_from_settings(config_path)
    if initial_state:
        state.update(initial_state)
    state["config_path"] = config_path
    resolved_thread_id = thread_id or str(uuid4())
    state["thread_id"] = resolved_thread_id

    if resume is None:
        state["user_request"] = user_request
        station_params = regex_extract_stations(user_request)
        if not state.get("start_station") and station_params.get("start_station"):
            state["start_station"] = station_params["start_station"]
        if not state.get("end_station") and station_params.get("end_station"):
            state["end_station"] = station_params["end_station"]

    if use_graph_v2:
        from .checkpointing import open_sqlite_checkpointer
        from .graph_v2 import DEFAULT_RECURSION_LIMIT, build_graph_v2

        def controller_invoke(system: str, user: str) -> str:
            llm = get_controller_llm(config_path)
            return llm.invoke([("system", system), ("user", user)]).content

        output_dir = state.get("output_dir") or "outputs"
        # 显式给出单次 invoke 的超级步上限：langgraph 默认上限极大（1.2.x 为 10007），
        # 一旦调度出现死循环就会长时间空转并持续写 checkpoint，这里把它变成可见的
        # GraphRecursionError（checkpoint 与已生成成果均保留，可继续 resume）。
        recursion_limit = int(state.get("max_graph_steps") or 0) or DEFAULT_RECURSION_LIMIT
        invoke_config = {
            "configurable": {"thread_id": resolved_thread_id},
            "recursion_limit": max(1, recursion_limit),
        }
        with open_sqlite_checkpointer(output_dir) as (checkpointer, checkpoint_path):
            graph = build_graph_v2(
                llm_invoke=controller_invoke,
                checkpointer=checkpointer,
            )
            if resume is None:
                graph_input = state
            else:
                recovery = _resume_recovery_patch(graph, invoke_config, state)
                graph_input = (
                    Command(resume=resume, update=recovery)
                    if recovery
                    else Command(resume=resume)
                )
            try:
                result = graph.invoke(graph_input, config=invoke_config)
            except Exception:
                # 先把真实异常打全，避免外层收尾动作把关键信息盖掉。
                logging.getLogger(__name__).exception(
                    "graph-v2 运行异常（thread_id=%s）", resolved_thread_id
                )
                raise
        result = dict(result)
        result["thread_id"] = resolved_thread_id
        result["checkpoint_path"] = checkpoint_path
    else:
        if resume is not None:
            raise ValueError("断点恢复仅支持 --graph-v2。")
        graph = build_graph()
        result = graph.invoke(state)

    run_logger = AgentLogger.from_state(result)
    log_path = run_logger.save(result.get("output_dir") or state.get("output_dir") or "outputs")
    result["agent_log_path"] = log_path

    # 同步把日志路径写进日志本身
    run_logger.log_file(file_type="agent_log_path", path=log_path, producer="run_agent")
    result["agent_log"] = run_logger.to_dict()

    return result
