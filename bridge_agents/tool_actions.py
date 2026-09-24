from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .capacity_batch import (
    aggregate_capacity_check_results,
    collect_capacity_check_tasks,
    safe_capacity_task_dir_name,
)
from .joint_reinforcement import aggregate_axial_check_results
from .pier_group import (
    build_side_route_map,
    collect_vertical_profiles,
    compute_pier_groups,
)
from .state import AgentState
from .prompt_registry import render_prompt
from .utils import (
    extract_json_object,
    get_controller_llm,
    get_revision_llm,
    get_structural_llm,
    invoke_tool,
    json_safe,
    metric_float,
    state_json,
    tool_failed,
    write_json,
    write_text,
)

logger = logging.getLogger(__name__)
_SAMPLE_SELECTOR_CACHE: Dict[str, Any] = {}

# ============================================================
# 可选导入：沿用你当前 tools 目录下的真实工具
# ============================================================
try:
    from tools.data_loader_tool import load_and_crop_project_data, load_and_crop_multi_route_project_data
except Exception:  # pragma: no cover
    load_and_crop_project_data = None
    load_and_crop_multi_route_project_data = None

try:
    from tools.drawing_mask_preprocess_tool import drawing_crop_and_mask_tool
except Exception:  # pragma: no cover
    drawing_crop_and_mask_tool = None

try:
    from tools.obstacle_semantic_extractor_tool import obstacle_semantic_extractor_tool
except Exception:  # pragma: no cover
    obstacle_semantic_extractor_tool = None

try:
    from tools.sample_selector_tool import SampleSelectorTool
except Exception:  # pragma: no cover
    SampleSelectorTool = None

try:
    from tools.design_generation_tool import generate_design
except Exception:  # pragma: no cover
    generate_design = None

try:
    from tools.collision_detection_tool import collision_detection_tool
except Exception:  # pragma: no cover
    collision_detection_tool = None

try:
    from tools.revision_instruction_tool import generate_revision_instruction as revision_instruction_tool
except Exception:  # pragma: no cover
    revision_instruction_tool = None

try:
    from tools.layout_revision_prompt_tool import layout_revision_prompt_builder_tool
except Exception:  # pragma: no cover
    layout_revision_prompt_builder_tool = None

# 后续结构设计工具：当前允许不存在，便于先搭框架
try:
    from tools.design_unit_extractor_tool import extract_design_units_tool
except Exception:  # pragma: no cover
    extract_design_units_tool = None

try:
    from tools.dimension_design_tool import dimension_design_tool
except Exception:  # pragma: no cover
    dimension_design_tool = None

try:
    from tools.reinforcement_design_tool import reinforcement_design_tool
except Exception:  # pragma: no cover
    reinforcement_design_tool = None

try:
    from tools.capacity_check_tool import capacity_check_tool
except Exception:  # pragma: no cover
    capacity_check_tool = None

try:
    from tools.reinforcement_drawing_tool import reinforcement_drawing_tool
except Exception:  # pragma: no cover
    reinforcement_drawing_tool = None

# ============================================================
# 设计输入组织与结果提取
# ============================================================
def _extract_obstacle_payload(obstacle_result: Optional[Dict[str, Any]]) -> Any:
    if not obstacle_result:
        return []
    for key in ["obstacle_grouped", "obstacle_data", "obstacle_list", "obstacles"]:
        if obstacle_result.get(key):
            return obstacle_result[key]
    if isinstance(obstacle_result, dict):
        return obstacle_result
    return []


def _clean_route_for_prompt(route_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(route_data, dict):
        return {}

    cleaned = {
        "线路名称": route_data.get("线路名称"),
        "线路编号": route_data.get("线路编号"),
        "桩号范围": route_data.get("截取范围", route_data.get("桩号范围")),
        "平曲线结构": route_data.get("平曲线结构", []),
        "横断面信息": route_data.get("横断面信息", []),
    }

    vertical = route_data.get("纵断面结构")
    if isinstance(vertical, dict):
        cleaned["纵断面结构"] = {
            "桩号范围": vertical.get("桩号范围", cleaned.get("桩号范围")),
            "设计线高程序列": vertical.get("设计线高程序列", []),
            "地形线高程序列": vertical.get("地形线高程序列", []),
        }
    else:
        cleaned["纵断面结构"] = {
            "桩号范围": cleaned.get("桩号范围"),
            "设计线高程序列": route_data.get("设计线高程序列", []),
            "地形线高程序列": route_data.get("地形线高程序列", []),
        }

    return {k: v for k, v in cleaned.items() if v not in [None, "", [], {}]}


def build_design_input_from_state(state: AgentState) -> Dict[str, Any]:
    cropped = state.get("cropped_data") or {}
    obstacle_result = state.get("obstacle_extractor_result") or {}
    if not obstacle_result and state.get("obstacle_json_path"):
        obstacle_result = _load_json_if_exists(state.get("obstacle_json_path"))

    semantic_obstacles = _extract_obstacle_payload(obstacle_result)
    original_obstacles = cropped.get("障碍物信息", cropped.get("构造物信息", [])) if isinstance(cropped, dict) else []
    obstacle_payload = semantic_obstacles if semantic_obstacles else original_obstacles

    wrapped: Dict[str, Any] = {
        "线路类型": cropped.get("线路类型", "整体式") if isinstance(cropped, dict) else "整体式",
    }

    if obstacle_payload:
        wrapped["障碍物信息"] = obstacle_payload

    if isinstance(cropped, dict) and "线路数据" not in cropped:
        wrapped["K"] = _clean_route_for_prompt(cropped)

    route_data_map = cropped.get("线路数据") if isinstance(cropped, dict) else None
    if isinstance(route_data_map, dict):
        for route_key in ["K", "Z", "Z1", "Z2"]:
            if route_key in route_data_map:
                wrapped[route_key] = _clean_route_for_prompt(route_data_map[route_key])

    if isinstance(cropped, dict):
        for route_key in ["Z", "Z1", "Z2"]:
            if route_key in cropped and route_key not in wrapped:
                wrapped[route_key] = _clean_route_for_prompt(cropped[route_key])

    return wrapped


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 文件顶层必须是对象: {path}")
    return data


def extract_design_payload(design_result: Any) -> Any:
    """从工具返回中提取真正的设桥布跨 JSON。"""
    if not isinstance(design_result, dict):
        return design_result
    if "设桥总览" in design_result:
        return design_result

    for key in ["design_json", "result", "data", "design", "layout_result", "current_layout_result"]:
        value = design_result.get(key)
        if isinstance(value, dict) and "设桥总览" in value:
            return value
        if isinstance(value, dict):
            nested = value.get("design_json")
            if isinstance(nested, dict) and "设桥总览" in nested:
                return nested

    candidate_keys = [
        "saved_json_path", "design_json_path", "output_json_path", "result_json_path",
        "json_path", "final_json_path", "output_path", "saved_output_json_path",
    ]
    output_files = design_result.get("output_files") if isinstance(design_result.get("output_files"), dict) else {}
    for source in [design_result, output_files]:
        for key in candidate_keys:
            path = source.get(key)
            if path and os.path.exists(str(path)):
                loaded = _load_json_file(str(path))
                if "设桥总览" in loaded:
                    return loaded

    for text_key in ["design_text", "response_text", "raw_response", "revision_result_raw"]:
        text_value = design_result.get(text_key)
        if isinstance(text_value, str) and text_value.strip():
            parsed = extract_json_object(text_value)
            if "设桥总览" in parsed:
                return parsed

    raise ValueError("design_result 中未找到可读取的设桥 JSON。")


def get_sample_selector(few_shots_dir: str) -> Any:
    if SampleSelectorTool is None:
        raise ImportError("未能导入 SampleSelectorTool，请检查 tools/sample_selector_tool.py。")
    if few_shots_dir not in _SAMPLE_SELECTOR_CACHE:
        _SAMPLE_SELECTOR_CACHE[few_shots_dir] = SampleSelectorTool(few_shots_dir=few_shots_dir)
    return _SAMPLE_SELECTOR_CACHE[few_shots_dir]


# ============================================================
# InitialDesignAgent 使用的固定流程工具动作
# ============================================================
def _discover_route_prefixes(data_path: str) -> List[str]:
    """Scan data_path for PM files and return sorted unique file prefixes.

    Returns empty list when data_path is missing or contains no PM files.
    """
    if not data_path or not os.path.isdir(data_path):
        return []
    import glob as _glob
    candidates: List[str] = []
    for pattern in ["*.pm", "*.PM"]:
        for path in _glob.glob(os.path.join(data_path, pattern)):
            if os.path.isfile(path):
                candidates.append(os.path.splitext(os.path.basename(path))[0])
    return sorted(set(candidates))


def load_data_action(state: AgentState) -> Dict[str, Any]:
    if load_and_crop_project_data is None:
        return {"error": "未能导入 load_and_crop_project_data，请检查 tools/data_loader_tool.py。"}
    try:
        data_path = str(state.get("data_path") or "")
        file_prefix = str(state.get("file_prefix") or "")
        start_station = state.get("start_station")
        end_station = state.get("end_station")
        output_dir = state.get("output_dir")

        # ---- multi-route auto-detection ----
        prefixes = _discover_route_prefixes(data_path)
        use_multi_route = (
            len(prefixes) >= 2
            and load_and_crop_multi_route_project_data is not None
        )

        if use_multi_route:
            route_ids = ["K", "Z", "Z1", "Z2"]
            route_configs: List[Dict[str, str]] = []
            for idx, pfx in enumerate(prefixes):
                rid = route_ids[idx] if idx < len(route_ids) else f"Z{idx}"
                route_configs.append({
                    "route_id": rid,
                    "base_dir": data_path,
                    "file_prefix": pfx,
                })

            route_type = "分离式" if len(prefixes) == 2 else "多线路"
            logger.info(
                "检测到 %d 条线路 (%s)，自动切换多线路模式: %s",
                len(prefixes),
                ", ".join(prefixes),
                {c["route_id"]: c["file_prefix"] for c in route_configs},
            )
            multi_payload: Dict[str, Any] = {
                "route_configs": route_configs,
                "route_type": route_type,
                "start_station": start_station,
                "end_station": end_station,
                "output_dir": output_dir,
            }
            structures_filename = state.get("structures_filename")
            if structures_filename is not None:
                multi_payload["structures_filename"] = structures_filename
            result = invoke_tool(load_and_crop_multi_route_project_data, multi_payload)
        else:
            payload: Dict[str, Any] = {
                "base_dir": data_path,
                "file_prefix": file_prefix,
                "start_station": start_station,
                "end_station": end_station,
                "output_dir": output_dir,
            }
            structures_filename = state.get("structures_filename")
            if structures_filename is not None:
                payload["structures_filename"] = structures_filename
            result = invoke_tool(load_and_crop_project_data, payload)

        if tool_failed(result):
            return {"data_loader_result": result, "error": result.get("error", "数据加载失败") if isinstance(result, dict) else "数据加载失败"}
        output_files = result.get("output_files", {}) if isinstance(result, dict) else {}
        plane_paths = result.get("plane_paths", {}) if isinstance(result.get("plane_paths"), dict) else {}
        return {
            "data_loader_result": result,
            "cropped_data": result,
            "plane_paths": plane_paths if plane_paths else None,  # multi-route: {K: path, Z: path}
            "plane_json_path": (
                state.get("plane_json_path")
                or output_files.get("plane_json_path")
                or result.get("plane_json_path")
                or result.get("plane_path")
                or plane_paths.get("K")
                or result.get("平曲线文件")
            ),
            "error": None,
        }
    except Exception as e:
        logger.error("数据加载失败：%s", e, exc_info=True)
        return {"error": str(e)}


def drawing_crop_and_mask_action(state: AgentState) -> Dict[str, Any]:
    if drawing_crop_and_mask_tool is None:
        return {"error": "未能导入 drawing_crop_and_mask_tool，请检查 tools/drawing_mask_preprocess_tool.py。"}
    try:
        result = invoke_tool(drawing_crop_and_mask_tool, {
            "design_state_json": state_json(dict(state)),
            "input_drawing_path": state.get("input_drawing_path") or state.get("drawing_path"),
            "plane_json_path": state.get("plane_json_path"),
            "start_k": state.get("start_station"),
            "end_k": state.get("end_station"),
            "output_dir": state.get("output_dir"),
            "config_file": state.get("config_file"),
            "checkpoint_file": state.get("checkpoint_file"),
            "device": state.get("device") or "cuda:0",
            "converter_path": state.get("converter_path"),
            "dxf_output_dir": state.get("dxf_output_dir"),
            "swap_xy": True if state.get("swap_xy") is None else state.get("swap_xy"),
            "buffer": state.get("buffer") or 200.0,
            "resolution": state.get("resolution") or 0.5,
            "max_image_pixels": state.get("max_image_pixels") or 20000,
            "expand_inserts": bool(state.get("expand_inserts") or False),
            "patch_size": state.get("patch_size") or 256,
            "stride": state.get("stride") or 256,
            "background_id": state.get("background_id") or 0,
            "layer_config_json": state.get("layer_config_json"),
        })
        if tool_failed(result):
            return {"drawing_mask_result": result, "error": result.get("error", "图纸裁剪与掩码生成失败") if isinstance(result, dict) else "图纸裁剪与掩码生成失败"}
        output_files = result.get("output_files", {}) if isinstance(result, dict) else {}
        return {
            "drawing_mask_result": result,
            "png_path": output_files.get("png_path") or result.get("png_path"),
            "jpg_path": output_files.get("jpg_path") or result.get("jpg_path"),
            "pgw_path": output_files.get("pgw_path") or result.get("pgw_path"),
            "mask_path": output_files.get("mask_path") or result.get("mask_path"),
            "plane_json_path": result.get("plane_json_path") or state.get("plane_json_path"),
            "error": None,
        }
    except Exception as e:
        logger.error("图纸裁剪与掩码生成失败：%s", e, exc_info=True)
        return {"error": str(e)}


def obstacle_semantic_extractor_action(state: AgentState) -> Dict[str, Any]:
    if obstacle_semantic_extractor_tool is None:
        return {"error": "未能导入 obstacle_semantic_extractor_tool，请检查 tools/obstacle_semantic_extractor_tool.py。"}
    try:
        result = invoke_tool(obstacle_semantic_extractor_tool, {
            "preprocessed_route_data_json": state_json(state.get("cropped_data") or state.get("data_loader_result") or {}),
            "mask_path": state.get("mask_path"),
            "pgw_path": state.get("pgw_path"),
            "output_dir": state.get("output_dir"),
            "roadbed_type": state.get("roadbed_type") or "integrated",
            "start_k": state.get("start_station"),
            "end_k": state.get("end_station"),
            "section_name": state.get("section_name") or "section",
            "route_scan_config_json": state.get("route_scan_config_json"),
            "scan_width_integrated": state.get("scan_width_integrated") or 26.5,
            "scan_width_separated": state.get("scan_width_separated") or 13.25,
            "sample_step": state.get("sample_step") or 0.5,
            "step_k": state.get("step_k") or 0.5,
            "clip_to_route_range": True if state.get("clip_to_route_range") is None else state.get("clip_to_route_range"),
            "save_visualization": True if state.get("save_visualization") is None else state.get("save_visualization"),
            "show_visualization": bool(state.get("show_visualization") or False),
            "preview_wait_ms": state.get("preview_wait_ms") or 0,
        })
        if tool_failed(result):
            return {"obstacle_extractor_result": result, "error": result.get("error", "障碍物语义提取失败") if isinstance(result, dict) else "障碍物语义提取失败"}
        output_files = result.get("output_files", {}) if isinstance(result, dict) else {}
        plane_paths = result.get("plane_paths", {}) if isinstance(result.get("plane_paths"), dict) else {}
        return {
                "obstacle_extractor_result": result,
                "obstacle_json_path": output_files.get("obstacle_json_path") or result.get("obstacle_json_path"),
                "merged_visualization_path": output_files.get("merged_visualization_path") or result.get("merged_visualization_path"),
                "plane_json_path": state.get("plane_json_path") or plane_paths.get("K"),
                "mask_path": result.get("mask_path") or state.get("mask_path"),
                "pgw_path": result.get("pgw_path") or state.get("pgw_path"),
                "error": None,
        }
    except Exception as e:
        logger.error("障碍物语义提取失败：%s", e, exc_info=True)
        return {"error": str(e)}


def select_samples_action(state: AgentState) -> Dict[str, Any]:
    try:
        few_shots_dir = state.get("few_shots_dir")
        if not few_shots_dir:
            return {"few_shots": [], "error": "few_shots_dir 为空。"}
        design_input = state.get("design_input") or build_design_input_from_state(state)
        selector = get_sample_selector(few_shots_dir)
        result = selector._run(design_input=design_input, k=int(state.get("few_shots_k") or 3))
        return {"few_shots": result, "design_input": design_input, "error": None}
    except Exception as e:
        logger.error("样本选择失败：%s", e, exc_info=True)
        return {"few_shots": [], "error": str(e)}


def generate_layout_design_action(state: AgentState) -> Dict[str, Any]:
    if generate_design is None:
        return {"error": "未能导入 generate_design，请检查 tools/design_generation_tool.py。"}
    try:
        design_input = state.get("design_input") or build_design_input_from_state(state)
        result = invoke_tool(generate_design, {
            "design_input": design_input,
            "few_shots": state.get("few_shots", []),
            "standards_path": state.get("standards_path"),
            "config_path": state.get("config_path"),
            "output_dir": state.get("output_dir"),
        })
        error = None
        if isinstance(result, dict) and result.get("success") is False:
            error = result.get("error") or "设桥布跨生成失败。"
        return {
            "design_input": design_input,
            "design_result": result,
            "layout_result": extract_design_payload(result),
            "error": error,
        }
    except Exception as e:
        logger.error("设桥布跨设计生成失败：%s", e, exc_info=True)
        return {"error": str(e)}


# ============================================================
# LayoutRevisionAgent 使用的工具动作
# ============================================================
def passes_collision_threshold(metrics: Dict[str, Any]) -> bool:
    return (
        metric_float(metrics, "total_intrusion_depth_columns") <= 10.0
        and metric_float(metrics, "avg_intrusion_depth_columns") <= 1.2
        and metric_float(metrics, "avg_overlap_ratio_columns") <= 0.5
        and metric_float(metrics, "conflict_column_rate") <= 0.05
    )


def run_collision_detection_action(state: AgentState) -> Dict[str, Any]:
    if collision_detection_tool is None:
        return {"error": "未能导入 collision_detection_tool，请检查 tools/collision_detection_tool.py。"}
    try:
        design_result = state.get("layout_result") or state.get("design_result") or state.get("existing_layout_result")
        design_result = extract_design_payload(design_result)
        if not design_result:
            return {"error": "无设桥布跨结果，无法进行碰撞检测。"}
        for key in ["plane_json_path", "mask_path", "pgw_path"]:
            if not state.get(key):
                return {"error": f"缺少 {key}。"}

        collision_output_dir = state.get("collision_output_dir") or os.path.join(state.get("output_dir") or ".", "collision_detection")
        result = invoke_tool(collision_detection_tool, {
            "design_result": design_result,
            "plane_paths": state.get("plane_paths") or {},
            "plane_json_path": state.get("plane_json_path"),
            "mask_path": state.get("mask_path"),
            "pgw_path": state.get("pgw_path"),
            "output_dir": collision_output_dir,
            "offset_left_m": state.get("collision_offset_left_m") or 7.0,
            "offset_right_m": state.get("collision_offset_right_m") or 7.0,
            "column_half_spacing_m": state.get("collision_column_half_spacing_m") or 3.45,
            "check_radius_m": state.get("collision_check_radius_m") or 1.25,
            "mask_threshold": state.get("collision_mask_threshold") or 127,
            "save_visualization": True if state.get("collision_save_visualization") is None else state.get("collision_save_visualization"),
        })
        if tool_failed(result):
            return {"collision_result": result, "error": result.get("error", "碰撞检测失败") if isinstance(result, dict) else "碰撞检测失败"}

        output_files = result.get("output_files", {}) if isinstance(result, dict) else {}
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        collision_items = result.get("collision_items", []) if isinstance(result, dict) else []
        return {
            "layout_result": design_result,
            "collision_result": result,
            "verification_result": {
                "status": "collision_detected" if result.get("has_collision") else "passed",
                "has_collision": result.get("has_collision"),
                "metrics": metrics,
                "collision_items": collision_items,
                "message": result.get("message"),
                "output_files": output_files,
            },
            "collision_metrics": metrics,
            "collision_items": collision_items,
            "collision_report_json_path": output_files.get("report_json_path"),
            "collision_report_txt_path": output_files.get("report_txt_path"),
            "collision_metrics_json_path": output_files.get("metrics_json_path"),
            "collision_visualization_path": output_files.get("visualization_path"),
            "error": None,
        }
    except Exception as e:
        logger.error("碰撞检测失败：%s", e, exc_info=True)
        return {"error": str(e)}


def generate_revision_instruction_action(state: AgentState) -> Dict[str, Any]:
    if revision_instruction_tool is None:
        return {"error": "未能导入 revision_instruction_tool，请检查 tools/revision_instruction_tool.py。"}
    try:
        report_path = state.get("collision_report_json_path")
        layout_payload = state.get("layout_result") or state.get("design_result") or state.get("existing_layout_result")
        layout_payload = extract_design_payload(layout_payload)
        collision_items = list(state.get("collision_items") or [])
        if not collision_items and report_path and os.path.exists(str(report_path)):
            report_data = _load_json_file(str(report_path))
            if isinstance(report_data, list):
                collision_items = report_data
            elif isinstance(report_data, dict):
                collision_items = list(report_data.get("collision_items") or [])
        result = invoke_tool(revision_instruction_tool, {
            "collision_report_path": None,
            "collision_report": {
                "collision_items": collision_items,
                "metrics": state.get("collision_metrics") or {},
            },
            "layout_result": layout_payload,
        })
        if tool_failed(result):
            return {"error": result.get("error", "修正指令生成失败") if isinstance(result, dict) else "修正指令生成失败"}
        instruction_text = result.get("revision_instruction")
        if not instruction_text:
            return {"error": "revision_instruction_tool 未返回 revision_instruction。"}
        human_feedback = str(state.get("human_review_feedback") or "").strip()
        if human_feedback:
            instruction_text = (
                instruction_text.rstrip()
                + "\n\n## 人工复核补充要求\n"
                + human_feedback
                + "\n"
            )

        iteration_index = int(state.get("iteration_index") or 0)
        output_dir = state.get("output_dir") or os.getcwd()
        instruction_path = os.path.join(output_dir, "revision_instructions", f"revision_instruction_round_{iteration_index + 1}.txt")
        write_text(instruction_path, instruction_text)
        return {
            "revision_instruction": instruction_text,
            "revision_instruction_path": instruction_path,
            "revision_engineering_metrics": result.get("engineering_metrics"),
            "error": None,
        }
    except Exception as e:
        logger.error("修正指令生成失败：%s", e, exc_info=True)
        return {"error": str(e)}


def build_revision_prompt_action(state: AgentState) -> Dict[str, Any]:
    if layout_revision_prompt_builder_tool is None:
        return {"error": "未能导入 layout_revision_prompt_builder_tool，请检查 tools/layout_revision_prompt_tool.py。"}
    try:
        result = invoke_tool(layout_revision_prompt_builder_tool, {
            "design_state_json": state_json(dict(state)),
        })
        if tool_failed(result):
            return {"error": result.get("error", "修正 Prompt 构造失败") if isinstance(result, dict) else "修正 Prompt 构造失败"}
        return {
            "revision_prompt": result.get("revision_prompt"),
            "revision_prompt_path": result.get("revision_prompt_path"),
            "error": None,
        }
    except Exception as e:
        logger.error("修正 Prompt 构造失败：%s", e, exc_info=True)
        return {"error": str(e)}


def generate_revised_layout_action(state: AgentState) -> Dict[str, Any]:
    try:
        prompt_text = state.get("revision_prompt")
        if not prompt_text:
            return {"error": "缺少 revision_prompt，无法调用修正 LLM。"}

        iteration_index = int(state.get("iteration_index") or 0)
        output_dir = state.get("output_dir") or os.getcwd()
        result_path = os.path.join(output_dir, "revision_results", f"revision_design_round_{iteration_index + 1}.json")

        llm = get_revision_llm(state.get("config_path") or "config/settings.yaml")
        raw_repairs = state.get("llm_max_format_repairs")
        # 注意：不能用 `x or 1`，否则显式配置 0（不重试）会被当成 1。
        max_format_repairs = max(0, int(raw_repairs)) if raw_repairs is not None else 1
        total_attempts = max_format_repairs + 1

        raw_dir = os.path.join(output_dir, "revision_results")
        system_prompt = (
            "你是一名桥梁工程设计师。请根据修正 Prompt 输出完整、可解析、可复检的设桥布跨 JSON。"
            "只能输出 JSON，不得输出解释文字；字符串值中不得出现未转义的换行、制表符等控制字符。"
        )
        messages: List[Any] = [
            ("system", system_prompt),
            ("user", prompt_text),
        ]

        raw = ""
        revised: Optional[Dict[str, Any]] = None
        parse_error: Optional[Exception] = None
        attempts_used = 0
        for attempt in range(1, total_attempts + 1):
            attempts_used = attempt
            response = llm.invoke(messages)
            raw = str(response.content).strip()
            try:
                candidate = extract_json_object(raw)
                if "设桥总览" not in candidate:
                    raise ValueError("修正输出缺少 '设桥总览' 字段。")
                revised = candidate
                break
            except Exception as exc:  # noqa: BLE001 - 需要留证并决定是否重试
                parse_error = exc
                revised = None
                # 每次失败都把 LLM 原始输出落盘，便于定位（例如字符串内的裸控制字符）。
                try:
                    os.makedirs(raw_dir, exist_ok=True)
                    raw_path = os.path.join(
                        raw_dir,
                        f"revision_design_round_{iteration_index + 1}_raw_failed_attempt_{attempt}.txt",
                    )
                    with open(raw_path, "w", encoding="utf-8") as handle:
                        handle.write(raw)
                    logger.error("修正方案解析失败（第 %s/%s 次），原始输出已保存：%s", attempt, total_attempts, raw_path)
                except OSError as write_error:
                    logger.warning("修正原始输出落盘失败：%s", write_error)
                if attempt >= total_attempts:
                    break
                # 自动格式修复重试：只要求模型修 JSON 格式，不改变设计意图。
                messages = [
                    (
                        "system",
                        "你负责修复 JSON 输出格式。保持原设计语义不变，只输出可解析的 JSON 对象，"
                        "不得输出说明文字；字符串值中不得出现未转义的换行、制表符等控制字符。",
                    ),
                    (
                        "user",
                        f"原始任务：\n{prompt_text}\n\n上次输出：\n{raw}\n\n解析错误：{parse_error}",
                    ),
                ]

        if revised is None:
            raise ValueError(
                f"修正方案 JSON 解析失败（已自动尝试 {attempts_used} 次）：{parse_error}"
            )
        write_json(result_path, revised)

        history = list(state.get("revision_history") or [])
        history.append({
            "round": iteration_index + 1,
            "action": "generate_revised_layout",
            "result_path": result_path,
            "format_attempts": attempts_used,
        })
        return {
            "revision_result_raw": raw,
            "revision_result_path": result_path,
            "revision_format_attempts": attempts_used,
            "layout_result": revised,
            "design_result": revised,
            "revision_history": history,
            "error": None,
        }
    except Exception as e:
        logger.error("修正方案生成失败：%s", e, exc_info=True)
        return {"error": str(e)}


# ============================================================
# StructuralDesignAgent 使用的工具动作
# ============================================================
def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _id_list(value: Any) -> List[str]:
    """把状态里的 ID 列表统一成非空字符串列表。"""
    return [str(item) for item in _safe_list(value) if str(item).strip()]


def _dimension_retry_unit_ids(state: AgentState, revision_context: Dict[str, Any]) -> Optional[List[str]]:
    """返回尺寸设计本轮要重跑的"单元编号"。

    尺寸设计与配筋设计使用两套 ID 命名空间：
    - 尺寸设计：单元编号（如 2-2）；
    - 配筋设计：设计组任务号（如 2-2-1-G1）。
    尺寸侧只接受前者；历史上把 state.failed_task_ids（可能存的是配筋任务号）直接当
    单元号传入，导致尺寸设计工具按 ID 匹配失败并让整个阶段判失败。
    """
    explicit = _id_list(revision_context.get("dimension_retry_unit_ids"))
    if explicit:
        return explicit
    dimension_batch = _safe_dict(state.get("dimension_batch_status"))
    if dimension_batch:
        if dimension_batch.get("stage_complete") is False:
            # 批次未完成但拿不到具体失败单元时返回 None，交由工具整体重跑。
            return _id_list(dimension_batch.get("failed_unit_ids")) or None
        return None
    # 旧状态没有尺寸批次信息：保持历史行为，回退到单一 failed_task_ids 字段。
    return _id_list(state.get("failed_task_ids")) or None


def _structural_evidence_state_update(
    state: AgentState,
    tool_result: Dict[str, Any],
    *,
    stage: str,
) -> Dict[str, Any]:
    merged = dict(state.get("evidence_bundles") or {})
    trace: List[Dict[str, Any]] = []
    bundles = tool_result.get("evidence_bundles")
    if not isinstance(bundles, dict):
        bundles = {}
    for task_id, raw_context in bundles.items():
        if not isinstance(raw_context, dict):
            continue
        scoped_id = f"{stage}:{task_id}"
        merged[scoped_id] = raw_context
        trace.append({
            "stage": stage,
            "task_id": str(task_id),
            "retrieval_status": raw_context.get("retrieval_status"),
            "evidence_ids": list(raw_context.get("evidence_ids") or []),
            "query_hash": raw_context.get("query_hash") or "",
        })
    return {"evidence_bundles": merged, "code_trace": trace}


def _station_to_float(station: Any) -> Optional[float]:
    """将 K12+345.6 / 12345.6 宽松转换为 m。"""
    if station in [None, ""]:
        return None
    if isinstance(station, (int, float)):
        return float(station)
    text = str(station).strip().upper().replace(" ", "")
    m = None
    import re
    # K12+345.6、K-1+520、12+345.6 均尽量兼容
    m = re.search(r"K?(-?\d+)\+(\d+(?:\.\d+)?)", text)
    if m:
        km = float(m.group(1))
        offset = float(m.group(2))
        if km < 0:
            return km * 1000.0 - offset
        return km * 1000.0 + offset
    try:
        return float(text)
    except ValueError:
        return None


def _float_to_station(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    sign = "-" if value < 0 else ""
    abs_v = abs(float(value))
    km = int(abs_v // 1000)
    meter = abs_v - km * 1000
    return f"K{sign}{km}+{meter:06.3f}".rstrip("0").rstrip(".")


def _first_value(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] not in [None, "", [], {}]:
            return data[key]
    return default


def _parse_span_list(span_expr: Any) -> List[float]:
    """宽松解析跨径组合。支持 3×30+40、[30,30,40]、3*30m 等。"""
    if isinstance(span_expr, list):
        spans: List[float] = []
        for item in span_expr:
            if isinstance(item, (int, float)):
                spans.append(float(item))
            elif isinstance(item, dict):
                v = _first_value(item, ["span", "跨径", "长度", "span_length_m"])
                if isinstance(v, (int, float)):
                    spans.append(float(v))
                else:
                    spans.extend(_parse_span_list(v))
            else:
                spans.extend(_parse_span_list(item))
        return spans
    if not isinstance(span_expr, str):
        return []

    import re
    text = span_expr.replace(" ", "").replace("＋", "+").replace("×", "x").replace("*", "x")
    spans: List[float] = []
    used_ranges = []
    for m in re.finditer(r"(\d+)x(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE):
        count = int(m.group(1))
        length = float(m.group(2))
        spans.extend([length] * count)
        used_ranges.append(m.span())

    # 去掉已经作为 n×L 解析过的片段，再解析孤立数字。
    remaining = []
    last = 0
    for start, end in used_ranges:
        remaining.append(text[last:start])
        last = end
    remaining.append(text[last:])
    remain_text = "+".join(remaining)
    for m in re.finditer(r"(?<!\d)(\d+(?:\.\d+)?)(?:m|米)?", remain_text, flags=re.IGNORECASE):
        value = float(m.group(1))
        if 5 <= value <= 250:
            spans.append(value)
    return spans


def _find_bridge_candidates(layout_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从不同可能的布跨结果 schema 中提取桥梁对象列表。"""
    candidates: List[Dict[str, Any]] = []
    direct_keys = ["桥梁列表", "桥梁布设", "桥梁方案", "bridges", "bridge_list"]
    for key in direct_keys:
        value = layout_result.get(key)
        if isinstance(value, list):
            candidates.extend([x for x in value if isinstance(x, dict)])

    # 有些结果将桥梁对象放在“设桥总览”的子字段里。
    overview = layout_result.get("设桥总览")
    if isinstance(overview, dict):
        for key in direct_keys:
            value = overview.get(key)
            if isinstance(value, list):
                candidates.extend([x for x in value if isinstance(x, dict)])

    # 兜底：递归寻找含有跨径组合/起终点/桥名特征的字典。
    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            keys = set(obj.keys())
            key_text = " ".join(str(k) for k in keys)
            if any(x in key_text for x in ["跨径", "桥名", "桥梁名称", "起点", "终点", "span"]):
                if any(x in key_text for x in ["跨径", "span"]):
                    candidates.append(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    if not candidates:
        walk(layout_result)

    # 去重，避免同一个 dict 被递归重复加入。
    unique: List[Dict[str, Any]] = []
    seen = set()
    for item in candidates:
        ident = id(item)
        if ident not in seen:
            seen.add(ident)
            unique.append(item)
    return unique


def _extract_width(layout_result: Dict[str, Any], bridge: Dict[str, Any]) -> Any:
    width = _first_value(bridge, ["桥面宽度", "单幅桥面宽度", "bridge_width", "width", "桥宽"])
    if width not in [None, "", [], {}]:
        return width
    overview = _safe_dict(layout_result.get("设桥总览"))
    return _first_value(overview, ["桥面宽度", "单幅桥面宽度", "bridge_width", "width", "桥宽"], default="未知")


def _extract_pier_table(bridge: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ["逐墩表", "墩位表", "桥墩表", "pier_table", "piers", "pier_list"]:
        value = bridge.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def _make_pier_roles(pier_count: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(pier_count):
        if i == 0 or i == pier_count - 1:
            role = "桥台"
        elif i == 1 or i == pier_count - 2:
            role = "边墩"
        else:
            role = "中间墩"
        rows.append({"pier_index": i, "pier_name": f"P{i}", "pier_role": role})
    return rows


def _partition_spans(spans: List[float]) -> List[Dict[str, Any]]:
    """兜底分联：优先按整桥；跨数较多时按 4~5 跨拆分，后续可由正式工具替换。"""
    if not spans:
        return [{"link_index": 1, "span_indices": [], "span_combination": [], "partition_basis": "未识别跨径，暂按整桥作为一个设计单元"}]
    if len(spans) <= 5:
        return [{"link_index": 1, "span_indices": list(range(1, len(spans) + 1)), "span_combination": spans, "partition_basis": "跨数不多，兜底按整桥作为一个设计单元"}]

    links: List[Dict[str, Any]] = []
    start = 0
    link_index = 1
    while start < len(spans):
        end = min(start + 5, len(spans))
        # 避免最后只剩 1 跨时过度拆分。
        if len(spans) - end == 1:
            end = len(spans)
        links.append({
            "link_index": link_index,
            "span_indices": list(range(start + 1, end + 1)),
            "span_combination": spans[start:end],
            "partition_basis": "兜底分联：按连续 4~5 跨控制，正式工程分联逻辑后续由工具替换",
        })
        start = end
        link_index += 1
    return links


def _fallback_extract_design_units(layout_result: Dict[str, Any]) -> Dict[str, Any]:
    bridges = _find_bridge_candidates(layout_result)
    if not bridges:
        bridges = [{"桥梁名称": "Bridge-1", "跨径组合": _first_value(layout_result, ["跨径组合", "span_arrangement", "spans"], [])}]

    design_units: List[Dict[str, Any]] = []
    bridge_summaries: List[Dict[str, Any]] = []
    global_unit_index = 1

    for b_idx, bridge in enumerate(bridges, start=1):
        name = _first_value(bridge, ["桥梁名称", "桥名", "name", "bridge_name"], default=f"Bridge-{b_idx}")
        start_station = _first_value(bridge, ["起点桩号", "起点", "start_station", "bridge_start"])
        end_station = _first_value(bridge, ["终点桩号", "终点", "end_station", "bridge_end"])
        span_expr = _first_value(bridge, ["跨径组合", "跨径布置", "span_arrangement", "spans", "span_combination"])
        spans = _parse_span_list(span_expr)
        pier_table = _extract_pier_table(bridge)
        pier_count = max(len(pier_table), len(spans) + 1 if spans else 0)
        pier_roles = []
        if pier_table:
            for idx, pier in enumerate(pier_table):
                pier_roles.append({
                    "pier_index": idx,
                    "pier_name": _first_value(pier, ["墩号", "桥墩编号", "pier_name", "pier_id"], default=f"P{idx}"),
                    "pier_station": _first_value(pier, ["桩号", "墩位桩号", "station", "pier_station"]),
                    "pier_role": _first_value(pier, ["墩位角色", "role", "pier_role"], default=None),
                })
            # 补齐角色
            fallback_roles = _make_pier_roles(len(pier_roles))
            for i, row in enumerate(pier_roles):
                if not row.get("pier_role"):
                    row["pier_role"] = fallback_roles[i]["pier_role"]
        else:
            pier_roles = _make_pier_roles(pier_count if pier_count else 2)

        width = _extract_width(layout_result, bridge)
        links = _partition_spans(spans)
        bridge_summaries.append({
            "bridge_index": b_idx,
            "bridge_name": name,
            "start_station": start_station,
            "end_station": end_station,
            "span_combination": spans,
            "deck_width": width,
            "pier_count": len(pier_roles),
            "link_count": len(links),
        })
        for link in links:
            span_indices = link.get("span_indices") or []
            # 第 i 跨对应 P{i-1}~P{i}，因此联内墩台范围取首跨起点墩到末跨终点墩。
            if span_indices:
                p_start = max(0, int(span_indices[0]) - 1)
                p_end = min(len(pier_roles) - 1, int(span_indices[-1]))
                unit_piers = pier_roles[p_start:p_end + 1]
            else:
                unit_piers = pier_roles
            design_units.append({
                "unit_id": f"U{global_unit_index}",
                "bridge_index": b_idx,
                "bridge_name": name,
                "link_index": link["link_index"],
                "start_station": start_station,
                "end_station": end_station,
                "span_indices": span_indices,
                "span_combination": link.get("span_combination") or [],
                "deck_width": width,
                "pier_roles": unit_piers,
                "partition_basis": link.get("partition_basis"),
            })
            global_unit_index += 1

    return {
        "status": "fallback_extracted",
        "message": "未接入正式 design_unit_extractor_tool，已根据布跨结果进行兜底解析，可用于跑通结构设计流程。",
        "bridge_summaries": bridge_summaries,
        "design_units": design_units,
    }


def _default_dimension_result(design_units: Dict[str, Any]) -> Dict[str, Any]:
    units = _safe_list(design_units.get("design_units"))
    unit_results: List[Dict[str, Any]] = []
    for unit in units:
        spans = [float(x) for x in _safe_list(unit.get("span_combination")) if isinstance(x, (int, float))]
        max_span = max(spans) if spans else None
        deck_width = unit.get("deck_width")
        girder_type = "预应力混凝土T梁" if (max_span or 30) <= 40 else "预应力混凝土小箱梁/连续梁待判定"
        pier_count = len(_safe_list(unit.get("pier_roles")))
        cap_length = None
        try:
            cap_length = round(float(deck_width) + 1.6, 2)
        except Exception:
            cap_length = "按桥面宽度+两侧构造悬臂确定"
        unit_results.append({
            "unit_id": unit.get("unit_id"),
            "bridge_name": unit.get("bridge_name"),
            "link_index": unit.get("link_index"),
            "design_basis": "兜底尺寸设计，仅用于流程联调；正式成果应由 dimension_design_tool 或结构设计 LLM 生成。",
            "superstructure": {
                "girder_type": girder_type,
                "span_combination": spans,
                "support_rows_per_pier": 2,
            },
            "substructure_dimension": {
                "cap_beam": {
                    "length_m": cap_length,
                    "width_m": 1.8 if (max_span or 30) <= 40 else 2.2,
                    "height_m": 1.6 if (max_span or 30) <= 40 else 2.0,
                    "cantilever_length_m": "按支座及边梁外缘控制",
                },
                "pier": {
                    "type": "双柱式桥墩" if pier_count >= 3 else "桥台/边墩按墩位角色分别确定",
                    "column_count": 2,
                    "column_diameter_m": 1.4 if (max_span or 30) <= 40 else 1.6,
                },
                "foundation": {
                    "type": "桩基础",
                    "pile_count_per_pier": 2,
                    "pile_diameter_m": 1.6 if (max_span or 30) <= 40 else 1.8,
                },
            },
        })
    return {
        "status": "fallback_dimension_design",
        "message": "尺寸设计工具/LLM不可用时的兜底结果，仅保证结构设计链路可运行。",
        "unit_dimension_results": unit_results,
    }


def _default_reinforcement_result(dimension_result: Dict[str, Any]) -> Dict[str, Any]:
    unit_results = _safe_list(dimension_result.get("unit_dimension_results"))
    type_library: Dict[str, Dict[str, Any]] = {}
    mapping: List[Dict[str, Any]] = []
    for idx, unit in enumerate(unit_results, start=1):
        sub = _safe_dict(unit.get("substructure_dimension"))
        pier = _safe_dict(sub.get("pier"))
        cap = _safe_dict(sub.get("cap_beam"))
        key = f"cap_{cap.get('width_m')}_{cap.get('height_m')}_pier_{pier.get('column_diameter_m')}"
        if key not in type_library:
            type_library[key] = {
                "type_id": f"R{len(type_library) + 1}",
                "dimension_signature": key,
                "design_basis": "兜底配筋，仅用于流程联调；正式配筋应由 reinforcement_design_tool 或结构设计 LLM 生成。",
                "cap_beam_reinforcement": {
                    "main_bars": "待正式设计",
                    "stirrups": "待正式设计",
                    "additional_bars": "待正式设计",
                },
                "pier_column_reinforcement": {
                    "longitudinal_bars": "待正式设计",
                    "hoops": "待正式设计",
                    "plastic_hinge_zone_detailing": "待 ModelingCheckAgent 与抗震构造要求进一步校核",
                },
                "foundation_reinforcement": {
                    "pile_main_bars": "待正式设计",
                    "pile_hoops": "待正式设计",
                },
            }
        mapping.append({
            "unit_id": unit.get("unit_id"),
            "bridge_name": unit.get("bridge_name"),
            "link_index": unit.get("link_index"),
            "reinforcement_type_id": type_library[key]["type_id"],
        })
    return {
        "status": "fallback_reinforcement_design",
        "message": "按照相同尺寸归并为同类配筋对象，但未进行正式配筋计算。",
        "reinforcement_type_library": list(type_library.values()),
        "unit_reinforcement_mapping": mapping,
    }


def _invoke_structural_llm_json(state: AgentState, system_prompt: str, user_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        llm = get_structural_llm(state.get("config_path") or "config/settings.yaml")
        response = llm.invoke([
            ("system", system_prompt),
            ("user", json.dumps(json_safe(user_payload), ensure_ascii=False, indent=2)),
        ])
        return extract_json_object(response.content)
    except Exception as e:
        logger.warning("结构设计 LLM 调用失败，启用兜底结果：%s", e)
        return None


def extract_design_units_action(state: AgentState) -> Dict[str, Any]:
    """设计单元提取。优先调用正式工具，缺失时执行可运行兜底解析。"""
    try:
        layout_result = state.get("layout_result") or state.get("existing_layout_result")
        if not layout_result:
            return {"error": "缺少 layout_result 或 existing_layout_result，无法提取结构设计单元。"}
        layout_result = extract_design_payload(layout_result)

        if extract_design_units_tool is not None:
            result = invoke_tool(extract_design_units_tool, {
                "layout_result": layout_result,
                "design_input": state.get("design_input") or {},
                "output_dir": state.get("output_dir"),
            })
            if tool_failed(result):
                return {"error": result.get("error", "设计单元提取失败") if isinstance(result, dict) else "设计单元提取失败"}
            return {"layout_result": layout_result, "design_units": result, "error": None}

        result = _fallback_extract_design_units(layout_result)
        return {"layout_result": layout_result, "design_units": result, "error": None}
    except Exception as e:
        logger.error("设计单元提取失败：%s", e, exc_info=True)
        return {"error": str(e)}


def dimension_design_action(state: AgentState) -> Dict[str, Any]:
    try:
        design_units = _safe_dict(state.get("design_units"))
        if not design_units:
            return {"error": "缺少 design_units，无法进行尺寸设计。"}

        if dimension_design_tool is not None:
            revision_context = _safe_dict(state.get("revision_context"))
            retry_dimension = revision_context.get("target_step") == "dimension_design"
            result = invoke_tool(dimension_design_tool, {
                "design_units": design_units,
                "layout_result": state.get("layout_result"),
                "standards_path": state.get("standards_path"),
                "config_path": state.get("config_path"),
                "output_dir": state.get("output_dir"),
                "retry_unit_ids": (
                    _dimension_retry_unit_ids(state, revision_context) if retry_dimension else None
                ),
                # 合并基准同样要回退到磁盘：返修失效已把内存里的尺寸成果清空，
                # 直接用 state 取到的恒为 None，会让尺寸成果坍缩成只剩被返修的那一联。
                "existing_dimension_design_result": (
                    _load_dimension_summary(state) if retry_dimension else None
                ),
            })
            if tool_failed(result):
                return {"error": result.get("error", "下部结构尺寸设计失败") if isinstance(result, dict) else "下部结构尺寸设计失败"}
            errors = list(result.get("errors") or []) if isinstance(result, dict) else []
            return {
                "dimension_design_result": result,
                "dimension_batch_status": {
                    "expected_unit_count": result.get("expected_unit_count"),
                    "completed_unit_count": result.get("completed_unit_count", result.get("unit_result_count")),
                    "failed_unit_count": result.get("failed_unit_count", len(errors)),
                    "failed_unit_ids": [item.get("单元编号") for item in errors if isinstance(item, dict) and item.get("单元编号")],
                    "stage_complete": result.get("stage_complete", not errors),
                    "partial_result_available": result.get("partial_result_available", bool(errors)),
                },
                **_structural_evidence_state_update(
                    state,
                    result,
                    stage="dimension_design",
                ),
                "error": None,
            }

        skill_text = (
            render_prompt(
                "agents.structural_design.v1",
                {},
                config_path=state.get("config_path") or "config/settings.yaml",
            ).system_content
        )
        llm_result = _invoke_structural_llm_json(state, """
你是 StructuralDesignAgent 内部的下部结构尺寸设计 LLM。
你必须基于设计单元完成一联一设计的尺寸设计，输出严格 JSON。
这里只生成尺寸设计结果，不进行承载力验算，不输出 Markdown。
""".strip(), {
            "agent_skill": skill_text,
            "design_units": design_units,
            "layout_result_summary": state.get("layout_result") or state.get("existing_layout_result"),
            "standards_path": state.get("standards_path"),
            "revision_context": state.get("revision_context"),
        })
        if llm_result:
            llm_result.setdefault("status", "llm_dimension_design")
            return {"dimension_design_result": llm_result, "error": None}

        return {"dimension_design_result": _default_dimension_result(design_units), "error": None}
    except Exception as e:
        logger.error("下部结构尺寸设计失败：%s", e, exc_info=True)
        return {"error": str(e)}


def compute_pier_groups_action(state: AgentState) -> Dict[str, Any]:
    """桥墩结构设计组归并：墩高核实、柱位盖梁高度与净高、高度审计。"""
    try:
        layout_result = (
            state.get("layout_result")
            or state.get("existing_layout_result")
            or state.get("final_layout_result")
        )
        design_units = _safe_dict(state.get("design_units"))
        dimension_result = _safe_dict(state.get("dimension_design_result"))
        if not layout_result or not design_units or not dimension_result:
            return {"error": "缺少 layout_result、design_units 或 dimension_design_result，无法归并桥墩设计组。"}

        cropped = state.get("cropped_data") or state.get("data_loader_result") or {}
        profiles = collect_vertical_profiles(cropped)
        route_map = build_side_route_map(layout_result, profiles)
        side_profiles = {
            side: profiles[route]
            for side, route in route_map.items()
            if route in profiles
        }

        super_height = state.get("superstructure_height_m")
        foundation_offset = float(state.get("foundation_top_offset_m") or 0.0)

        result = compute_pier_groups(
            layout_result,
            design_units,
            dimension_result,
            vertical_profiles=side_profiles,
            super_height=super_height,
            foundation_top_offset_m=foundation_offset,
        )
        output_dir = state.get("output_dir") or "outputs"
        pier_group_path = os.path.join(
            output_dir,
            "structural_design",
            "pier_group",
            "pier_group_result.json",
        )
        write_json(pier_group_path, result)
        return {
            "pier_group_result": result,
            "pier_group_result_path": pier_group_path,
            "error": None,
        }
    except Exception as e:
        logger.error("桥墩设计组归并失败：%s", e, exc_info=True)
        return {"error": str(e)}


def reinforcement_design_action(state: AgentState) -> Dict[str, Any]:
    try:
        dimension_result = _safe_dict(state.get("dimension_design_result"))
        design_units = _safe_dict(state.get("design_units"))
        if not dimension_result:
            return {"error": "缺少 dimension_design_result，无法进行配筋设计。"}

        if reinforcement_design_tool is not None:
            revision_context = _safe_dict(state.get("revision_context"))
            retry_dimension = revision_context.get("target_step") == "dimension_design"
            rework_active = bool(revision_context)
            # 配筋重跑范围（任务号命名空间）：本轮显式范围 ∪ 上一轮生成失败 ∪ 本轮验算未通过
            retry_task_ids = _reinforcement_retry_task_ids(state, revision_context) if rework_active else []
            # 尺寸变更会使其配筋失效：尺寸返修时按"受影响单元"重跑配筋
            # （工具按 task.单元编号 匹配），而不是把配筋任务号塞给单元号参数。
            retry_unit_ids = (
                _dimension_retry_unit_ids(state, revision_context) if retry_dimension else None
            )
            # 合并基准：返修只重做部分分组，其余分组沿用上一轮结果；内存被返修失效清空时
            # 必须回退到磁盘上的上一轮汇总，否则结果集会只剩被返修的那一组。
            existing_result = _safe_dict(state.get("reinforcement_design_result"))
            if rework_active and not existing_result:
                existing_result = _load_reinforcement_summary(state)
            result = invoke_tool(reinforcement_design_tool, {
                "dimension_design_result": dimension_result,
                "design_units": design_units,
                "standards_path": state.get("standards_path"),
                "config_path": state.get("config_path"),
                "output_dir": state.get("output_dir"),
                "retry_task_ids": retry_task_ids or None,
                "retry_unit_ids": retry_unit_ids or None,
                "existing_reinforcement_design_result": existing_result if rework_active else None,
                "pier_group_result": state.get("pier_group_result"),
            })
            if tool_failed(result):
                return {"error": result.get("error", "配筋设计失败") if isinstance(result, dict) else "配筋设计失败"}
            output_files = result.get("output_files", {}) if isinstance(result, dict) else {}
            errors = list(result.get("errors") or []) if isinstance(result, dict) else []
            return {
                "reinforcement_design_result": result,
                "reinforcement_batch_status": {
                    "expected_task_count": result.get("expected_task_count"),
                    "completed_task_count": result.get("completed_task_count", result.get("task_result_count")),
                    "failed_task_count": result.get("failed_task_count", len(errors)),
                    "failed_task_ids": [item.get("task_id") for item in errors if isinstance(item, dict) and item.get("task_id")],
                    "stage_complete": result.get("stage_complete", not errors),
                    "partial_result_available": result.get("partial_result_available", bool(errors)),
                },
                "reinforcement_yaml_path": output_files.get("reinforcement_yaml_path") or state.get("reinforcement_yaml_path"),
                "reinforcement_yaml_paths": output_files.get("reinforcement_yaml_paths"),
                "opensees_force_json_path": (
                    output_files.get("opensees_force_json_path")
                    or output_files.get("internal_force_json_path")
                    or state.get("opensees_force_json_path")
                ),
                "internal_force_output_paths": output_files.get("internal_force_output_paths"),
                **_structural_evidence_state_update(
                    state,
                    result,
                    stage="reinforcement_design",
                ),
                "error": None,
            }

        skill_text = (
            render_prompt(
                "agents.structural_design.v1",
                {},
                config_path=state.get("config_path") or "config/settings.yaml",
            ).system_content
        )
        llm_result = _invoke_structural_llm_json(state, """
你是 StructuralDesignAgent 内部的下部结构配筋设计 LLM。
你必须根据尺寸设计结果，先按相同尺寸归并构件类型，再给出各类构件配筋方案及映射关系。
这里只生成配筋设计结果，不进行最终承载力验算，不输出 Markdown，只输出严格 JSON。
""".strip(), {
            "agent_skill": skill_text,
            "design_units": design_units,
            "dimension_design_result": dimension_result,
            "standards_path": state.get("standards_path"),
            "revision_context": state.get("revision_context"),
        })
        if llm_result:
            llm_result.setdefault("status", "llm_reinforcement_design")
            return {"reinforcement_design_result": llm_result, "error": None}

        return {"reinforcement_design_result": _default_reinforcement_result(dimension_result), "error": None}
    except Exception as e:
        logger.error("配筋设计失败：%s", e, exc_info=True)
        return {"error": str(e)}


# ============================================================
# ModelingCheckAgent 使用的工具动作
# ============================================================
def _pick_nested_path(*sources: Any, keys: List[str]) -> Optional[str]:
    """从多层工具返回中宽松提取路径。"""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if value not in [None, "", [], {}]:
                return str(value)
        for nested_key in ["output_files", "files", "paths", "result"]:
            nested = source.get(nested_key)
            if isinstance(nested, dict):
                value = _pick_nested_path(nested, keys=keys)
                if value:
                    return value
    return None


def _load_json_if_exists(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(str(path)):
        return {}
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _existing_path_or_none(path: Optional[str]) -> Optional[str]:
    if path and os.path.exists(str(path)):
        return str(path)
    return None


def _max_control_utilization(control_sections: Dict[str, Any]) -> Dict[str, Any]:
    result = {"max_utilization": None, "control_name": None, "util_type": None, "x_m": None}
    for name, row in (control_sections or {}).items():
        if not isinstance(row, dict):
            continue
        for key in ["util_M_pos", "util_M_neg", "util_V"]:
            value = row.get(key)
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if result["max_utilization"] is None or v > float(result["max_utilization"]):
                result = {"max_utilization": v, "control_name": name, "util_type": key, "x_m": row.get("x_m")}
    return result


def _fallback_modeling_feedback_decision(
    check_result: Dict[str, Any],
    *,
    reinforcement_already_retried: bool = False,
) -> Dict[str, Any]:
    overall = _safe_dict(check_result.get("overall_check"))
    if overall.get("all_ok") is True:
        return {
            "overall_status": "pass",
            "next_action": "pass",
            "target_agent": "END",
            "target_step": None,
            "control_reason": "承载力验算 overall_check.all_ok=true。",
            "requires_rerun_check": False,
            "revision_instruction": "验算通过，无需修正。",
        }

    control_sections = _safe_dict(check_result.get("control_sections"))
    util = _max_control_utilization(control_sections)
    max_u = util.get("max_utilization")
    if max_u is None:
        # 批次汇总的 utilization_summary 是权威控制项；control_sections 缺失/裁剪时回退到这里。
        summary_util = _safe_dict(check_result.get("utilization_summary"))
        max_u = summary_util.get("max_utilization") or util.get("max_utilization")
    try:
        max_u_float = float(max_u) if max_u is not None else 1.0
    except (TypeError, ValueError):
        max_u_float = 1.0

    failed_items = [k for k in ["M_pos_ok", "M_neg_ok", "V_ok"] if overall.get(k) is False]
    compression_zone_exceeded = (
        overall.get("compression_zone_ok") is False
        or any(
            row.get("compression_zone_limit_exceeded") is True
            for row in control_sections.values()
            if isinstance(row, dict)
        )
    )
    if compression_zone_exceeded:
        next_action = "revise_dimension"
        target_step = "dimension_design"
        decision_basis = "compression_zone_limit_exceeded"
        instruction = "受压区高度超过相对界限受压区高度，返回尺寸设计阶段调整截面有效高度或宽度，再重新配筋。"
    elif reinforcement_already_retried and max_u_float > 1.0:
        # 配筋返修后承载力仍未通过：同一截面下反复调整纵筋/箍筋无法收敛，
        # 应升级到尺寸设计加大截面，避免结构↔验算无限返修循环。
        next_action = "revise_dimension"
        target_step = "dimension_design"
        decision_basis = "reinforcement_rework_exhausted"
        instruction = "配筋返修后承载力仍未通过（控制利用率仍超限），返回尺寸设计阶段加大截面有效高度或宽度，再重新配筋。"
    else:
        next_action = "revise_reinforcement"
        target_step = "reinforcement_design"
        decision_basis = "capacity_or_reinforcement_failure"
        instruction = "承载力验算未通过且受压区界限校核有效，返回配筋设计阶段调整纵筋、箍筋或局部加强钢筋。"

    reason = (
        f"overall_check 未通过；失败项={failed_items}；"
        f"控制利用率={max_u_float:.3f}，控制位置={util.get('control_name')}，类型={util.get('util_type')}。"
    )
    return {
        "overall_status": next_action,
        "next_action": next_action,
        "target_agent": "StructuralDesignAgent",
        "target_step": target_step,
        "control_reason": reason,
        "requires_rerun_check": True,
        "revision_instruction": instruction,
        "decision_basis": decision_basis,
        "control_utilization": util,
        "failed_items": failed_items,
    }


def run_capacity_check_action(state: AgentState) -> Dict[str, Any]:
    """对全部成功配筋分组逐组执行盖梁承载力验算并聚合结果。"""
    if capacity_check_tool is None:
        return {"error": "未能导入 capacity_check_tool，请检查 tools/capacity_check_tool.py。"}

    try:
        tasks = collect_capacity_check_tasks(state)
        output_dir = str(
            state.get("capacity_check_output_dir")
            or os.path.join(state.get("output_dir") or "outputs", "capacity_check")
        )
        multiple_tasks = len(tasks) > 1
        task_results: List[Dict[str, Any]] = []

        for task in tasks:
            task_id = str(task["task_id"])
            task_output_dir = (
                os.path.join(output_dir, safe_capacity_task_dir_name(task_id))
                if multiple_tasks
                else output_dir
            )
            result = invoke_tool(capacity_check_tool, {
                "reinforcement_yaml_path": task["reinforcement_yaml_path"],
                "opensees_force_json_path": task["opensees_force_json_path"],
                "output_dir": task_output_dir,
                "capacity_script_path": state.get("capacity_check_script_path"),
                "opensees_script_path": state.get("opensees_script_path"),
                "combination_name": state.get("capacity_check_combination_name") or "ULS_basic",
                "gamma_0": state.get("capacity_check_gamma_0") or 1.10,
                "apply_gamma0_to_demand": True if state.get("capacity_check_apply_gamma0") is None else state.get("capacity_check_apply_gamma0"),
                "auto_run_opensees_if_missing": False,
            })
            if not isinstance(result, dict):
                result = {"success": False, "error": "承载力验算工具返回非字典结果。"}
            task_results.append({
                **task,
                "output_dir": task_output_dir,
                "result": result,
            })

        aggregate = aggregate_capacity_check_results(task_results)
        source_batch = _safe_dict(state.get("reinforcement_batch_status"))
        aggregate["source_reinforcement_batch_status"] = source_batch
        aggregate["accepted_partial_input"] = source_batch.get("human_accepted") is True
        batch_summary_path = os.path.join(output_dir, "capacity_check_batch_summary.json")
        task_summary_paths = [
            _safe_dict(item.get("result")).get("output_files", {}).get("capacity_check_summary_path")
            for item in task_results
            if isinstance(_safe_dict(item.get("result")).get("output_files"), dict)
        ]
        task_summary_paths = [path for path in task_summary_paths if path]
        aggregate["output_files"] = {
            "capacity_check_summary_path": batch_summary_path,
            "capacity_check_summary_paths": task_summary_paths,
            "task_output_dirs": [item["output_dir"] for item in task_results],
        }
        write_json(batch_summary_path, aggregate)

        batch_status = {
            "expected_task_count": aggregate["expected_task_count"],
            "completed_task_count": aggregate["completed_task_count"],
            "failed_task_count": aggregate["failed_task_count"],
            "failed_task_ids": list(aggregate["failed_task_ids"]),
            "stage_complete": aggregate["stage_complete"],
        }
        update: Dict[str, Any] = {
            "capacity_check_result": aggregate,
            "capacity_check_results": aggregate["task_results"],
            "capacity_batch_status": batch_status,
            "check_result": aggregate,
            "reinforcement_yaml_path": tasks[0]["reinforcement_yaml_path"],
            "reinforcement_yaml_paths": [task["reinforcement_yaml_path"] for task in tasks],
            "opensees_force_json_path": tasks[0]["opensees_force_json_path"],
            "internal_force_output_paths": [task["opensees_force_json_path"] for task in tasks],
            "capacity_check_summary_path": batch_summary_path,
            "capacity_check_summary_paths": task_summary_paths,
            "compliance_matrix": aggregate.get("compliance_matrix"),
            "unresolved_code_items": aggregate.get("unresolved_code_items") or [],
            "error": None,
        }
        if not aggregate["stage_complete"]:
            update["error"] = (
                "承载力批量验算未完整执行：失败任务="
                f"{aggregate['failed_task_ids']}"
            )
        return update
    except Exception as e:
        logger.error("承载力验算失败：%s", e, exc_info=True)
        return {"error": str(e)}


def _collect_unit_ids(payload: Any) -> List[str]:
    """递归收集工程成果里出现过的"单元编号"，用于把任务号降级映射到尺寸单元号。"""
    found: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "单元编号" and str(value or "").strip():
                    found.append(str(value).strip())
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return sorted(set(found))


def _map_task_ids_to_unit_ids(task_ids: List[str], unit_ids: List[str]) -> List[str]:
    """把配筋任务号（如 2-2-3-G2）按前缀降级映射到尺寸单元号（如 2-2）。"""
    known = {str(item) for item in unit_ids if str(item).strip()}
    mapped: List[str] = []
    for raw in task_ids:
        task_id = str(raw).strip()
        if not task_id:
            continue
        if task_id in known:
            mapped.append(task_id)
            continue
        parts = task_id.split("-")
        for cut in range(len(parts) - 1, 0, -1):
            candidate = "-".join(parts[:cut])
            if candidate in known:
                mapped.append(candidate)
                break
    return sorted(set(mapped))


def _failed_check_task_ids(check_result: Dict[str, Any]) -> List[str]:
    """从承载力批次汇总中取出"验算未通过"的任务号。

    优先使用批次汇总自带的 failed_check_task_ids（aggregate_capacity_check_results
    在验算层面判定的失败任务）；其余分支兼容旧格式或被裁剪过的汇总。
    """
    named = check_result.get("failed_check_task_ids")
    if isinstance(named, list) and named:
        return sorted({str(item).strip() for item in named if str(item).strip()})
    failed: List[str] = []
    for row in check_result.get("task_results") or []:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id") or "").strip()
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        overall = result.get("overall_check") if isinstance(result.get("overall_check"), dict) else {}
        if task_id and (overall.get("all_ok") is False or result.get("stage_complete") is False):
            failed.append(task_id)
    if failed:
        return sorted(set(failed))
    utilization = check_result.get("utilization_summary")
    overall = check_result.get("overall_check") if isinstance(check_result.get("overall_check"), dict) else {}
    control_task_id = ""
    if isinstance(utilization, dict):
        control_task_id = str(utilization.get("task_id") or "").strip()
        if not control_task_id:
            control_name = str(utilization.get("control_name") or "")
            if "::" in control_name:
                control_task_id = control_name.split("::", 1)[0].strip()
    if control_task_id and (overall.get("all_ok") is False or not overall):
        return [control_task_id]
    # 兜底：批次报告里直接给出的失败任务清单
    for key in ("failed_task_ids", "failed_tasks"):
        values = check_result.get(key)
        if isinstance(values, list) and values:
            return sorted({str(item) for item in values if str(item).strip()})
    return []


def _modeling_rework_retry_scope(
    state: AgentState,
    check_result: Dict[str, Any],
    next_action: str,
) -> Dict[str, Any]:
    """给出结构返修的最小重做范围（按步骤分命名空间）。

    尺寸设计用"单元编号"，配筋设计用"设计组任务号"，两者不能混用；
    不写清范围会让结构阶段整批重跑（2026-09-15 运行即缺少该字段）。
    """
    failed_task_ids = _failed_check_task_ids(check_result)
    # 单元号要在"尺寸成果"与"设计单元"两处取并集：尺寸设计失败过的单元不会出现在
    # 尺寸成果里，只取一处会漏掉它们，进而退化成整体重跑。
    unit_ids = sorted(
        set(_collect_unit_ids(state.get("dimension_design_result")))
        | set(_collect_unit_ids(state.get("design_units")))
    )
    scope: Dict[str, Any] = {
        "reinforcement_retry_task_ids": failed_task_ids,
        "dimension_retry_unit_ids": _map_task_ids_to_unit_ids(failed_task_ids, unit_ids),
    }
    if next_action == "revise_dimension" and not scope["dimension_retry_unit_ids"]:
        # 尺寸返修但无法定位单元：留空表示整体重跑，避免误传配筋任务号。
        scope["dimension_retry_unit_ids"] = []
    if not scope["reinforcement_retry_task_ids"]:
        scope["reinforcement_retry_task_ids"] = []
    return scope


def _load_first_json(paths: List[str]) -> Dict[str, Any]:
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload:
            return payload
    return {}


def _load_dimension_summary(state: AgentState) -> Dict[str, Any]:
    """上一轮尺寸成果汇总：优先内存，其次 state 记录路径 / output_dir 标准位置。

    尺寸返修只重做失败单元，其余单元必须沿用上一轮结果；而返修失效逻辑会把内存里的
    `dimension_design_result` 清空（见 dependencies.ARTIFACT_RUNTIME_KEYS），因此磁盘上的
    上一轮汇总就是唯一的合并基准。尺寸侧缺了这一步回退，合并基准恒为 None，尺寸成果会
    坍缩成"只剩被返修的那一联"；下游配筋再按坍缩结果重算任务清单，已验算通过的分组
    就被静默丢弃（2026-09-15 示例项目K29：配筋 20 组 → 4 组、最终只出 1 张图）。
    """
    payload = _safe_dict(state.get("dimension_design_result"))
    if payload:
        return payload
    candidates: List[str] = []
    if state.get("dimension_design_result_path"):
        candidates.append(str(state["dimension_design_result_path"]))
    output_dir = str(state.get("output_dir") or "").strip()
    if output_dir:
        candidates.append(
            os.path.join(
                output_dir,
                "structural_design",
                "dimension_design",
                "dimension_design_result.json",
            )
        )
    return _load_first_json(candidates)


def _load_reinforcement_summary(state: AgentState) -> Dict[str, Any]:
    """上一轮配筋成果汇总：优先内存，其次 state 记录路径 / output_dir 标准位置。

    返修只重做部分分组，其余分组必须沿用上一轮结果；而返修失效逻辑会把内存里的
    `reinforcement_design_result` 清空，因此磁盘上的上一轮汇总就是唯一的合并基准。
    """
    payload = _safe_dict(state.get("reinforcement_design_result"))
    if payload:
        return payload
    candidates: List[str] = []
    if state.get("reinforcement_design_result_path"):
        candidates.append(str(state["reinforcement_design_result_path"]))
    output_dir = str(state.get("output_dir") or "").strip()
    if output_dir:
        candidates.append(
            os.path.join(
                output_dir,
                "structural_design",
                "reinforcement_design",
                "reinforcement_design_result.json",
            )
        )
    return _load_first_json(candidates)


def _reinforcement_result_root(payload: Dict[str, Any]) -> Dict[str, Any]:
    root = payload.get("任务3_下部结构配筋设计结果") if isinstance(payload, dict) else None
    return root if isinstance(root, dict) else (payload if isinstance(payload, dict) else {})


def _previous_reinforcement_failure_ids(state: AgentState) -> List[str]:
    """上一轮配筋批次里未完成（生成失败）的任务号。"""
    root = _reinforcement_result_root(_load_reinforcement_summary(state))
    ids: List[str] = []
    for item in root.get("失败分组") or []:
        if isinstance(item, dict) and str(item.get("task_id") or "").strip():
            ids.append(str(item["task_id"]).strip())
    if not ids:
        ids = _id_list(root.get("failed_task_ids"))
    if not ids:
        ids = _id_list(_safe_dict(state.get("reinforcement_batch_status")).get("failed_task_ids"))
    return sorted(set(ids))


def _load_capacity_summary(state: AgentState) -> Dict[str, Any]:
    """读取承载力批次汇总：优先内存，其次 state 记录路径 / Handoff 成果路径 / output_dir 标准位置。"""
    for key in ("check_result", "capacity_check_result"):
        payload = _safe_dict(state.get(key))
        if payload:
            return payload
    candidates: List[str] = []
    handoff = _safe_dict(state.get("latest_handoff"))
    for artifact in handoff.get("produced_artifacts") or []:
        if isinstance(artifact, dict) and str(artifact.get("artifact_type") or "") == "check_result":
            if artifact.get("path"):
                candidates.append(str(artifact["path"]))
    if state.get("capacity_check_summary_path"):
        candidates.append(str(state["capacity_check_summary_path"]))
    output_dir = str(state.get("output_dir") or "").strip()
    if output_dir:
        candidates.append(os.path.join(output_dir, "capacity_check", "capacity_check_batch_summary.json"))
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload:
            return payload
    return {}


def _reinforcement_retry_task_ids(state: AgentState, revision_context: Dict[str, Any]) -> List[str]:
    """配筋返修要重做的任务号（设计组命名空间）。

    取并集（不是"第一个非空即用"）：本轮返修上下文显式下发的任务 + 上一轮生成失败的分组
    + 本轮验算未通过的分组。少取任何一个都会让批次重新变成"未完成"，
    于是又回到人工复核（2026-09-15 的 示例项目K29 运行即卡在这里）。
    """
    ids: List[str] = []

    def _add(values: List[str]) -> None:
        for item in values:
            if item and item not in ids:
                ids.append(item)

    _add(_id_list(revision_context.get("reinforcement_retry_task_ids")))
    batch = _safe_dict(state.get("reinforcement_batch_status"))
    if batch.get("stage_complete") is False:
        _add(_id_list(batch.get("failed_task_ids")))
    summary = _load_capacity_summary(state)
    if summary:
        _add(_failed_check_task_ids(summary))
    _add(_previous_reinforcement_failure_ids(state))
    if not ids and revision_context.get("target_step") == "reinforcement_design":
        _add(_id_list(state.get("failed_task_ids")))
    return sorted(set(ids))


def _reinforcement_already_retried(state: AgentState) -> bool:
    """是否已经对同一批成果做过配筋返修（用于决定是否升级到尺寸设计）。"""
    rounds = state.get("stage_revision_rounds")
    if isinstance(rounds, dict) and int(rounds.get("structural_design") or 0) >= 1:
        return True
    return int(state.get("check_iteration_index") or 0) >= 2


def generate_modeling_feedback_action(state: AgentState) -> Dict[str, Any]:
    """根据确定性验算信号生成 pass / revise_reinforcement / revise_dimension 决策。"""
    try:
        check_result = _safe_dict(state.get("check_result") or state.get("capacity_check_result"))
        if not check_result:
            return {"error": "缺少 check_result，无法生成验算反馈决策。"}

        decision = _fallback_modeling_feedback_decision(
            check_result,
            reinforcement_already_retried=_reinforcement_already_retried(state),
        )
        decision["decision_source"] = "deterministic_rule"

        axial_summary = aggregate_axial_check_results(
            state.get("reinforcement_design_result")
        )
        if axial_summary.get("has_slenderness_review"):
            decision = {
                "overall_status": "revise_dimension",
                "next_action": "revise_dimension",
                "target_agent": "StructuralDesignAgent",
                "target_step": "dimension_design",
                "control_reason": (
                    "墩柱长细比超出表5.3.1适用范围，"
                    f"任务={axial_summary['slenderness_review_task_ids']}。"
                ),
                "revision_instruction": "墩柱长细比超出稳定系数表适用范围，返回尺寸设计调整柱径或净高。",
                "decision_basis": "column_slenderness_out_of_range",
                "decision_source": "deterministic_axial_check",
                "requires_rerun_check": True,
            }
        elif axial_summary.get("has_axial_failure"):
            decision = {
                "overall_status": "revise_reinforcement",
                "next_action": "revise_reinforcement",
                "target_agent": "StructuralDesignAgent",
                "target_step": "reinforcement_design",
                "control_reason": (
                    "墩柱普通轴心受压承载力不足，"
                    f"任务={axial_summary['axial_fail_task_ids']}。"
                ),
                "revision_instruction": "墩柱轴压承载力不足，返回联合配筋设计调整纵向主筋。",
                "decision_basis": "column_axial_capacity_failure",
                "decision_source": "deterministic_axial_check",
                "requires_rerun_check": True,
            }

        allowed = {"pass", "revise_reinforcement", "revise_dimension"}
        next_action = str(decision.get("next_action") or decision.get("overall_status") or "").strip()
        if next_action not in allowed:
            decision = _fallback_modeling_feedback_decision(
                check_result,
                reinforcement_already_retried=_reinforcement_already_retried(state),
            )
            next_action = decision["next_action"]

        if next_action == "pass":
            decision.update({
                "overall_status": "pass",
                "next_action": "pass",
                "target_agent": "END",
                "target_step": None,
                "requires_rerun_check": False,
            })
            revision_context = None
        else:
            target_step = "reinforcement_design" if next_action == "revise_reinforcement" else "dimension_design"
            decision.update({
                "overall_status": next_action,
                "next_action": next_action,
                "target_agent": "StructuralDesignAgent",
                "target_step": target_step,
                "requires_rerun_check": True,
            })
            revision_context = {
                "source_agent": "ModelingCheckAgent",
                "target_agent": "StructuralDesignAgent",
                "target_step": target_step,
                "revision_type": next_action,
                "control_reason": decision.get("control_reason"),
                "revision_instruction": decision.get("revision_instruction"),
                "control_sections": check_result.get("control_sections"),
                "overall_check": check_result.get("overall_check"),
                "utilization_summary": check_result.get("utilization_summary"),
                "requires_rerun_check": True,
                **_modeling_rework_retry_scope(state, check_result, next_action),
            }

        output_dir = state.get("output_dir") or "outputs"
        decision_path = os.path.join(output_dir, "modeling_check", "feedback_decision.json")
        write_json(decision_path, decision)
        revision_context_path = None
        if revision_context:
            revision_context_path = os.path.join(output_dir, "modeling_check", "revision_context.json")
            write_json(revision_context_path, revision_context)

        return {
            "feedback_decision": decision,
            "feedback_decision_path": decision_path,
            "revision_context": revision_context,
            "revision_context_path": revision_context_path,
            "modeling_revision_instruction": decision.get("revision_instruction"),
            "error": None,
        }
    except Exception as e:
        logger.error("验算反馈决策生成失败：%s", e, exc_info=True)
        return {"error": str(e)}
def generate_reinforcement_drawings_action(state: AgentState) -> Dict[str, Any]:
    if reinforcement_drawing_tool is None:
        return {"error": "未能导入 reinforcement_drawing_tool。"}
    try:
        result = reinforcement_drawing_tool(
            pier_group_result=state.get("pier_group_result") or {},
            dimension_design_result=state.get("dimension_design_result") or {},
            reinforcement_design_result=state.get("reinforcement_design_result") or {},
            capacity_check_result=(
                state.get("capacity_check_result") or state.get("check_result")
            ),
            output_dir=state.get("output_dir") or "outputs",
            allow_unverified=bool(state.get("accepted_risks")),
            accepted_risks=state.get("accepted_risks") or [],
        )
        return {
            "drawing_package_result": result,
            "drawing_index_path": result.get("drawing_index_path"),
            "design_manifest_path": result.get("design_manifest_path"),
            "cad_script_paths": result.get("cad_script_paths") or [],
            "drawing_preview_paths": result.get("drawing_preview_paths") or [],
        }
    except Exception as exc:
        return {"error": str(exc)}
