from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from bridge_agents.batch_execution import run_indexed_batch
from bridge_agents.llm_generation_guard import (
    StructuredGenerationError,
    generate_structured_output,
)
from bridge_agents.prompt_audit import artifact_attempt, record_prompt_audit, reserve_artifact_path
from tools.dimension_design_prompt_tool import build_dimension_design_prompt_tool

load_dotenv()


# ============================================================
# 通用函数
# ============================================================
def _load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick(settings: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in settings and settings[key] not in [None, ""]:
            return settings[key]

    sections = [
        "paths", "data", "layout", "prompts", "llm", "agent",
        "structure", "structural_design", "dimension_design", "retrieval",
    ]
    for section in sections:
        value = settings.get(section)
        if isinstance(value, dict):
            for key in keys:
                if key in value and value[key] not in [None, ""]:
                    return value[key]
    return default


def _resolve_api_key(api_config: Dict[str, Any]) -> str:
    api_key = api_config.get("api_key")
    if api_key == "ENV":
        env_name = api_config.get("api_key_env", "DEEPSEEK_API_KEY")
        api_key = os.getenv(env_name)
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未找到 LLM API Key，请检查 settings.yaml 或环境变量。")
    return api_key


def _build_llm(config_path: str = "config/settings.yaml") -> ChatOpenAI:
    settings = _load_settings(config_path)
    llm_settings = settings.get("llm", {})
    role_config = llm_settings.get("dimension") or llm_settings.get("structure") or llm_settings.get("controller") or llm_settings
    if not isinstance(role_config, dict):
        role_config = {}

    return ChatOpenAI(
        model=role_config.get("model", "deepseek-chat"),
        openai_api_key=_resolve_api_key(role_config),
        openai_api_base=role_config.get("base_url", "https://api.deepseek.com"),
        temperature=role_config.get("temperature", 0),
    )


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


def _write_json(path: Union[str, Path], data: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, ensure_ascii=False, indent=2)
    return str(path)


def _write_text(path: Union[str, Path], text: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")
    return str(path)


def _extract_json_object(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(content[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("LLM 输出 JSON 顶层必须是对象。")
    return data


def _extract_single_unit_inputs(design_units: Any) -> List[Dict[str, Any]]:
    """兼容 design_unit_extractor_tool 的包装结果和裸结果。"""
    if not design_units:
        return []

    if isinstance(design_units, dict):
        if isinstance(design_units.get("single_unit_inputs"), list):
            return [u for u in design_units["single_unit_inputs"] if isinstance(u, dict)]

        if isinstance(design_units.get("design_units_result"), dict):
            return _extract_single_unit_inputs(design_units["design_units_result"])

        root = design_units.get("任务1_设计单元提取结果", design_units)
        if isinstance(root, dict):
            bridge_list = root.get("桥梁列表", [])
            bridge_keys = ["桥梁编号", "桥梁名称", "桥型", "上部结构类型"]
            single_units: List[Dict[str, Any]] = []
            for bridge in bridge_list if isinstance(bridge_list, list) else []:
                if not isinstance(bridge, dict):
                    continue
                for unit in bridge.get("单联尺寸设计输入", []) or []:
                    if isinstance(unit, dict):
                        for key in bridge_keys:
                            if key not in unit and key in bridge:
                                unit[key] = bridge[key]
                        single_units.append(unit)
            return single_units

    return []


def _group_unit_results_by_bridge(unit_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bridge_map: Dict[str, Dict[str, Any]] = {}

    for item in unit_results:
        unit_input = item.get("single_unit_input", {}) or {}
        bridge_id = str(unit_input.get("桥梁编号", "未知桥梁"))
        if bridge_id not in bridge_map:
            bridge_map[bridge_id] = {
                "桥梁编号": bridge_id,
                "桥型": unit_input.get("桥型"),
                "单元尺寸设计结果": [],
            }

        result = item.get("dimension_result", {}) or {}
        bridge_map[bridge_id]["单元尺寸设计结果"].append({
            "单元编号": unit_input.get("单元编号"),
            "联号": unit_input.get("联号"),
            "本联信息": unit_input.get("本联信息"),
            "桥面宽度信息": unit_input.get("桥面宽度信息"),
            "分组尺寸设计结果": result.get("分组尺寸设计结果", []),
            "桥墩尺寸映射关系": result.get("桥墩尺寸映射关系", []),
        })

    return list(bridge_map.values())


def _extract_existing_unit_results(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("dimension_design_result"), dict):
        return _extract_existing_unit_results(value["dimension_design_result"])
    root = value.get("任务2_下部结构尺寸设计结果", value)
    rows = root.get("单元原始结果") if isinstance(root, dict) else None
    return [dict(row) for row in (rows or []) if isinstance(row, dict)]


def _match_retry_unit_ids(
    retry_ids: set,
    known_unit_ids: List[str],
) -> tuple:
    """把重试 ID 匹配到尺寸设计单元编号，返回 (命中的单元号集合, 未匹配的 ID 集合)。

    优先精确匹配；匹配不上时按前缀逐级降级，用于吸收上游误传的其它命名空间 ID
    （例如把配筋设计任务号 2-2-1-G1 降级为尺寸设计单元号 2-2）。
    """
    known = {str(unit_id) for unit_id in known_unit_ids if str(unit_id)}
    matched = set()
    unresolved = set()
    for raw_id in sorted(retry_ids):
        unit_id = str(raw_id)
        if unit_id in known:
            matched.add(unit_id)
            continue
        parts = unit_id.split("-")
        hit = ""
        for cut in range(len(parts), 0, -1):
            candidate = "-".join(parts[:cut])
            if candidate in known:
                hit = candidate
                break
        if hit:
            matched.add(hit)
        else:
            unresolved.add(unit_id)
    return matched, unresolved


# ============================================================
# 对外工具函数
# ============================================================
def dimension_design_tool(
    design_units: Dict[str, Any],
    layout_result: Optional[Dict[str, Any]] = None,
    standards_path: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    output_dir: Optional[str] = None,
    samples_yaml_path: Optional[str] = None,
    template_json_path: Optional[str] = None,
    max_examples_per_group: Optional[int] = None,
    max_workers: Optional[int] = None,
    max_format_repairs: Optional[int] = None,
    max_regenerations: Optional[int] = None,
    retry_unit_ids: Optional[List[str]] = None,
    existing_dimension_design_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """以有界并发逐联完成下部结构尺寸设计。

    输入：design_units，即 design_unit_extractor_tool 的输出。
    输出：任务2_下部结构尺寸设计结果。
    """
    single_unit_inputs = _extract_single_unit_inputs(design_units)
    if not single_unit_inputs:
        return {"success": False, "error": "design_units 中未找到 单联尺寸设计输入。"}

    settings = _load_settings(config_path)
    samples_yaml_path = samples_yaml_path or _pick(
        settings,
        "dimension_samples_yaml_path",
        "structure_samples_yaml_path",
        "pier_samples_yaml_path",
        default=None,
    )
    prompt_id = _pick(settings, "cap_dimension_design_prompt_id", default="tasks.cap_dimension_design.v1")
    max_examples_per_group = int(max_examples_per_group or _pick(settings, "max_examples_per_group", default=3))
    max_workers = int(max_workers or _pick(settings, "dimension_max_workers", default=4))
    max_format_repairs = int(
        max_format_repairs
        if max_format_repairs is not None
        else _pick(settings, "dimension_max_format_repairs", default=1)
    )
    max_regenerations = int(
        max_regenerations
        if max_regenerations is not None
        else _pick(settings, "dimension_max_regenerations", default=2)
    )
    retry_ids = {str(unit_id) for unit_id in (retry_unit_ids or []) if unit_id}
    known_unit_ids = [
        str(unit.get("单元编号") or "")
        for unit in single_unit_inputs
        if str(unit.get("单元编号") or "")
    ]
    matched_ids, unresolved_ids = _match_retry_unit_ids(retry_ids, known_unit_ids)
    retry_fallback = False
    retry_note = ""
    if unresolved_ids:
        if matched_ids:
            # 部分 ID 能对上（例如配筋任务号按前缀降级到尺寸单元号）：先按已匹配单元重跑。
            retry_note = (
                "部分重试 ID 未匹配到尺寸设计单元，已按可匹配单元重跑："
                f"unmatched={sorted(unresolved_ids)}"
            )
        else:
            # 一个都对不上（多为上游传错命名空间）：整体重跑，不让整个阶段因 ID 不匹配而失败。
            matched_ids = set(known_unit_ids)
            retry_fallback = True
            retry_note = (
                "重试 ID 未匹配到任何尺寸设计单元，已降级为整体重跑："
                f"unmatched={sorted(unresolved_ids)}"
            )
    effective_retry_ids = matched_ids if retry_ids else set()
    selected_unit_inputs = [
        unit
        for unit in single_unit_inputs
        if not effective_retry_ids or str(unit.get("单元编号") or "") in effective_retry_ids
    ]
    existing_results = _extract_existing_unit_results(existing_dimension_design_result)
    retained_results = [
        row
        for row in existing_results
        if str((row.get("single_unit_input") or {}).get("单元编号") or "")
        not in effective_retry_ids
    ] if effective_retry_ids else []

    if not samples_yaml_path:
        return {"success": False, "error": "缺少 dimension_samples_yaml_path，请在 settings.yaml 的 structure 或 dimension_design 中配置。"}

    output_root = Path(output_dir or _pick(settings, "output_dir", default="outputs"))
    stage_dir = output_root / "structural_design" / "dimension_design"
    stage_dir.mkdir(parents=True, exist_ok=True)

    llm = _build_llm(config_path)

    def process_unit(indexed_unit: tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        index, single_unit = indexed_unit
        unit_id = str(single_unit.get("单元编号") or f"unit_{index}")
        safe_unit_id = unit_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        prompt_payload: Optional[Dict[str, Any]] = None
        prompt_txt_path: Optional[str] = None
        audit_attempt = 1

        try:
            prompt_payload = build_dimension_design_prompt_tool(
                single_unit_input=single_unit,
                samples_yaml_path=str(samples_yaml_path),
                template_json_path=template_json_path,
                max_examples_per_group=max_examples_per_group,
                config_path=config_path,
                prompt_id=str(prompt_id),
            )
            prompt_json = prompt_payload["prompt_json"]
            prompt_text = prompt_payload["llm_prompt_text"]

            prompt_json_target = reserve_artifact_path(stage_dir / f"prompt_{safe_unit_id}.json")
            prompt_txt_target = reserve_artifact_path(stage_dir / f"prompt_{safe_unit_id}.txt")
            prompt_json_path = _write_json(prompt_json_target, prompt_json)
            prompt_txt_path = _write_text(prompt_txt_target, prompt_text)
            audit_attempt = artifact_attempt(prompt_txt_target)
            record_prompt_audit(
                output_root,
                prompt_id=prompt_payload.get("prompt_id"),
                version=prompt_payload.get("prompt_version"),
                template_sha256=prompt_payload.get("template_sha256"),
                stage="dimension_design",
                scope_id=unit_id,
                attempt=audit_attempt,
                rendered_path=prompt_txt_path,
                status="rendered",
                metadata={
                    "match_debug": prompt_payload.get("match_debug", []),
                },
            )

            system_prompt = (
                "你是一名桥梁结构设计工程师，负责根据单联设计输入和示例样本"
                "生成下部结构尺寸设计结果。只输出 JSON。"
            )

            def invoke_generation(user_prompt: str) -> str:
                response = llm.invoke([
                    ("system", system_prompt),
                    ("user", user_prompt),
                ])
                return str(response.content or "").strip()

            generation = generate_structured_output(
                invoke=invoke_generation,
                parse=_extract_json_object,
                initial_prompt=prompt_text,
                repair_prompt=lambda raw, error: (
                    "请只修复下面内容的 JSON 格式，保持工程数值和语义不变。"
                    "输出必须是一个完整 JSON 对象，不要解释。\n"
                    f"解析错误：{error}\n待修复内容：\n{raw}"
                ),
                regeneration_prompt=lambda error: (
                    f"{prompt_text}\n\n上一次输出无法解析：{error}\n"
                    "请重新完成本单元设计，只输出完整 JSON 对象。"
                ),
                max_format_repairs=max_format_repairs,
                max_regenerations=max_regenerations,
            )
            dimension_result = generation.value
            raw_response_paths = []
            for attempt in generation.attempts:
                raw_path = stage_dir / (
                    f"response_{safe_unit_id}_attempt_{attempt.index}_{attempt.kind}.txt"
                )
                raw_response_paths.append(_write_text(raw_path, attempt.raw_response))

            result_path = _write_json(stage_dir / f"dimension_result_{safe_unit_id}.json", dimension_result)
            record_prompt_audit(
                output_root,
                prompt_id=prompt_payload.get("prompt_id"),
                version=prompt_payload.get("prompt_version"),
                template_sha256=prompt_payload.get("template_sha256"),
                stage="dimension_design",
                scope_id=unit_id,
                attempt=audit_attempt,
                rendered_path=prompt_txt_path,
                status="completed",
            )

            return {
                "unit_index": index,
                "single_unit_input": single_unit,
                "dimension_result": dimension_result,
                "match_debug": prompt_payload.get("match_debug", []),
                "compatibility_warnings": prompt_payload.get("compatibility_warnings", []),
                "prompt_id": prompt_payload.get("prompt_id"),
                "prompt_version": prompt_payload.get("prompt_version"),
                "template_sha256": prompt_payload.get("template_sha256"),
                "generation_attempts": [
                    {
                        "kind": attempt.kind,
                        "index": attempt.index,
                        "error": attempt.error,
                        "raw_response_path": raw_response_paths[position],
                    }
                    for position, attempt in enumerate(generation.attempts)
                ],
                "output_files": {
                    "prompt_json_path": prompt_json_path,
                    "prompt_txt_path": prompt_txt_path,
                    "dimension_result_path": result_path,
                    "raw_response_paths": raw_response_paths,
                },
            }

        except Exception as exc:
            if isinstance(exc, StructuredGenerationError):
                for attempt in exc.attempts:
                    _write_text(
                        stage_dir
                        / f"response_{safe_unit_id}_attempt_{attempt.index}_{attempt.kind}.txt",
                        attempt.raw_response,
                    )
            record_prompt_audit(
                output_root,
                prompt_id=prompt_payload.get("prompt_id") if prompt_payload else str(prompt_id),
                version=prompt_payload.get("prompt_version") if prompt_payload else None,
                template_sha256=prompt_payload.get("template_sha256") if prompt_payload else None,
                stage="dimension_design",
                scope_id=unit_id,
                attempt=audit_attempt,
                rendered_path=prompt_txt_path,
                status="failed",
                metadata={"error": str(exc)},
            )
            raise

    original_indexes = {
        str(unit.get("单元编号") or ""): index
        for index, unit in enumerate(single_unit_inputs, start=1)
    }
    indexed_units = [
        (original_indexes.get(str(unit.get("单元编号") or ""), index), unit)
        for index, unit in enumerate(selected_unit_inputs, start=1)
    ]
    batch_results = run_indexed_batch(
        indexed_units,
        process_unit,
        max_workers=max_workers,
    )
    unit_results: List[Dict[str, Any]] = list(retained_results)
    errors: List[Dict[str, Any]] = []
    for batch_result in batch_results:
        index, single_unit = indexed_units[batch_result.index]
        if batch_result.error:
            error_item = {
                "unit_index": index,
                "单元编号": str(single_unit.get("单元编号") or f"unit_{index}"),
                "error": batch_result.error,
            }
            if "StructuredGenerationError" in batch_result.error:
                error_item["failure_type"] = "structured_generation_exhausted"
            errors.append(error_item)
        elif batch_result.value is not None:
            unit_results.append(batch_result.value)

    unit_results.sort(key=lambda item: int(item.get("unit_index") or 0))

    expected_unit_count = len(single_unit_inputs)
    completed_unit_count = len(unit_results)
    failed_unit_count = len(errors)
    stage_complete = completed_unit_count == expected_unit_count and failed_unit_count == 0
    partial_result_available = 0 < completed_unit_count < expected_unit_count
    summary = {
        "任务2_下部结构尺寸设计结果": {
            "桥梁列表": _group_unit_results_by_bridge(unit_results),
            "单元原始结果": unit_results,
            "失败单元": errors,
            "expected_unit_count": expected_unit_count,
            "completed_unit_count": completed_unit_count,
            "failed_unit_count": failed_unit_count,
            "stage_complete": stage_complete,
            "partial_result_available": partial_result_available,
        }
    }
    if retry_note:
        summary["任务2_下部结构尺寸设计结果"]["重试说明"] = retry_note

    summary_path = _write_json(stage_dir / "dimension_design_result.json", summary)
    evidence_bundles = {
        str(item.get("single_unit_input", {}).get("单元编号") or item.get("unit_index")):
            item.get("evidence_context")
        for item in unit_results
        if item.get("evidence_context")
    }

    if errors and not unit_results:
        return {
            "success": False,
            "error": "所有单联尺寸设计均失败。",
            "errors": errors,
            "expected_unit_count": expected_unit_count,
            "completed_unit_count": completed_unit_count,
            "failed_unit_count": failed_unit_count,
            "stage_complete": stage_complete,
            "partial_result_available": partial_result_available,
            "evidence_bundles": evidence_bundles,
            "output_files": {"dimension_design_result_path": summary_path},
        }

    return {
        "success": True,
        "dimension_design_result": summary,
        "unit_result_count": completed_unit_count,
        "expected_unit_count": expected_unit_count,
        "completed_unit_count": completed_unit_count,
        "failed_unit_count": failed_unit_count,
        "stage_complete": stage_complete,
        "partial_result_available": partial_result_available,
        "errors": errors,
        "retry_fallback": retry_fallback,
        "retry_note": retry_note,
        "evidence_bundles": evidence_bundles,
        "output_files": {
            "dimension_design_result_path": summary_path,
        },
    }
