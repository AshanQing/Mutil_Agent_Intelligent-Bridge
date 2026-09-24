from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from bridge_agents.prompt_audit import artifact_attempt, record_prompt_audit, reserve_artifact_path
from bridge_agents.prompt_registry import render_external_prompt, render_prompt

load_dotenv()


# ============================================================
# 通用 I/O 与 LLM 工具
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
        "structure", "structural_design", "design_unit", "dimension_design",
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
    # 优先使用结构设计专用模型，其次 controller。
    role_config = llm_settings.get("structure") or llm_settings.get("dimension") or llm_settings.get("controller") or llm_settings
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


def _read_text(path: Optional[Union[str, Path]], default: str = "") -> str:
    if not path:
        return default
    path = Path(path)
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


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


def _collect_single_unit_inputs(design_units_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    root = design_units_result.get("任务1_设计单元提取结果", design_units_result)
    bridge_list = root.get("桥梁列表", []) if isinstance(root, dict) else []
    single_units: List[Dict[str, Any]] = []

    bridge_keys = ["桥梁编号", "桥梁名称", "桥型", "上部结构类型"]
    for bridge in bridge_list:
        if not isinstance(bridge, dict):
            continue
        for unit in bridge.get("单联尺寸设计输入", []) or []:
            if isinstance(unit, dict):
                # 从桥梁层向下传播字段，避免下游缺字段
                for key in bridge_keys:
                    if key not in unit and key in bridge:
                        unit[key] = bridge[key]
                single_units.append(unit)

    return single_units


ALLOWED_PIER_ROLES = {"桥台", "边墩", "中间墩"}


def _parse_station(value: Any) -> Optional[float]:
    text = str(value or "").strip().upper().replace(" ", "")
    match = re.search(r"[KZ]?(\d+)\+(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1)) * 1000.0 + float(match.group(2))
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_span_combo(value: Any) -> List[float]:
    text = str(value or "").replace("×", "x").replace("X", "x").replace("*", "x")
    spans: List[float] = []

    def expand(match: re.Match[str]) -> str:
        spans.extend([float(match.group(2))] * int(match.group(1)))
        return "+"

    remaining = re.sub(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)", expand, text)
    spans.extend(float(item) for item in re.findall(r"\d+(?:\.\d+)?", remaining))
    return spans


def _pier_id(value: Any) -> str:
    match = re.search(r"\d+", str(value or ""))
    return match.group(0) if match else str(value or "").strip()


def _layout_bridge_catalog(layout_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """按桥位编号索引布跨方案。

    真实产物是 ``{"设桥总览": {"桥位列表": [...]}}``，工具入参也可能是被其它键名
    包了一层的工具返回；这里逐层解包到出现 ``桥位列表`` 为止。早期版本只认
    ``layout_result`` 等键名而漏掉 ``设桥总览``，导致 catalog 恒为空、每条设计单元
    都被判"无法映射"，重试永远不可能成功。
    """
    payload = layout_result
    for _ in range(4):
        if not isinstance(payload, dict) or "桥位列表" in payload:
            break
        for key in ("设桥总览", "layout_result", "design_result", "result", "output"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
        else:
            break

    catalog: Dict[str, Dict[str, Any]] = {}
    for bridge in payload.get("桥位列表", []) or []:
        if not isinstance(bridge, dict):
            continue
        bridge_id = str(bridge.get("桥位编号") or bridge.get("桥梁编号") or "").strip()
        unified = bridge.get("统一布跨方案")
        if bridge_id and isinstance(unified, dict):
            catalog[bridge_id] = unified
        for side in bridge.get("分幅布跨方案列表", []) or []:
            if not isinstance(side, dict):
                continue
            side_label = str(side.get("幅别") or side.get("侧别") or "").strip()
            suffix = "R" if "右" in side_label else "L" if "左" in side_label else side_label
            if bridge_id and suffix:
                catalog[f"{bridge_id}-{suffix}"] = side
    return catalog


def validate_design_units_result(
    layout_result: Dict[str, Any],
    design_units_result: Dict[str, Any],
) -> Dict[str, Any]:
    """校验设计单元的工程语义，阻止非法角色和失真的分联进入下游。"""
    errors: List[str] = []
    warnings: List[str] = []
    catalog = _layout_bridge_catalog(layout_result)
    root = design_units_result.get("任务1_设计单元提取结果", design_units_result)
    bridges = root.get("桥梁列表", []) if isinstance(root, dict) else []
    if not isinstance(bridges, list) or not bridges:
        return {"valid": False, "errors": ["结果中未找到桥梁列表。"], "warnings": []}

    for bridge in bridges:
        if not isinstance(bridge, dict):
            errors.append("桥梁列表中存在非对象条目。")
            continue
        bridge_id = str(bridge.get("桥梁编号") or "").strip()
        source = catalog.get(bridge_id)
        if source is None:
            errors.append(f"桥梁编号 {bridge_id or '<空>'} 无法映射到设桥布跨结果。")
            source_piers: set[str] = set()
        else:
            source_piers = {
                _pier_id(item.get("墩号"))
                for item in source.get("墩位与墩高", []) or []
                if isinstance(item, dict)
            }

        units = bridge.get("设计单元列表", []) or []
        downstream_units = bridge.get("单联尺寸设计输入", []) or []
        if not units:
            errors.append(f"桥梁 {bridge_id} 缺少设计单元列表。")
        if not downstream_units:
            errors.append(f"桥梁 {bridge_id} 缺少单联尺寸设计输入。")

        unit_by_id = {
            str(unit.get("单元编号")): unit
            for unit in units
            if isinstance(unit, dict) and unit.get("单元编号") is not None
        }
        downstream_by_id = {
            str(unit.get("单元编号")): unit
            for unit in downstream_units
            if isinstance(unit, dict) and unit.get("单元编号") is not None
        }
        if set(unit_by_id) != set(downstream_by_id):
            errors.append(f"桥梁 {bridge_id} 的设计单元列表与单联尺寸设计输入编号不一致。")

        unit_lengths: List[float] = []
        unit_pier_roles: Dict[str, List[str]] = {}
        previous_end: Optional[float] = None
        for unit_id, unit in unit_by_id.items():
            info = unit.get("本联信息") if isinstance(unit.get("本联信息"), dict) else {}
            spans = _parse_span_combo(info.get("跨径组合"))
            declared_count = info.get("跨数")
            if not spans:
                errors.append(f"单元 {unit_id} 无法解析跨径组合。")
                continue
            try:
                count_matches = declared_count is not None and int(declared_count) == len(spans)
            except (TypeError, ValueError):
                count_matches = False
            if not count_matches:
                errors.append(f"单元 {unit_id} 的跨数与跨径组合不一致。")
            length = sum(spans)
            unit_lengths.append(length)
            start = _parse_station(info.get("起点桩号"))
            end = _parse_station(info.get("终点桩号"))
            if start is None or end is None or end <= start:
                errors.append(f"单元 {unit_id} 的起终点桩号无效。")
            elif not math.isclose(end - start, length, abs_tol=0.5):
                errors.append(f"单元 {unit_id} 的桩号长度与跨径合计不一致。")
            if previous_end is not None and start is not None and not math.isclose(start, previous_end, abs_tol=0.5):
                errors.append(f"单元 {unit_id} 未与前一联连续衔接。")
            previous_end = end

            if max(spans) <= 30.0 and length > 120.0 + 0.5:
                errors.append(f"单元 {unit_id} 为30m及以下标准跨，联长不得超过120m。")
            elif max(spans) <= 40.0 and length > 200.0 + 0.5:
                errors.append(f"单元 {unit_id} 为40m及以下标准跨，联长不得超过200m。")

            for pier in unit.get("桥墩信息", []) or []:
                if not isinstance(pier, dict):
                    continue
                role = str(pier.get("墩位角色") or "").strip()
                raw_pier_id = str(pier.get("原始墩号") or "")
                normalized_id = _pier_id(raw_pier_id)
                if role not in ALLOWED_PIER_ROLES:
                    errors.append(f"单元 {unit_id} 的墩位 {raw_pier_id} 使用非法角色“{role}”。")
                if source_piers and normalized_id not in source_piers:
                    errors.append(f"单元 {unit_id} 引用了布跨结果中不存在的墩号 {raw_pier_id}。")
                unit_pier_roles.setdefault(normalized_id, []).append(role)

            downstream = downstream_by_id.get(unit_id, {})
            d_info = downstream.get("本联信息") if isinstance(downstream.get("本联信息"), dict) else {}
            try:
                downstream_count_matches = int(d_info.get("跨数") or -1) == len(spans)
            except (TypeError, ValueError):
                downstream_count_matches = False
            if d_info and (_parse_span_combo(d_info.get("跨径组合")) != spans or not downstream_count_matches):
                errors.append(f"单元 {unit_id} 的下游尺寸输入与设计单元跨径信息不一致。")
            for group in downstream.get("本联设计分组", []) or []:
                if not isinstance(group, dict):
                    continue
                role = str(group.get("墩位角色") or "").strip()
                if role not in ALLOWED_PIER_ROLES:
                    errors.append(f"单元 {unit_id} 的设计分组使用非法角色“{role}”。")
                for raw_pier_id in group.get("包含桥墩号列表", []) or []:
                    if source_piers and _pier_id(raw_pier_id) not in source_piers:
                        errors.append(f"单元 {unit_id} 的设计分组引用不存在的墩号 {raw_pier_id}。")

            grouped_pier_ids: set[str] = set()
            for group in downstream.get("本联设计分组", []) or []:
                if not isinstance(group, dict):
                    continue
                for raw_pier_id in group.get("包含桥墩号列表", []) or []:
                    grouped_pier_ids.add(_pier_id(raw_pier_id))
            uncovered_pier_ids = [
                _pier_id(pier.get("原始墩号"))
                for pier in unit.get("桥墩信息", []) or []
                if isinstance(pier, dict)
                and str(pier.get("墩位角色") or "").strip() != "桥台"
                and _pier_id(pier.get("原始墩号")) not in grouped_pier_ids
            ]
            if uncovered_pier_ids:
                errors.append(
                    f"单元 {unit_id} 的本联设计分组未覆盖桥墩: {sorted(uncovered_pier_ids)}"
                )

        for normalized_id, roles in unit_pier_roles.items():
            if len(roles) > 1 and any(role != "边墩" for role in roles):
                errors.append(f"桥梁 {bridge_id} 的连接墩 {normalized_id} 必须在相邻单元中标记为边墩。")
        if len(unit_lengths) > 1 and max(unit_lengths) - min(unit_lengths) > 40.5:
            warnings.append(f"桥梁 {bridge_id} 的相邻设计单元长度差较大，应复核均衡分联依据。")

    return {"valid": not errors, "errors": errors, "warnings": warnings}


# ============================================================
# 对外工具函数
# ============================================================
def extract_design_units_tool(
    layout_result: Dict[str, Any],
    design_input: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    prompt_path: Optional[str] = None,
) -> Dict[str, Any]:
    """从设桥布跨结果中提取结构设计单元。

    输入：
    - layout_result: 修正后的设桥布跨 JSON。
    - output_dir: 输出目录。
    - config_path: 配置文件路径。
    - prompt_path: 可选，设计单元划分 Prompt 模板路径。

    输出：
    - design_units_result: LLM 输出的完整设计单元提取结果。
    - single_unit_inputs: 后续尺寸设计逐联输入列表。
    """
    if not layout_result:
        return {"success": False, "error": "layout_result 为空，无法进行设计单元提取。"}

    settings = _load_settings(config_path)
    output_root = Path(output_dir or _pick(settings, "output_dir", default="outputs"))
    stage_dir = output_root / "structural_design" / "design_units"

    prompt_id = _pick(settings, "design_unit_extraction_prompt_id", default="tasks.design_unit_extraction.v1")
    try:
        llm = _build_llm(config_path)
    except Exception as e:
        return {"success": False, "error": f"设计单元提取 LLM 初始化失败：{e}"}
    validation_feedback_json = ""
    result: Dict[str, Any] = {}
    validation: Dict[str, Any] = {"valid": False, "errors": [], "warnings": []}
    prompt_file = ""
    rendered_prompt = None
    for attempt in (1, 2):
        prompt_context = {
            "layout_result_json": json.dumps(_json_safe(layout_result), ensure_ascii=False, indent=2),
            "validation_feedback_json": validation_feedback_json,
        }
        rendered_prompt = (
            render_external_prompt(prompt_path, prompt_context)
            if prompt_path
            else render_prompt(str(prompt_id), prompt_context, config_path=config_path)
        )
        prompt_text = rendered_prompt.user_content or rendered_prompt.system_content
        prompt_target = reserve_artifact_path(stage_dir / f"design_unit_extraction_prompt_attempt_{attempt}.txt")
        prompt_file = _write_text(prompt_target, prompt_text)
        compatibility_prompt = stage_dir / "design_unit_extraction_prompt.txt"
        if not compatibility_prompt.exists():
            _write_text(compatibility_prompt, prompt_text)
        audit_attempt = artifact_attempt(prompt_target)
        record_prompt_audit(
            output_root,
            prompt_id=rendered_prompt.prompt_id,
            version=rendered_prompt.version,
            template_sha256=rendered_prompt.template_sha256,
            stage="design_unit_extraction",
            scope_id="all_bridges",
            attempt=audit_attempt,
            rendered_path=prompt_file,
            status="rendered",
            metadata={"semantic_attempt": attempt},
        )

        try:
            response = llm.invoke([
                (
                    "system",
                    "你是一名桥梁工程设计师，负责从设桥布跨结果中提取按联划分的下部结构设计单元。只输出 JSON。",
                ),
                ("user", prompt_text),
            ])
            result = _extract_json_object(response.content.strip())
        except Exception as e:
            record_prompt_audit(
                output_root,
                prompt_id=rendered_prompt.prompt_id,
                version=rendered_prompt.version,
                template_sha256=rendered_prompt.template_sha256,
                stage="design_unit_extraction",
                scope_id="all_bridges",
                attempt=audit_attempt,
                rendered_path=prompt_file,
                status="failed",
                metadata={"semantic_attempt": attempt, "error": str(e)},
            )
            return {
                "success": False,
                "error": f"设计单元提取 LLM 调用或 JSON 解析失败：{e}",
                "prompt_path": prompt_file,
                "prompt_id": rendered_prompt.prompt_id,
            }

        validation = validate_design_units_result(layout_result, result)
        if validation["valid"]:
            record_prompt_audit(
                output_root,
                prompt_id=rendered_prompt.prompt_id,
                version=rendered_prompt.version,
                template_sha256=rendered_prompt.template_sha256,
                stage="design_unit_extraction",
                scope_id="all_bridges",
                attempt=audit_attempt,
                rendered_path=prompt_file,
                status="completed",
                metadata={"semantic_attempt": attempt},
            )
            break
        record_prompt_audit(
            output_root,
            prompt_id=rendered_prompt.prompt_id,
            version=rendered_prompt.version,
            template_sha256=rendered_prompt.template_sha256,
            stage="design_unit_extraction",
            scope_id="all_bridges",
            attempt=audit_attempt,
            rendered_path=prompt_file,
            status="semantic_failed",
            metadata={"semantic_attempt": attempt, "errors": validation["errors"]},
        )
        validation_feedback_json = json.dumps(
            {"校验错误": validation["errors"], "上一版结果": _json_safe(result)},
            ensure_ascii=False,
            indent=2,
        )

    if not validation["valid"]:
        return {
            "success": False,
            "error": "设计单元结果连续两次未通过工程语义校验。",
            "manual_review_required": True,
            "design_units_result": result,
            "prompt_path": prompt_file,
            "semantic_validation": {**validation, "repair_attempted": True},
        }

    single_unit_inputs = _collect_single_unit_inputs(result)
    if not single_unit_inputs:
        return {
            "success": False,
            "error": "设计单元提取结果中未找到 单联尺寸设计输入。",
            "design_units_result": result,
            "prompt_path": prompt_file,
            "semantic_validation": {**validation, "repair_attempted": bool(validation_feedback_json)},
        }

    result_file = _write_json(stage_dir / "design_units_result.json", result)
    single_units_file = _write_json(stage_dir / "single_unit_inputs.json", single_unit_inputs)

    return {
        "success": True,
        "design_units_result": result,
        "single_unit_inputs": single_unit_inputs,
        "prompt_id": rendered_prompt.prompt_id,
        "prompt_version": rendered_prompt.version,
        "template_sha256": rendered_prompt.template_sha256,
        "semantic_validation": {**validation, "repair_attempted": bool(validation_feedback_json)},
        "output_files": {
            "prompt_path": prompt_file,
            "design_units_result_path": result_file,
            "single_unit_inputs_path": single_units_file,
        },
    }
