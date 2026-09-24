from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from bridge_agents.batch_execution import run_indexed_batch
from bridge_agents.joint_reinforcement import (
    build_axial_check_summary,
    compute_axial_capacity_check,
    compute_total_vertical_load,
    inject_column_heights,
    summarize_pier_axial,
    validate_joint_reinforcement,
)
from bridge_agents.prompt_audit import artifact_attempt, record_prompt_audit, reserve_artifact_path
from tools.reinforcement_design_prompt_tool import (
    build_reinforcement_design_prompt_tool,
    extract_reinforcement_tasks_from_dimension_result,
    safe_filename,
)
from tools.cap_load_design_tool import cap_load_design_tool
from tools.cap_internal_force_analysis_tool import cap_internal_force_analysis_tool
from tools.cap_force_control_info_tool import cap_force_control_info_tool

load_dotenv()


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
        "paths", "data", "prompts", "llm", "agent",
        "structure", "structural_design", "reinforcement_design",
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
    role_config = (
        llm_settings.get("reinforcement")
        or llm_settings.get("structure")
        or llm_settings.get("controller")
        or llm_settings
    )
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


def _strip_code_fence(text: str) -> str:
    content = (text or "").strip()
    for fence in ["```yaml", "```yml", "```json", "```"]:
        if fence in content:
            return content.split(fence, 1)[1].split("```", 1)[0].strip()
    return content


def _parse_yaml_or_json(text: str) -> Dict[str, Any]:
    content = _strip_code_fence(text)
    try:
        data = yaml.safe_load(content)
    except Exception:
        data = None
    if isinstance(data, dict):
        return data
    try:
        data = json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(content[start:end + 1])
        else:
            raise ValueError("LLM 输出无法解析为 YAML/JSON 对象。")
    if not isinstance(data, dict):
        raise ValueError("LLM 输出顶层必须是对象。")
    return data


def _group_reinforcement_results(task_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bridge_map: Dict[str, Dict[str, Any]] = {}
    for item in task_results:
        task = item.get("reinforcement_task", {}) or {}
        bridge_id = str(task.get("桥梁编号", "未知桥梁"))
        if bridge_id not in bridge_map:
            bridge_map[bridge_id] = {"桥梁编号": bridge_id, "单元配筋设计结果": []}
        bridge_map[bridge_id]["单元配筋设计结果"].append({
            "单元编号": task.get("单元编号"),
            "联号": task.get("联号"),
            "分组编号": task.get("分组编号"),
            "包含桥墩号列表": task.get("包含桥墩号列表", []),
            "墩位角色": task.get("墩位角色"),
            "配筋设计结果": item.get("reinforcement_result"),
            "内力控制信息": item.get("force_control_info"),
            "桥墩配筋映射关系": [
                {"桥墩号": pier_id, "所属分组编号": task.get("分组编号"), "配筋任务ID": task.get("task_id")}
                for pier_id in (task.get("包含桥墩号列表") or [])
            ],
        })
    return list(bridge_map.values())


def _extract_existing_results(payload: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract prior successful and failed group records from nested tool output."""
    if not isinstance(payload, dict):
        return [], []
    root = payload.get("任务3_下部结构配筋设计结果")
    if isinstance(root, dict):
        return (
            [item for item in root.get("分组原始结果", []) if isinstance(item, dict)],
            [item for item in root.get("失败分组", []) if isinstance(item, dict)],
        )
    nested = payload.get("reinforcement_design_result")
    if isinstance(nested, dict):
        return _extract_existing_results(nested)
    return [], []


def _compute_pier_axial_summary(
    task: Dict[str, Any],
    force_payload: Dict[str, Any],
    load_payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """从盖梁分析结果与配筋任务计算墩柱控制轴力与竖向平衡校验。"""
    pier_info = task.get("墩柱净高信息") or {}
    net_height = pier_info.get("controlling_net_height_m")
    if net_height is None:
        return None
    internal = (force_payload.get("internal_force_output") or {})
    dead_case = internal.get("dead_case_full_beam") or {}
    reactions = dead_case.get("column_reactions_kN") or []
    task_input = task.get("桥墩尺寸信息") or {}
    geom = task_input.get("墩柱几何信息") or {}
    column_count = int(geom.get("column_count") or 0)
    # 尺寸设计任务中的 column_diameter 与 column_height 均采用 m。
    diameter_m_raw = geom.get("column_diameter")
    diameter_m = float(diameter_m_raw) if diameter_m_raw else 0.0
    total_vertical = compute_total_vertical_load(load_payload.get("load_design_output"))
    return summarize_pier_axial(
        reactions,
        column_count,
        diameter_m,
        float(net_height),
        total_vertical,
    )


def _compute_axial_check(
    task: Dict[str, Any],
    pier_axial: Optional[Dict[str, Any]],
    reinforcement_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """基于配筋结果与控制轴力执行墩柱轴压验算。"""
    if not pier_axial:
        return None
    reinforcement = reinforcement_result.get("reinforcement") or {}
    pier_column = reinforcement.get("pier_column") or {}
    task_input = task.get("桥墩尺寸信息") or {}
    geom = task_input.get("墩柱几何信息") or {}
    diameter_m = geom.get("column_diameter")
    net_height = (task.get("墩柱净高信息") or {}).get("controlling_net_height_m")
    if diameter_m is None or net_height is None:
        return None
    effective_length = task.get("墩柱计算长度信息") or {}
    return compute_axial_capacity_check(
        pier_axial["controlling_axial_force_kN"],
        float(diameter_m) * 1000.0,
        float(net_height),
        pier_column.get("longitudinal_bars", []) or [],
        effective_length_factor=effective_length.get("effective_length_factor"),
        rotational_restraint_coefficient=effective_length.get(
            "rotational_restraint_coefficient"
        ),
        horizontal_restraint_coefficient=effective_length.get(
            "horizontal_restraint_coefficient"
        ),
    )


def reinforcement_design_tool(
    dimension_design_result: Dict[str, Any],
    design_units: Optional[Dict[str, Any]] = None,
    standards_path: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    output_dir: Optional[str] = None,
    template_yaml_path: Optional[str] = None,
    samples_yaml_path: Optional[str] = None,
    max_examples: Optional[int] = None,
    max_workers: Optional[int] = None,
    max_format_repairs: Optional[int] = None,
    retry_task_ids: Optional[List[str]] = None,
    retry_unit_ids: Optional[List[str]] = None,
    existing_reinforcement_design_result: Optional[Dict[str, Any]] = None,
    pier_group_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """根据尺寸设计结果逐分组完成盖梁配筋设计。

    新流程：尺寸分组 → 荷载设计 → OpenSees盖梁内力计算 → 内力控制信息提取 → Prompt组装 → LLM配筋。
    """
    settings = _load_settings(config_path)

    prompt_id = _pick(settings, "cap_reinforcement_design_prompt_id", default="tasks.cap_reinforcement_design.v1")
    samples_yaml_path = samples_yaml_path or _pick(
        settings,
        "reinforcement_samples_yaml_path",
        "cap_rebar_samples_yaml_path",
        default=None,
    )
    column_samples_yaml_path = _pick(
        settings,
        "reinforcement_column_samples_yaml_path",
        "column_samples_yaml_path",
        default=None,
    )
    max_examples = int(max_examples or _pick(settings, "reinforcement_max_examples", "max_examples", default=2))
    max_workers = int(max_workers or _pick(settings, "reinforcement_max_workers", default=3))
    max_format_repairs = int(
        max_format_repairs
        if max_format_repairs is not None
        else _pick(settings, "reinforcement_max_format_repairs", default=1)
    )

    default_cover_mm = {
        "cover_x": float(_pick(settings, "cover_x_mm", default=50.0)),
        "cover_y": float(_pick(settings, "cover_y_mm", default=50.0)),
        "cover_z": float(_pick(settings, "cover_z_mm", default=60.0)),
    }

    all_tasks = extract_reinforcement_tasks_from_dimension_result(
        dimension_design_result=dimension_design_result,
        default_cover_mm=default_cover_mm,
    )
    all_tasks = inject_column_heights(all_tasks, pier_group_result)
    column_height_diagnostics = [
        {
            "task_id": task.get("task_id"),
            "missing_pier_groups": (task.get("墩柱净高信息") or {}).get("missing_pier_groups") or [],
        }
        for task in all_tasks
        if (task.get("墩柱净高信息") or {}).get("missing_pier_groups")
    ]
    if not all_tasks:
        return {"success": False, "error": "dimension_design_result 中未提取到可配筋的桥墩尺寸分组。"}
    retry_ids = {str(task_id) for task_id in (retry_task_ids or []) if task_id}
    retry_units = {str(unit_id) for unit_id in (retry_unit_ids or []) if unit_id}
    tasks = [
        task for task in all_tasks
        if (
            (not retry_ids and not retry_units)
            or str(task.get("task_id")) in retry_ids
            or str(task.get("单元编号")) in retry_units
        )
    ]
    if (retry_ids or retry_units) and not tasks:
        return {
            "success": False,
            "error": (
                "待重试配筋任务未在当前尺寸设计中找到: "
                f"task_ids={sorted(retry_ids)}, unit_ids={sorted(retry_units)}"
            ),
        }

    output_root = Path(output_dir or _pick(settings, "output_dir", default="outputs"))
    stage_dir = output_root / "structural_design" / "reinforcement_design"
    stage_dir.mkdir(parents=True, exist_ok=True)

    previous_results, previous_errors = _extract_existing_results(
        existing_reinforcement_design_result
    )
    prepared_tasks: List[Dict[str, Any]] = []
    evidence_bundles: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = [
        item for item in previous_errors
        if str(item.get("task_id")) not in retry_ids
        and str(item.get("单元编号")) not in retry_units
    ] if (retry_ids or retry_units) else []

    # OpenSeesPy maintains global model state. Keep all deterministic preprocessing
    # on the caller thread and only parallelize LLM generation below.
    for index, task in enumerate(tasks, start=1):
        task_id = safe_filename(task.get("task_id") or f"reinforcement_task_{index}")
        task_dir = stage_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        task_context_path = _write_json(
            reserve_artifact_path(task_dir / f"task_context_{task_id}.json"),
            {
                "task_index": index,
                "reinforcement_task": task,
                "template_yaml_path": template_yaml_path,
                "samples_yaml_path": samples_yaml_path,
                "config_path": config_path,
            },
        )
        prompt_payload: Optional[Dict[str, Any]] = None
        prompt_txt_path: Optional[str] = None
        audit_attempt = artifact_attempt(task_context_path)
        record_prompt_audit(
            output_root,
            prompt_id=str(prompt_id),
            version=None,
            template_sha256=None,
            stage="reinforcement_design",
            scope_id=task_id,
            attempt=audit_attempt,
            rendered_path=None,
            status="preprocessing",
            metadata={"task_context_path": task_context_path},
        )
        try:
            load_payload = cap_load_design_tool(
                reinforcement_task=task,
                output_dir=task_dir,
                task_id=task_id,
                write_files=True,
            )
            if not load_payload.get("success"):
                raise RuntimeError(f"荷载设计失败: {load_payload.get('error')}")

            force_payload = cap_internal_force_analysis_tool(
                analysis_load_input=load_payload["analysis_load_input"],
                output_dir=str(task_dir),
                task_id=task_id,
                save_figures=True,
            )
            if not force_payload.get("success"):
                raise RuntimeError(f"盖梁内力计算失败: {force_payload.get('error')}")

            control_payload = cap_force_control_info_tool(
                internal_force_output=force_payload["internal_force_output"],
                output_dir=task_dir,
                task_id=task_id,
                combo_name=_pick(settings, "force_combo_name", default=None),
                shear_ratio=float(_pick(settings, "force_shear_ratio", default=0.75)),
                symmetry_tolerance=float(_pick(settings, "force_symmetry_tolerance", default=0.08)),
                write_files=True,
            )
            if not control_payload.get("success"):
                raise RuntimeError(f"内力控制信息提取失败: {control_payload.get('error')}")

            force_control_info = control_payload["force_control_info"]
            pier_axial = _compute_pier_axial_summary(task, force_payload, load_payload)
            prompt_payload = build_reinforcement_design_prompt_tool(
                reinforcement_task=task,
                template_yaml_path=template_yaml_path,
                samples_yaml_path=str(samples_yaml_path) if samples_yaml_path else None,
                column_samples_yaml_path=str(column_samples_yaml_path) if column_samples_yaml_path else None,
                max_examples=max_examples,
                force_control_info=force_control_info,
                config_path=config_path,
                prompt_id=str(prompt_id),
            )
            prompt_yaml = prompt_payload["prompt_yaml"]
            prompt_text = prompt_payload["llm_prompt_text"]

            prompt_yaml_target = reserve_artifact_path(task_dir / f"prompt_{task_id}.yaml")
            prompt_txt_target = reserve_artifact_path(task_dir / f"prompt_{task_id}.txt")
            prompt_yaml_path = _write_text(prompt_yaml_target, yaml.safe_dump(prompt_yaml, allow_unicode=True, sort_keys=False, indent=2))
            prompt_txt_path = _write_text(prompt_txt_target, prompt_text)
            audit_attempt = artifact_attempt(prompt_txt_target)
            record_prompt_audit(
                output_root,
                prompt_id=prompt_payload.get("prompt_id"),
                version=prompt_payload.get("prompt_version"),
                template_sha256=prompt_payload.get("template_sha256"),
                stage="reinforcement_design",
                scope_id=task_id,
                attempt=audit_attempt,
                rendered_path=prompt_txt_path,
                status="rendered",
                metadata={
                    "task_context_path": task_context_path,
                },
            )
            prepared_tasks.append({
                "task_index": index,
                "reinforcement_task": task,
                "task_id": task_id,
                "task_dir": task_dir,
                "task_context_path": task_context_path,
                "prompt_payload": prompt_payload,
                "prompt_text": prompt_text,
                "prompt_txt_path": prompt_txt_path,
                "audit_attempt": audit_attempt,
                "base_output_files": {
                    **load_payload.get("output_files", {}),
                    **force_payload.get("output_files", {}),
                    **control_payload.get("output_files", {}),
                    "prompt_yaml_path": prompt_yaml_path,
                    "prompt_txt_path": prompt_txt_path,
                },
                "load_design_result": load_payload.get("load_design_output"),
                "analysis_load_input": load_payload.get("analysis_load_input"),
                "internal_force_output_summary": {
                    "available_combinations": list((force_payload.get("internal_force_output") or {}).get("combined_envelopes_full_beam", {}).keys()),
                    "note_moment_sign": (force_payload.get("internal_force_output") or {}).get("note_moment_sign"),
                },
                "force_control_info": force_control_info,
                "pier_axial": pier_axial,
            })

        except Exception as exc:
            record_prompt_audit(
                output_root,
                prompt_id=prompt_payload.get("prompt_id") if prompt_payload else str(prompt_id),
                version=prompt_payload.get("prompt_version") if prompt_payload else None,
                template_sha256=prompt_payload.get("template_sha256") if prompt_payload else None,
                stage="reinforcement_design",
                scope_id=task_id,
                attempt=audit_attempt,
                rendered_path=prompt_txt_path,
                status="failed" if prompt_txt_path else "preprocessing_failed",
                metadata={"task_context_path": task_context_path, "error": str(exc)},
            )
            errors.append({
                "task_index": index,
                "task_id": task.get("task_id"),
                "桥梁编号": task.get("桥梁编号"),
                "单元编号": task.get("单元编号"),
                "分组编号": task.get("分组编号"),
                "error": str(exc),
                "task_context_path": task_context_path,
            })

    llm = _build_llm(config_path) if prepared_tasks else None

    def generate_reinforcement(prepared: Dict[str, Any]) -> Dict[str, Any]:
        task = prepared["reinforcement_task"]
        task_id = prepared["task_id"]
        task_dir = prepared["task_dir"]
        prompt_payload = prepared["prompt_payload"]
        prompt_text = prepared["prompt_text"]
        prompt_txt_path = prepared["prompt_txt_path"]
        audit_attempt = prepared["audit_attempt"]
        parse_error: Optional[Exception] = None
        previous_raw = ""

        try:
            for format_attempt in range(1, max_format_repairs + 2):
                if format_attempt == 1:
                    messages = [
                        (
                            "system",
                            "你是一名桥梁下部结构配筋设计工程师。请根据盖梁尺寸、荷载内力控制信息和示例样本，输出规则化、参数化的盖梁配筋 YAML。只输出 YAML。",
                        ),
                        ("user", prompt_text),
                    ]
                else:
                    messages = [
                        (
                            "system",
                            "你负责修复配筋设计输出格式。保持原设计语义，只输出可解析的 YAML 对象，不得输出说明文字。",
                        ),
                        (
                            "user",
                            f"原始任务：\n{prompt_text}\n\n上次输出：\n{previous_raw}\n\n解析错误：{parse_error}",
                        ),
                    ]

                response = llm.invoke(messages)
                previous_raw = str(response.content).strip()
                raw_target = reserve_artifact_path(
                    task_dir / f"raw_response_{task_id}_attempt_{format_attempt}.txt"
                )
                _write_text(raw_target, previous_raw)
                try:
                    reinforcement_result = _parse_yaml_or_json(previous_raw)
                    validation = validate_joint_reinforcement(reinforcement_result)
                    if not validation["complete"]:
                        missing = "、".join(validation["missing"])
                        raise ValueError(f"联合配筋结果缺少: {missing}")
                    break
                except Exception as exc:
                    parse_error = exc
                    if format_attempt > max_format_repairs:
                        raise ValueError(
                            f"配筋输出经过 {max_format_repairs} 次格式修复后仍无法解析或缺失盖梁/墩柱配筋: {exc}"
                        ) from exc

            axial_check = _compute_axial_check(
                task,
                prepared.get("pier_axial"),
                reinforcement_result,
            )
            result_yaml_path = _write_text(
                task_dir / f"reinforcement_result_{task_id}.yaml",
                yaml.safe_dump(
                    reinforcement_result,
                    allow_unicode=True,
                    sort_keys=False,
                    indent=2,
                ),
            )
            result_json_path = _write_json(
                task_dir / f"reinforcement_result_{task_id}.json",
                reinforcement_result,
            )
            record_prompt_audit(
                output_root,
                prompt_id=prompt_payload.get("prompt_id"),
                version=prompt_payload.get("prompt_version"),
                template_sha256=prompt_payload.get("template_sha256"),
                stage="reinforcement_design",
                scope_id=task_id,
                attempt=audit_attempt,
                rendered_path=prompt_txt_path,
                status="completed",
                metadata={"format_attempts": format_attempt},
            )
            return {
                "task_index": prepared["task_index"],
                "reinforcement_task": task,
                "load_design_result": prepared["load_design_result"],
                "analysis_load_input": prepared["analysis_load_input"],
                "internal_force_output_summary": prepared["internal_force_output_summary"],
                "force_control_info": prepared["force_control_info"],
                "pier_axial": prepared.get("pier_axial"),
                "axial_check": axial_check,
                "reinforcement_result": reinforcement_result,
                "matched_sample_count": prompt_payload.get("matched_sample_count", 0),
                "sample_load_mode": prompt_payload.get("sample_load_mode"),
                "prompt_id": prompt_payload.get("prompt_id"),
                "prompt_version": prompt_payload.get("prompt_version"),
                "template_sha256": prompt_payload.get("template_sha256"),
                "output_files": {
                    **prepared["base_output_files"],
                    "reinforcement_result_yaml_path": result_yaml_path,
                    "reinforcement_result_json_path": result_json_path,
                },
            }
        except Exception as exc:
            record_prompt_audit(
                output_root,
                prompt_id=prompt_payload.get("prompt_id"),
                version=prompt_payload.get("prompt_version"),
                template_sha256=prompt_payload.get("template_sha256"),
                stage="reinforcement_design",
                scope_id=task_id,
                attempt=audit_attempt,
                rendered_path=prompt_txt_path,
                status="failed",
                metadata={
                    "task_context_path": prepared["task_context_path"],
                    "error": str(exc),
                },
            )
            raise

    batch_results = run_indexed_batch(
        prepared_tasks,
        generate_reinforcement,
        max_workers=max_workers,
    )
    task_results: List[Dict[str, Any]] = []
    for batch_result in batch_results:
        prepared = prepared_tasks[batch_result.index]
        if batch_result.error:
            task = prepared["reinforcement_task"]
            errors.append({
                "task_index": prepared["task_index"],
                "task_id": task.get("task_id"),
                "桥梁编号": task.get("桥梁编号"),
                "单元编号": task.get("单元编号"),
                "分组编号": task.get("分组编号"),
                "error": batch_result.error,
                "task_context_path": prepared["task_context_path"],
            })
        elif batch_result.value is not None:
            task_results.append(batch_result.value)

    all_task_ids = {str(task.get("task_id")) for task in all_tasks}
    if retry_ids or retry_units:
        retained_results = [
            item for item in previous_results
            if str((item.get("reinforcement_task") or {}).get("task_id")) not in retry_ids
            and str((item.get("reinforcement_task") or {}).get("单元编号")) not in retry_units
        ]
    else:
        # 整批重跑不合并上一轮成果（避免用旧成果顶替本轮生成失败的任务），
        # 只保留"已不在本轮任务清单里"的分组，交给下面的 orphan 处理。
        retained_results = [
            item for item in previous_results
            if str((item.get("reinforcement_task") or {}).get("task_id")) not in all_task_ids
        ]
    result_by_task_id = {
        str((item.get("reinforcement_task") or {}).get("task_id")): item
        for item in [*retained_results, *task_results]
    }
    # 沿用上一轮配筋成果的分组里，有些已不在当前尺寸成果的任务清单中（尺寸成果坍缩：
    # 只剩被返修的那几联）。只按 all_tasks 重排会把它们无痕删掉，expected/completed 一起
    # 缩水而 stage_complete 仍为 True（2026-09-15 示例项目K29：配筋 20 组 → 4 组、最终只出
    # 1 张图）。这里显式沿用这些组，并计入失败清单，让批次停在人工复核而不是静默丢成果。
    orphan_task_ids = sorted(
        task_id
        for task_id in result_by_task_id
        if task_id and task_id not in all_task_ids
    )
    task_results = [
        result_by_task_id[task_id]
        for task in all_tasks
        if (task_id := str(task.get("task_id"))) in result_by_task_id
    ] + [result_by_task_id[task_id] for task_id in orphan_task_ids]
    for task_id in orphan_task_ids:
        orphan = result_by_task_id[task_id]
        task = orphan.get("reinforcement_task") or {}
        errors.append({
            "task_index": orphan.get("task_index"),
            "task_id": task_id,
            "桥梁编号": task.get("桥梁编号"),
            "单元编号": task.get("单元编号"),
            "分组编号": task.get("分组编号"),
            "error": (
                "当前尺寸设计成果中已不存在该分组（尺寸成果坍缩）：已沿用上一轮配筋成果，"
                "需人工复核其尺寸依据。"
            ),
            "orphaned_from_dimension_result": True,
        })

    expected_task_count = len(all_tasks) + len(orphan_task_ids)
    completed_task_count = len(task_results)
    failed_task_count = len(errors)
    stage_complete = completed_task_count == expected_task_count and failed_task_count == 0
    partial_result_available = 0 < completed_task_count < expected_task_count
    summary = {
        "任务3_下部结构配筋设计结果": {
            "桥梁列表": _group_reinforcement_results(task_results),
            "分组原始结果": task_results,
            "失败分组": errors,
            "尺寸成果缺失分组": orphan_task_ids,
            "expected_task_count": expected_task_count,
            "completed_task_count": completed_task_count,
            "failed_task_count": failed_task_count,
            "stage_complete": stage_complete,
            "partial_result_available": partial_result_available,
        }
    }
    summary_path = _write_json(stage_dir / "reinforcement_design_result.json", summary)
    summary_yaml_path = _write_text(stage_dir / "reinforcement_design_result.yaml", yaml.safe_dump(summary, allow_unicode=True, sort_keys=False, indent=2))
    # 墩柱轴压验算汇总表：逐组 d/H/k/λ/φ/利用率/结论 + 整体计数，供成果展示与人工复核。
    axial_summary = build_axial_check_summary(summary)
    axial_summary_path = _write_json(stage_dir / "axial_check_summary.json", axial_summary)
    axial_summary_yaml_path = _write_text(
        stage_dir / "axial_check_summary.yaml",
        yaml.safe_dump(axial_summary, allow_unicode=True, sort_keys=False, indent=2),
    )
    reinforcement_yaml_paths = [
        item.get("output_files", {}).get("reinforcement_result_yaml_path")
        for item in task_results
        if item.get("output_files", {}).get("reinforcement_result_yaml_path")
    ]
    internal_force_output_paths = [
        item.get("output_files", {}).get("internal_force_output_path")
        for item in task_results
        if item.get("output_files", {}).get("internal_force_output_path")
    ]
    common_output_files = {
        "reinforcement_design_result_path": summary_path,
        "reinforcement_design_result_yaml_path": summary_yaml_path,
        "axial_check_summary_path": axial_summary_path,
        "axial_check_summary_yaml_path": axial_summary_yaml_path,
        "reinforcement_yaml_path": reinforcement_yaml_paths[0] if reinforcement_yaml_paths else None,
        "reinforcement_yaml_paths": reinforcement_yaml_paths,
        "opensees_force_json_path": internal_force_output_paths[0] if internal_force_output_paths else None,
        "internal_force_json_path": internal_force_output_paths[0] if internal_force_output_paths else None,
        "internal_force_output_paths": internal_force_output_paths,
    }

    if errors and not task_results:
        return {
            "success": False,
            "error": "所有配筋分组设计均失败。",
            "errors": errors,
            "column_height_diagnostics": column_height_diagnostics,
            "expected_task_count": expected_task_count,
            "completed_task_count": completed_task_count,
            "failed_task_count": failed_task_count,
            "stage_complete": stage_complete,
            "partial_result_available": partial_result_available,
            "evidence_bundles": evidence_bundles,
            "output_files": common_output_files,
        }

    return {
        "success": True,
        "reinforcement_design_result": summary,
        "column_height_diagnostics": column_height_diagnostics,
        "task_result_count": completed_task_count,
        "expected_task_count": expected_task_count,
        "completed_task_count": completed_task_count,
        "failed_task_count": failed_task_count,
        "stage_complete": stage_complete,
        "partial_result_available": partial_result_available,
        "errors": errors,
        "evidence_bundles": evidence_bundles,
        "output_files": common_output_files,
    }
