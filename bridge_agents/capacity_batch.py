from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


FLEXURE_FORMULA_ID = "F_3362_CH05_5_2_2_1_BENDING_CAPACITY_TENSION_FLANGE"
XI_B_TABLE_ID = "T_3362_CH05_5_2_1_RELATIVE_LIMIT_COMPRESSION_ZONE_HEIGHT"
SHEAR_LIMIT_FORMULA_ID = "F_3362_CH08_8_4_4_CAP_BEAM_SHEAR_CAPACITY"
SHEAR_REINFORCED_FORMULA_ID = "F_3362_CH08_8_4_5_CAP_BEAM_INCLINED_SHEAR"


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _group_results_from_payload(payload: Any) -> List[Dict[str, Any]]:
    current = _safe_dict(payload)
    if not current:
        return []

    root = current.get("任务3_下部结构配筋设计结果")
    if isinstance(root, dict):
        return [item for item in root.get("分组原始结果", []) if isinstance(item, dict)]

    nested = current.get("reinforcement_design_result")
    if isinstance(nested, dict):
        return _group_results_from_payload(nested)
    return []


def _path_task_id(path: str, *, prefix: str) -> str:
    candidate = Path(str(path))
    stem = candidate.stem
    if stem.startswith(prefix):
        suffix = stem[len(prefix) :].strip()
        if suffix:
            return suffix
    parent_name = candidate.parent.name.strip()
    return parent_name


def _unique_path_map(paths: Sequence[str], *, prefix: str, label: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw_path in paths:
        path = str(raw_path or "").strip()
        if not path:
            continue
        task_id = _path_task_id(path, prefix=prefix)
        if not task_id:
            raise ValueError(f"{label} 无法提取 task_id: {path}")
        if task_id in result:
            raise ValueError(f"{label} 出现重复 task_id: {task_id}")
        result[task_id] = path
    return result


def collect_capacity_check_tasks(state: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Collect paired reinforcement/force inputs in stable reinforcement-task order."""
    group_results = _group_results_from_payload(state.get("reinforcement_design_result"))
    if group_results:
        tasks: List[Dict[str, str]] = []
        seen: set[str] = set()
        for index, item in enumerate(group_results, start=1):
            reinforcement_task = _safe_dict(item.get("reinforcement_task"))
            output_files = _safe_dict(item.get("output_files"))
            task_id = str(
                reinforcement_task.get("task_id")
                or item.get("task_id")
                or f"capacity-task-{index:03d}"
            ).strip()
            reinforcement_path = str(
                output_files.get("reinforcement_result_yaml_path") or ""
            ).strip()
            force_path = str(output_files.get("internal_force_output_path") or "").strip()
            if task_id in seen:
                raise ValueError(f"承载力验算输入出现重复 task_id: {task_id}")
            if not reinforcement_path or not force_path:
                raise ValueError(
                    f"承载力验算任务 {task_id} 缺少配筋 YAML 或 OpenSees 内力文件。"
                )
            seen.add(task_id)
            tasks.append(
                {
                    "task_id": task_id,
                    "reinforcement_yaml_path": reinforcement_path,
                    "opensees_force_json_path": force_path,
                }
            )
        return tasks

    reinforcement_paths = [
        str(path) for path in (state.get("reinforcement_yaml_paths") or []) if path
    ]
    force_paths = [
        str(path) for path in (state.get("internal_force_output_paths") or []) if path
    ]
    if reinforcement_paths or force_paths:
        reinforcement_by_id = _unique_path_map(
            reinforcement_paths,
            prefix="reinforcement_result_",
            label="配筋 YAML 路径列表",
        )
        force_by_id = _unique_path_map(
            force_paths,
            prefix="internal_force_output_full_beam_",
            label="OpenSees 内力路径列表",
        )
        if set(reinforcement_by_id) != set(force_by_id):
            missing_force = sorted(set(reinforcement_by_id) - set(force_by_id))
            missing_reinforcement = sorted(set(force_by_id) - set(reinforcement_by_id))
            raise ValueError(
                "承载力验算多路径无法按 task_id 一一配对："
                f"缺内力={missing_force}，缺配筋={missing_reinforcement}。"
            )
        return [
            {
                "task_id": task_id,
                "reinforcement_yaml_path": reinforcement_by_id[task_id],
                "opensees_force_json_path": force_by_id[task_id],
            }
            for task_id in reinforcement_by_id
        ]

    reinforcement_path = str(state.get("reinforcement_yaml_path") or "").strip()
    force_path = str(state.get("opensees_force_json_path") or "").strip()
    if not reinforcement_path:
        raise ValueError("缺少 reinforcement_yaml_path，无法进行承载力验算。")
    if not force_path:
        raise ValueError("缺少 opensees_force_json_path，无法进行承载力验算。")
    task_id = _path_task_id(reinforcement_path, prefix="reinforcement_result_") or "capacity-task-001"
    return [
        {
            "task_id": task_id,
            "reinforcement_yaml_path": reinforcement_path,
            "opensees_force_json_path": force_path,
        }
    ]


def safe_capacity_task_dir_name(task_id: str) -> str:
    safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(task_id)).strip("._")
    return safe_name or "capacity-task"


def _max_utilization(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _safe_dict(result.get("utilization_summary"))
    if summary.get("max_utilization") is not None:
        return dict(summary)

    maximum: Dict[str, Any] = {
        "max_utilization": None,
        "control_name": None,
        "util_type": None,
        "x_m": None,
    }
    for name, row in _safe_dict(result.get("control_sections")).items():
        if not isinstance(row, dict):
            continue
        for key in ("util_M_pos", "util_M_neg", "util_V"):
            value = row.get(key)
            if value is None:
                continue
            try:
                utilization = float(value)
            except (TypeError, ValueError):
                continue
            if maximum["max_utilization"] is None or utilization > float(maximum["max_utilization"]):
                maximum = {
                    "max_utilization": utilization,
                    "control_name": name,
                    "util_type": key,
                    "x_m": row.get("x_m"),
                }
    return maximum


def _compliance_status(value: Any, *, execution_ok: bool) -> str:
    if not execution_ok or value is None:
        return "missing_input"
    return "pass" if value is True else "fail"


def build_capacity_compliance_matrix(
    task_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    limitations: List[str] = []
    definitions = (
        (
            "5.2.2:flexure_positive",
            "M_pos_ok",
            "正弯矩正截面抗弯承载力",
            "5.2.2",
            [FLEXURE_FORMULA_ID, XI_B_TABLE_ID],
        ),
        (
            "5.2.2:flexure_negative",
            "M_neg_ok",
            "负弯矩正截面抗弯承载力",
            "5.2.2",
            [FLEXURE_FORMULA_ID, XI_B_TABLE_ID],
        ),
        (
            "5.2.1:compression_zone_limit",
            "compression_zone_ok",
            "相对界限受压区高度",
            "5.2.1",
            [XI_B_TABLE_ID],
        ),
        (
            "8.4.4-8.4.5:shear",
            "V_ok",
            "盖梁斜截面抗剪承载力",
            "8.4.4/8.4.5",
            [SHEAR_LIMIT_FORMULA_ID, SHEAR_REINFORCED_FORMULA_ID],
        ),
    )
    for item in task_results:
        task_id = str(item.get("task_id") or "")
        result = _safe_dict(item.get("result"))
        overall = _safe_dict(result.get("overall_check"))
        execution_ok = result.get("success") is not False and not result.get("error")
        utilization = _max_utilization(result)
        for suffix, result_key, check_name, clause, source_ids in definitions:
            status = _compliance_status(overall.get(result_key), execution_ok=execution_ok)
            entries.append(
                {
                    "task_id": task_id,
                    "code_item_id": f"{task_id}:{suffix}",
                    "standard": "JTG 3362-2018",
                    "clause": clause,
                    "check_name": check_name,
                    "status": status,
                    "source_entity_ids": list(source_ids),
                    "result_key": result_key,
                    "result_value": overall.get(result_key),
                    "control_utilization": utilization if result_key in {"M_pos_ok", "M_neg_ok", "V_ok"} else None,
                    "message": (
                        "验算通过" if status == "pass"
                        else "验算未通过" if status == "fail"
                        else "缺少可执行验算结果"
                    ),
                }
            )
        summary = _safe_dict(result.get("summary"))
        coverage = _safe_dict(summary.get("formula_coverage"))
        for limitation in coverage.get("not_covered") or []:
            text = str(limitation)
            if text and text not in limitations:
                limitations.append(text)
    return {
        "schema_version": "capacity_compliance_v1",
        "entries": entries,
        "coverage_limitations": limitations,
    }


def aggregate_capacity_check_results(task_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    expected_count = len(task_results)
    execution_failed_ids: List[str] = []
    failed_check_ids: List[str] = []
    control_sections: Dict[str, Any] = {}
    maximum: Dict[str, Any] = {
        "max_utilization": None,
        "control_name": None,
        "util_type": None,
        "x_m": None,
        "task_id": None,
    }
    component_values: Dict[str, List[bool]] = {
        "M_pos_ok": [],
        "M_neg_ok": [],
        "V_ok": [],
        "compression_zone_ok": [],
    }

    normalized_results: List[Dict[str, Any]] = []
    for item in task_results:
        task_id = str(item.get("task_id") or "")
        result = _safe_dict(item.get("result"))
        success = result.get("success") is not False and not result.get("error")
        overall = _safe_dict(result.get("overall_check"))
        if not success:
            execution_failed_ids.append(task_id)
        elif overall.get("all_ok") is not True:
            failed_check_ids.append(task_id)

        for key in component_values:
            if key in overall:
                component_values[key].append(overall.get(key) is True)

        for name, row in _safe_dict(result.get("control_sections")).items():
            namespaced = f"{task_id}::{name}"
            enriched = dict(row) if isinstance(row, dict) else {"value": row}
            enriched.setdefault("task_id", task_id)
            enriched.setdefault("source_control_name", name)
            control_sections[namespaced] = enriched

        task_maximum = _max_utilization(result)
        value = task_maximum.get("max_utilization")
        if value is not None and (
            maximum["max_utilization"] is None
            or float(value) > float(maximum["max_utilization"])
        ):
            maximum = {**task_maximum, "task_id": task_id}

        normalized_results.append(
            {
                "task_id": task_id,
                "reinforcement_yaml_path": item.get("reinforcement_yaml_path"),
                "opensees_force_json_path": item.get("opensees_force_json_path"),
                "output_dir": item.get("output_dir"),
                "check_result": result,
            }
        )

    completed_count = expected_count - len(execution_failed_ids)
    stage_complete = expected_count > 0 and not execution_failed_ids
    overall_check = {
        key: bool(values) and all(values) and stage_complete
        for key, values in component_values.items()
        if values
    }
    overall_check["all_ok"] = (
        stage_complete
        and not failed_check_ids
        and all(
            _safe_dict(_safe_dict(item.get("result")).get("overall_check")).get("all_ok") is True
            for item in task_results
        )
    )

    compliance_matrix = build_capacity_compliance_matrix(task_results)
    unresolved_code_items = [
        entry
        for entry in compliance_matrix["entries"]
        if entry["status"] in {"fail", "manual_review", "missing_input"}
    ]
    return {
        "success": stage_complete,
        "check_type": "cap_beam_capacity_envelope_batch",
        "expected_task_count": expected_count,
        "completed_task_count": completed_count,
        "failed_task_count": len(execution_failed_ids),
        "failed_task_ids": execution_failed_ids,
        "failed_check_task_ids": failed_check_ids,
        "stage_complete": stage_complete,
        "overall_check": overall_check,
        "control_sections": control_sections,
        "utilization_summary": maximum,
        "task_results": normalized_results,
        "compliance_matrix": compliance_matrix,
        "unresolved_code_items": unresolved_code_items,
        "error": None if stage_complete else "部分承载力验算任务执行失败。",
    }
