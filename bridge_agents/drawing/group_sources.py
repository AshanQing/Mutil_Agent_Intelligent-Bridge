from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import DrawingGroupSource


class DrawingSourceError(ValueError):
    pass


class DrawingSourceMissingError(DrawingSourceError):
    pass


class DrawingSourceConflictError(DrawingSourceError):
    pass


def _safe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalized_members(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _extract_dimension_groups(payload: Any) -> list[dict[str, Any]]:
    current = _safe_mapping(payload)
    if not current:
        return []

    nested = current.get("dimension_design_result")
    if isinstance(nested, Mapping):
        return _extract_dimension_groups(nested)

    root = current.get("任务2_下部结构尺寸设计结果")
    if isinstance(root, Mapping):
        rows: list[dict[str, Any]] = []
        for item in root.get("单元原始结果", []) or []:
            item_map = _safe_mapping(item)
            unit_input = _safe_mapping(item_map.get("single_unit_input"))
            result = _safe_mapping(item_map.get("dimension_result"))
            for group in result.get("分组尺寸设计结果", []) or []:
                if isinstance(group, Mapping):
                    rows.append(
                        {
                            "bridge_id": str(unit_input.get("桥梁编号") or "").strip(),
                            "unit_id": str(unit_input.get("单元编号") or "").strip(),
                            "group_id": str(group.get("分组编号") or "").strip(),
                            "member_piers": _normalized_members(
                                group.get("包含桥墩号列表")
                            ),
                            "dimension": dict(group),
                        }
                    )
        if rows:
            return rows

        for bridge in root.get("桥梁列表", []) or []:
            bridge_map = _safe_mapping(bridge)
            bridge_id = str(bridge_map.get("桥梁编号") or "").strip()
            for unit in bridge_map.get("单元尺寸设计结果", []) or []:
                unit_map = _safe_mapping(unit)
                unit_id = str(unit_map.get("单元编号") or "").strip()
                for group in unit_map.get("分组尺寸设计结果", []) or []:
                    if isinstance(group, Mapping):
                        rows.append(
                            {
                                "bridge_id": bridge_id,
                                "unit_id": unit_id,
                                "group_id": str(group.get("分组编号") or "").strip(),
                                "member_piers": _normalized_members(
                                    group.get("包含桥墩号列表")
                                ),
                                "dimension": dict(group),
                            }
                        )
        return rows
    return []


def _extract_reinforcement_rows(payload: Any) -> list[dict[str, Any]]:
    current = _safe_mapping(payload)
    if not current:
        return []
    nested = current.get("reinforcement_design_result")
    if isinstance(nested, Mapping):
        return _extract_reinforcement_rows(nested)
    root = current.get("任务3_下部结构配筋设计结果")
    if isinstance(root, Mapping):
        return [
            dict(item)
            for item in root.get("分组原始结果", []) or []
            if isinstance(item, Mapping)
        ]
    return []


def _extract_capacity_rows(payload: Any) -> list[dict[str, Any]]:
    current = _safe_mapping(payload)
    if not current:
        return []
    for nested_key in ("capacity_check_result", "check_result"):
        nested = current.get(nested_key)
        if isinstance(nested, Mapping) and nested is not current:
            rows = _extract_capacity_rows(nested)
            if rows:
                return rows
    return [
        dict(item)
        for item in current.get("task_results", []) or []
        if isinstance(item, Mapping)
    ]


def _capacity_status(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "missing"
    result = _safe_mapping(row.get("check_result") or row.get("result"))
    if not result:
        return "missing"
    if result.get("success") is False or result.get("error"):
        return "failed"
    all_ok = _safe_mapping(result.get("overall_check")).get("all_ok")
    if all_ok is True:
        return "passed"
    if all_ok is False:
        return "failed"
    return "missing"


def _row_has_no_effective_column(row: Mapping[str, Any]) -> bool:
    """该行是否可确定没有有效柱身（净高非正/柱径非正 → 桥台或埋入式墩）。

    与 joint_reinforcement 的聚合判定同规则，用于把历史成果里旧版本写入的
    manual_review（实为"无柱身、验算不适用"）正确归类，避免误判为 missing 而拒绝出图。
    """
    task = _safe_mapping(row.get("reinforcement_task")) or row
    geom = _safe_mapping(_safe_mapping(task.get("桥墩尺寸信息")).get("墩柱几何信息"))
    info = _safe_mapping(task.get("墩柱净高信息"))
    net_height = info.get("controlling_net_height_m")
    diameter = geom.get("column_diameter")
    if net_height is None and diameter is None:
        return False
    try:
        if net_height is not None and float(net_height) <= 0:
            return True
        if diameter is not None and float(diameter) <= 0:
            return True
    except (TypeError, ValueError):
        return False
    return False


def _axial_status(row: Mapping[str, Any]) -> str:
    axial = _safe_mapping(row.get("axial_check"))
    if not axial:
        return "missing"
    if axial.get("status") == "not_applicable" or _row_has_no_effective_column(row):
        # 该组没有有效柱身（桥台/埋入式墩）：柱身稳定验算不适用，不参与出图准入判定。
        return "not_applicable"
    if axial.get("status") != "computed":
        return "missing"
    return "passed" if axial.get("check_ok") is True else "failed"


def _combined_check_status(capacity: str, axial: str) -> str:
    if "failed" in {capacity, axial}:
        return "failed"
    if axial == "not_applicable":
        # 柱身维度不适用时只看承载力验算结论
        return capacity
    if capacity == axial == "passed":
        return "passed"
    return "missing"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_source_files(
    items: Iterable[tuple[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for label, raw_path in items:
        value = str(raw_path or "").strip()
        if not value:
            continue
        paths[label] = value
        path = Path(value)
        if path.is_file():
            hashes[label] = _file_sha256(path)
    return paths, hashes


def collect_drawing_group_sources(
    *,
    pier_group_result: Mapping[str, Any],
    dimension_design_result: Mapping[str, Any],
    reinforcement_design_result: Mapping[str, Any],
    capacity_check_result: Mapping[str, Any] | None,
) -> list[DrawingGroupSource]:
    """以配筋任务行为驱动装配绘图源（每张图 = 一个配筋任务）。

    历史实现以 pier_group 的确定性归并组为驱动，再按“桥墩成员集合完全相等”
    去匹配尺寸分组；但 pier_group 会拆分/去重连接墩（如边墩独立成组），与
    尺寸设计“前后联连接墩复用同一分组”的成员口径不一致，导致出图装配报
    “缺少匹配的尺寸设计结果”。本实现改为直接消费 reinforcement_design_result
    的“分组原始结果”（任务行自带 桥梁/单元/分组编号 G1/G2/task_id），与
    尺寸设计、承载力验算均为同一分组口径、天然一一对应；
    pier_group_result 仅作为净高等上下文参考（墩柱净高已注入任务，可缺失）。
    """
    dimensions = _extract_dimension_groups(dimension_design_result)
    dim_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for dim in dimensions:
        key = (
            str(dim.get("bridge_id") or "").strip(),
            str(dim.get("unit_id") or "").strip(),
            str(dim.get("group_id") or "").strip(),
        )
        if not all(key):
            continue
        if key in dim_by_key:
            raise DrawingSourceConflictError(
                f"尺寸设计分组 {key[0]}-{key[1]}-{key[2]} 存在多个分组行。"
            )
        dim_by_key[key] = dim

    reinforcement_rows = _extract_reinforcement_rows(reinforcement_design_result)
    if not reinforcement_rows:
        raise DrawingSourceMissingError(
            "配筋设计结果中未找到 分组原始结果 任务行，无法装配绘图源。"
        )
    capacity_by_task = {
        str(item.get("task_id") or "").strip(): item
        for item in _extract_capacity_rows(capacity_check_result)
        if str(item.get("task_id") or "").strip()
    }
    pier_groups = [
        dict(item)
        for item in (pier_group_result or {}).get("design_groups", []) or []
        if isinstance(item, Mapping)
    ]

    seen_task_ids: set[str] = set()
    sources: list[DrawingGroupSource] = []
    for row in reinforcement_rows:
        task = _safe_mapping(row.get("reinforcement_task"))
        bridge_id = str(task.get("桥梁编号") or "").strip()
        unit_id = str(task.get("单元编号") or "").strip()
        group_no = str(task.get("分组编号") or "").strip()
        task_id = str(task.get("task_id") or row.get("task_id") or "").strip()
        if not task_id:
            raise DrawingSourceMissingError(
                f"配筋任务（{bridge_id}-{unit_id}-{group_no}）缺少 task_id。"
            )
        if task_id in seen_task_ids:
            raise DrawingSourceConflictError(
                f"配筋任务 {task_id} 对应多个配筋行。"
            )
        seen_task_ids.add(task_id)

        dimension = dim_by_key.get((bridge_id, unit_id, group_no))
        if dimension is None:
            raise DrawingSourceMissingError(
                f"配筋任务 {task_id}（{bridge_id}-{unit_id}-{group_no}）"
                "缺少匹配的尺寸设计结果。"
            )

        members = _normalized_members(task.get("包含桥墩号列表"))
        if not members:
            members = _normalized_members(dimension.get("member_piers"))
        if not members:
            raise DrawingSourceMissingError(
                f"配筋任务 {task_id} 缺少桥墩成员信息。"
            )

        reinforcement = dict(_safe_mapping(row.get("reinforcement_result")))
        axial_check = dict(_safe_mapping(row.get("axial_check")))
        output_files = _safe_mapping(row.get("output_files"))
        source_paths, source_hashes = _existing_source_files(
            (
                ("reinforcement_yaml", output_files.get("reinforcement_result_yaml_path")),
                ("reinforcement_json", output_files.get("reinforcement_result_json_path")),
                ("internal_force", output_files.get("internal_force_output_path")),
            )
        )
        pier_ref = next(
            (
                dict(group)
                for group in pier_groups
                if str(group.get("bridge_id") or "").strip() == bridge_id
                and str(group.get("unit_id") or "").strip() == unit_id
            ),
            {},
        )
        capacity_status = _capacity_status(capacity_by_task.get(task_id))
        axial_status = _axial_status(row)
        sources.append(
            DrawingGroupSource(
                design_group_id=task_id,
                task_id=task_id,
                member_piers=list(members),
                dimension=dict(dimension["dimension"]),
                reinforcement=reinforcement,
                check_status=_combined_check_status(capacity_status, axial_status),
                design_context={
                    "pier_group": dict(pier_ref),
                    "reinforcement_task": dict(task),
                    "axial_check": axial_check,
                },
                source_paths=source_paths,
                source_hashes=source_hashes,
            )
        )
    return sources
