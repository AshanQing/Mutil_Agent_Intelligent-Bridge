from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from tools.cap_internal_force_analysis_tool import cap_internal_force_analysis_tool
from tools.reinforcement_design_tool import (
    _compute_axial_check,
    _compute_pier_axial_summary,
)
from tools.reinforcement_drawing_tool import reinforcement_drawing_tool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层必须是对象。")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    finally:
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()


def recompute_existing_outputs(output_dir: str | Path) -> dict[str, Any]:
    """复算既有配筋任务的反力、轴力、长细比，并重建绘图成果。"""
    root = Path(output_dir).resolve()
    reinforcement_root = root / "structural_design" / "reinforcement_design"
    aggregate_json = reinforcement_root / "reinforcement_design_result.json"
    aggregate_yaml = reinforcement_root / "reinforcement_design_result.yaml"
    payload = _load_json(aggregate_json)
    result_root = payload.get("任务3_下部结构配筋设计结果") or {}
    rows = result_root.get("分组原始结果") or []
    if not rows:
        raise ValueError("既有配筋汇总中没有分组原始结果。")

    summaries: list[dict[str, Any]] = []
    for row in rows:
        task = row.get("reinforcement_task") or {}
        task_id = str(task.get("task_id") or row.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("配筋分组缺少 task_id。")
        group_dir = reinforcement_root / task_id
        analysis_input = _load_json(group_dir / f"analysis_load_input_{task_id}.json")
        saved_load_output = _load_json(
            group_dir / f"load_design_output_{task_id}.json"
        )
        # 运行期工具返回对象比落盘 JSON 多一层 load_design_output 包装；
        # 恢复既有成果时重建该层，保持与 _compute_pier_axial_summary 契约一致。
        load_payload = {"load_design_output": saved_load_output}
        force_payload = cap_internal_force_analysis_tool(
            analysis_input,
            output_dir=str(group_dir),
            task_id=task_id,
            save_figures=True,
        )
        if not force_payload.get("success"):
            raise RuntimeError(f"{task_id} 内力复算失败：{force_payload.get('error')}")
        pier_axial = _compute_pier_axial_summary(task, force_payload, load_payload)
        axial_check = _compute_axial_check(
            task,
            pier_axial,
            row.get("reinforcement_result") or {},
        )
        row["pier_axial"] = pier_axial
        row["axial_check"] = axial_check
        summaries.append(
            {
                "task_id": task_id,
                "column_reactions_kN": (pier_axial or {}).get("column_reactions_kN"),
                "controlling_axial_force_kN": (pier_axial or {}).get(
                    "controlling_axial_force_kN"
                ),
                "slenderness_ratio": (axial_check or {}).get("slenderness_ratio"),
                "status": (axial_check or {}).get("status"),
            }
        )

    _atomic_write(
        aggregate_json,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        aggregate_yaml,
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
    )

    pier_group = _load_json(
        root / "structural_design" / "pier_group" / "pier_group_result.json"
    )
    dimension = _load_json(
        root / "structural_design" / "dimension_design" / "dimension_design_result.json"
    )
    capacity = _load_json(root / "capacity_check" / "capacity_check_batch_summary.json")
    manifest_path = root / "deliverables" / "design_manifest.json"
    accepted_risks = (
        (_load_json(manifest_path).get("accepted_risks") or [])
        if manifest_path.exists()
        else []
    )
    drawing_result = reinforcement_drawing_tool(
        pier_group_result=pier_group,
        dimension_design_result=dimension,
        reinforcement_design_result=payload,
        capacity_check_result=capacity,
        output_dir=root,
        allow_unverified=bool(accepted_risks),
        accepted_risks=accepted_risks,
    )
    return {
        "task_count": len(summaries),
        "tasks": summaries,
        "drawing_result": drawing_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="复算既有配筋成果的反力/轴压检查并重建配筋图。"
    )
    parser.add_argument("output_dir", help="一次完整运行的输出目录")
    args = parser.parse_args()
    result = recompute_existing_outputs(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
