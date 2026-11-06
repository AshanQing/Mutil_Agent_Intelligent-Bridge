"""Read engineering evidence for the user-facing Graph V2 workbench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CapacityEvidence:
    source_path: Path | None = None
    completed_tasks: int | None = None
    total_tasks: int | None = None
    execution_failures: int | None = None
    engineering_pass: bool | None = None
    control_group: str = ""
    max_utilization: float | None = None
    can_accept_modeling_risk: bool = False


def _count(value: Any) -> int | None:
    try:
        return max(0, int(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def read_capacity_evidence(output_dir: str | Path) -> CapacityEvidence:
    root = Path(output_dir) / "capacity_check"
    for name in ("capacity_check_batch_summary.json", "capacity_check_summary.json"):
        path = root / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or not payload:
            continue
        overall = payload.get("overall_check")
        overall = overall if isinstance(overall, dict) else {}
        utilization = payload.get("utilization_summary")
        utilization = utilization if isinstance(utilization, dict) else {}
        try:
            max_utilization = float(utilization["max_utilization"])
        except (KeyError, TypeError, ValueError):
            max_utilization = None
        completed = _count(payload.get("completed_task_count"))
        total = _count(payload.get("expected_task_count"))
        failures = _count(payload.get("failed_task_count"))
        complete = (
            payload.get("stage_complete") is True
            and bool(overall)
            and overall.get("all_ok") is not None
            and (completed is None or total is None or completed == total)
        )
        return CapacityEvidence(
            source_path=path,
            completed_tasks=completed,
            total_tasks=total,
            execution_failures=failures,
            engineering_pass=bool(overall["all_ok"]) if overall.get("all_ok") is not None else None,
            control_group=str(utilization.get("task_id") or ""),
            max_utilization=max_utilization,
            can_accept_modeling_risk=complete,
        )
    return CapacityEvidence()


def read_human_decisions(output_dir: str | Path) -> tuple[dict[str, Any], ...]:
    path = Path(output_dir) / "human_review" / "human_review_decisions.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("action"):
            rows.append(row)
    return tuple(rows)
