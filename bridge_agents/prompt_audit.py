from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union


PathLike = Union[str, Path]
_AUDIT_WRITE_LOCK = threading.Lock()


def reserve_artifact_path(path: PathLike) -> Path:
    """返回不会覆盖既有文件的路径。"""
    candidate = Path(path)
    if not candidate.exists():
        return candidate
    attempt = 2
    while True:
        versioned = candidate.with_name(f"{candidate.stem}_attempt_{attempt}{candidate.suffix}")
        if not versioned.exists():
            return versioned
        attempt += 1


def artifact_attempt(path: PathLike) -> int:
    match = re.search(r"_attempt_(\d+)$", Path(path).stem)
    return int(match.group(1)) if match else 1


def record_prompt_audit(
    output_dir: PathLike,
    *,
    prompt_id: Optional[str],
    version: Optional[str],
    template_sha256: Optional[str],
    stage: str,
    scope_id: str,
    attempt: int,
    rendered_path: Optional[PathLike],
    status: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """将一次 Prompt 生命周期事件追加到统一 JSONL 审计文件。"""
    audit_path = Path(output_dir) / "logs" / "prompt_trace.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "prompt_id": prompt_id,
        "version": version,
        "template_sha256": template_sha256,
        "stage": stage,
        "scope_id": scope_id,
        "attempt": int(attempt),
        "rendered_path": str(rendered_path) if rendered_path else None,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
    }
    with _AUDIT_WRITE_LOCK:
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return audit_path
