from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


SCHEMA_VERSION = 1
LEDGER_RELATIVE_PATH = os.path.join("human_review", "human_review_decisions.jsonl")
ACCEPTANCE_ACTIONS = {
    "accept_and_continue",
    "accept_partial_and_continue",
    "accept_check_and_finish",
}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sha256_file(path: str) -> Optional[str]:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subject_for_review(state: Mapping[str, Any], review_type: str) -> Dict[str, Any]:
    candidates = {
        "layout_collision_review": (
            "layout_result",
            state.get("revision_result_path")
            or state.get("latest_revision_result_path")
            or state.get("layout_revision_result_path")
            or state.get("final_layout_result_path"),
        ),
        "structural_batch_review": (
            "reinforcement_design_result",
            state.get("reinforcement_design_result_path")
            or state.get("structural_design_result_path"),
        ),
        "modeling_check_review": (
            "capacity_check_result",
            state.get("capacity_check_summary_path"),
        ),
    }
    state_key, raw_path = candidates.get(review_type, (None, None))
    path = str(raw_path or "").strip()
    return {
        "state_key": state_key,
        "path": path or None,
        "sha256": _sha256_file(path) if path else None,
    }


def persist_human_review_decision(
    state: Mapping[str, Any],
    update: Mapping[str, Any],
) -> Optional[str]:
    """Append one human decision and its reviewed-artifact fingerprint."""
    output_dir = str(state.get("output_dir") or "").strip()
    decision = _safe_dict(update.get("human_review_decision"))
    review_type = str(update.get("human_review_scope") or "").strip()
    if not output_dir or not decision or not review_type:
        return None

    accepted_risks = list(update.get("accepted_risks") or [])
    previous_risk_count = len(list(state.get("accepted_risks") or []))
    accepted_risk = (
        accepted_risks[previous_risk_count]
        if len(accepted_risks) > previous_risk_count
        else None
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": decision.get("decision_id"),
        "review_type": review_type,
        "action": decision.get("action"),
        "feedback": decision.get("feedback") or "",
        "extra_rounds": decision.get("extra_rounds"),
        "decided_at": decision.get("decided_at"),
        "subject": _subject_for_review(state, review_type),
        "accepted_risk": accepted_risk,
        "accepted_state": {
            "reinforcement_batch_status": update.get("reinforcement_batch_status"),
            "task_status": update.get("task_status"),
        },
    }
    ledger_path = Path(output_dir) / LEDGER_RELATIVE_PATH
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return str(ledger_path)


def _load_records(ledger_path: Path) -> list[Dict[str, Any]]:
    if not ledger_path.is_file():
        return []
    records: list[Dict[str, Any]] = []
    with ledger_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"人工复核审计账本第 {line_number} 行不是合法 JSON: {ledger_path}"
                ) from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _subject_path_candidates(
    review_type: str,
    discovered: Mapping[str, Any],
) -> list[str]:
    """该复核类型下"被复核成果"的候选路径。

    同一复核类型在成果恢复路径上可能解析到不同的等价文件（例如结构类复核
    既可能记 reinforcement_design_result.json，也可能记 structural_design_result.json），
    因此恢复时必须逐个候选比对，否则会把人工已接受的风险静默丢掉。
    """
    if review_type == "layout_collision_review":
        keys = (
            "layout_revision_result_path",
            "latest_revision_result_path",
            "existing_layout_result_path",
        )
    elif review_type == "structural_batch_review":
        keys = ("reinforcement_design_result_path", "structural_design_result_path")
    elif review_type == "modeling_check_review":
        keys = ("capacity_check_summary_path",)
    else:
        keys = ()
    return [str(discovered[key]).strip() for key in keys if discovered.get(key)]


def _current_subject_path(
    review_type: str,
    discovered: Mapping[str, Any],
) -> Optional[str]:
    candidates = _subject_path_candidates(review_type, discovered)
    return candidates[0] if candidates else None


def _subject_matches(record: Mapping[str, Any], discovered: Mapping[str, Any]) -> bool:
    review_type = str(record.get("review_type") or "")
    subject = _safe_dict(record.get("subject"))
    reviewed_path = str(subject.get("path") or "").strip()
    expected_hash = str(subject.get("sha256") or "").strip()
    state_key = str(subject.get("state_key") or "").strip()
    if reviewed_path and expected_hash:
        reviewed_key = os.path.normcase(os.path.abspath(reviewed_path))
        for candidate in _subject_path_candidates(review_type, discovered):
            if os.path.normcase(os.path.abspath(candidate)) != reviewed_key:
                continue
            if _sha256_file(candidate) == expected_hash:
                return True
        return False
    # 账本没有记录物理定位（例如验算类复核以 state_key 记录"被复核的是哪份成果"，
    # subject.path/sha256 为空）时，退化为"该成果当前仍然存在即视为同一次接受"。
    # 否则人工接受决定会在重新运行时被静默丢弃，流程又退回到"无法继续"
    # （2026-09-16 示例项目K31 重跑丢失 modeling_check 接受决定即为此原因）。
    if state_key:
        return _state_key_present(state_key, discovered)
    return False


# state_key -> 该成果在成果发现结果中的候选路径键
_STATE_KEY_PATH_KEYS: Dict[str, tuple[str, ...]] = {
    "capacity_check_result": ("capacity_check_summary_path",),
    "check_result": ("capacity_check_summary_path",),
    "reinforcement_design_result": (
        "reinforcement_design_result_path",
        "structural_design_result_path",
    ),
    "layout_result": (
        "layout_revision_result_path",
        "latest_revision_result_path",
        "existing_layout_result_path",
    ),
}


def _state_key_present(state_key: str, discovered: Mapping[str, Any]) -> bool:
    """判断账本里以 state_key 记录的成果在当前运行中是否仍然存在。"""
    if discovered.get(state_key):
        return True
    for key in _STATE_KEY_PATH_KEYS.get(state_key, ()):
        value = str(discovered.get(key) or "").strip()
        if value and Path(value).is_file():
            return True
    return False


def load_persisted_human_review_state(
    output_dir: Any,
    discovered: Mapping[str, Any],
) -> Dict[str, Any]:
    """Restore only the latest still-applicable acceptance for each review scope."""
    if not output_dir:
        return {}
    ledger_path = Path(str(output_dir)) / LEDGER_RELATIVE_PATH
    records = _load_records(ledger_path)
    if not records:
        return {}

    latest_by_scope: Dict[str, Dict[str, Any]] = {}
    history = []
    for record in records:
        review_type = str(record.get("review_type") or "")
        if review_type:
            latest_by_scope[review_type] = record
        history.append(
            {
                "review_type": review_type,
                "decision": {
                    "decision_id": record.get("decision_id"),
                    "action": record.get("action"),
                    "feedback": record.get("feedback") or "",
                    "extra_rounds": record.get("extra_rounds"),
                    "decided_at": record.get("decided_at"),
                },
                "subject": record.get("subject") or {},
                "restored_from": str(ledger_path),
            }
        )

    patch: Dict[str, Any] = {
        "human_review_history": history,
        "human_review_decision_path": str(ledger_path),
    }
    accepted_risks = []
    for review_type, record in latest_by_scope.items():
        action = str(record.get("action") or "")
        if action not in ACCEPTANCE_ACTIONS or not _subject_matches(record, discovered):
            continue
        risk = record.get("accepted_risk")
        if isinstance(risk, dict):
            accepted_risks.append(risk)

        current_path = _current_subject_path(review_type, discovered)
        if review_type == "layout_collision_review" and action == "accept_and_continue":
            layout_result = (
                discovered.get("layout_revision_result")
                or discovered.get("latest_revision_result")
                or discovered.get("existing_layout_result")
            )
            patch.update(
                {
                    "human_override": True,
                    "unresolved_manual_review": False,
                    "layout_revision_completed": True,
                    "layout_revision_result": layout_result,
                    "final_layout_result": layout_result,
                    "layout_revision_result_path": current_path,
                    "final_layout_result_path": current_path,
                    "task_status": "layout_revision_completed",
                }
            )
        elif review_type == "structural_batch_review" and action == "accept_partial_and_continue":
            accepted_state = _safe_dict(record.get("accepted_state"))
            batch_status = _safe_dict(accepted_state.get("reinforcement_batch_status"))
            batch_status.update({"human_accepted": True, "accepted_for_workflow": True})
            patch.update(
                {
                    "human_override": True,
                    "unresolved_manual_review": False,
                    "reinforcement_batch_status": batch_status,
                    "task_status": "structural_design_completed_with_risk",
                }
            )
        elif review_type == "modeling_check_review" and action == "accept_check_and_finish":
            patch.update(
                {
                    "human_override": True,
                    "unresolved_manual_review": False,
                    "task_status": "completed_with_accepted_risks",
                }
            )
    if accepted_risks:
        patch["accepted_risks"] = accepted_risks
    return patch
