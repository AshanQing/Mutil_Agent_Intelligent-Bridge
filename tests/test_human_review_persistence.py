from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from bridge_agents.agent import _load_initial_state_from_settings
from bridge_agents.contracts import StageName
from bridge_agents.graph_v2 import build_graph_v2


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(tmp_path: Path, output_dir: Path) -> Path:
    route_dir = tmp_path / "route"
    route_dir.mkdir()
    (route_dir / "K.pm").write_text("", encoding="utf-8")
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"user_request": "请完成全流程设计"},
                "paths": {
                    "data_path": str(route_dir),
                    "output_dir": str(output_dir),
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _layout_review_state(output_dir: Path, revision_path: Path) -> dict:
    return {
        "output_dir": str(output_dir),
        "user_request": "请完成全流程设计",
        "user_intent": "full_design",
        "task_status": "manual_review_required",
        "layout_result": {"桥位列表": [{"桥位编号": 5}]},
        "latest_revision_result_path": str(revision_path),
        "iteration_index": 3,
        "max_revision_rounds": 3,
        "collision_metrics": {
            "conflict_column_rate": 0.0656,
            "has_collision": True,
        },
        "latest_handoff": {
            "stage": StageName.LAYOUT_REVISION.value,
            "status": "manual_review",
            "message": "自动修正未收敛",
        },
    }


def _unused_stage_runners() -> dict:
    return {
        stage.value: lambda state: {"task_status": f"{stage.value}_completed"}
        for stage in (
            StageName.INITIAL_DESIGN,
            StageName.LAYOUT_REVISION,
            StageName.STRUCTURAL_DESIGN,
            StageName.MODELING_CHECK,
        )
    }


def test_graph_persists_layout_acceptance_as_audit_record(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    revision_path = output_dir / "revision_results" / "revision_design_round_3.json"
    _write_json(revision_path, {"桥位列表": [{"桥位编号": 5}]})
    state = _layout_review_state(output_dir, revision_path)
    graph = build_graph_v2(
        llm_invoke=lambda system, user: json.dumps(
            {
                "intent": "full_design",
                "next_stage": StageName.MANUAL_REVIEW.value,
                "stage_objective": "人工复核",
                "reason": "测试",
                "task_complete": False,
            },
            ensure_ascii=False,
        ),
        stage_runners=_unused_stage_runners(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "persist-layout-review"}}

    paused = graph.invoke(state, config=config)
    assert paused["__interrupt__"]
    graph.invoke(
        Command(resume={"action": "accept_and_continue", "feedback": "接受当前布跨"}),
        config=config,
    )

    ledger_path = output_dir / "human_review" / "human_review_decisions.jsonl"
    assert ledger_path.is_file()
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["review_type"] == "layout_collision_review"
    assert records[0]["action"] == "accept_and_continue"
    assert records[0]["subject"] == {
        "state_key": "layout_result",
        "path": str(revision_path),
        "sha256": _sha256(revision_path),
    }


def _write_layout_acceptance_record(
    output_dir: Path,
    revision_path: Path,
    *,
    subject_hash: str,
) -> None:
    ledger_path = output_dir / "human_review" / "human_review_decisions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "decision_id": "layout-accept-001",
        "review_type": "layout_collision_review",
        "action": "accept_and_continue",
        "feedback": "接受当前布跨",
        "decided_at": "2026-08-24T00:00:00+00:00",
        "subject": {
            "state_key": "layout_result",
            "path": str(revision_path),
            "sha256": subject_hash,
        },
        "accepted_risk": {
            "scope": "layout_revision",
            "reason": "接受当前布跨",
            "accepted_at": "2026-08-24T00:00:00+00:00",
        },
    }
    ledger_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def test_initial_state_restores_matching_layout_acceptance(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    revision_path = output_dir / "revision_results" / "revision_design_round_3.json"
    metrics_path = output_dir / "collision_detection" / "collision_metrics_round_3.json"
    _write_json(revision_path, {"桥位列表": [{"桥位编号": 5}]})
    _write_json(metrics_path, {"has_collision": True})
    _write_layout_acceptance_record(
        output_dir,
        revision_path,
        subject_hash=_sha256(revision_path),
    )

    state = _load_initial_state_from_settings(str(_settings(tmp_path, output_dir)))

    assert state["layout_revision_completed"] is True
    assert state["layout_revision_result"] == {"桥位列表": [{"桥位编号": 5}]}
    assert state["unresolved_manual_review"] is False
    assert state["task_status"] == "layout_revision_completed"
    assert state["human_review_history"][0]["decision"]["action"] == "accept_and_continue"
    assert state["accepted_risks"][0]["scope"] == "layout_revision"


def test_initial_state_ignores_layout_acceptance_after_artifact_changes(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    revision_path = output_dir / "revision_results" / "revision_design_round_3.json"
    metrics_path = output_dir / "collision_detection" / "collision_metrics_round_3.json"
    _write_json(revision_path, {"桥位列表": [{"桥位编号": 5}]})
    _write_json(metrics_path, {"has_collision": True})
    _write_layout_acceptance_record(
        output_dir,
        revision_path,
        subject_hash="0" * 64,
    )

    state = _load_initial_state_from_settings(str(_settings(tmp_path, output_dir)))

    assert state["layout_revision_completed"] is False
    assert state["unresolved_manual_review"] is True
    assert state["task_status"] == "manual_review_required"
    assert state.get("accepted_risks") in (None, [])


def test_initial_state_ignores_acceptance_for_a_different_artifact_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    current_revision_path = output_dir / "revision_results" / "revision_design_round_3.json"
    reviewed_path = tmp_path / "reviewed_elsewhere" / "revision_design_round_3.json"
    metrics_path = output_dir / "collision_detection" / "collision_metrics_round_3.json"
    payload = {"桥位列表": [{"桥位编号": 5}]}
    _write_json(current_revision_path, payload)
    _write_json(reviewed_path, payload)
    _write_json(metrics_path, {"has_collision": True})
    _write_layout_acceptance_record(
        output_dir,
        reviewed_path,
        subject_hash=_sha256(reviewed_path),
    )

    state = _load_initial_state_from_settings(str(_settings(tmp_path, output_dir)))

    assert state["layout_revision_completed"] is False
    assert state["unresolved_manual_review"] is True


def _write_acceptance_record(
    output_dir: Path,
    *,
    review_type: str,
    action: str,
    state_key: str,
    subject_path: Path,
    risk_scope: str,
    accepted_state: dict | None = None,
) -> None:
    ledger_path = output_dir / "human_review" / "human_review_decisions.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "decision_id": f"{risk_scope}-accept-001",
        "review_type": review_type,
        "action": action,
        "feedback": "人工确认",
        "decided_at": "2026-08-24T00:00:00+00:00",
        "subject": {
            "state_key": state_key,
            "path": str(subject_path),
            "sha256": _sha256(subject_path),
        },
        "accepted_risk": {
            "scope": risk_scope,
            "reason": "人工确认",
            "accepted_at": "2026-08-24T00:00:00+00:00",
        },
        "accepted_state": accepted_state or {},
    }
    ledger_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def test_initial_state_restores_structural_partial_acceptance(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    reinforcement_path = (
        output_dir
        / "structural_design"
        / "reinforcement_design"
        / "reinforcement_design_result.json"
    )
    _write_json(reinforcement_path, {"status": "manual_review_required"})
    _write_acceptance_record(
        output_dir,
        review_type="structural_batch_review",
        action="accept_partial_and_continue",
        state_key="reinforcement_design_result",
        subject_path=reinforcement_path,
        risk_scope="structural_design",
        accepted_state={
            "reinforcement_batch_status": {
                "expected_task_count": 18,
                "completed_task_count": 15,
                "failed_task_count": 3,
                "failed_task_ids": ["T3", "T8", "T17"],
                "stage_complete": False,
            }
        },
    )

    state = _load_initial_state_from_settings(str(_settings(tmp_path, output_dir)))

    assert state["reinforcement_batch_status"]["human_accepted"] is True
    assert state["reinforcement_batch_status"]["accepted_for_workflow"] is True
    assert state["reinforcement_batch_status"]["failed_task_ids"] == ["T3", "T8", "T17"]
    assert state["task_status"] == "structural_design_completed_with_risk"
    assert state["accepted_risks"][0]["scope"] == "structural_design"


def test_structural_acceptance_survives_subject_path_variant(tmp_path: Path) -> None:
    """账本记录的 subject 是结构设计汇总、恢复时解析到配筋汇总时，风险不得被丢掉。

    实际运行中同一复核类型的两条候选路径都合法（2026-09-16 示例项目K29 即因此
    把人工已接受的部分成果风险静默丢失，交付清单里没有任何风险记录）。
    """
    output_dir = tmp_path / "run"
    structural_path = output_dir / "structural_design" / "structural_design_result.json"
    reinforcement_path = (
        output_dir / "structural_design" / "reinforcement_design" / "reinforcement_design_result.json"
    )
    _write_json(structural_path, {"status": "manual_review_required"})
    _write_json(reinforcement_path, {"status": "manual_review_required"})
    _write_acceptance_record(
        output_dir,
        review_type="structural_batch_review",
        action="accept_partial_and_continue",
        state_key="reinforcement_design_result",
        subject_path=structural_path,
        risk_scope="structural_design",
    )

    state = _load_initial_state_from_settings(str(_settings(tmp_path, output_dir)))

    assert state["accepted_risks"][0]["scope"] == "structural_design"


def test_restored_modeling_acceptance_finishes_without_llm(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    capacity_path = output_dir / "capacity_check" / "capacity_check_batch_summary.json"
    _write_json(
        capacity_path,
        {
            "check_type": "cap_beam_capacity_envelope_batch",
            "stage_complete": True,
            "overall_check": {"all_ok": False},
        },
    )
    _write_acceptance_record(
        output_dir,
        review_type="modeling_check_review",
        action="accept_check_and_finish",
        state_key="capacity_check_result",
        subject_path=capacity_path,
        risk_scope="modeling_check",
    )
    state = _load_initial_state_from_settings(str(_settings(tmp_path, output_dir)))
    state.update({"user_request": "请完成全流程设计", "user_intent": "full_design"})
    calls = 0

    def unexpected_llm(system: str, user: str) -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("恢复的验算接受终态不得再次调用协调器 LLM")

    graph = build_graph_v2(
        llm_invoke=unexpected_llm,
        stage_runners=_unused_stage_runners(),
    )
    result = graph.invoke(state)

    assert calls == 0
    assert result["task_status"] == "completed_with_accepted_risks"
    assert result["final_summary"]["accepted_risks"][0]["scope"] == "modeling_check"
