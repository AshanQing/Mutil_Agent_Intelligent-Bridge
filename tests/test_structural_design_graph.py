from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from bridge_agents.stage_agents import StructuralDesignAgent


def test_structural_stage_plan_regenerates_after_invalid_json(monkeypatch) -> None:
    responses = iter([
        "not-json",
        "still-not-json",
        json.dumps(_stage_plan("summarize_structural_design_result"), ensure_ascii=False),
    ])

    class FakeLlm:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return type("Response", (), {"content": next(responses)})()

    llm = FakeLlm()
    agent = StructuralDesignAgent()
    monkeypatch.setattr(agent, "_load_skill", lambda _state: "skill")
    monkeypatch.setattr("bridge_agents.stage_agents.get_structural_llm", lambda _path: llm)

    plan = agent._make_stage_plan({"user_request": "结构设计"})

    assert plan["required_steps"] == ["summarize_structural_design_result"]
    assert llm.calls == 3


def _stage_plan(*steps: str, can_execute: bool = True) -> Dict[str, Any]:
    return {
        "stage": "structural_design",
        "agent": "StructuralDesignAgent",
        "can_execute": can_execute,
        "required_steps": list(steps),
        "skipped_steps": [],
    }


def test_structural_design_builds_explicit_pipeline_graph() -> None:
    graph = StructuralDesignAgent()._build_structural_design_graph()

    assert set(graph.get_graph().nodes) >= {
        "plan_structural_design",
        "prepare_structural_action",
        "execute_structural_action",
        "complete_structural_design",
    }


def test_structural_design_graph_executes_planned_actions_in_order(monkeypatch, tmp_path: Path) -> None:
    agent = StructuralDesignAgent()
    calls: List[str] = []

    monkeypatch.setattr(
        agent,
        "_make_stage_plan",
        lambda working: _stage_plan(
            "extract_design_units",
            "dimension_design",
            "reinforcement_design",
            "summarize_structural_design_result",
        ),
    )
    monkeypatch.setattr(agent, "_log_action_start", lambda working, action: None)
    monkeypatch.setattr(agent, "_log_action_end", lambda working, action, update: None)

    def fake_run_action(action_spec, working: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(action_spec.name)
        if action_spec.name == "extract_design_units":
            return {"design_units": {"units": ["U1"]}, "error": None}
        if action_spec.name == "dimension_design":
            assert working["design_units"] == {"units": ["U1"]}
            return {"dimension_design_result": {"U1": {"diameter": 1.8}}, "error": None}
        if action_spec.name == "compute_pier_groups":
            assert working["dimension_design_result"] == {"U1": {"diameter": 1.8}}
            return {"pier_group_result": {"design_groups": []}, "error": None}
        assert working["dimension_design_result"] == {"U1": {"diameter": 1.8}}
        return {
            "reinforcement_design_result": {"U1": {"bars": 24}},
            "reinforcement_yaml_path": "reinforcement.yaml",
            "opensees_force_json_path": "forces.json",
            "error": None,
        }

    monkeypatch.setattr(agent, "_run_action", fake_run_action)

    result = agent.run({
        "layout_result": {"bridges": ["B1"]},
        "output_dir": str(tmp_path),
    })

    assert calls == ["extract_design_units", "dimension_design", "compute_pier_groups", "reinforcement_design"]
    assert result["task_status"] == "structural_design_completed"
    assert result["structural_design_result"]["handoff_to_modeling_check_agent"]["ready"] is True
    assert Path(result["structural_design_result_path"]).is_file()
    assert [entry["step"] for entry in result["structural_design_history"]] == [
        "stage_planning",
        "extract_design_units",
        "dimension_design",
        "compute_pier_groups",
        "reinforcement_design",
        "summarize_structural_design_result",
    ]
    assert not any(key.startswith("structural_design_graph_") for key in result)


def test_structural_design_graph_stops_after_action_error(monkeypatch, tmp_path: Path) -> None:
    agent = StructuralDesignAgent()
    calls: List[str] = []

    monkeypatch.setattr(
        agent,
        "_make_stage_plan",
        lambda working: _stage_plan(
            "extract_design_units",
            "dimension_design",
            "summarize_structural_design_result",
        ),
    )
    monkeypatch.setattr(agent, "_log_action_start", lambda working, action: None)
    monkeypatch.setattr(agent, "_log_action_end", lambda working, action, update: None)

    def fake_run_action(action_spec, working: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(action_spec.name)
        return {"error": "layout data unavailable"}

    monkeypatch.setattr(agent, "_run_action", fake_run_action)

    result = agent.run({"output_dir": str(tmp_path)})

    assert calls == ["extract_design_units"]
    assert result["task_status"] == "failed"
    assert result["message"] == "StructuralDesignAgent 在 extract_design_units 阶段失败。"
    failed_entry = result["structural_design_history"][-1]
    assert {key: failed_entry[key] for key in ("step", "status", "expected_key", "error")} == {
        "step": "extract_design_units",
        "status": "failed",
        "expected_key": "design_units",
        "error": "layout data unavailable",
    }
    # 时间戳保证同一动作重复失败也各自留痕（append_unique 不会把重复失败吞掉）。
    assert failed_entry["recorded_at"]


def test_structural_design_graph_rejects_plan_that_cannot_execute(monkeypatch) -> None:
    agent = StructuralDesignAgent()
    monkeypatch.setattr(
        agent,
        "_run_action",
        lambda action_spec, working: (_ for _ in ()).throw(AssertionError("action must not run")),
    )

    result = agent.run({
        "structural_stage_plan": {
            **_stage_plan("extract_design_units", "summarize_structural_design_result", can_execute=False),
            "notes": "missing layout",
        }
    })

    assert result["task_status"] == "failed"
    assert result["error"] == "missing layout"
    assert [entry["step"] for entry in result["structural_design_history"]] == ["stage_planning"]


def test_structural_design_graph_rejects_unregistered_plan_action(monkeypatch) -> None:
    agent = StructuralDesignAgent()
    monkeypatch.setattr(
        agent,
        "_run_action",
        lambda action_spec, working: (_ for _ in ()).throw(AssertionError("action must not run")),
    )

    result = agent.run({
        "structural_stage_plan": _stage_plan("unknown_action", "summarize_structural_design_result")
    })

    assert result["task_status"] == "failed"
    assert result["message"] == "StructuralDesignAgent required_steps 解析失败。"
    assert result["error"] == "StructuralDesignAgent 阶段计划包含非法步骤: unknown_action"


def test_structural_design_graph_requires_summary_marker_after_actions(monkeypatch) -> None:
    agent = StructuralDesignAgent()
    calls: List[str] = []

    monkeypatch.setattr(agent, "_log_action_start", lambda working, action: None)
    monkeypatch.setattr(agent, "_log_action_end", lambda working, action, update: None)

    def fake_run_action(action_spec, working: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(action_spec.name)
        return {"design_units": {"units": ["U1"]}, "error": None}

    monkeypatch.setattr(agent, "_run_action", fake_run_action)

    result = agent.run({
        "structural_stage_plan": _stage_plan("extract_design_units"),
        "layout_result": {"bridges": ["B1"]},
    })

    assert calls == ["extract_design_units"]
    assert result["task_status"] == "failed"
    assert result["error"] == "required_steps 缺少 summarize_structural_design_result。"


def test_structural_design_graph_replans_when_revision_context_exists(monkeypatch, tmp_path: Path) -> None:
    agent = StructuralDesignAgent()
    planning_calls: List[Dict[str, Any]] = []
    action_calls: List[str] = []

    def fake_plan(working: Dict[str, Any]) -> Dict[str, Any]:
        planning_calls.append(working)
        return _stage_plan("reinforcement_design", "summarize_structural_design_result")

    monkeypatch.setattr(agent, "_make_stage_plan", fake_plan)
    monkeypatch.setattr(agent, "_log_action_start", lambda working, action: None)
    monkeypatch.setattr(agent, "_log_action_end", lambda working, action, update: None)

    def fake_run_action(action_spec, working: Dict[str, Any]) -> Dict[str, Any]:
        action_calls.append(action_spec.name)
        assert working["dimension_design_result"] == {"U1": {"diameter": 1.8}}
        return {"reinforcement_design_result": {"U1": {"bars": 28}}, "error": None}

    monkeypatch.setattr(agent, "_run_action", fake_run_action)

    result = agent.run({
        "structural_stage_plan": _stage_plan(
            "extract_design_units",
            "dimension_design",
            "reinforcement_design",
            "summarize_structural_design_result",
        ),
        "revision_context": {"target_step": "reinforcement_design"},
        "layout_result": {"bridges": ["B1"]},
        "design_units": {"units": ["U1"]},
        "dimension_design_result": {"U1": {"diameter": 1.8}},
        "output_dir": str(tmp_path),
    })

    assert len(planning_calls) == 1
    assert action_calls == ["reinforcement_design"]
    assert result["structural_stage_plan"]["required_steps"] == [
        "reinforcement_design",
        "summarize_structural_design_result",
    ]
    assert result["task_status"] == "structural_design_completed"


def test_structural_design_partial_reinforcement_requires_manual_review(
    monkeypatch, tmp_path: Path
) -> None:
    agent = StructuralDesignAgent()
    monkeypatch.setattr(
        agent,
        "_make_stage_plan",
        lambda working: _stage_plan(
            "reinforcement_design", "summarize_structural_design_result"
        ),
    )
    monkeypatch.setattr(agent, "_log_action_start", lambda *_args: None)
    monkeypatch.setattr(agent, "_log_action_end", lambda *_args: None)
    monkeypatch.setattr(
        agent,
        "_run_action",
        lambda _spec, _working: {
            "reinforcement_design_result": {"partial": True},
            "reinforcement_batch_status": {
                "expected_task_count": 18,
                "completed_task_count": 15,
                "failed_task_count": 3,
                "failed_task_ids": ["T3", "T8", "T17"],
                "stage_complete": False,
                "partial_result_available": True,
            },
            "error": None,
        },
    )

    result = agent.run(
        {
            "layout_result": {"bridges": ["B1"]},
            "design_units": {"units": ["U1"]},
            "dimension_design_result": {"U1": {}},
            "output_dir": str(tmp_path),
        }
    )

    assert result["task_status"] == "manual_review_required"
    assert result["unresolved_manual_review"] is True
    assert result["structural_stage_summary"]["status"] == "manual_review_required"
    assert result["failed_task_ids"] == ["T3", "T8", "T17"]
    assert result["structural_design_result"]["handoff_to_modeling_check_agent"][
        "ready"
    ] is False
