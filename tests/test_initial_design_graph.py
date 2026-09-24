from __future__ import annotations

from typing import Any, Dict, List

from bridge_agents.stage_agents import InitialDesignAgent


def _stage_plan(*steps: str) -> Dict[str, Any]:
    return {
        "stage": "initial_design",
        "agent": "InitialDesignAgent",
        "can_execute": True,
        "required_steps": list(steps),
        "skipped_steps": [],
    }


def test_initial_design_builds_explicit_pipeline_graph() -> None:
    graph = InitialDesignAgent()._build_initial_design_graph()

    assert set(graph.get_graph().nodes) >= {
        "plan_initial_design",
        "prepare_initial_action",
        "execute_initial_action",
        "complete_initial_design",
    }


def test_initial_design_graph_executes_planned_actions_in_order(monkeypatch) -> None:
    agent = InitialDesignAgent()
    calls: List[str] = []

    monkeypatch.setattr(
        agent,
        "_make_stage_plan",
        lambda working: _stage_plan("load_data", "generate_layout_design"),
    )
    monkeypatch.setattr(agent, "_log_action_start", lambda working, action: None)
    monkeypatch.setattr(agent, "_log_action_end", lambda working, action, update: None)

    def fake_run_action(action_spec, working: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(action_spec.name)
        if action_spec.name == "load_data":
            return {"cropped_data": {"route": "K1"}, "error": None}
        assert working["cropped_data"] == {"route": "K1"}
        return {"layout_result": {"bridges": ["B1"]}, "error": None}

    monkeypatch.setattr(agent, "_run_action", fake_run_action)

    result = agent.run({"user_request": "生成布跨方案"})

    assert calls == ["load_data", "generate_layout_design"]
    assert result["layout_result"] == {"bridges": ["B1"]}
    assert result["task_status"] == "initial_design_completed"
    assert [entry["step"] for entry in result["initial_design_history"]] == [
        "stage_planning",
        "load_data",
        "generate_layout_design",
    ]
    assert not any(key.startswith("initial_design_graph_") for key in result)


def test_initial_design_graph_stops_after_action_error(monkeypatch) -> None:
    agent = InitialDesignAgent()
    calls: List[str] = []

    monkeypatch.setattr(
        agent,
        "_make_stage_plan",
        lambda working: _stage_plan("load_data", "generate_layout_design"),
    )
    monkeypatch.setattr(agent, "_log_action_start", lambda working, action: None)
    monkeypatch.setattr(agent, "_log_action_end", lambda working, action, update: None)

    def fake_run_action(action_spec, working: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(action_spec.name)
        return {"error": "route data unavailable"}

    monkeypatch.setattr(agent, "_run_action", fake_run_action)

    result = agent.run({"user_request": "生成布跨方案"})

    assert calls == ["load_data"]
    assert result["task_status"] == "failed"
    assert result["message"] == "InitialDesignAgent 在 load_data 阶段失败。"
    assert result["initial_design_history"][-1] == {
        "step": "load_data",
        "status": "failed",
        "expected_key": "cropped_data",
        "error": "route data unavailable",
    }


def test_initial_design_graph_rejects_plan_that_cannot_execute(monkeypatch) -> None:
    agent = InitialDesignAgent()
    monkeypatch.setattr(
        agent,
        "_run_action",
        lambda action_spec, working: (_ for _ in ()).throw(AssertionError("action must not run")),
    )

    result = agent.run({
        "initial_design_stage_plan": {
            **_stage_plan("load_data"),
            "can_execute": False,
            "notes": "missing route range",
        }
    })

    assert result["task_status"] == "failed"
    assert result["error"] == "missing route range"
    assert [entry["step"] for entry in result["initial_design_history"]] == ["stage_planning"]


def test_initial_design_graph_rejects_unregistered_plan_action(monkeypatch) -> None:
    agent = InitialDesignAgent()
    monkeypatch.setattr(
        agent,
        "_run_action",
        lambda action_spec, working: (_ for _ in ()).throw(AssertionError("action must not run")),
    )

    result = agent.run({"initial_design_stage_plan": _stage_plan("unknown_action")})

    assert result["task_status"] == "failed"
    assert result["message"] == "InitialDesignAgent required_steps 解析失败。"
    assert result["error"] == "InitialDesignAgent 阶段计划包含非法步骤: unknown_action"


def test_initial_design_graph_wraps_planning_failure(monkeypatch) -> None:
    agent = InitialDesignAgent()

    def fail_planning(working: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("controller unavailable")

    monkeypatch.setattr(agent, "_make_stage_plan", fail_planning)

    result = agent.run({"user_request": "生成布跨方案"})

    assert result["active_agent"] == "InitialDesignAgent"
    assert result["task_status"] == "failed"
    assert result["message"] == "InitialDesignAgent 阶段计划生成失败。"
    assert result["error"] == "controller unavailable"


def test_initial_design_graph_preserves_existing_history_without_duplicates(monkeypatch) -> None:
    agent = InitialDesignAgent()
    previous = {"step": "load_data", "status": "completed", "expected_key": "cropped_data", "error": None}

    monkeypatch.setattr(agent, "_log_action_start", lambda working, action: None)
    monkeypatch.setattr(agent, "_log_action_end", lambda working, action, update: None)
    monkeypatch.setattr(
        agent,
        "_run_action",
        lambda action_spec, working: {"cropped_data": {"route": "K1"}, "error": None},
    )

    result = agent.run({
        "initial_design_stage_plan": _stage_plan("load_data"),
        "initial_design_history": [previous],
    })

    assert result["initial_design_history"].count(previous) == 1
    assert [entry["step"] for entry in result["initial_design_history"]] == [
        "load_data",
        "stage_planning",
    ]
