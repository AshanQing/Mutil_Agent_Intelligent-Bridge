from __future__ import annotations

from typing import Any, Dict, List

from bridge_agents.stage_agents import LayoutRevisionAgent


def _passing_metrics() -> Dict[str, float]:
    return {
        "total_intrusion_depth_columns": 0.0,
        "avg_intrusion_depth_columns": 0.0,
        "avg_overlap_ratio_columns": 0.0,
        "conflict_column_rate": 0.0,
    }


def test_layout_revision_graph_finishes_when_metrics_already_pass(monkeypatch) -> None:
    agent = LayoutRevisionAgent()
    calls: List[str] = []

    monkeypatch.setattr(agent, "_activate_skill", lambda working: "skill")
    monkeypatch.setattr("bridge_agents.stage_agents.get_controller_llm", lambda config_path: object())

    def fake_execute(action: str, working: Dict[str, Any]):
        calls.append(action)
        return (
            {
                "task_status": "layout_revision_completed",
                "layout_revision_completed": True,
                "layout_revision_result": working.get("layout_result"),
                "final_layout_result": working.get("layout_result"),
                "layout_result": working.get("layout_result"),
                "error": None,
            },
            {"action": action, "success": True},
            True,
        )

    monkeypatch.setattr(agent, "_execute_action", fake_execute)

    result = agent.run({
        "user_intent": "layout_check_revision",
        "layout_result": {"bridge": "layout"},
        "collision_metrics": _passing_metrics(),
        "max_react_steps": 3,
    })

    assert calls == ["finish_revision"]
    assert result["task_status"] == "layout_revision_completed"
    assert result["layout_revision_completed"] is True
    assert result["revision_history"][0]["action"] == "finish_revision"


def test_layout_revision_graph_falls_back_and_loops_until_pass(monkeypatch) -> None:
    agent = LayoutRevisionAgent()
    calls: List[str] = []

    class BrokenLLM:
        def invoke(self, messages):
            raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(agent, "_activate_skill", lambda working: "skill")
    monkeypatch.setattr("bridge_agents.stage_agents.get_controller_llm", lambda config_path: BrokenLLM())

    def fake_execute(action: str, working: Dict[str, Any]):
        calls.append(action)
        if action == "run_collision_detection":
            return (
                {"collision_metrics": _passing_metrics(), "collision_items": [], "error": None},
                {"action": action, "success": True, "metrics": _passing_metrics()},
                False,
            )
        if action == "finish_revision":
            return (
                {
                    "task_status": "layout_revision_completed",
                    "layout_revision_completed": True,
                    "layout_revision_result": working.get("layout_result"),
                    "final_layout_result": working.get("layout_result"),
                    "error": None,
                },
                {"action": action, "success": True},
                True,
            )
        raise AssertionError(f"Unexpected action: {action}")

    monkeypatch.setattr(agent, "_execute_action", fake_execute)

    result = agent.run({
        "user_intent": "layout_check_revision",
        "layout_result": {"bridge": "layout"},
        "max_react_steps": 4,
    })

    assert calls == ["run_collision_detection", "finish_revision"]
    assert result["task_status"] == "layout_revision_completed"
    assert [item["action"] for item in result["revision_history"]] == [
        "run_collision_detection",
        "finish_revision",
    ]
