from __future__ import annotations

from typing import Any, Dict, List

from bridge_agents.stage_agents import ModelingCheckAgent


def _passing_check_result() -> Dict[str, Any]:
    return {"overall_check": {"all_ok": True}, "control_sections": {}, "utilization_summary": {}}


def _failing_check_result() -> Dict[str, Any]:
    return {"overall_check": {"all_ok": False}, "control_sections": {}, "utilization_summary": {}}


def test_modeling_check_graph_finishes_when_check_already_passed(monkeypatch) -> None:
    agent = ModelingCheckAgent()
    calls: List[str] = []

    monkeypatch.setattr(agent, "_activate_skill", lambda working: "skill")
    monkeypatch.setattr("bridge_agents.stage_agents.get_controller_llm", lambda config_path: object())

    def fake_execute(action: str, working: Dict[str, Any]):
        calls.append(action)
        return (
            {
                "feedback_decision": {
                    "overall_status": "pass",
                    "next_action": "pass",
                    "target_agent": "END",
                    "requires_rerun_check": False,
                },
                "task_status": "modeling_check_passed",
                "message": "passed",
                "error": None,
            },
            {"action": action, "success": True},
            True,
        )

    monkeypatch.setattr(agent, "_execute_action", fake_execute)

    result = agent.run({
        "check_result": _passing_check_result(),
        "max_modeling_check_steps": 3,
    })

    assert calls == ["finish_check"]
    assert result["task_status"] == "modeling_check_passed"
    assert result["feedback_decision"]["next_action"] == "pass"
    assert result["modeling_check_history"][0]["action"] == "finish_check"


def test_modeling_check_graph_generates_feedback_then_finishes(monkeypatch) -> None:
    agent = ModelingCheckAgent()
    calls: List[str] = []

    class BrokenLLM:
        def invoke(self, messages):
            raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(agent, "_activate_skill", lambda working: "skill")
    monkeypatch.setattr("bridge_agents.stage_agents.get_controller_llm", lambda config_path: BrokenLLM())

    def fake_execute(action: str, working: Dict[str, Any]):
        calls.append(action)
        if action == "run_capacity_check":
            return (
                {"check_result": _failing_check_result(), "capacity_check_result": _failing_check_result(), "error": None},
                {"action": action, "success": True},
                False,
            )
        if action == "generate_revision_instruction":
            decision = {
                "overall_status": "revise_reinforcement",
                "next_action": "revise_reinforcement",
                "target_agent": "StructuralDesignAgent",
                "target_step": "reinforcement_design",
                "requires_rerun_check": True,
            }
            return (
                {
                    "feedback_decision": decision,
                    "revision_context": {"target_step": "reinforcement_design"},
                    "check_iteration_index": 1,
                    "error": None,
                },
                {"action": action, "success": True, "feedback_decision": decision},
                False,
            )
        if action == "finish_check":
            return (
                {
                    "feedback_decision": working.get("feedback_decision"),
                    "task_status": "modeling_check_revision_required",
                    "message": "revision required",
                    "error": None,
                },
                {"action": action, "success": True},
                True,
            )
        raise AssertionError(f"Unexpected action: {action}")

    monkeypatch.setattr(agent, "_execute_action", fake_execute)

    result = agent.run({
        "reinforcement_yaml_path": "reinforcement.yaml",
        "opensees_force_json_path": "force.json",
        "max_modeling_check_steps": 5,
    })

    assert calls == ["run_capacity_check", "generate_revision_instruction", "finish_check"]
    assert result["task_status"] == "modeling_check_revision_required"
    assert result["feedback_decision"]["next_action"] == "revise_reinforcement"
    assert [item["action"] for item in result["modeling_check_history"]] == calls
