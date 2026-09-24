from __future__ import annotations

from langgraph.graph import END, StateGraph

from bridge_agents.state import AgentState
from bridge_agents.state_reducers import append_unique


def test_append_unique_accepts_full_or_delta_lists() -> None:
    first = {"step": "load_data", "status": "completed"}
    second = {"step": "generate_layout_design", "status": "completed"}

    assert append_unique([first], [first, second]) == [first, second]
    assert append_unique(None, second) == [second]


def test_agent_state_reducers_merge_history_fields_without_duplicates() -> None:
    prompt_a = {"prompt_id": "agents.task_allocation.v1", "rendered_at": "t1"}
    prompt_b = {"prompt_id": "agents.initial_design.v1", "rendered_at": "t2"}
    history_a = {"react_step": 1, "action": "run_collision_detection"}
    history_b = {"react_step": 2, "action": "finish_revision"}
    review_a = {"decision_id": "r1", "action": "continue_revision"}
    review_b = {"decision_id": "r2", "action": "accept_and_continue"}

    def first_node(state: AgentState) -> AgentState:
        return {
            "prompt_trace": [prompt_a],
            "revision_history": [history_a],
            "modeling_check_history": [history_a],
            "agent_events": [history_a],
            "human_review_history": [review_a],
        }

    def second_node(state: AgentState) -> AgentState:
        return {
            "prompt_trace": [prompt_a, prompt_b],
            "revision_history": [history_a, history_b],
            "modeling_check_history": [history_a, history_b],
            "agent_events": [history_a, history_b],
            "human_review_history": [review_a, review_b],
        }

    builder = StateGraph(AgentState)
    builder.add_node("first", first_node)
    builder.add_node("second", second_node)
    builder.set_entry_point("first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    result = builder.compile().invoke({})

    assert result["prompt_trace"] == [prompt_a, prompt_b]
    assert result["revision_history"] == [history_a, history_b]
    assert result["modeling_check_history"] == [history_a, history_b]
    assert result["agent_events"] == [history_a, history_b]
    assert result["human_review_history"] == [review_a, review_b]
