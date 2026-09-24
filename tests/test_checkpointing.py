from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.graph import END, StateGraph

from bridge_agents.agent import run_agent
from bridge_agents.checkpointing import (
    NULL_TASK_ID,
    clear_stale_resume_writes,
    open_sqlite_checkpointer,
    stale_resume_writes,
)
from bridge_agents.state import AgentState


def _build_small_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("finish", lambda state: {"task_status": "checkpointed"})
    builder.set_entry_point("finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)


def test_sqlite_checkpointer_restores_state_after_reopen(tmp_path: Path) -> None:
    config = {"configurable": {"thread_id": "persistent-review-1"}}

    with open_sqlite_checkpointer(tmp_path) as (checkpointer, checkpoint_path):
        graph = _build_small_graph(checkpointer)
        result = graph.invoke({"user_request": "保存我"}, config=config)
        assert result["task_status"] == "checkpointed"

    assert Path(checkpoint_path).exists()

    with open_sqlite_checkpointer(tmp_path) as (checkpointer, reopened_path):
        graph = _build_small_graph(checkpointer)
        snapshot = graph.get_state(config)

    assert reopened_path == checkpoint_path
    assert snapshot.values["user_request"] == "保存我"
    assert snapshot.values["task_status"] == "checkpointed"


def test_run_agent_pauses_and_resumes_from_sqlite_in_a_new_call(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        f"paths:\n  output_dir: '{output_dir.as_posix()}'\n",
        encoding="utf-8",
    )
    thread_id = "run-agent-review-1"
    review_state = {
        "output_dir": str(output_dir),
        "task_status": "manual_review_required",
        "iteration_index": 3,
        "max_revision_rounds": 3,
        "layout_result": {"设桥总览": {}},
        "collision_metrics": {"conflict_column_rate": 0.0656},
        "latest_handoff": {
            "stage": "layout_revision",
            "status": "manual_review",
            "message": "达到自动修正轮次上限。",
        },
    }

    paused = run_agent(
        "继续全流程设计",
        config_path=str(config_path),
        initial_state=review_state,
        use_graph_v2=True,
        thread_id=thread_id,
    )

    assert paused["thread_id"] == thread_id
    assert paused["__interrupt__"][0].value["review_type"] == "layout_collision_review"
    assert Path(paused["checkpoint_path"]).exists()

    resumed = run_agent(
        "",
        config_path=str(config_path),
        use_graph_v2=True,
        thread_id=thread_id,
        resume={"action": "abort"},
    )

    assert resumed["task_status"] == "cancelled"
    assert resumed["thread_id"] == thread_id
    assert resumed["error"] is None


def test_stale_pending_writes_are_detected_and_cleared(tmp_path: Path) -> None:
    """恢复残值（__resume__/__error__ 与重复的全局输入补丁）必须能被发现并清除。

    这两类残值都会让后续恢复失真：前者让新决策被旧值顶掉，后者因同一 channel
    在一次恢复里被写两次而抛 InvalidUpdateError。
    """
    output_dir = tmp_path / "run"
    thread_id = "stale-writes-1"
    config = {"configurable": {"thread_id": thread_id}}

    with open_sqlite_checkpointer(output_dir) as (checkpointer, _path):
        graph = _build_small_graph(checkpointer)
        graph.invoke({"user_request": "保存我"}, config=config)
        row = checkpointer.conn.execute(
            "SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints "
            "WHERE thread_id=? AND checkpoint_ns='' ORDER BY rowid DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        stale_rows = [
            (row[0], row[1], row[2], NULL_TASK_ID, 0, "collision_metrics", "json", b"{}"),
            (row[0], row[1], row[2], NULL_TASK_ID, 1, "collision_metrics", "json", b"{}"),
            (row[0], row[1], row[2], "task-x", 0, "__resume__", "json", b"{}"),
            (row[0], row[1], row[2], "task-x", 1, "__error__", "json", b"{}"),
        ]
        checkpointer.conn.executemany(
            "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            stale_rows,
        )
        checkpointer.conn.commit()

    detected = stale_resume_writes(output_dir, thread_id)
    channels = sorted(item["channel"] for item in detected)
    assert channels == ["__error__", "__resume__", "collision_metrics", "collision_metrics"]

    cleared = clear_stale_resume_writes(output_dir, thread_id)
    assert len(cleared) == 4
    assert stale_resume_writes(output_dir, thread_id) == []

    connection = sqlite3.connect(str(Path(output_dir) / "checkpoints" / "graph_v2.sqlite"))
    try:
        remaining = connection.execute(
            "SELECT COUNT(*) FROM writes WHERE thread_id=? AND channel IN "
            "('__resume__','__error__','collision_metrics')",
            (thread_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert remaining == 0
