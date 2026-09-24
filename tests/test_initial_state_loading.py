from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

from bridge_agents.agent import _load_initial_state_from_settings, run_agent


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_initial_state_uses_output_dir_instead_of_configured_stage_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    route_dir = tmp_path / "route"
    route_dir.mkdir()
    (route_dir / "K.pm").write_text("", encoding="utf-8")

    discovered_layout = {"source": "output_dir"}
    configured_layout = {"source": "configured_path"}
    configured_path = tmp_path / "configured" / "design_result.json"

    _write_json(output_dir / "design_run_20260723_120000" / "design_result.json", discovered_layout)
    _write_json(configured_path, configured_layout)

    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_path": str(route_dir),
                    "output_dir": str(output_dir),
                    "existing_layout_result_path": str(configured_path),
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    state = _load_initial_state_from_settings(str(cfg))

    assert state["file_prefix"] == "K"
    assert state["layout_result"] == discovered_layout
    assert state["existing_layout_result"] == discovered_layout


def test_initial_state_discovers_preprocess_artifacts_from_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    route_dir = tmp_path / "route"
    route_dir.mkdir()
    (route_dir / "K.pm").write_text("", encoding="utf-8")

    plane_path = output_dir / "plane_from_loader" / "K_plane.json"
    mask_path = output_dir / "mask" / "obstacle_mask_K1_000_K2_000.png"
    pgw_path = output_dir / "pred_K1_000_K2_000_fast.pgw"
    obstacle_path = output_dir / "obstacle_semantic" / "complete_obstacles_grouped_section.json"

    _write_json(plane_path, {"plane": True})
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.write_bytes(b"")
    pgw_path.write_text("1\n0\n0\n-1\n0\n0\n", encoding="utf-8")
    _write_json(obstacle_path, {"obstacles": []})

    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_path": str(route_dir),
                    "output_dir": str(output_dir),
                    "plane_json_path": str(tmp_path / "ignored_plane.json"),
                    "mask_path": str(tmp_path / "ignored_mask.png"),
                    "pgw_path": str(tmp_path / "ignored.pgw"),
                    "obstacle_json_path": str(tmp_path / "ignored_obstacles.json"),
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    state = _load_initial_state_from_settings(str(cfg))

    assert state["plane_json_path"] == str(plane_path)
    assert state["mask_path"] == str(mask_path)
    assert state["pgw_path"] == str(pgw_path)
    assert state["obstacle_json_path"] == str(obstacle_path)


def test_initial_state_restores_latest_manual_review_round(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    route_dir = tmp_path / "route"
    route_dir.mkdir()
    (route_dir / "K.pm").write_text("", encoding="utf-8")

    initial_layout = {"source": "initial"}
    round_two = {"source": "revision-2"}
    round_three = {"source": "revision-3"}
    final_metrics = {
        "conflict_column_rate": 0.0656,
        "total_intrusion_depth_columns": 12.597,
    }
    _write_json(
        output_dir / "design_run_20260821_120000" / "design_result.json",
        initial_layout,
    )
    round_two_path = output_dir / "revision_results" / "revision_design_round_2.json"
    round_three_path = output_dir / "revision_results" / "revision_design_round_3.json"
    _write_json(round_two_path, round_two)
    _write_json(round_three_path, round_three)
    newer_timestamp = time.time() + 60
    os.utime(round_two_path, (newer_timestamp, newer_timestamp))
    _write_json(
        output_dir / "collision_detection" / "collision_metrics_design_result_20260821_150340.json",
        final_metrics,
    )

    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"paths": {"data_path": str(route_dir), "output_dir": str(output_dir)}},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    state = _load_initial_state_from_settings(str(cfg))

    assert state["existing_layout_result"] == initial_layout
    assert state["layout_result"] == round_three
    assert state["latest_revision_result"] == round_three
    assert state["iteration_index"] == 3
    assert state["collision_metrics"] == final_metrics
    assert state["layout_revision_completed"] is False
    assert state["task_status"] == "manual_review_required"
    assert state["unresolved_manual_review"] is True


def test_initial_state_prefers_capacity_batch_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    route_dir = tmp_path / "route"
    route_dir.mkdir()
    (route_dir / "K.pm").write_text("", encoding="utf-8")

    legacy_summary = {"check_type": "single", "overall_check": {"all_ok": True}}
    batch_summary = {
        "check_type": "cap_beam_capacity_envelope_batch",
        "expected_task_count": 2,
        "completed_task_count": 2,
        "failed_task_count": 0,
        "stage_complete": True,
        "overall_check": {"all_ok": False},
    }
    _write_json(
        output_dir / "capacity_check" / "unit-A" / "capacity_check_summary.json",
        legacy_summary,
    )
    batch_path = output_dir / "capacity_check" / "capacity_check_batch_summary.json"
    _write_json(batch_path, batch_summary)

    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"paths": {"data_path": str(route_dir), "output_dir": str(output_dir)}},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    state = _load_initial_state_from_settings(str(cfg))

    assert state["capacity_check_summary_path"] == str(batch_path)
    assert state["capacity_check_result"] == batch_summary
    assert state["check_result"] == batch_summary
    assert state["capacity_batch_status"] == {
        "expected_task_count": 2,
        "completed_task_count": 2,
        "failed_task_count": 0,
        "failed_task_ids": [],
        "stage_complete": True,
    }


def test_run_agent_extracts_station_range_before_graph_v2(monkeypatch, tmp_path: Path) -> None:
    import bridge_agents.agent as agent_module
    import bridge_agents.checkpointing as checkpointing_module
    import bridge_agents.graph_v2 as graph_v2_module

    captured = {}

    class FakeGraph:
        def invoke(self, state, config=None):
            captured.update(state)
            captured["invoke_config"] = config
            return dict(state)

    @contextmanager
    def fake_checkpointer(output_dir):
        yield object(), str(tmp_path / "checkpoint.sqlite")

    monkeypatch.setattr(
        agent_module,
        "_load_initial_state_from_settings",
        lambda config_path: {
            "output_dir": str(tmp_path / "output"),
            "agent_log": {"events": [], "files": []},
        },
    )
    monkeypatch.setattr(checkpointing_module, "open_sqlite_checkpointer", fake_checkpointer)
    monkeypatch.setattr(
        graph_v2_module,
        "build_graph_v2",
        lambda **kwargs: FakeGraph(),
    )

    result = run_agent(
        "请对示例高速K1+451-K3+500段进行全流程设计任务。",
        config_path="config/test.yaml",
        use_graph_v2=True,
        thread_id="station-test-thread",
    )

    assert captured["start_station"] == "K1+451"
    assert captured["end_station"] == "K3+500"
    invoke_config = captured["invoke_config"]
    assert invoke_config["configurable"] == {"thread_id": "station-test-thread"}
    # graph-v2 必须显式给出单次 invoke 的超级步上限（调度死循环的最后一道保险）。
    assert invoke_config["recursion_limit"] >= 1
    assert result["thread_id"] == "station-test-thread"
    assert result["checkpoint_path"] == str(tmp_path / "checkpoint.sqlite")
