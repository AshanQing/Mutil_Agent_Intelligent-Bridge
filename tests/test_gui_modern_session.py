from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bridge_agents.gui_modern_session import (
    ModernGuiForm,
    find_latest_snapshot,
    load_form_from_yaml,
    load_form_state,
    route_range_from_request,
    save_form_state,
)


def test_form_round_trip_excludes_authorization_and_thread_id(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    form = ModernGuiForm(
        user_request="完成全流程设计",
        data_path="D:/route",
        drawing_path="D:/input.dxf",
        output_dir="D:/output/run",
        file_prefix="K",
        thread_id="secret-thread",
        external_authorized=True,
    )

    save_form_state(form, state_path)
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    restored = load_form_state(state_path)

    assert "thread_id" not in stored["values"]
    assert "external_authorized" not in stored["values"]
    assert restored.user_request == "完成全流程设计"
    assert restored.external_authorized is False
    assert restored.thread_id != "secret-thread"


def test_load_form_from_yaml_reads_only_public_run_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"user_request": "设计桥梁"},
                "paths": {
                    "data_path": "data/route",
                    "input_drawing_path": "input.dxf",
                    "output_dir": "output/demo",
                    "plane_json_path": "private-intermediate.json",
                },
                "route_data": {"file_prefix": "K"},
                "agent": {"max_revision_rounds": 4},
                "modeling": {"max_check_revision_rounds": 5},
                "code_rag": {"enabled": True},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    form = load_form_from_yaml(config_path)

    assert form.user_request == "设计桥梁"
    assert form.data_path == "data/route"
    assert form.file_prefix == "K"
    assert form.max_revision_rounds == 4
    assert form.max_check_revision_rounds == 5
    assert form.code_rag_enabled is True
    assert not hasattr(form, "plane_json_path")


def test_find_latest_snapshot_prefers_matching_thread(tmp_path: Path) -> None:
    run_configs = tmp_path / "run_configs"
    run_configs.mkdir()
    (run_configs / "other.yaml").write_text("task: {}", encoding="utf-8")
    expected = run_configs / "bridge-001.yaml"
    expected.write_text("task: {}", encoding="utf-8")

    assert find_latest_snapshot(tmp_path, "bridge-001") == expected


@pytest.mark.parametrize(
    ("user_request", "expected"),
    [
        ("完成K29+000-K31+200桥梁全部设计", "K29+000 — K31+200"),
        ("完成 K1+600—K3+100 全流程桥梁设计并输出配筋图", "K1+600 — K3+100"),
        ("完成K29+000_K31+200桥梁设计", "K29+000 — K31+200"),
        ("完成全流程桥梁设计", ""),
        ("", ""),
    ],
)
def test_route_range_from_request_reads_stations(user_request: str, expected: str) -> None:
    assert route_range_from_request(user_request) == expected
