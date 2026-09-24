from __future__ import annotations

import json
from pathlib import Path

from bridge_agents.prompt_registry import render_prompt
from tools.layout_revision_prompt_tool import (
    build_compact_collision_context,
    build_layout_revision_prompt_from_state,
)


COLLISION_ITEMS = [
    {
        "bridge_id": "2",
        "scheme_type": "分幅布跨方案",
        "side_label": "左幅",
        "pier_id": "3",
        "col_name": "Left-2",
        "station": "K7+100",
        "station_label": "K7+100.000",
        "k_val": 7100.0,
        "x": 480700.0,
        "y": 3021000.0,
        "r_m": 1.25,
        "res": {
            "status": "COLLISION",
            "intrusion_depth_m": 2.0,
            "overlap_ratio": 0.5,
            "center_dist_m": 1.0,
        },
    },
    {
        "bridge_id": "1",
        "scheme_type": "分幅布跨方案",
        "side_label": "右幅",
        "pier_id": "2",
        "col_name": "Right-1",
        "station": "K6+600",
        "station_label": "K6+600.000",
        "k_val": 6600.0,
        "x": 480600.0,
        "y": 3020600.0,
        "r_m": 1.25,
        "res": {
            "status": "COLLISION",
            "intrusion_depth_m": 1.5,
            "overlap_ratio": 0.25,
            "center_dist_m": 0.5,
        },
    },
]


def test_compact_collision_context_keeps_risk_fields_and_stable_order() -> None:
    metrics = {
        "conflict_column_rate": 0.1,
        "total_columns": 20,
    }

    forward = build_compact_collision_context(metrics, COLLISION_ITEMS)
    reversed_result = build_compact_collision_context(metrics, list(reversed(COLLISION_ITEMS)))

    assert forward == reversed_result
    assert forward["collision_metrics"] == {
        "conflict_column_rate": 0.1,
        "total_columns": 20,
    }
    assert forward["collision_items"][0] == {
        "bridge_id": "1",
        "side_label": "右幅",
        "pier_id": "2",
        "col_name": "Right-1",
        "station": "K6+600.000",
        "status": "COLLISION",
        "intrusion_depth_m": 1.5,
        "overlap_ratio": 0.25,
    }
    assert "x" not in forward["collision_items"][0]
    assert "center_dist_m" not in forward["collision_items"][0]


def test_revision_template_places_static_contract_before_dynamic_layout() -> None:
    rendered = render_prompt(
        "tasks.layout_revision_design.v1",
        {
            "ROLE_BLOCK": "STATIC_ROLE_MARKER",
            "STANDARDS_BLOCK": "STATIC_STANDARDS_MARKER",
            "DESIGN_INPUT_JSON": "STABLE_DESIGN_INPUT_MARKER",
            "PREVIOUS_LAYOUT_JSON": "DYNAMIC_PREVIOUS_LAYOUT_MARKER",
            "REVISION_INSTRUCTION_TEXT": "DYNAMIC_INSTRUCTION_MARKER",
            "REVISION_ADVICE_JSON": "DYNAMIC_COLLISION_MARKER",
            "OUTPUT_SCHEMA_BLOCK": "STATIC_OUTPUT_SCHEMA_MARKER",
        },
        config_path="config/settings.yaml",
    )
    prompt = rendered.user_content or rendered.system_content

    previous_index = prompt.index("DYNAMIC_PREVIOUS_LAYOUT_MARKER")
    assert prompt.index("STATIC_ROLE_MARKER") < previous_index
    assert prompt.index("STATIC_STANDARDS_MARKER") < previous_index
    assert prompt.index("修正原则") < previous_index
    assert prompt.index("STATIC_OUTPUT_SCHEMA_MARKER") < previous_index
    assert prompt.index("STABLE_DESIGN_INPUT_MARKER") < previous_index
    assert previous_index < prompt.index("DYNAMIC_COLLISION_MARKER")
    assert previous_index < prompt.index("DYNAMIC_INSTRUCTION_MARKER")


def test_revision_prompt_uses_compact_collision_context(tmp_path) -> None:
    state = {
        "config_path": "config/settings.yaml",
        "output_dir": str(tmp_path),
        "iteration_index": 1,
        "layout_result": {
            "设桥总览": {"说明": "PREVIOUS_LAYOUT_MARKER"},
            "桥位列表": [],
        },
        "design_input": {"线路": "A1"},
        "collision_metrics": {"conflict_column_rate": 0.1},
        "collision_items": COLLISION_ITEMS,
        "revision_instruction": "仅调整冲突墩位。",
    }

    forward = build_layout_revision_prompt_from_state(state)["revision_prompt"]
    state["collision_items"] = list(reversed(COLLISION_ITEMS))
    reversed_prompt = build_layout_revision_prompt_from_state(state)["revision_prompt"]

    assert forward == reversed_prompt
    assert '"station": "K6+600.000"' in forward
    assert '"intrusion_depth_m": 1.5' in forward
    assert '"x": 480600.0' not in forward
    assert '"center_dist_m": 0.5' not in forward

    rows = [
        json.loads(line)
        for line in (Path(tmp_path) / "logs/prompt_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[-1]["stage"] == "layout_revision"
    assert rows[-1]["status"] == "rendered"
