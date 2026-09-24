from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from bridge_agents.tool_actions import run_capacity_check_action


def _write_inputs(root: Path, task_id: str) -> tuple[str, str]:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    reinforcement_path = task_dir / f"reinforcement_result_{task_id}.yaml"
    force_path = task_dir / f"internal_force_output_full_beam_{task_id}.json"
    reinforcement_path.write_text("盖梁配筋: {}\n", encoding="utf-8")
    force_path.write_text("{}", encoding="utf-8")
    return str(reinforcement_path), str(force_path)


def _detailed_reinforcement_state(
    output_dir: Path,
    pairs: List[tuple[str, str, str]],
) -> Dict[str, Any]:
    group_results = []
    for task_id, reinforcement_path, force_path in pairs:
        group_results.append(
            {
                "task_index": len(group_results) + 1,
                "reinforcement_task": {
                    "task_id": task_id,
                    "桥梁编号": "1-R",
                    "单元编号": "1-R-1",
                    "分组编号": "G1",
                },
                "output_files": {
                    "reinforcement_result_yaml_path": reinforcement_path,
                    "internal_force_output_path": force_path,
                },
            }
        )
    return {
        "output_dir": str(output_dir),
        "capacity_check_output_dir": str(output_dir / "capacity_check"),
        "reinforcement_design_result": {
            "success": True,
            "reinforcement_design_result": {
                "任务3_下部结构配筋设计结果": {
                    "分组原始结果": group_results,
                    "expected_task_count": len(group_results),
                    "completed_task_count": len(group_results),
                    "failed_task_count": 0,
                    "stage_complete": True,
                }
            },
        },
        "reinforcement_batch_status": {
            "expected_task_count": len(group_results),
            "completed_task_count": len(group_results),
            "failed_task_count": 0,
            "stage_complete": True,
        },
    }


def _passing_result(max_utilization: float) -> Dict[str, Any]:
    return {
        "success": True,
        "check_type": "cap_beam_capacity_envelope",
        "overall_check": {
            "M_pos_ok": True,
            "M_neg_ok": True,
            "V_ok": True,
            "compression_zone_ok": True,
            "all_ok": True,
        },
        "control_sections": {
            "max_positive_moment_utilization": {
                "x_m": 1.25,
                "util_M_pos": max_utilization,
                "util_M_neg": 0.2,
                "util_V": 0.3,
            }
        },
        "utilization_summary": {
            "max_utilization": max_utilization,
            "control_name": "max_positive_moment_utilization",
            "util_type": "util_M_pos",
            "x_m": 1.25,
        },
        "output_files": {},
        "summary": {
            "formula_coverage": {
                "executed": [
                    "F_3362_CH05_5_2_2_1_BENDING_CAPACITY_TENSION_FLANGE",
                    "F_3362_CH08_8_4_4_CAP_BEAM_SHEAR_CAPACITY",
                    "F_3362_CH08_8_4_5_CAP_BEAM_INCLINED_SHEAR",
                ],
                "not_covered": ["裂缝", "挠度"],
            }
        },
        "error": None,
    }


def test_capacity_action_checks_every_reinforcement_group(monkeypatch, tmp_path: Path) -> None:
    first = _write_inputs(tmp_path / "inputs", "unit-A")
    second = _write_inputs(tmp_path / "inputs", "unit-B")
    state = _detailed_reinforcement_state(
        tmp_path,
        [("unit-A", *first), ("unit-B", *second)],
    )
    calls: List[Dict[str, Any]] = []

    def fake_capacity_check_tool(**kwargs):
        calls.append(kwargs)
        task_id = Path(kwargs["reinforcement_yaml_path"]).parent.name
        return _passing_result(0.72 if task_id == "unit-A" else 0.83)

    monkeypatch.setattr(
        "bridge_agents.tool_actions.capacity_check_tool",
        fake_capacity_check_tool,
    )

    result = run_capacity_check_action(state)

    assert result["error"] is None
    assert [Path(call["reinforcement_yaml_path"]).parent.name for call in calls] == [
        "unit-A",
        "unit-B",
    ]
    assert [Path(call["opensees_force_json_path"]).parent.name for call in calls] == [
        "unit-A",
        "unit-B",
    ]
    assert [Path(call["output_dir"]).name for call in calls] == ["unit-A", "unit-B"]
    assert result["capacity_batch_status"] == {
        "expected_task_count": 2,
        "completed_task_count": 2,
        "failed_task_count": 0,
        "failed_task_ids": [],
        "stage_complete": True,
    }
    assert result["check_result"]["overall_check"]["all_ok"] is True
    assert result["check_result"]["utilization_summary"]["max_utilization"] == 0.83
    assert result["check_result"]["utilization_summary"]["task_id"] == "unit-B"
    assert Path(result["capacity_check_summary_path"]).name == "capacity_check_batch_summary.json"
    assert len(result["compliance_matrix"]["entries"]) == 8
    assert result["compliance_matrix"]["coverage_limitations"] == ["裂缝", "挠度"]
    assert result["unresolved_code_items"] == []


def test_capacity_action_aggregates_engineering_failure(monkeypatch, tmp_path: Path) -> None:
    first = _write_inputs(tmp_path / "inputs", "unit-A")
    second = _write_inputs(tmp_path / "inputs", "unit-B")
    state = _detailed_reinforcement_state(
        tmp_path,
        [("unit-A", *first), ("unit-B", *second)],
    )

    def fake_capacity_check_tool(**kwargs):
        task_id = Path(kwargs["reinforcement_yaml_path"]).parent.name
        result = _passing_result(0.75 if task_id == "unit-A" else 1.18)
        if task_id == "unit-B":
            result["overall_check"].update({"M_pos_ok": False, "all_ok": False})
        return result

    monkeypatch.setattr(
        "bridge_agents.tool_actions.capacity_check_tool",
        fake_capacity_check_tool,
    )

    result = run_capacity_check_action(state)

    assert result["error"] is None
    assert result["capacity_batch_status"]["stage_complete"] is True
    assert result["check_result"]["overall_check"] == {
        "M_pos_ok": False,
        "M_neg_ok": True,
        "V_ok": True,
        "compression_zone_ok": True,
        "all_ok": False,
    }
    assert result["check_result"]["failed_check_task_ids"] == ["unit-B"]
    assert result["check_result"]["utilization_summary"]["max_utilization"] == 1.18
    assert result["check_result"]["utilization_summary"]["task_id"] == "unit-B"
    assert set(result["check_result"]["control_sections"]) == {
        "unit-A::max_positive_moment_utilization",
        "unit-B::max_positive_moment_utilization",
    }
    unresolved = result["unresolved_code_items"]
    assert any(
        item["task_id"] == "unit-B"
        and item["code_item_id"].endswith("flexure_positive")
        and item["status"] == "fail"
        for item in unresolved
    )


def test_capacity_action_rejects_unpaired_multi_path_inputs(monkeypatch, tmp_path: Path) -> None:
    first = _write_inputs(tmp_path / "inputs", "unit-A")
    second = _write_inputs(tmp_path / "inputs", "unit-B")
    calls: List[Dict[str, Any]] = []

    def fake_capacity_check_tool(**kwargs):
        calls.append(kwargs)
        return _passing_result(0.5)

    monkeypatch.setattr(
        "bridge_agents.tool_actions.capacity_check_tool",
        fake_capacity_check_tool,
    )

    result = run_capacity_check_action(
        {
            "output_dir": str(tmp_path),
            "reinforcement_yaml_paths": [first[0], second[0]],
            "internal_force_output_paths": [first[1]],
        }
    )

    assert calls == []
    assert "无法按 task_id 一一配对" in result["error"]


def test_capacity_action_keeps_single_path_compatibility(monkeypatch, tmp_path: Path) -> None:
    reinforcement_path, force_path = _write_inputs(tmp_path / "inputs", "unit-A")

    monkeypatch.setattr(
        "bridge_agents.tool_actions.capacity_check_tool",
        lambda **kwargs: _passing_result(0.64),
    )

    result = run_capacity_check_action(
        {
            "output_dir": str(tmp_path),
            "capacity_check_output_dir": str(tmp_path / "capacity_check"),
            "reinforcement_yaml_path": reinforcement_path,
            "opensees_force_json_path": force_path,
        }
    )

    assert result["error"] is None
    assert result["capacity_batch_status"]["expected_task_count"] == 1
    assert result["check_result"]["overall_check"]["all_ok"] is True
