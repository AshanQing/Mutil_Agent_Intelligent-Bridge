from __future__ import annotations

from tools.revision_instruction_tool import DirectFeedbackGenerator
from bridge_agents.prompt_registry import render_prompt
from bridge_agents import tool_actions


def _collision(
    pier_id: str,
    station: float,
    *,
    bridge_id: str = "1",
    side_label: str = "右幅",
    depth: float = 1.0,
) -> dict:
    return {
        "bridge_id": bridge_id,
        "side_label": side_label,
        "pier_id": pier_id,
        "k_val": station,
        "station_label": f"K0+{station:06.1f}",
        "res": {
            "status": "COLLISION",
            "intrusion_depth_m": depth,
            "overlap_ratio": 0.4,
        },
    }


def _layout() -> dict:
    return {
        "桥位列表": [
            {
                "桥位编号": "1",
                "统一布跨方案": {
                    "设桥信息": {
                        "跨径组合": "8×30",
                        "桥型": "预应力混凝土先简支后连续T梁",
                    }
                },
            }
        ]
    }


def test_collision_clusters_do_not_merge_separated_sides() -> None:
    generator = DirectFeedbackGenerator(
        [
            _collision("1", 100.0, side_label="右幅"),
            _collision("1", 100.0, side_label="左幅"),
        ],
        layout_result=_layout(),
    )

    clusters = generator._cluster_collisions()

    assert len(clusters) == 2
    assert {cluster[0]["side_label"] for cluster in clusters} == {"右幅", "左幅"}


def test_multiple_pier_collision_requires_review_without_invented_clear_span() -> None:
    report = [_collision(str(index), 30.0 * index) for index in range(1, 7)]
    generator = DirectFeedbackGenerator(report, layout_result=_layout())

    payload = generator.to_metrics_payload()
    instruction = generator.generate_prompt()

    strategy = payload["cluster_strategy_summary"][0]["strategy"]
    assert strategy["necessity_level"] == "manual_review"
    assert strategy["required_clear_span"] is None
    assert "估计避障净跨需求" not in instruction
    assert "策略C" not in instruction
    assert "165.0m" not in instruction
    assert "碰撞点不能作为连续障碍物边界" in instruction


def test_single_pier_collision_stays_in_standard_span_action_space() -> None:
    generator = DirectFeedbackGenerator([_collision("3", 90.0)], layout_result=_layout())

    strategy = generator.to_metrics_payload()["cluster_strategy_summary"][0]["strategy"]

    assert strategy["necessity_level"] == "minor"
    assert strategy["allowed_spans"] == [20.0, 25.0, 30.0, 35.0, 40.0]
    assert strategy["required_clear_span"] is None


def test_abutment_feedback_does_not_introduce_transition_pier_role() -> None:
    generator = DirectFeedbackGenerator(
        [_collision("0 (桥台)", 0.0)],
        layout_result=_layout(),
    )

    instruction = generator.generate_prompt()

    assert "桥梁边界调整" in instruction
    assert "过渡墩" not in instruction


def test_revision_task_contract_routes_unbounded_multi_pier_collision_to_review() -> None:
    rendered = render_prompt(
        "tasks.layout_revision_design.v1",
        {
            "ROLE_BLOCK": "桥梁工程师",
            "STANDARDS_BLOCK": "标准跨径20m至40m。",
            "DESIGN_INPUT_JSON": "{}",
            "PREVIOUS_LAYOUT_JSON": "{}",
            "REVISION_ADVICE_JSON": "{}",
            "REVISION_INSTRUCTION_TEXT": "多墩碰撞且缺少连续障碍物边界，转人工复核。",
            "OUTPUT_SCHEMA_BLOCK": "只输出JSON。",
        },
    )
    prompt = rendered.user_content

    assert "删除中间冲突墩位并合并跨径" not in prompt
    assert "不得根据碰撞墩位的纵向包络推导净跨径" in prompt


def test_station_format_does_not_add_a_fourth_meter_digit() -> None:
    assert DirectFeedbackGenerator._format_station(7890.0) == "K7+890.0"


def test_revision_action_keeps_metrics_when_report_file_exists(
    monkeypatch, tmp_path
) -> None:
    report_path = tmp_path / "collision_report.json"
    report_path.write_text("[]", encoding="utf-8")
    captured = {}

    def fake_tool(**payload):
        captured.update(payload)
        generator = DirectFeedbackGenerator(
            payload["collision_report"],
            layout_result=payload["layout_result"],
        )
        return {
            "success": True,
            "revision_instruction": generator.generate_prompt(),
            "engineering_metrics": generator.to_metrics_payload(),
        }

    monkeypatch.setattr(tool_actions, "revision_instruction_tool", fake_tool)
    metrics = {
        "conflict_column_count": 2,
        "conflict_column_rate": 0.125,
        "avg_intrusion_depth_columns": 1.2,
    }
    result = tool_actions.generate_revision_instruction_action(
        {
            "collision_report_json_path": str(report_path),
            "collision_items": [_collision("1", 100.0)],
            "collision_metrics": metrics,
            "layout_result": {"设桥总览": _layout()},
            "output_dir": str(tmp_path),
        }
    )

    assert result["error"] is None
    assert captured["collision_report"]["metrics"] == metrics
    assert captured["collision_report"]["collision_items"][0]["pier_id"] == "1"
    assert result["revision_engineering_metrics"]["collision_metrics"][
        "conflict_column_rate"
    ] == 0.125
