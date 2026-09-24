from __future__ import annotations

from pathlib import Path

from bridge_agents.agent import final_output_node as compatible_final_output
from bridge_agents.drawing.package import build_final_deliverables
from bridge_agents.graph_v2 import final_output_node as graph_v2_final_output


def test_shared_final_deliverables_generates_grouped_drawings(
    drawing_pipeline_state,
) -> None:
    update = build_final_deliverables(drawing_pipeline_state)

    assert update["drawing_package_result"]["success"] is True
    assert Path(update["drawing_index_path"]).is_file()
    # 分页链路：每设计组两张图纸（盖梁配筋详图/墩柱配筋详图）
    assert len(update["cad_script_paths"]) == 2
    assert len(update["drawing_preview_paths"]) == 2


def test_both_final_output_nodes_include_shared_drawing_paths(
    drawing_pipeline_state,
) -> None:
    state = {
        **drawing_pipeline_state,
        "task_status": "modeling_check_passed",
        "coordinator_decision": {"task_complete": True},
    }

    compatible = compatible_final_output(state)
    graph_v2 = graph_v2_final_output(state)

    assert len(compatible["cad_script_paths"]) == 2
    assert len(graph_v2["cad_script_paths"]) == 2
    assert compatible["final_summary"]["drawing_index_path"]
    assert graph_v2["final_summary"]["drawing_index_path"]


def test_graph_v2_final_output_fails_when_drawing_group_cannot_be_written(
    drawing_pipeline_state,
) -> None:
    state = dict(drawing_pipeline_state)
    state["pier_group_result"] = {
        "design_groups": [{
            **drawing_pipeline_state["pier_group_result"]["design_groups"][0],
            "design_group_id": "../unsafe",
        }]
    }

    result = graph_v2_final_output(state)

    assert result["task_status"] == "failed"
    assert "绘图" in result["message"]


def test_non_drawing_final_clears_inapplicable_drawing_invalidation() -> None:
    update = build_final_deliverables({
        "invalidated_artifacts": ["drawing_package", "final_deliverables"],
        "minimal_rework_path": ["drawing_package", "final_deliverables"],
    })

    assert update["invalidated_artifacts"] == []
    assert update["minimal_rework_path"] == []
