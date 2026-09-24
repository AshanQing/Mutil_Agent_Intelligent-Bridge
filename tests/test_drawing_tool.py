from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.reinforcement_drawing_tool import reinforcement_drawing_tool


def test_drawing_tool_collects_group_sources_and_returns_paths(
    tmp_path: Path,
    drawing_source,
) -> None:
    source = drawing_source
    reinforcement_task = {
        **source.design_context["reinforcement_task"],
        "桥梁编号": "B1",
        "单元编号": "U1",
        "分组编号": "G1",
    }
    result = reinforcement_drawing_tool(
        pier_group_result={
            "design_groups": [source.design_context["pier_group"] | {
                "design_group_id": source.design_group_id,
                "bridge_id": "B1",
                "unit_id": "U1",
                "dimension_group_id": "G1",
                "member_piers": source.member_piers,
            }]
        },
        dimension_design_result={"任务2_下部结构尺寸设计结果": {
            "单元原始结果": [{
                "single_unit_input": {"桥梁编号": "B1", "单元编号": "U1"},
                "dimension_result": {"分组尺寸设计结果": [{
                    **source.dimension,
                    "包含桥墩号列表": source.member_piers,
                }]},
            }]
        }},
        reinforcement_design_result={"任务3_下部结构配筋设计结果": {
            "分组原始结果": [{
                "task_id": source.task_id,
                "reinforcement_task": reinforcement_task,
                "reinforcement_result": source.reinforcement,
                "axial_check": {"status": "computed", "check_ok": True},
            }]
        }},
        capacity_check_result={
            "task_results": [{
                "task_id": source.task_id,
                "result": {"overall_check": {"all_ok": True}},
            }]
        },
        output_dir=tmp_path,
    )

    assert result["success"] is True
    assert Path(result["drawing_index_path"]).is_file()
    # 分页链路：每设计组两张图纸（盖梁配筋详图/墩柱配筋详图）
    assert len(result["cad_script_paths"]) == 2
    assert len(result["drawing_preview_paths"]) == 2


def test_export_cli_help_exits_successfully() -> None:
    completed = subprocess.run(
        [sys.executable, "run_export_drawings.py", "--help"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0
    assert "--allow-unverified" in completed.stdout
