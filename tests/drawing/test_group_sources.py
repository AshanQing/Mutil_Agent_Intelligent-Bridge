from __future__ import annotations

from copy import deepcopy

import pytest

from bridge_agents.drawing.group_sources import (
    DrawingSourceConflictError,
    DrawingSourceMissingError,
    collect_drawing_group_sources,
)


def _pier_group_result() -> dict:
    return {
        "design_groups": [
            {
                "design_group_id": "B1-U1-1",
                "bridge_id": "B1",
                "unit_id": "U1",
                "member_piers": ["P1", "P2"],
                "assumed_member_piers": [],
                "merge_fields": {
                    "盖梁尺寸": {"长度": 12000, "宽度": 2200},
                    "墩柱尺寸": {"数量": 2, "直径": 1600},
                },
                "controlling_pier_id": "P2",
                "controlling_net_height_m": 10.5,
                "member_net_heights_m": {"P1": 9.8, "P2": 10.5},
            }
        ]
    }


def _dimension_design_result() -> dict:
    return {
        "任务2_下部结构尺寸设计结果": {
            "单元原始结果": [
                {
                    "single_unit_input": {"桥梁编号": "B1", "单元编号": "U1"},
                    "dimension_result": {
                        "分组尺寸设计结果": [
                            {
                                "分组编号": "G1",
                                "包含桥墩号列表": ["P1", "P2"],
                                "盖梁尺寸": {"长度": 12000, "宽度": 2200},
                                "墩柱尺寸": {"数量": 2, "直径": 1600},
                            }
                        ]
                    },
                }
            ]
        }
    }


def _reinforcement_row(*, task_id: str = "B1-U1-G1", bar_count: int = 24) -> dict:
    return {
        "reinforcement_task": {
            "task_id": task_id,
            "桥梁编号": "B1",
            "单元编号": "U1",
            "分组编号": "G1",
            "包含桥墩号列表": ["P1", "P2"],
            "桥墩尺寸信息": {
                "盖梁几何信息": {
                    "cap_length": 12.0,
                    "cap_width": 2.2,
                    "cap_height_mid": 1.8,
                    "cap_height_end": 1.1,
                    "cantilever_length": 1.5,
                },
                "墩柱几何信息": {
                    "column_count": 2,
                    "column_diameter": 1.6,
                    "column_spacing": 6.5,
                    "column_height": 10.5,
                },
                "保护层与控制参数": {
                    "cover_x": 50,
                    "cover_y": 50,
                    "cover_z": 60,
                },
            },
            "墩柱净高信息": {
                "controlling_net_height_m": 10.5,
                "member_net_heights_m": {"P1": 9.8, "P2": 10.5},
            },
        },
        "reinforcement_result": {
            "reinforcement": {
                "pier_cap": {"longitudinal_bars": [{"dia": 32, "count": 12}]},
                "pier_column": {
                    "longitudinal_bars": [{"dia": 32, "count": bar_count}]
                },
            }
        },
        "axial_check": {"status": "computed", "check_ok": True},
        "output_files": {},
    }


def _reinforcement_design_result(*rows: dict) -> dict:
    return {
        "reinforcement_design_result": {
            "任务3_下部结构配筋设计结果": {
                "分组原始结果": list(rows),
            }
        }
    }


def _capacity_check_result(*, all_ok: bool = True) -> dict:
    return {
        "task_results": [
            {
                "task_id": "B1-U1-G1",
                "check_result": {
                    "success": True,
                    "overall_check": {"all_ok": all_ok},
                },
            }
        ]
    }


def test_collects_one_source_for_all_member_piers() -> None:
    sources = collect_drawing_group_sources(
        pier_group_result=_pier_group_result(),
        dimension_design_result=_dimension_design_result(),
        reinforcement_design_result=_reinforcement_design_result(_reinforcement_row()),
        capacity_check_result=_capacity_check_result(),
    )

    assert len(sources) == 1
    assert sources[0].design_group_id == "B1-U1-G1"
    assert sources[0].task_id == "B1-U1-G1"
    assert sources[0].member_piers == ["P1", "P2"]
    assert sources[0].dimension["盖梁尺寸"]["长度"] == 12000
    assert sources[0].reinforcement["reinforcement"]["pier_column"][
        "longitudinal_bars"
    ][0]["count"] == 24
    assert sources[0].check_status == "passed"
    assert sources[0].design_context["reinforcement_task"]["桥墩尺寸信息"][
        "墩柱几何信息"
    ]["column_height"] == 10.5
    assert sources[0].design_context["pier_group"][
        "controlling_net_height_m"
    ] == 10.5


def test_axial_not_applicable_does_not_downgrade_group() -> None:
    """柱身不适用（净高非正/无柱身）时，出图状态只看承载力结论。"""
    row = _reinforcement_row()
    row["axial_check"] = {"status": "not_applicable", "check_ok": None}

    sources = collect_drawing_group_sources(
        pier_group_result=_pier_group_result(),
        dimension_design_result=_dimension_design_result(),
        reinforcement_design_result=_reinforcement_design_result(row),
        capacity_check_result=_capacity_check_result(),
    )

    assert sources[0].check_status == "passed"

    failed_capacity = collect_drawing_group_sources(
        pier_group_result=_pier_group_result(),
        dimension_design_result=_dimension_design_result(),
        reinforcement_design_result=_reinforcement_design_result(row),
        capacity_check_result=_capacity_check_result(all_ok=False),
    )
    assert failed_capacity[0].check_status == "failed"


def test_legacy_manual_review_with_non_positive_net_height_is_not_applicable() -> None:
    """历史成果里旧版本写入的 manual_review（净高非正）不得被判成 missing 而拒绝出图。"""
    row = _reinforcement_row()
    row["axial_check"] = {"status": "manual_review", "slenderness_ratio": None}
    row["reinforcement_task"]["墩柱净高信息"] = {"controlling_net_height_m": -0.91}

    sources = collect_drawing_group_sources(
        pier_group_result=_pier_group_result(),
        dimension_design_result=_dimension_design_result(),
        reinforcement_design_result=_reinforcement_design_result(row),
        capacity_check_result=_capacity_check_result(),
    )

    assert sources[0].check_status == "passed"


def test_combined_z_pattern_resolves_to_union() -> None:
    """配筋 LLM 会用 "pattern_1+pattern_3" 表达叠加布置，解析器需按并集处理。"""
    from bridge_agents.drawing.reinforcement_adapter import (
        DrawingExpressionError,
        resolve_z_positions,
    )

    patterns = {"full": [1, 2, 3, 4], "pattern_1": [1, 3], "pattern_3": [2, 4]}

    assert resolve_z_positions(patterns, "pattern_1") == [1, 3]
    assert resolve_z_positions(patterns, "pattern_1+pattern_3") == [1, 2, 3, 4]
    with pytest.raises(DrawingExpressionError, match="未知 z_pattern"):
        resolve_z_positions(patterns, "pattern_9")


def test_missing_reinforcement_mapping_is_rejected() -> None:
    with pytest.raises(DrawingSourceMissingError, match="配筋"):
        collect_drawing_group_sources(
            pier_group_result=_pier_group_result(),
            dimension_design_result=_dimension_design_result(),
            reinforcement_design_result=_reinforcement_design_result(),
            capacity_check_result=_capacity_check_result(),
        )


def test_member_piers_follow_task_not_pier_group() -> None:
    # A 驱动：装配源以配筋任务行为主，pier_group 仅作净高上下文；
    # 即使 pier 组把边墩假定进成员，source 成员仍以任务行为准。
    pier_groups = _pier_group_result()
    pier_groups["design_groups"][0]["member_piers"] = ["P1", "P2", "P3"]
    pier_groups["design_groups"][0]["assumed_member_piers"] = ["P3"]

    sources = collect_drawing_group_sources(
        pier_group_result=pier_groups,
        dimension_design_result=_dimension_design_result(),
        reinforcement_design_result=_reinforcement_design_result(_reinforcement_row()),
        capacity_check_result=_capacity_check_result(),
    )

    assert sources[0].member_piers == ["P1", "P2"]
    assert sources[0].design_context["pier_group"]["member_net_heights_m"] == {
        "P1": 9.8,
        "P2": 10.5,
    }


def test_pier_group_split_different_from_dimension_still_assembles() -> None:
    # 复刻真实故障：pier_group 把边墩独立成组，而 dimension 把连接墩并进 G2，
    # 旧实现按成员集合匹配会抛“缺少匹配的尺寸设计结果”；任务行驱动下必须正常出两源。
    dimension_design = {
        "任务2_下部结构尺寸设计结果": {
            "单元原始结果": [
                {
                    "single_unit_input": {"桥梁编号": "B1", "单元编号": "U1"},
                    "dimension_result": {
                        "分组尺寸设计结果": [
                            {
                                "分组编号": "G1",
                                "包含桥墩号列表": ["P1", "P2"],
                                "盖梁尺寸": {"长度": 12000, "宽度": 2200},
                                "墩柱尺寸": {"数量": 2, "直径": 1600},
                            },
                            {
                                "分组编号": "G2",
                                "包含桥墩号列表": ["P3", "P4"],
                                "盖梁尺寸": {"长度": 12000, "宽度": 2200},
                                "墩柱尺寸": {"数量": 2, "直径": 1600},
                            },
                        ]
                    },
                }
            ]
        }
    }
    # pier_group：连接墩 P4 归属其他单元，这里只出现中间墩 [P1,P2] 与边墩 [P3]。
    pier_groups = _pier_group_result()
    pier_groups["design_groups"] = [
        {
            **pier_groups["design_groups"][0],
            "design_group_id": "B1-U1-1",
            "member_piers": ["P1", "P2"],
        },
        {
            **pier_groups["design_groups"][0],
            "design_group_id": "B1-U1-2",
            "member_piers": ["P3"],
        },
    ]

    def row(group_no: str, members: list[str], task_id: str) -> dict:
        row_payload = _reinforcement_row(task_id=task_id)
        task = row_payload["reinforcement_task"]
        task["分组编号"] = group_no
        task["包含桥墩号列表"] = members
        return row_payload

    rows = [
        row("G1", ["P1", "P2"], "B1-U1-G1"),
        row("G2", ["P3", "P4"], "B1-U1-G2"),
    ]
    capacity = {
        "task_results": [
            {"task_id": tid, "check_result": {"success": True, "overall_check": {"all_ok": True}}}
            for tid in ("B1-U1-G1", "B1-U1-G2")
        ]
    }

    sources = collect_drawing_group_sources(
        pier_group_result=pier_groups,
        dimension_design_result=dimension_design,
        reinforcement_design_result=_reinforcement_design_result(*rows),
        capacity_check_result=capacity,
    )

    assert len(sources) == 2
    by_group = {s.design_group_id: s for s in sources}
    assert by_group["B1-U1-G1"].member_piers == ["P1", "P2"]
    assert by_group["B1-U1-G2"].member_piers == ["P3", "P4"]


def test_existing_reinforcement_file_is_hashed(tmp_path) -> None:
    reinforcement_path = tmp_path / "reinforcement.yaml"
    reinforcement_path.write_bytes(b"bars\n")
    row = _reinforcement_row()
    row["output_files"]["reinforcement_result_yaml_path"] = str(reinforcement_path)

    sources = collect_drawing_group_sources(
        pier_group_result=_pier_group_result(),
        dimension_design_result=_dimension_design_result(),
        reinforcement_design_result=_reinforcement_design_result(row),
        capacity_check_result=_capacity_check_result(),
    )

    assert sources[0].source_paths["reinforcement_yaml"] == str(reinforcement_path)
    assert sources[0].source_hashes["reinforcement_yaml"] == (
        "96fc702edd90c38e1b9af4a8ced6839eb0bcad1d29ebbc59d3ffccb4fff788a7"
    )


def test_conflicting_reinforcement_for_same_group_is_rejected() -> None:
    conflicting = _reinforcement_row(bar_count=28)
    with pytest.raises(DrawingSourceConflictError, match="多个配筋"):
        collect_drawing_group_sources(
            pier_group_result=_pier_group_result(),
            dimension_design_result=_dimension_design_result(),
            reinforcement_design_result=_reinforcement_design_result(
                _reinforcement_row(), conflicting
            ),
            capacity_check_result=_capacity_check_result(),
        )


@pytest.mark.parametrize(
    ("capacity_all_ok", "axial_check", "expected"),
    [
        (False, {"status": "computed", "check_ok": True}, "failed"),
        (True, {"status": "computed", "check_ok": False}, "failed"),
        (True, {"status": "manual_review"}, "missing"),
    ],
)
def test_combines_cap_and_column_check_status(
    capacity_all_ok: bool,
    axial_check: dict,
    expected: str,
) -> None:
    row = _reinforcement_row()
    row["axial_check"] = deepcopy(axial_check)

    sources = collect_drawing_group_sources(
        pier_group_result=_pier_group_result(),
        dimension_design_result=_dimension_design_result(),
        reinforcement_design_result=_reinforcement_design_result(row),
        capacity_check_result=_capacity_check_result(all_ok=capacity_all_ok),
    )

    assert sources[0].check_status == expected
