from __future__ import annotations

import math

import pytest

from bridge_agents.pier_group import (
    build_side_route_map,
    build_design_groups,
    cap_height_at_column_x,
    cap_height_at_pier,
    collect_vertical_profiles,
    compute_net_height,
    compute_pier_groups,
    compute_layout_pier_height,
    extract_pier_rows,
    interpolate_elevation,
    parse_pier_no,
    parse_station,
    resolve_design_bridge_id,
    superstructure_height_m,
    verify_pier_height,
)
from bridge_agents.tool_actions import compute_pier_groups_action


class TestParseStation:
    def test_k_station(self):
        assert parse_station("K12+373.500") == pytest.approx(12373.5)

    def test_z_station(self):
        assert parse_station("Z20+531.3") == pytest.approx(20531.3)

    def test_plain_number(self):
        assert parse_station("12373.5") == pytest.approx(12373.5)

    def test_unparseable_returns_none(self):
        assert parse_station("") is None
        assert parse_station("abc") is None


class TestInterpolateElevation:
    def test_hits_exact_point(self):
        points = [{"桩号": 100.0, "高程": 10.0}, {"桩号": 200.0, "高程": 20.0}]
        assert interpolate_elevation(points, 100.0) == pytest.approx(10.0)

    def test_linear_interpolation(self):
        points = [{"桩号": 100.0, "高程": 10.0}, {"桩号": 200.0, "高程": 20.0}]
        assert interpolate_elevation(points, 150.0) == pytest.approx(15.0)

    def test_unsorted_input_is_sorted(self):
        points = [{"桩号": 200.0, "高程": 20.0}, {"桩号": 100.0, "高程": 10.0}]
        assert interpolate_elevation(points, 150.0) == pytest.approx(15.0)

    def test_out_of_range_returns_none(self):
        points = [{"桩号": 100.0, "高程": 10.0}, {"桩号": 200.0, "高程": 20.0}]
        assert interpolate_elevation(points, 50.0) is None
        assert interpolate_elevation(points, 250.0) is None


class TestSuperstructureHeight:
    def test_t_beam_40m(self):
        assert superstructure_height_m("预应力砼连续T梁（先简支后连续）", 40.0) == pytest.approx(3.0)

    def test_t_beam_30m(self):
        assert superstructure_height_m("预应力砼连续T梁", 30.0) == pytest.approx(2.5)

    def test_t_beam_20m(self):
        assert superstructure_height_m("预应力砼T梁", 20.0) == pytest.approx(1.8)

    def test_box_girder(self):
        assert superstructure_height_m("预应力砼小箱梁", 30.0) == pytest.approx(2.2)

    def test_interpolated_mid_span(self):
        assert superstructure_height_m("预应力砼连续T梁", 35.0) == pytest.approx(2.75)


class TestCapHeightAtColumnX:
    CAP = {"h_mid": 1.8, "h_end": 0.95, "cap_length": 24.0, "cantilever": 1.3}

    def test_middle_constant_section_uses_mid(self):
        assert cap_height_at_column_x(1.8, 0.95, 24.0, 1.3, 0.0) == pytest.approx(1.8)
        assert cap_height_at_column_x(1.8, 0.95, 24.0, 1.3, 9.0) == pytest.approx(1.8)

    def test_cantilever_end_uses_end(self):
        assert cap_height_at_column_x(1.8, 0.95, 24.0, 1.3, 12.0) == pytest.approx(0.95)
        assert cap_height_at_column_x(1.8, 0.95, 24.0, 1.3, -12.0) == pytest.approx(0.95)

    def test_cantilever_midpoint_interpolates(self):
        inner_half = 12.0 - 1.3
        x = inner_half + 0.65
        expected = 0.95 + (1.8 - 0.95) * 0.5
        assert cap_height_at_column_x(1.8, 0.95, 24.0, 1.3, x) == pytest.approx(expected)


class TestComputeLayoutPierHeight:
    def test_positive_difference(self):
        assert compute_layout_pier_height(13.52, 0.0, 3.0) == pytest.approx(10.52)

    def test_absolute_value(self):
        assert compute_layout_pier_height(0.0, 13.52, 3.0) == pytest.approx(10.52)


def _split_layout_result() -> dict:
    return {
        "设桥总览": {},
        "桥位列表": [
            {
                "桥位编号": 1,
                "统一布跨方案": None,
                "分幅布跨方案列表": [
                    {
                        "幅别": "右幅",
                        "方案编号": "R1",
                        "设桥信息": {"桥型": "预应力砼连续T梁（先简支后连续）"},
                        "墩位与墩高": [
                            {
                                "墩号": "0 (桥台)",
                                "桩号": "K6+500.000",
                                "墩位类型": "桥台",
                                "本幅墩高(m)": 0.0,
                                "墩位校核说明": "桥台",
                            },
                            {
                                "墩号": "1 (桥墩)",
                                "桩号": "K6+520.000",
                                "墩位类型": "桥墩",
                                "本幅墩高(m)": 10.5,
                                "墩位校核说明": "高差13.52m，上部结构取3.0m",
                            },
                        ],
                    },
                    {
                        "幅别": "左幅",
                        "方案编号": "L1",
                        "设桥信息": {"桥型": "预应力砼连续T梁"},
                        "墩位与墩高": [
                            {
                                "墩号": "0 (桥台)",
                                "桩号": "K6+510.000",
                                "墩位类型": "桥台",
                                "本幅墩高(m)": 0.0,
                            },
                        ],
                    },
                ],
            }
        ],
    }


class TestParsePierNo:
    def test_extracts_number_from_label(self):
        assert parse_pier_no("0 (桥台)") == "0"
        assert parse_pier_no("12 (桥墩)") == "12"

    def test_plain_number(self):
        assert parse_pier_no("3") == "3"


class TestExtractPierRows:
    def test_extracts_split_sides(self):
        rows = extract_pier_rows(_split_layout_result())
        assert len(rows) == 3
        assert rows[0]["pier_no"] == "0"
        assert rows[0]["side"] == "右幅"
        assert rows[0]["station"] == pytest.approx(6500.0)
        assert rows[1]["layout_height_m"] == pytest.approx(10.5)
        assert rows[1]["bridge_type"] == "预应力砼连续T梁（先简支后连续）"

    def test_unwraps_nested_payload(self):
        nested = {"output": _split_layout_result()}
        rows = extract_pier_rows(nested)
        assert len(rows) == 3


class TestVerifyPierHeight:
    PROFILE = {
        "设计线高程序列": [
            {"桩号": 100.0, "高程": 13.52},
            {"桩号": 200.0, "高程": 13.52},
        ],
        "地形线高程序列": [
            {"桩号": 100.0, "高程": 0.0},
            {"桩号": 200.0, "高程": 0.0},
        ],
    }

    def test_matches_when_within_tolerance(self):
        row = {"pier_no": "1", "station": 150.0, "layout_height_m": 10.52, "side": "右幅", "bridge_id": "1"}
        result = verify_pier_height(row, self.PROFILE, super_height=3.0)
        assert result["status"] == "verified"
        assert result["computed_layout_height_m"] == pytest.approx(10.52)
        assert result["height_mismatch"] is False

    def test_flags_mismatch_beyond_tolerance(self):
        row = {"pier_no": "1", "station": 150.0, "layout_height_m": 12.0, "side": "右幅", "bridge_id": "1"}
        result = verify_pier_height(row, self.PROFILE, super_height=3.0, tolerance=0.5)
        assert result["status"] == "verified"
        assert result["height_mismatch"] is True

    def test_reports_missing_station(self):
        row = {"pier_no": "1", "station": None, "layout_height_m": 10.5, "side": "右幅", "bridge_id": "1"}
        result = verify_pier_height(row, self.PROFILE, super_height=3.0)
        assert result["status"] == "station_missing"

    def test_reports_elevation_out_of_range(self):
        row = {"pier_no": "1", "station": 999.0, "layout_height_m": 10.5, "side": "右幅", "bridge_id": "1"}
        result = verify_pier_height(row, self.PROFILE, super_height=3.0)
        assert result["status"] == "elevation_missing"


def _cap(cap_length=24.0, cap_width=2.0, h_mid=1.8, h_end=0.95, cantilever=1.3):
    return {"长度": cap_length, "宽度": cap_width, "中高": h_mid, "端高": h_end, "悬臂": cantilever}


def _column(col_type="柱式墩", count=3, diameter=1.8, spacing=9.0):
    return {"墩柱类型": col_type, "墩柱尺寸": {"数量": count, "直径": diameter, "中心间距": spacing}}


def _group(gid, pier_nos, role, cap, col):
    return {
        "分组编号": gid,
        "包含桥墩号列表": pier_nos,
        "墩位角色": role,
        **col,
        "是否双排支座": True,
        "T梁布置": {"T梁排数": 2, "前跨": {"T梁数量": 11, "T梁间距": 2.25}},
        "盖梁尺寸": cap,
    }


def _unit(unit_id, spans, piers, groups, mapping):
    return {
        "单元编号": unit_id,
        "本联信息": {"跨径组合": spans, "跨数": len(spans.split("+"))},
        "桥面宽度信息": {"宽度类型": "标准宽度", "起点宽度": 25.0, "终点宽度": 25.0},
        "桥墩信息": piers,
        "分组尺寸设计结果": groups,
        "桥墩尺寸映射关系": mapping,
    }


def _pier(no, role, connection=False):
    return {"原始墩号": no, "桩号": f"K1+{int(no) * 10:03d}.000", "墩位角色": role, "是否连接墩": connection}


def _design_units() -> dict:
    return {
        "任务1_设计单元提取结果": {
            "桥梁列表": [
                {
                    "桥梁编号": "1",
                    "桥型": "装配式预应力砼T梁",
                    "设计单元列表": [
                        _unit(
                            "1-1",
                            "40+40",
                            [_pier("0", "桥台"), _pier("1", "中间墩"), _pier("2", "中间墩"), _pier("3", "边墩", True)],
                            [],
                            [],
                        ),
                        _unit(
                            "1-2",
                            "40+40",
                            [_pier("3", "边墩", True), _pier("4", "中间墩"), _pier("5", "桥台")],
                            [],
                            [],
                        ),
                    ],
                }
            ]
        }
    }


def _dimension_result() -> dict:
    cap_mid = _cap()
    cap_abutment = _cap(h_mid=1.6, h_end=1.6, cantilever=0.0)
    col_mid = _column()
    col_edge = _column(diameter=2.0)
    col_abutment = _column(col_type="肋板式桥台", count=0, diameter=0.0, spacing=0.0)
    return {
        "任务2_下部结构尺寸设计结果": {
            "桥梁列表": [
                {
                    "桥梁编号": "1",
                    "桥型": "装配式预应力砼T梁",
                    "单元尺寸设计结果": [
                        {
                            "单元编号": "1-1",
                            "分组尺寸设计结果": [
                                _group("G1", ["0"], "桥台", cap_abutment, col_abutment),
                                _group("G2", ["1", "2"], "中间墩", cap_mid, col_mid),
                                _group("G3", ["3"], "边墩", cap_mid, col_edge),
                            ],
                            "桥墩尺寸映射关系": [
                                {"桥墩号": "0", "所属分组编号": "G1"},
                                {"桥墩号": "1", "所属分组编号": "G2"},
                                {"桥墩号": "2", "所属分组编号": "G2"},
                                {"桥墩号": "3", "所属分组编号": "G3"},
                            ],
                        },
                        {
                            "单元编号": "1-2",
                            "分组尺寸设计结果": [
                                _group("G4", ["3"], "边墩", cap_mid, col_edge),
                                _group("G5", ["4"], "中间墩", cap_mid, col_mid),
                                _group("G6", ["5"], "桥台", cap_abutment, col_abutment),
                            ],
                            "桥墩尺寸映射关系": [
                                {"桥墩号": "3", "所属分组编号": "G4"},
                                {"桥墩号": "4", "所属分组编号": "G5"},
                                {"桥墩号": "5", "所属分组编号": "G6"},
                            ],
                        },
                    ],
                }
            ]
        }
    }


class TestBuildDesignGroups:
    def test_groups_within_unit_and_does_not_merge_across_units(self):
        result = build_design_groups(_design_units(), _dimension_result())
        groups = result["design_groups"]
        assert len(groups) == 6

    def test_same_role_and_dimension_merge(self):
        result = build_design_groups(_design_units(), _dimension_result())
        groups = result["design_groups"]
        mid_group_1_1 = next(g for g in groups if g["unit_id"] == "1-1" and g["pier_role"] == "中间墩")
        assert sorted(mid_group_1_1["member_piers"]) == ["1", "2"]

    def test_different_dimension_splits(self):
        result = build_design_groups(_design_units(), _dimension_result())
        groups = result["design_groups"]
        roles_1_1 = sorted(g["pier_role"] for g in groups if g["unit_id"] == "1-1")
        assert roles_1_1 == ["中间墩", "桥台", "边墩"]

    def test_connection_pier_with_dimension_in_both_units_belongs_to_both(self):
        # 连接墩（如 2-2 单元的墩1）若本联尺寸设计明确为其分配了分组，
        # 必须在本联独立成组，否则该分组的净高/配筋没有数据来源。
        result = build_design_groups(_design_units(), _dimension_result())
        groups = result["design_groups"]
        pier3_units = sorted(g["unit_id"] for g in groups if "3" in g["member_piers"])
        assert pier3_units == ["1-1", "1-2"]

    def test_unmapped_edge_pier_falls_back_to_adjacent_group(self):
        units = _design_units()
        dims = _dimension_result()
        # 移除单元 1-1 里边墩 3 的尺寸分组映射，模拟设计单元提取漏掉边墩
        unit = dims["任务2_下部结构尺寸设计结果"]["桥梁列表"][0]["单元尺寸设计结果"][0]
        unit["分组尺寸设计结果"] = [
            g for g in unit["分组尺寸设计结果"] if g["分组编号"] != "G3"
        ]
        unit["桥墩尺寸映射关系"] = [
            m for m in unit["桥墩尺寸映射关系"] if m["桥墩号"] != "3"
        ]
        result = build_design_groups(units, dims)
        groups = result["design_groups"]
        edge_found = any(
            "3" in g["member_piers"] and g["unit_id"] == "1-1"
            for g in groups
        )
        assert edge_found is True
        assumed = [
            g
            for g in groups
            if "3" in (g.get("assumed_member_piers") or [])
        ]
        assert len(assumed) == 1
        assert assumed[0]["pier_role"] == "中间墩"


    def test_extract_tool_envelope_layout_yields_same_groups(self):
        # extract_design_units_tool 返回 {"success": ..., "design_units_result": <任务1 wrapper>} 信封，
        # 而 pier_group 必须与 dimension 工具一样先解信封，否则归并出 0 分组（无墩柱净高）。
        envelope = {
            "success": True,
            "design_units_result": _design_units(),
            "single_unit_inputs": [],
            "prompt_id": "tasks.design_unit_extraction.v1",
        }
        result = build_design_groups(envelope, _dimension_result())
        groups = result["design_groups"]
        assert len(groups) == 6
        assert sorted(g["unit_id"] for g in groups) == ["1-1", "1-1", "1-1", "1-2", "1-2", "1-2"]
        mid_group_1_1 = next(g for g in groups if g["unit_id"] == "1-1" and g["pier_role"] == "中间墩")
        assert sorted(mid_group_1_1["member_piers"]) == ["1", "2"]

    def test_bare_design_unit_root_without_task1_wrapper(self):
        # 极端情况：任务1内容体直接作为顶层（无信封、无任务1包装）。
        bare = _design_units()["任务1_设计单元提取结果"]
        result = build_design_groups(bare, _dimension_result())
        assert len(result["design_groups"]) == 6


class TestCapHeightAtPier:
    def test_multi_column_inside_middle_uses_mid(self):
        dim = {"盖梁尺寸": _cap(), "墩柱尺寸": {"数量": 3, "直径": 1.8, "中心间距": 9.0}}
        assert cap_height_at_pier(dim) == pytest.approx(1.8)

    def test_no_column_returns_zero(self):
        dim = {"盖梁尺寸": _cap(), "墩柱尺寸": {"数量": 0, "直径": 0.0, "中心间距": 0.0}}
        assert cap_height_at_pier(dim) == pytest.approx(0.0)


class TestComputeNetHeight:
    def test_subtracts_cap_height_and_offset(self):
        assert compute_net_height(12.0, 1.8, 0.0) == pytest.approx(10.2)
        assert compute_net_height(12.0, 1.8, 0.5) == pytest.approx(9.7)


def _pier_group_layout_result() -> dict:
    piers = []
    heights = {"0": 0.0, "1": 12.0, "2": 12.0, "3": 12.0, "4": 12.0, "5": 0.0}
    for no in ["0", "1", "2", "3", "4", "5"]:
        piers.append({
            "墩号": f"{no} (桥台)" if no in ("0", "5") else f"{no} (桥墩)",
            "桩号": f"K1+{int(no) * 10:03d}.000",
            "墩位类型": "桥台" if no in ("0", "5") else "桥墩",
            "本幅墩高(m)": heights[no],
        })
    return {
        "桥位列表": [
            {
                "桥位编号": 1,
                "统一布跨方案": None,
                "分幅布跨方案列表": [
                    {
                        "幅别": "右幅",
                        "方案编号": "R1",
                        "设桥信息": {"桥型": "装配式预应力砼T梁"},
                        "墩位与墩高": piers,
                    }
                ],
            }
        ]
    }


def _pier_group_profile() -> dict:
    return {
        "设计线高程序列": [
            {"桩号": 1000.0, "高程": 15.0},
            {"桩号": 1060.0, "高程": 15.0},
        ],
        "地形线高程序列": [
            {"桩号": 1000.0, "高程": 0.0},
            {"桩号": 1060.0, "高程": 0.0},
        ],
    }


class TestComputePierGroups:
    def test_controlling_net_height_and_members(self):
        result = compute_pier_groups(
            _pier_group_layout_result(),
            _design_units(),
            _dimension_result(),
            {"右幅": _pier_group_profile()},
            super_height=3.0,
        )
        groups = result["design_groups"]
        assert len(groups) == 6
        mid_group = next(g for g in groups if g["unit_id"] == "1-1" and g["pier_role"] == "中间墩")
        assert mid_group["member_net_heights_m"] == {"1": pytest.approx(10.2), "2": pytest.approx(10.2)}
        assert mid_group["controlling_net_height_m"] == pytest.approx(10.2)
        assert mid_group["controlling_pier_id"] == "1"

    def test_height_audit_covers_every_pier_once(self):
        result = compute_pier_groups(
            _pier_group_layout_result(),
            _design_units(),
            _dimension_result(),
            {"右幅": _pier_group_profile()},
            super_height=3.0,
        )
        # 墩3 是连接墩，在 1-1（G3）与 1-2（G4）两个设计组各审计一次。
        audit_piers = sorted(a["pier_no"] for a in result["height_audit"])
        assert audit_piers == ["0", "1", "2", "3", "3", "4", "5"]

    def test_abutment_has_zero_net_height(self):
        result = compute_pier_groups(
            _pier_group_layout_result(),
            _design_units(),
            _dimension_result(),
            {"右幅": _pier_group_profile()},
            super_height=3.0,
        )
        abutment = next(g for g in result["design_groups"] if g["pier_role"] == "桥台")
        assert abutment["controlling_net_height_m"] == pytest.approx(0.0)


class TestCollectVerticalProfiles:
    def test_multi_route_map(self):
        cropped = {
            "线路数据": {
                "K": {"纵断面结构": {"设计线高程序列": [{"桩号": 1.0, "高程": 10.0}], "地形线高程序列": [{"桩号": 1.0, "高程": 0.0}]}},
                "Z": {"纵断面结构": {"设计线高程序列": [{"桩号": 1.0, "高程": 9.0}], "地形线高程序列": [{"桩号": 1.0, "高程": 0.0}]}},
            }
        }
        profiles = collect_vertical_profiles(cropped)
        assert set(profiles) == {"K", "Z"}
        assert profiles["K"]["设计线高程序列"][0]["高程"] == pytest.approx(10.0)

    def test_single_route_falls_back_to_k(self):
        cropped = {
            "纵断面结构": {
                "设计线高程序列": [{"桩号": 1.0, "高程": 10.0}],
                "地形线高程序列": [{"桩号": 1.0, "高程": 0.0}],
            }
        }
        profiles = collect_vertical_profiles(cropped)
        assert set(profiles) == {"K"}


class TestBuildSideRouteMap:
    def test_uses_explicit_line_field(self):
        layout = {
            "桥位列表": [
                {
                    "桥位编号": 1,
                    "分幅布跨方案列表": [
                        {"幅别": "左幅", "线路": "Z线"},
                        {"幅别": "右幅", "线路": "K线"},
                    ],
                }
            ]
        }
        profiles = {"K": {}, "Z": {}}
        route_map = build_side_route_map(layout, profiles)
        assert route_map == {"左幅": "Z", "右幅": "K"}

    def test_defaults_when_no_line_field(self):
        layout = {
            "桥位列表": [
                {
                    "桥位编号": 1,
                    "分幅布跨方案列表": [
                        {"幅别": "左幅"},
                        {"幅别": "右幅"},
                    ],
                }
            ]
        }
        profiles = {"K": {}, "Z": {}}
        route_map = build_side_route_map(layout, profiles)
        assert route_map == {"左幅": "Z", "右幅": "K"}


class TestComputePierGroupsAction:
    def test_writes_pier_group_result_to_state(self):
        state = {
            "layout_result": _pier_group_layout_result(),
            "design_units": _design_units(),
            "dimension_design_result": _dimension_result(),
            "cropped_data": {
                "线路数据": {
                    "K": {"纵断面结构": _pier_group_profile()},
                }
            },
        }
        update = compute_pier_groups_action(state)
        assert update["error"] is None
        result = update["pier_group_result"]
        assert len(result["design_groups"]) == 6
        assert len(result["height_audit"]) == 7

    def test_returns_error_without_required_state(self):
        update = compute_pier_groups_action({})
        assert update["error"] is not None


class TestResolveDesignBridgeId:
    def test_split_side_maps_to_suffixed_bridge(self):
        assert resolve_design_bridge_id("1", "右幅", {"1-R", "1-L"}) == "1-R"
        assert resolve_design_bridge_id("1", "左幅", {"1-R", "1-L"}) == "1-L"

    def test_monolithic_falls_back_to_plain_id(self):
        assert resolve_design_bridge_id("1", "", {"1"}) == "1"

    def test_unknown_side_falls_back(self):
        assert resolve_design_bridge_id("1", "右幅", {"1"}) == "1"
