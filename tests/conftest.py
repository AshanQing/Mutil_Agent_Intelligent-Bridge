from __future__ import annotations

import pytest

from bridge_agents.drawing import (
    CircleEntity,
    DimensionEntity,
    DrawingDocument,
    DrawingGroupSource,
    DrawingLayer,
    GeometryView,
    LineEntity,
    SectionGeometry,
    TextEntity,
)


@pytest.fixture(autouse=True)
def isolate_gui_form_state(tmp_path_factory, monkeypatch):
    """把界面表单状态的落盘位置改到临时目录。

    窗口的 `start_task()` 会顺手把当前表单存成 `output/.graph_v2_gui_modern_state.json`，
    那是开发者本机真实的界面状态文件：测试一旦触发保存，就会把它覆盖成 pytest 的
    tmp 路径，下次启动界面预填的是一堆已经不存在的目录。`save_form_state` 在调用时
    才取 `default_state_path()`，所以换掉这个函数就能整条改道。
    """
    target = tmp_path_factory.mktemp("gui_form_state") / "state.json"
    monkeypatch.setattr(
        "bridge_agents.gui_modern_session.default_state_path", lambda: target
    )
    return target


@pytest.fixture
def drawing_source() -> DrawingGroupSource:
    return DrawingGroupSource(
        design_group_id="B1-U1-1",
        task_id="B1-U1-G1",
        member_piers=["P1", "P2"],
        dimension={
            "分组编号": "G1",
            "盖梁尺寸": {
                "长度": 12.0,
                "宽度": 2.2,
                "中高": 1.8,
                "端高": 1.1,
                "悬臂": 1.5,
            },
            "墩柱尺寸": {"数量": 2, "直径": 1.6, "中心间距": 6.5},
        },
        reinforcement={
            "reinforcement": {
                "pier_cap": {
                    "z_patterns": {
                        "full": [1, 2, 3, 4],
                        "pattern_1": [1, 3],
                        "pattern_2": [2, 4],
                    },
                    "skeleton_definitions": {
                        "skeleton_1": {
                            "name": "主骨架",
                            "function": "顶底通长主筋 + 弯起钢筋",
                            "rebar_groups": ["N1", "N2", "N3", "N5", "N6"],
                            "z_pattern": "pattern_1",
                        },
                        "skeleton_2": {
                            "name": "加强骨架",
                            "function": "顶底通长主筋",
                            "rebar_groups": ["N1", "N2"],
                            "z_pattern": "pattern_2",
                        },
                    },
                    "skeleton_layout": {
                        "total_positions": 4,
                        "odd_positions_use": "skeleton_1",
                        "even_positions_use": "skeleton_2",
                    },
                    "longitudinal_bars": [
                        {
                            "id": "N1",
                            "category": "主筋",
                            "subtype": "顶部通长主筋",
                            "dia": 28,
                            "mirror": "longitudinal",
                            "z_pattern": "full",
                            "range_definition": {
                                "x_start_expr": "-(cap_length / 2 - cover_x)",
                                "x_end_expr": 0,
                            },
                            "layer_definition": {
                                "y_ref": "top_cover",
                                "y_offset": 0,
                            },
                        },
                        {
                            "id": "N2",
                            "category": "主筋",
                            "subtype": "底部通长主筋",
                            "dia": 28,
                            "mirror": "longitudinal",
                            "z_pattern": "full",
                            "range_definition": {
                                "x_start_expr": "-(cap_length / 2 - cover_x)",
                                "x_end_expr": 0,
                            },
                            "layer_definition": {
                                "y_ref": "bottom_cover",
                                "y_offset": 0,
                            },
                        },
                        {
                            "id": "N3",
                            "category": "弯起钢筋",
                            "subtype": "底部上弯起钢筋",
                            "dia": 28,
                            "mirror": "longitudinal",
                            "z_pattern": "full",
                            "control_definition": {
                                "key_points": {
                                    "left_bend_point": {
                                        "x_expr": "pier_centerline_x - 1000",
                                        "y_ref": "top_cover",
                                        "y_offset": "dia * 2",
                                    },
                                    "right_bend_point": {
                                        "x_expr": "pier_centerline_x + 1600",
                                        "y_ref": "top_cover",
                                        "y_offset": "dia * 2",
                                    },
                                },
                                "branch_rules": {
                                    "left_branch": {
                                        "direction": "left_down",
                                        "until": {
                                            "y_ref": "bottom_cover",
                                            "y_offset": "dia * 1",
                                        },
                                    },
                                    "right_branch": {
                                        "direction": "right_down",
                                        "until": {
                                            "y_ref": "bottom_cover",
                                            "y_offset": "dia * 1",
                                        },
                                    },
                                },
                            },
                        },
                        {
                            "id": "N5",
                            "category": "弯起钢筋",
                            "subtype": "独立斜筋",
                            "dia": 28,
                            "mirror": "longitudinal",
                            "z_pattern": "full",
                            "control_definition": {
                                "key_points": {
                                    "bend_start_point": {
                                        "x_expr": "pier_centerline_x + 950",
                                        "y_ref": "top_cover",
                                        "y_offset": "dia * 2",
                                    }
                                },
                                "branch_rules": {
                                    "main_branch": {
                                        "direction": "right_down",
                                        "until": {
                                            "y_ref": "bottom_cover",
                                            "y_offset": "dia * 2",
                                        },
                                    }
                                },
                            },
                        },
                        {
                            "id": "N6",
                            "category": "弯起钢筋",
                            "subtype": "圆弧过渡弯起钢筋",
                            "dia": 28,
                            "mirror": "longitudinal",
                            "z_pattern": "full",
                            "control_definition": {
                                "bend_angle": 45,
                                "key_points": {
                                    "arc_left_point": {
                                        "x_expr": (
                                            "pier_centerline_x - radius * "
                                            "math.tan(math.radians(bend_angle))"
                                        )
                                    },
                                    "arc_right_point": {
                                        "x_expr": (
                                            "pier_centerline_x + radius * "
                                            "math.tan(math.radians(bend_angle))"
                                        )
                                    },
                                    "right_bottom_end": {
                                        "x_expr": "pier_centerline_x + 1300",
                                        "y_ref": "bottom_cover",
                                        "y_offset": "dia",
                                    },
                                },
                                "arc_definition": {"radius": 350},
                                "branch_rules": {
                                    "left_branch": {
                                        "direction": "left_down",
                                        "until": {
                                            "y_ref": "bottom_cover",
                                            "y_offset": "dia * 1",
                                        },
                                    },
                                },
                            },
                        },
                    ],
                    "stirrups": {
                        "longitudinal_distribution": [
                            {
                                "name": "左加密区",
                                "x_from": "-(cap_length / 2 - cover_x)",
                                "x_to": 0,
                                "spacing": 100,
                            },
                            {
                                "name": "右普通区",
                                "x_from": 0,
                                "x_to": "cap_length / 2 - cover_x",
                                "spacing": 150,
                            },
                        ],
                        "distribution_overview": {
                            "x_start_expr": "-(cap_length / 2 - cover_x)",
                            "x_end_expr": "cap_length / 2 - cover_x",
                            "segment_sequence": [
                                {"name": "左加密区", "spacing": 100, "count": 59},
                                {"name": "右普通区", "spacing": 150, "count": 40},
                            ],
                        },
                        "transverse_cage_definition": {"dia": 12},
                    },
                },
                "pier_column": {
                    "longitudinal_bars": [
                        {
                            "id": "C1",
                            "category": "主筋",
                            "subtype": "墩柱纵向主筋",
                            "dia": 28,
                            "count": 16,
                            "section_definition": {
                                "arrangement": "circular_uniform",
                                "bar_centerline_diameter": (
                                    "column_diameter - 2 * concrete_cover - dia"
                                ),
                            },
                            "range_definition": {
                                "y_start": 0,
                                "y_end": "column_height + 1200",
                            },
                        }
                    ],
                    "spiral_stirrups": [
                        {
                            "id": "C2",
                            "category": "箍筋",
                            "subtype": "墩柱螺旋箍筋",
                            "dia": 10,
                            "section_definition": {
                                "hoop_diameter": (
                                    "column_diameter - 2 * concrete_cover + dia"
                                )
                            },
                            "vertical_distribution": [
                                {
                                    "name": "柱底加密区",
                                    "y_from": 0,
                                    "y_to": 1000,
                                    "pitch": 100,
                                },
                                {
                                    "name": "柱身普通区",
                                    "y_from": 1000,
                                    "y_to": "column_height",
                                    "pitch": 200,
                                },
                            ],
                        }
                    ],
                    "ordinary_hoops": [],
                    "strengthening_bars": [],
                },
            }
        },
        check_status="passed",
        design_context={
            "pier_group": {
                "controlling_pier_id": "P2",
                "controlling_net_height_m": 10.5,
                "member_net_heights_m": {"P1": 9.8, "P2": 10.5},
            },
            "reinforcement_task": {
                "task_id": "B1-U1-G1",
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
                        "concrete_cover": 50,
                    },
                },
            },
        },
    )


@pytest.fixture
def simple_drawing_document() -> DrawingDocument:
    return DrawingDocument(
        design_group_id="B1-U1-G1",
        member_piers=["P1", "P2"],
        layers=[
            DrawingLayer(name="S-BEAM", line_type="DASHED"),
            DrawingLayer(name="R-MAIN", color_rgb=(255, 0, 0)),
            DrawingLayer(name="A-DIMS", color_rgb=(0, 255, 0)),
            DrawingLayer(name="A-TEXT", color_rgb=(255, 255, 0)),
        ],
        views=[
            GeometryView(
                view_id="cap-elevation",
                sections=[
                    SectionGeometry(
                        section_id="CAP-ELEVATION",
                        entities=[
                            LineEntity(
                                entity_id="cap-line-001",
                                layer="S-BEAM",
                                start=(0, 0),
                                end=(1000, 0),
                            ),
                            CircleEntity(
                                entity_id="cap-circle-001",
                                layer="R-MAIN",
                                center=(500, 300),
                                radius=20,
                            ),
                            TextEntity(
                                entity_id="cap-text-001",
                                layer="A-TEXT",
                                insertion=(0, 500),
                                text="盖梁\n配筋",
                                height=100,
                            ),
                            DimensionEntity(
                                entity_id="cap-dimension-001",
                                layer="A-DIMS",
                                start=(0, 0),
                                end=(1000, 0),
                                offset=-200,
                            ),
                        ],
                    )
                ],
            ),
            GeometryView(
                view_id="column-section",
                sections=[
                    SectionGeometry(
                        section_id="COLUMN-SECTION",
                        entities=[
                            CircleEntity(
                                entity_id="column-circle-001",
                                layer="S-BEAM",
                                center=(0, 0),
                                radius=400,
                            )
                        ],
                    )
                ],
            ),
        ],
    )


@pytest.fixture
def drawing_pipeline_state(tmp_path, drawing_source) -> dict:
    source = drawing_source
    task = {
        **source.design_context["reinforcement_task"],
        "桥梁编号": "B1",
        "单元编号": "U1",
        "分组编号": "G1",
    }
    return {
        "output_dir": str(tmp_path),
        "pier_group_result": {
            "design_groups": [{
                **source.design_context["pier_group"],
                "design_group_id": source.design_group_id,
                "bridge_id": "B1",
                "unit_id": "U1",
                "dimension_group_id": "G1",
                "member_piers": source.member_piers,
            }]
        },
        "dimension_design_result": {"任务2_下部结构尺寸设计结果": {
            "单元原始结果": [{
                "single_unit_input": {"桥梁编号": "B1", "单元编号": "U1"},
                "dimension_result": {"分组尺寸设计结果": [{
                    **source.dimension,
                    "包含桥墩号列表": source.member_piers,
                }]},
            }]
        }},
        "reinforcement_design_result": {"任务3_下部结构配筋设计结果": {
            "分组原始结果": [{
                "task_id": source.task_id,
                "reinforcement_task": task,
                "reinforcement_result": source.reinforcement,
                "axial_check": {"status": "computed", "check_ok": True},
            }]
        }},
        "capacity_check_result": {
            "task_results": [{
                "task_id": source.task_id,
                "result": {"overall_check": {"all_ok": True}},
            }]
        },
        "check_result": {
            "task_results": [{
                "task_id": source.task_id,
                "result": {"overall_check": {"all_ok": True}},
            }]
        },
        "accepted_risks": [],
        "invalidated_artifacts": [],
    }
