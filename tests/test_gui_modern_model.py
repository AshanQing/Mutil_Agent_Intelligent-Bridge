from __future__ import annotations

from bridge_agents.gui_modern_model import (
    build_dashboard_model,
    build_demo_dashboard_model,
)


def test_dashboard_model_maps_running_structural_design_snapshot() -> None:
    progress = {
        "running": True,
        "active_stage": "structural_design",
        "stage_states": {
            "initial_design": "completed",
            "layout_revision": "completed",
            "structural_design": "running",
            "modeling_check": "pending",
            "final_output": "pending",
        },
        "current_scope": "pier_group_03",
        "completed_groups": 4,
        "total_groups": 9,
        "recent_log": "pier_group_03 配筋完成",
        "latest_status": "completed",
    }

    model = build_dashboard_model(
        progress,
        drawings=[{"title": "桥墩一般构造图"}],
        project_name="示例项目9.8",
        route_range="K12+400 — K14+860",
    )

    assert model.project_name == "示例项目9.8"
    assert model.run_status == "运行中"
    assert model.active_agent == "结构设计智能体"
    assert model.progress_percent == 50
    assert model.stages[2].state == "running"
    assert model.metrics[1].value == "4 / 9"
    assert model.metrics[3].value == "1"
    assert model.timeline[0].title == "配筋设计进展"


def test_dashboard_model_surfaces_human_review_as_primary_action() -> None:
    model = build_dashboard_model(
        {
            "running": False,
            "active_stage": "modeling_check",
            "stage_states": {
                "initial_design": "completed",
                "layout_revision": "completed",
                "structural_design": "completed",
                "modeling_check": "review",
                "final_output": "pending",
            },
            "completed_groups": 8,
            "total_groups": 9,
        },
        review={
            "review_type": "modeling_check_review",
            "message": "1 个设计组连续验算失败，请决定重试或接受风险。",
            "failed_task_ids": ["pier_group_07"],
            "allowed_actions": ["continue_modeling_revision", "accept_check_and_finish"],
        },
    )

    assert model.run_status == "等待人工判断"
    assert model.progress_percent == 70
    assert model.review.required is True
    assert model.review.failed_ids == ("pier_group_07",)
    assert model.review.actions == ("重新设计并复验", "接受风险并结束")


def test_demo_model_is_explicitly_marked_as_demo_data() -> None:
    model = build_demo_dashboard_model()

    assert model.is_demo is True
    assert model.data_badge == "演示数据"
    assert model.project_name == "示例项目 9.8"


def test_dashboard_model_flags_delivery_before_modeling_check() -> None:
    model = build_dashboard_model(
        {
            "running": False,
            "active_stage": None,
            "stage_states": {
                "initial_design": "completed",
                "layout_revision": "completed",
                "structural_design": "completed",
                "modeling_check": "pending",
                "final_output": "completed",
            },
        }
    )

    assert model.run_status == "等待人工判断"
    assert model.stages[3].state == "review"
    assert model.review.required is True
    assert model.review.title == "阶段依赖不完整"
    assert "建模验算" in model.review.message
