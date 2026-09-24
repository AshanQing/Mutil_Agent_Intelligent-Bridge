from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from bridge_agents.gui_modern_model import build_dashboard_model, build_demo_dashboard_model
from bridge_agents.gui_modern_runtime import run_operation
from bridge_agents.gui_modern_session import ModernGuiForm
from bridge_agents.gui_modern_window import ModernBridgeWindow


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


class FakeController:
    def __init__(self) -> None:
        self.started = None
        self.resumed = None
        self.asked = None

    def start(self, config, *, base_config_path):
        self.started = (config, base_config_path)
        return {
            "task_status": "manual_review_required",
            "thread_id": config.thread_id,
            "config_snapshot_path": "D:/output/run_configs/test.yaml",
            "active_agent": "ModelingCheckAgent",
            "__interrupt__": [
                {
                    "value": {
                        "review_type": "modeling_check_review",
                        "message": "验算结果需要复核",
                        "available_actions": ["continue_modeling_revision", "accept_check_and_finish"],
                    }
                }
            ],
        }

    def resume(self, **kwargs):
        self.resumed = kwargs
        return {"task_status": "completed", "thread_id": kwargs["thread_id"]}

    def answer(self, question, **kwargs):
        self.asked = (question, kwargs)
        return {"answer": "盖梁验算满足要求。", "cited_artifact_ids": ["CAP-01"]}

    def assess(self, **kwargs):
        return {"overall_conclusion": "成果需要补充验算。"}


def sync_dispatch(name, operation, callback):
    callback(run_operation(name, operation))


def _form(tmp_path) -> ModernGuiForm:
    return ModernGuiForm(
        base_config_path="config/settings.yaml",
        user_request="完成全流程桥梁设计",
        data_path=str(tmp_path / "route"),
        drawing_path=str(tmp_path / "input.dxf"),
        output_dir=str(tmp_path / "output"),
        file_prefix="K",
        thread_id="bridge-test-001",
        external_authorized=True,
    )


def test_window_exposes_complete_workbench_navigation(app, tmp_path) -> None:
    window = ModernBridgeWindow(
        build_demo_dashboard_model(),
        controller=FakeController(),
        initial_form=_form(tmp_path),
        operation_dispatcher=sync_dispatch,
    )

    assert window.pages.count() == 6
    assert [button.text().split()[-1] for button in window.nav_buttons] == [
        "工程总览",
        "任务启动",
        "人工复核",
        "成果目录",
        "成果问答",
        "运行信息",
    ]


def test_start_then_review_resume_uses_controller(app, tmp_path) -> None:
    controller = FakeController()
    saved_forms = []
    reinforcement = tmp_path / "output" / "structural_design" / "reinforcement_design"
    for group in ("G1", "G2"):
        group_dir = reinforcement / group
        group_dir.mkdir(parents=True)
        (group_dir / f"reinforcement_result_{group}.json").write_text("{}", encoding="utf-8")
    window = ModernBridgeWindow(
        build_demo_dashboard_model(),
        controller=controller,
        initial_form=_form(tmp_path),
        operation_dispatcher=sync_dispatch,
        form_state_saver=saved_forms.append,
    )

    window.start_task()
    assert controller.started is not None
    assert saved_forms[0].thread_id == "bridge-test-001"
    assert window.model.metrics[1].value == "2 / 2"
    assert window.current_snapshot.endswith("test.yaml")
    assert window.current_interrupt["review_type"] == "modeling_check_review"
    assert window.review_action_buttons[0].isEnabled()

    window.review_feedback.setPlainText("重新计算控制截面")
    window.resume_task("continue_modeling_revision")
    assert controller.resumed["feedback"] == "重新计算控制截面"


def test_question_uses_recent_history_and_renders_citation(app, tmp_path) -> None:
    controller = FakeController()
    window = ModernBridgeWindow(
        build_demo_dashboard_model(),
        controller=controller,
        initial_form=_form(tmp_path),
        operation_dispatcher=sync_dispatch,
    )
    snapshot = tmp_path / "output" / "run_configs" / "test.yaml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("task: {}", encoding="utf-8")
    window.current_snapshot = str(snapshot)
    window.chat_input.setPlainText("盖梁验算通过了吗？")

    window.send_question()

    assert controller.asked[0] == "盖梁验算通过了吗？"
    assert controller.asked[1]["history"][-1]["role"] == "user"
    assert "CAP-01" in window.chat_view.toPlainText()


def test_generate_catalog_views_runs_in_operation_channel(app, tmp_path) -> None:
    form = _form(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    generated = []

    def builder(path):
        generated.append(path)
        return {"initial_plan": str(output_dir / "initial.png"), "final_plan": None}

    window = ModernBridgeWindow(
        build_demo_dashboard_model(),
        controller=FakeController(),
        initial_form=form,
        operation_dispatcher=sync_dispatch,
        catalog_view_builder=builder,
    )

    window.generate_catalog_views()

    assert generated == [str(output_dir)]
    assert "已生成 1 张目录视图" in window.run_log.toPlainText()


def _form_with_range(tmp_path, user_request: str) -> ModernGuiForm:
    form = _form(tmp_path)
    form.user_request = user_request
    return form


def test_route_range_comes_from_request_once_the_task_starts(app, tmp_path) -> None:
    """顶栏必须显示本次要设计的桩号范围，而不是一直挂着"路线范围未设置"。"""
    window = ModernBridgeWindow(
        build_dashboard_model({}),
        controller=FakeController(),
        initial_form=_form_with_range(tmp_path, "完成K29+000-K31+200桥梁全部设计"),
        operation_dispatcher=sync_dispatch,
    )
    assert window.model.route_range == "路线范围未设置"

    window.start_task()

    assert window.model.route_range == "K29+000 — K31+200"
    assert window.top_route_range.text() == "K29+000 — K31+200"


def test_route_range_tracks_edits_to_the_task_description(app, tmp_path) -> None:
    """改了任务描述里的桩号，运行中的下一次刷新就要跟着变，不能记住上一次的结果。"""
    window = ModernBridgeWindow(
        build_dashboard_model({}),
        controller=FakeController(),
        initial_form=_form_with_range(tmp_path, "完成K29+000-K31+200桥梁全部设计"),
        operation_dispatcher=sync_dispatch,
    )
    window.start_task()

    window.task_request.setPlainText("改为完成K1+600—K3+100桥梁设计")
    window.busy = True  # 进度定时器只在运行中刷新
    window.refresh_progress()

    assert window.model.route_range == "K1+600 — K3+100"


def test_explicit_route_range_wins_over_the_request(app, tmp_path) -> None:
    """--route-range 是显式指定，不能被任务描述里的桩号顶掉。"""
    window = ModernBridgeWindow(
        build_demo_dashboard_model(),
        controller=FakeController(),
        initial_form=_form_with_range(tmp_path, "完成K29+000-K31+200桥梁全部设计"),
        operation_dispatcher=sync_dispatch,
        route_range="K1+000 — K2+000",
    )

    window.start_task()

    assert window.model.route_range == "K1+000 — K2+000"
