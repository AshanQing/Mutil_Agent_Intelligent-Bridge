from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from bridge_agents.gui_modern_image_view import PanZoomImageView
from bridge_agents.gui_modern_model import build_dashboard_model
from bridge_agents.gui_modern_runtime import run_operation
from bridge_agents.gui_modern_session import ModernGuiForm
from bridge_agents.workbench_evidence import read_capacity_evidence, read_human_decisions
from bridge_agents.bridge_design_workbench import BridgeDesignWorkbenchWindow


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _capacity_batch(*, complete: bool = True) -> dict:
    return {
        "stage_complete": complete,
        "expected_task_count": 10,
        "completed_task_count": 10 if complete else 7,
        "failed_task_count": 0,
        "overall_check": {"M_pos_ok": True, "M_neg_ok": False, "V_ok": True, "all_ok": False},
        "utilization_summary": {"max_utilization": 1.444, "task_id": "2-2-2-G1"},
        "failed_check_task_ids": ["2-2-2-G1"],
    }


def test_capacity_evidence_separates_calculated_from_engineering_pass(tmp_path: Path) -> None:
    _write_json(tmp_path / "capacity_check/capacity_check_batch_summary.json", _capacity_batch())

    evidence = read_capacity_evidence(tmp_path)

    assert evidence.completed_tasks == 10
    assert evidence.total_tasks == 10
    assert evidence.execution_failures == 0
    assert evidence.engineering_pass is False
    assert evidence.control_group == "2-2-2-G1"
    assert evidence.max_utilization == 1.444
    assert evidence.can_accept_modeling_risk is True


def test_incomplete_or_missing_capacity_evidence_blocks_risk_acceptance(tmp_path: Path) -> None:
    assert read_capacity_evidence(tmp_path).can_accept_modeling_risk is False
    _write_json(tmp_path / "capacity_check/capacity_check_batch_summary.json", _capacity_batch(complete=False))
    assert read_capacity_evidence(tmp_path).can_accept_modeling_risk is False


def test_human_decisions_ignore_malformed_lines(tmp_path: Path) -> None:
    ledger = tmp_path / "human_review/human_review_decisions.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('bad json\n{"action":"accept_check_and_finish","decided_at":"2026-10-15T12:00:00"}\n', encoding="utf-8")
    assert [row["action"] for row in read_human_decisions(tmp_path)] == ["accept_check_and_finish"]


class ReviewController:
    def __init__(self) -> None:
        self.resumed: dict | None = None

    def resume(self, **kwargs):
        self.resumed = kwargs
        return {"task_status": "completed_with_accepted_risks", "thread_id": kwargs["thread_id"]}


def _window(tmp_path: Path, controller=None, dispatcher=None) -> BridgeDesignWorkbenchWindow:
    form = ModernGuiForm(
        user_request="K29+000—K31+200桥梁设计",
        data_path=str(tmp_path / "route"),
        drawing_path=str(tmp_path / "input.dxf"),
        output_dir=str(tmp_path),
        thread_id="bridge-test",
        external_authorized=True,
    )
    return BridgeDesignWorkbenchWindow(
        build_dashboard_model({}, project_name="测试工程"),
        initial_form=form,
        controller=controller or ReviewController(),
        operation_dispatcher=dispatcher or (lambda name, operation, callback: callback(run_operation(name, operation))),
        form_state_saver=lambda form: None,
    )


def test_six_modules_route_to_nine_workbench_states(app, tmp_path: Path) -> None:
    _write_json(tmp_path / "capacity_check/capacity_check_batch_summary.json", _capacity_batch())
    window = _window(tmp_path)
    try:
        assert window.pages.count() == 9
        assert [button.text().split()[-1] for button in window.nav_buttons] == [
            "工程总览", "任务启动", "人工复核", "成果中心", "成果问答", "运行记录",
        ]
        assert all(not button.text()[:1].isdigit() and not button.icon().isNull()
                   for button in window.nav_buttons)
        for state, index in {"task": 1, "confirm": 6, "missing": 7, "rework": 8, "log": 5}.items():
            window.show_page(state)
            assert window.pages.currentIndex() == index
    finally:
        window.close()


@pytest.mark.parametrize("name", ["reinforcement.svg", "column_reinforcement_detail.svg"])
def test_svg_preview_uses_declared_full_page_background(app, tmp_path: Path, name: str) -> None:
    svg = tmp_path / name
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="120" '
        'viewBox="0 0 160 120"><rect width="100%" height="100%" fill="#111827"/>'
        '<path d="M10 10 H150" stroke="#e0f2fe"/></svg>',
        encoding="utf-8",
    )
    image = PanZoomImageView._render_svg(str(svg)).toImage()
    corner = image.pixelColor(image.width() - 4, image.height() - 4)
    assert (corner.red(), corner.green(), corner.blue()) == (17, 24, 39)


def test_blank_task_inputs_are_not_marked_ready(app, tmp_path: Path) -> None:
    window = _window(tmp_path)
    try:
        window.form_edits["data_path"].clear()
        window.form_edits["drawing_path"].clear()
        window.form_edits["output_dir"].clear()
        assert "请选择" in window.preflight_route.text()
        assert "请选择" in window.preflight_drawing.text()
        assert "请选择" in window.preflight_output.text()
    finally:
        window.close()


def test_start_transitions_to_overview_while_graph_runs(app, tmp_path: Path) -> None:
    dispatched = []
    window = _window(tmp_path, dispatcher=lambda name, operation, callback: dispatched.append(name))
    try:
        window.show_page("task")
        window.start_task()
        assert dispatched == ["start"]
        assert window.pages.currentIndex() == 0
    finally:
        window.close()


def test_acceptance_needs_complete_evidence_and_two_acknowledgements(app, tmp_path: Path) -> None:
    controller = ReviewController()
    window = _window(tmp_path, controller)
    try:
        window.current_result = {"thread_id": "bridge-test"}
        window.current_snapshot = str(tmp_path / "run_configs/bridge-test.yaml")
        window.current_interrupt = {
            "review_type": "modeling_check_review",
            "message": "验算未通过",
            "available_actions": ["continue_modeling_revision", "accept_check_and_finish", "abort"],
        }
        window._render_interrupt(window.current_interrupt)
        window.show_page("confirm")
        assert window.pages.currentIndex() == 7
        assert not window.confirm_submit.isEnabled()

        _write_json(tmp_path / "capacity_check/capacity_check_batch_summary.json", _capacity_batch())
        window.show_page("confirm")
        assert window.pages.currentIndex() == 6
        assert not window.confirm_submit.isEnabled()
        window.confirm_check_metrics.setChecked(True)
        assert not window.confirm_submit.isEnabled()
        window.confirm_check_risk.setChecked(True)
        assert window.confirm_submit.isEnabled()

        window.confirm_note.setPlainText("已由负责人复核")
        window.confirm_submit.click()
        assert controller.resumed is not None
        assert controller.resumed["action"] == "accept_check_and_finish"
        assert controller.resumed["feedback"] == "已由负责人复核"
    finally:
        window.close()


def test_rework_screen_appears_after_rework_decision(app, tmp_path: Path) -> None:
    class ReworkController(ReviewController):
        def resume(self, **kwargs):
            self.resumed = kwargs
            return {"task_status": "running", "thread_id": kwargs["thread_id"], "message": "返回结构设计返修"}

    controller = ReworkController()
    window = _window(tmp_path, controller)
    try:
        window.current_result = {"thread_id": "bridge-test"}
        window.current_snapshot = "snapshot.yaml"
        window.current_interrupt = {
            "review_type": "modeling_check_review",
            "available_actions": ["continue_modeling_revision", "accept_check_and_finish"],
            "rework_target_label": "结构设计",
        }
        window._render_interrupt(window.current_interrupt)
        window.resume_task("continue_modeling_revision")
        assert controller.resumed is not None
        assert controller.resumed["action"] == "continue_modeling_revision"
        assert window.pages.currentIndex() == 8
    finally:
        window.close()


def test_existing_accepted_risk_and_actions_are_visible_when_opening_output(app, tmp_path: Path) -> None:
    _write_json(tmp_path / "capacity_check/capacity_check_batch_summary.json", _capacity_batch())
    ledger = tmp_path / "human_review/human_review_decisions.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"action":"accept_check_and_finish","decided_at":"2026-10-15T12:00:00"}\n', encoding="utf-8")
    _write_json(tmp_path / "logs/agent_run_log.json", {
        "run_id": "actual-run-id",
        "events": [{"time": "2026-10-15T12:34:00", "event_type": "action_end", "agent": "ModelingCheckAgent", "name": "run_capacity_check", "status": "completed"}],
    })
    window = _window(tmp_path)
    try:
        window.nav_buttons[5].click()
        assert "完成且带风险" in window.top_status_host.itemAt(0).widget().text()
        assert "actual-run-id" in window.log_identity.text()
        assert "承载力批量验算" in window.run_log.toPlainText()
        assert "[12:34:00]" in window.run_log.toPlainText()
    finally:
        window.close()
