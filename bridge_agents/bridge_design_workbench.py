"""Figma v1 bridge design workbench over the existing Graph V2 runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from bridge_agents.gui_modern_model import MetricModel
from bridge_agents.gui_modern_runtime import format_run_event
from bridge_agents.gui_modern_widgets import BridgeStageRail, MetricCard, StatusPill
from bridge_agents.gui_modern_theme import DEFAULT_THEME
from bridge_agents.gui_modern_window import ACTION_LABELS, ModernBridgeWindow
from bridge_agents.workbench_visuals import BlueprintPage, BlueprintSidebar, navigation_icon
from bridge_agents.workbench_evidence import CapacityEvidence, read_capacity_evidence, read_human_decisions


def _value(value: object, missing: str = "待生成") -> str:
    return missing if value is None else str(value)


class BridgeDesignWorkbenchWindow(ModernBridgeWindow):
    """Six user modules and three contextual decision states."""

    PAGE_NAMES = {"overview": 0, "task": 1, "review": 2, "results": 3, "qa": 4, "log": 5,
                  "confirm": 6, "missing": 7, "rework": 8}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._pending_review_action = ""
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Bridge Design Workbench · 桥梁设计工作台")
        self.resize(1440, 900)

    @staticmethod
    def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("panelCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(11)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        return card, layout

    @staticmethod
    def _muted(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("muted")
        label.setWordWrap(True)
        return label

    def _build_sidebar(self) -> QWidget:
        sidebar = BlueprintSidebar()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(214)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(19, 25, 19, 22)
        layout.setSpacing(7)
        brand = QLabel("BRIDGE / AI")
        brand.setObjectName("brand")
        caption = QLabel("智能桥梁设计工作台")
        caption.setObjectName("brandCaption")
        layout.addWidget(brand)
        layout.addWidget(caption)
        layout.addSpacing(22)
        self.nav_buttons = []
        for index, (key, label) in enumerate((
            ("overview", "工程总览"), ("task", "任务启动"), ("review", "人工复核"),
            ("results", "成果中心"), ("qa", "成果问答"), ("log", "运行记录"),
        )):
            button = QPushButton(label)
            button.setObjectName("navButton")
            button.setIcon(navigation_icon(key))
            button.setIconSize(QSize(20, 20))
            button.setCheckable(True)
            button.setMinimumHeight(41)
            button.clicked.connect(lambda checked=False, page=index: self._switch_page(page))
            layout.addWidget(button)
            self.nav_buttons.append(button)
        self.nav_buttons[0].setChecked(True)
        layout.addStretch()
        project_card = QFrame()
        project_card.setStyleSheet("background: #173A57; border-radius: 10px;")
        card_layout = QVBoxLayout(project_card)
        card_layout.setContentsMargins(12, 11, 12, 11)
        project_caption = QLabel("当前项目")
        project_caption.setStyleSheet("color: #8FA7BA; font-size: 10px;")
        self.sidebar_project_name = QLabel(self.model.project_name)
        self.sidebar_project_name.setWordWrap(True)
        self.sidebar_project_name.setStyleSheet("color: white; font-weight: 600;")
        self.sidebar_data_badge = QLabel(f"●  {self.model.data_badge}")
        badge_color = DEFAULT_THEME.warning if self.model.is_demo else DEFAULT_THEME.accent
        self.sidebar_data_badge.setStyleSheet(f"color: {badge_color}; font-size: 10px;")
        card_layout.addWidget(project_caption)
        card_layout.addWidget(self.sidebar_project_name)
        card_layout.addWidget(self.sidebar_data_badge)
        layout.addWidget(project_card)
        return sidebar

    @staticmethod
    def _page_container() -> tuple[QWidget, QVBoxLayout]:
        page = BlueprintPage()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 23, 28, 28)
        layout.setSpacing(18)
        return page, layout

    def _build_workspace(self) -> QWidget:
        workspace = super()._build_workspace()
        self.pages.addWidget(self._scroll_page(self._build_confirm_page()))
        self.pages.addWidget(self._scroll_page(self._build_missing_page()))
        self.pages.addWidget(self._scroll_page(self._build_rework_page()))
        return workspace

    def _build_task_page(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("START A DESIGN RUN", "发起桥梁方案设计", "填写原始资料和任务目标；已有中间成果由输出目录自动识别。"))
        row = QHBoxLayout()
        row.setSpacing(16)
        form_card, form_layout = self._card("工程任务 · 原始输入")
        self.task_request = QPlainTextEdit(self.form.user_request)
        self.task_request.setPlaceholderText("例如：对 K29+000—K31+200 进行全流程桥梁设计")
        self.task_request.setMinimumHeight(128)
        self.task_request.setMaximumHeight(175)
        form_layout.addWidget(self._field_label("任务描述"))
        form_layout.addWidget(self.task_request, 1)
        self.form_edits: dict[str, QLineEdit] = {}
        for key, title, kind in (("data_path", "路线资料目录", "dir"),
                                  ("drawing_path", "输入图纸", "drawing"),
                                  ("output_dir", "输出目录", "dir")):
            form_layout.addWidget(self._field_label(title))
            edit = QLineEdit(str(getattr(self.form, key)))
            self.form_edits[key] = edit
            field_row = QHBoxLayout()
            field_row.addWidget(edit, 1)
            choose = QPushButton("选择")
            choose.setObjectName("secondaryButton")
            choose.clicked.connect(lambda checked=False, name=key, mode=kind: self._choose_path(name, mode))
            field_row.addWidget(choose)
            form_layout.addLayout(field_row)
            edit.textChanged.connect(self._refresh_preflight)
        form_layout.addWidget(self._field_label("最大返修轮数"))
        self.revision_spin = QSpinBox()
        self.revision_spin.setRange(1, 20)
        self.revision_spin.setValue(self.form.max_revision_rounds)
        self.check_revision_spin = QSpinBox()
        self.check_revision_spin.setRange(1, 20)
        self.check_revision_spin.setValue(self.form.max_check_revision_rounds)
        self.revision_spin.valueChanged.connect(self.check_revision_spin.setValue)
        form_layout.addWidget(self.revision_spin)
        self.thread_edit = QLineEdit(self.form.thread_id)
        self.thread_edit.setReadOnly(True)
        self.rag_check = QCheckBox()
        self.rag_check.setChecked(self.form.code_rag_enabled)
        self.form_edits["base_config_path"] = QLineEdit(self.form.base_config_path)
        self.authorization_check = QCheckBox("同意向已配置的模型服务发送本次任务内容和必要成果摘要")
        self.authorization_check.setChecked(self.form.external_authorized)
        form_layout.addWidget(self.authorization_check)
        form_layout.addStretch()
        actions = QHBoxLayout()
        self.start_button = QPushButton("启动设计任务")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_task)
        self.stop_button = QPushButton("安全终止")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.request_stop)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        form_layout.addLayout(actions)
        row.addWidget(form_card, 7)

        guide_card, guide = self._card("启动前检查")
        self.preflight_route = self._muted("")
        self.preflight_drawing = self._muted("")
        self.preflight_output = self._muted("")
        for item in (self.preflight_route, self.preflight_drawing, self.preflight_output):
            guide.addWidget(item)
        guide.addWidget(self._muted("●  使用 Graph V2 协同设计"))
        guide.addSpacing(12)
        tip = QFrame()
        tip.setStyleSheet("background: #EAF4FB; border-radius: 10px;")
        tip_layout = QVBoxLayout(tip)
        tip_layout.addWidget(QLabel("中间成果自动恢复"))
        tip_layout.addWidget(self._muted("布跨、结构设计和验算成果会从输出目录发现，无需逐项指定。"))
        guide.addWidget(tip)
        guide.addSpacing(14)
        guide.addWidget(self._field_label("任务启动后的阶段"))
        for step in ("路线资料与初步方案", "布跨修正与结构设计", "模型验算、人工复核与交付"):
            guide.addWidget(self._muted(f"•  {step}"))
        guide.addStretch()
        config_button = QPushButton("选择基础运行配置")
        config_button.setObjectName("secondaryButton")
        config_button.clicked.connect(lambda: self._choose_path("base_config_path", "file"))
        guide.addWidget(config_button)
        self.run_status_label = self._muted("等待启动")
        guide.addWidget(self.run_status_label)
        row.addWidget(guide_card, 4)
        layout.addLayout(row, 1)
        self._refresh_preflight()
        return page

    def _refresh_preflight(self) -> None:
        if not hasattr(self, "preflight_route"):
            return
        route_text = self.form_edits["data_path"].text().strip()
        drawing_text = self.form_edits["drawing_path"].text().strip()
        output_text = self.form_edits["output_dir"].text().strip()
        route_ok = bool(route_text) and Path(route_text).is_dir()
        drawing_ok = bool(drawing_text) and Path(drawing_text).is_file()
        output_ok = bool(output_text) and Path(output_text).is_dir()
        self.preflight_route.setText("已选择 · 路线资料目录" if route_ok else "请选择有效的路线资料目录")
        self.preflight_drawing.setText("已选择 · 输入图纸" if drawing_ok else "请选择输入图纸")
        self.preflight_output.setText("输出目录已存在" if output_ok else "输出目录将在启动时创建" if output_text else "请选择输出目录")

    def _capacity(self) -> CapacityEvidence:
        return read_capacity_evidence(self.form_edits["output_dir"].text().strip())

    def _display_status(self) -> str:
        current = str(self.current_result.get("task_status") or "")
        if current == "completed_with_accepted_risks":
            return "完成且带风险"
        if current and current not in {"completed", "completed_with_accepted_risks"}:
            return self.model.run_status
        evidence = read_capacity_evidence(self.form.output_dir)
        if evidence.engineering_pass is False and any(
            row.get("action") == "accept_check_and_finish"
            for row in read_human_decisions(self.form.output_dir)
        ):
            return "完成且带风险"
        return self.model.run_status

    def _status_pill(self) -> StatusPill:
        status = self._display_status()
        tone = "warning" if status == "完成且带风险" or self.model.review.required else "success" if status == "已完成" else "primary"
        return StatusPill(f"●  {status}", tone)

    def _metric_row(self, evidence: CapacityEvidence) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(13)
        calculated = f"{_value(evidence.completed_tasks)} / {_value(evidence.total_tasks)}"
        verdict = "已通过" if evidence.engineering_pass is True else "未通过" if evidence.engineering_pass is False else "待验算"
        for metric in (
            MetricModel("验算任务已计算", calculated, "完成数量 / 应算数量", "primary"),
            MetricModel("执行失败", _value(evidence.execution_failures), "计算任务失败数", "success"),
            MetricModel("工程验算结论", verdict, "独立于任务计算状态", "warning" if evidence.engineering_pass is False else "neutral"),
        ):
            row.addWidget(MetricCard(metric), 1)
        return row

    def _build_overview(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("PROJECT OVERVIEW", "工程总览", "查看阶段进展、工程验算状态与需要处理的判断。"))
        if self.model.review.required:
            alert, line = self._card("等待工程师复核")
            alert.setObjectName("overviewReviewCard")
            alert.setStyleSheet("QFrame#overviewReviewCard { background: #FFF2DC; border: 1px solid #F0D4A8; border-radius: 12px; }")
            line.addWidget(self._muted(self.model.review.message))
            button = QPushButton("进入复核")
            button.setObjectName("primaryButton")
            button.clicked.connect(lambda: self.show_page("review"))
            line.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(alert)
        stage_card, stage_layout = self._card("五阶段设计链路")
        stage_layout.addWidget(BridgeStageRail(self.model.stages))
        layout.addWidget(stage_card)
        evidence = read_capacity_evidence(self.form.output_dir)
        layout.addLayout(self._metric_row(evidence))
        issue, issue_layout = self._card("当前控制问题")
        if evidence.engineering_pass is False:
            group = evidence.control_group or "控制设计组待核对"
            ratio = f"{evidence.max_utilization:.3f}" if evidence.max_utilization is not None else "待核对"
            issue_layout.addWidget(QLabel(f"{group} · 最大利用率 {ratio}"))
            issue_layout.addWidget(self._muted("验算未通过。批次计算完成不表示工程指标合格，请核对关联成果。"))
        else:
            issue_layout.addWidget(self._muted("暂无未通过的控制指标，或尚未生成完整验算结果。"))
        layout.addWidget(issue)
        timeline, timeline_layout = self._card("最近动作")
        log_path = Path(self.form.output_dir) / "logs" / "agent_run_log.json"
        try:
            events = json.loads(log_path.read_text(encoding="utf-8")).get("events") or []
        except (OSError, ValueError, AttributeError):
            events = []
        recent = [
            f"{str(event.get('time') or '')[11:16]}  {message}"
            for event in events if isinstance(event, dict)
            if (message := format_run_event(event))
        ]
        for message in recent[-3:][::-1]:
            timeline_layout.addWidget(self._muted(message))
        if not recent:
            timeline_layout.addWidget(self._muted("启动或恢复任务后，阶段动作会显示在这里。"))
        layout.addWidget(timeline)
        layout.addStretch()
        return page

    def _build_review_page(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("HUMAN IN THE LOOP", "人工复核", "按原因、对象、证据和动作完成工程判断。"))
        reason, reason_layout = self._card("当前复核原因")
        reason.setObjectName("reviewReasonCard")
        reason.setStyleSheet("QFrame#reviewReasonCard { background: #FFF2DC; border: 1px solid #F0D4A8; border-radius: 12px; }")
        self.review_title = QLabel("当前没有等待处理的人工复核")
        self.review_title.setStyleSheet("font-size: 17px; font-weight: 700;")
        self.review_message = self._muted("")
        reason_layout.addWidget(self.review_title)
        reason_layout.addWidget(self.review_message)
        layout.addWidget(reason)
        row = QHBoxLayout()
        row.setSpacing(16)
        evidence_card, evidence_layout = self._card("控制指标与判断依据")
        self.review_evidence = self._muted("")
        evidence_layout.addWidget(self.review_evidence)
        self.review_details = QPlainTextEdit()
        self.review_details.setReadOnly(True)
        self.review_details.setMinimumHeight(260)
        evidence_layout.addWidget(self.review_details, 1)
        link = QPushButton("查看关联成果")
        link.setObjectName("secondaryButton")
        link.clicked.connect(lambda: self.show_page("results"))
        evidence_layout.addWidget(link, 0, Qt.AlignmentFlag.AlignLeft)
        row.addWidget(evidence_card, 7)
        action_card, action_layout = self._card("选择处理方式")
        action_layout.addWidget(self._muted("只展示当前复核节点允许的动作。"))
        action_layout.addWidget(self._field_label("补充要求或决策理由"))
        self.review_feedback = QPlainTextEdit()
        self.review_feedback.setMinimumHeight(120)
        self.review_feedback.setMaximumHeight(185)
        action_layout.addWidget(self.review_feedback, 1)
        action_layout.addWidget(self._field_label("追加返修轮数"))
        self.extra_rounds_spin = QSpinBox()
        self.extra_rounds_spin.setRange(1, 20)
        action_layout.addWidget(self.extra_rounds_spin)
        self.review_actions_layout = QVBoxLayout()
        self.review_action_buttons: list[QPushButton] = []
        action_layout.addLayout(self.review_actions_layout)
        action_layout.addStretch()
        row.addWidget(action_card, 4)
        layout.addLayout(row, 1)
        self._render_interrupt(self.current_interrupt)
        return page

    def _render_interrupt(self, review: dict[str, Any] | None) -> None:
        while self.review_actions_layout.count():
            item = self.review_actions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.review_action_buttons = []
        evidence = self._capacity()
        if not review:
            self.review_title.setText("当前没有等待处理的人工复核")
            self.review_message.setText("任务到达人工判断节点后，这里会显示原因、证据和候选动作。")
            self.review_details.setPlainText("尚无复核上下文。")
            self.review_evidence.setText("承载力验算：" + ("已有结果" if evidence.source_path else "尚无结果"))
            return
        self.review_title.setText("需要工程师判断")
        self.review_message.setText(str(review.get("message") or "当前阶段需要选择后续处理方式。"))
        target = review.get("rework_target_label") or "待协调器判断"
        round_text = f"当前轮次：{_value(review.get('current_round'))} / {_value(review.get('max_revision_rounds'))}"
        self.review_evidence.setText(f"{round_text}    ·    返修方向：{target}")
        lines = [f"验算批次：{_value(evidence.completed_tasks)} / {_value(evidence.total_tasks)} 已计算",
                 f"执行失败：{_value(evidence.execution_failures)}",
                 "工程结论：" + ("通过" if evidence.engineering_pass is True else "未通过" if evidence.engineering_pass is False else "待核实"),
                 f"控制设计组：{evidence.control_group or '待核实'}",
                 f"最大利用率：{_value(evidence.max_utilization)}",
                 f"证据文件：{evidence.source_path.name if evidence.source_path else '缺失'}"]
        if review.get("failed_task_ids"):
            lines.append("失败任务：" + "、".join(str(item) for item in review["failed_task_ids"]))
        self.review_details.setPlainText("\n".join(lines))
        for action in review.get("available_actions") or []:
            name = str(action)
            title = "查看风险确认" if name == "accept_check_and_finish" else ACTION_LABELS.get(name, name)
            button = QPushButton(title)
            button.setObjectName("dangerButton" if name == "abort" else "secondaryButton" if name.startswith("accept") else "primaryButton")
            if name == "accept_check_and_finish":
                button.clicked.connect(lambda: self.show_page("confirm"))
            elif name == "abort":
                button.clicked.connect(self.request_stop)
            else:
                button.clicked.connect(lambda checked=False, value=name: self.resume_task(value))
            self.review_actions_layout.addWidget(button)
            self.review_action_buttons.append(button)
        self._refresh_review_buttons()

    def _build_confirm_page(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("DECISION CHECKPOINT", "确认接受已知风险", "提交后保留未通过结论、人工决定和证据来源。"))
        warning, warning_layout = self._card("工程验算未通过；接受动作不会改变验算结论")
        warning.setObjectName("riskWarningCard")
        warning.setStyleSheet("QFrame#riskWarningCard { background: #FFF2DC; border: 1px solid #F0D4A8; border-radius: 12px; }")
        warning_layout.addWidget(self._muted("只有承载力验算批次完整时，才允许提交本决定。"))
        layout.addWidget(warning)
        row = QHBoxLayout()
        row.setSpacing(16)
        facts, facts_layout = self._card("确认前核对")
        self.confirm_facts = self._muted("")
        facts_layout.addWidget(self.confirm_facts)
        facts_layout.addWidget(self._muted("提交后的任务会标记为“完成且带风险”；未通过项仍保留在交付成果中。"))
        row.addWidget(facts, 1)
        decision, decision_layout = self._card("工程师确认")
        self.confirm_check_metrics = QCheckBox("我已查看控制组与未通过指标")
        self.confirm_check_risk = QCheckBox("我理解交付成果会持续标记带风险")
        self.confirm_check_metrics.toggled.connect(self._refresh_confirm_submit)
        self.confirm_check_risk.toggled.connect(self._refresh_confirm_submit)
        decision_layout.addWidget(self.confirm_check_metrics)
        decision_layout.addWidget(self.confirm_check_risk)
        decision_layout.addWidget(self._field_label("补充决策说明（可选）"))
        self.confirm_note = QPlainTextEdit()
        self.confirm_note.setMaximumHeight(100)
        decision_layout.addWidget(self.confirm_note)
        buttons = QHBoxLayout()
        back = QPushButton("返回复核")
        back.setObjectName("secondaryButton")
        back.clicked.connect(lambda: self.show_page("review"))
        self.confirm_submit = QPushButton("记录风险并结束")
        self.confirm_submit.setObjectName("primaryButton")
        self.confirm_submit.setStyleSheet("QPushButton:disabled { background: #E7EDF1; color: #8797A5; border: 1px solid #D6E0E7; }")
        self.confirm_submit.setEnabled(False)
        self.confirm_submit.clicked.connect(self._submit_risk_decision)
        buttons.addWidget(back)
        buttons.addWidget(self.confirm_submit)
        decision_layout.addLayout(buttons)
        row.addWidget(decision, 1)
        layout.addLayout(row)
        layout.addStretch()
        return page

    def _build_missing_page(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("BLOCKED ACTION STATE", "当前不能接受风险并结束", "缺少完整承载力验算结果时，保留返修路径。"))
        banner, banner_body = self._card("接受风险动作已阻断")
        banner.setObjectName("missingRiskCard")
        banner.setStyleSheet("QFrame#missingRiskCard { background: #FDEBE9; border: 1px solid #F1C8C3; border-radius: 12px; }")
        self.missing_reason = self._muted("")
        banner_body.addWidget(self.missing_reason)
        layout.addWidget(banner)
        row = QHBoxLayout()
        row.setSpacing(16)
        evidence, evidence_body = self._card("证据检查")
        evidence_body.addWidget(self._muted("承载力批次摘要：缺失或未完成"))
        evidence_body.addWidget(self._muted("控制组验算明细：目前不可核实"))
        evidence_body.addWidget(self._muted("布跨与结构成果仍可在成果中心查看。"))
        evidence_body.addStretch()
        row.addWidget(evidence, 7)
        card, body = self._card("可继续的动作")
        unavailable = QPushButton("证据不足，暂不可提交")
        unavailable.setEnabled(False)
        body.addWidget(unavailable)
        body.addWidget(self._muted("可先补齐分析输入或继续返修，再重新验算。"))
        actions = QHBoxLayout()
        rework = QPushButton("继续返修")
        rework.setObjectName("primaryButton")
        rework.clicked.connect(self._continue_from_missing)
        back = QPushButton("返回复核摘要")
        back.setObjectName("secondaryButton")
        back.clicked.connect(lambda: self.show_page("review"))
        actions.addWidget(rework)
        actions.addWidget(back)
        body.addLayout(actions)
        body.addStretch()
        row.addWidget(card, 4)
        layout.addLayout(row)
        layout.addStretch()
        return page

    def _build_rework_page(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("RESUME THE DESIGN", "已提交继续返修", "人工补充要求已记录，后续沿当前运行线程继续。"))
        route = QHBoxLayout()
        route.setSpacing(12)
        for title, state in (("01 人工决定", "已保存"), ("02 结构设计", "待执行或执行中"),
                             ("03 重新验算", "待执行"), ("04 交付或再复核", "待判断")):
            step, step_body = self._card(title)
            step_body.addWidget(self._muted(state))
            route.addWidget(step, 1)
        layout.addLayout(route)
        card, body = self._card("本次返修")
        self.rework_detail = self._muted("")
        body.addWidget(self.rework_detail)
        body.addWidget(self._muted("重新验算完成后，系统将更新通过状态；仍需判断时再次进入人工复核。"))
        button = QPushButton("返回工程总览")
        button.setObjectName("primaryButton")
        button.clicked.connect(lambda: self.show_page("overview"))
        body.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(card)
        layout.addStretch()
        return page

    def _build_results_page(self) -> QWidget:
        page = super()._build_results_page()
        layout = page.layout()
        heading = page.findChild(QLabel, "pageTitle")
        if heading:
            heading.setText("成果与风险")
        banner, body = self._card("成果状态与风险")
        banner.setObjectName("resultsRiskCard")
        banner.setStyleSheet("QFrame#resultsRiskCard { background: #FDEBE9; border: 1px solid #F1C8C3; border-radius: 12px; }")
        self.results_risk = self._muted("")
        body.addWidget(self.results_risk)
        self.results_trace = self._muted("")
        body.addWidget(self.results_trace)
        ask = QPushButton("就此成果提问")
        ask.setObjectName("secondaryButton")
        ask.clicked.connect(lambda: self.show_page("qa"))
        body.addWidget(ask, 0, Qt.AlignmentFlag.AlignLeft)
        layout.insertWidget(1, banner)
        return page

    def _build_chat_page(self) -> QWidget:
        page = super()._build_chat_page()
        heading = page.findChild(QLabel, "pageTitle")
        if heading:
            heading.setText("成果问答")
        evidence, body = self._card("本轮证据")
        self.qa_evidence = self._muted("回答中的引用编号会随每轮问答显示；项目成果和规范证据均从当前运行目录检索。")
        body.addWidget(self.qa_evidence)
        body.addWidget(self._muted("问答仅解释已有成果和证据，不更改设计文件。"))
        body.addStretch()
        back = QPushButton("返回成果中心")
        back.setObjectName("secondaryButton")
        back.clicked.connect(lambda: self.show_page("results"))
        body.addWidget(back)
        page.layout().takeAt(1)
        middle = QHBoxLayout()
        middle.setSpacing(16)
        middle.addWidget(self.chat_view, 7)
        middle.addWidget(evidence, 3)
        page.layout().insertLayout(1, middle, 1)
        return page

    def _build_info_page(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("RUN TRACE", "运行记录", "核对任务身份、阶段动作、复核决定和最终交付状态。"))
        identity, grid_layout = self._card("本次运行身份")
        self.log_identity = self._muted("")
        grid_layout.addWidget(self.log_identity)
        layout.addWidget(identity)
        row = QHBoxLayout()
        row.setSpacing(16)
        actions, action_layout = self._card("阶段动作")
        self.run_log = QPlainTextEdit()
        self.run_log.setReadOnly(True)
        self.run_log.setPlaceholderText("启动后显示阶段动作和异常摘要。")
        action_layout.addWidget(self.run_log, 1)
        row.addWidget(actions, 7)
        decisions, decision_layout = self._card("复核决定")
        self.log_decisions = QPlainTextEdit()
        self.log_decisions.setReadOnly(True)
        decision_layout.addWidget(self.log_decisions, 1)
        result = QPushButton("查看成果与风险")
        result.setObjectName("primaryButton")
        result.clicked.connect(lambda: self.show_page("results"))
        decision_layout.addWidget(result)
        row.addWidget(decisions, 4)
        layout.addLayout(row, 1)
        return page

    def _refresh_context(self) -> None:
        evidence = self._capacity()
        if hasattr(self, "confirm_facts"):
            verdict = "通过" if evidence.engineering_pass is True else "未通过" if evidence.engineering_pass is False else "待核实"
            ratio = f"{evidence.max_utilization:.3f}" if evidence.max_utilization is not None else "待核实"
            self.confirm_facts.setText(
                f"计算状态：{_value(evidence.completed_tasks)} / {_value(evidence.total_tasks)}\n"
                f"工程结论：{verdict}\n控制对象：{evidence.control_group or '待核实'}\n"
                f"控制利用率：{ratio}\n"
                f"依据：{evidence.source_path.name if evidence.source_path else '承载力验算证据缺失'}"
            )
        if hasattr(self, "missing_reason"):
            self.missing_reason.setText("承载力批次摘要缺失或未完成；目前无法核实未通过范围。")
        decisions = read_human_decisions(self.form_edits["output_dir"].text().strip())
        if hasattr(self, "results_risk"):
            state = str(self.current_result.get("task_status") or "")
            risk = "当前交付含人工接受风险。" if state == "completed_with_accepted_risks" or any(
                row.get("action") == "accept_check_and_finish" for row in decisions
            ) else "当前无已接受的验算风险记录。"
            verdict = "验算未通过" if evidence.engineering_pass is False else "验算已通过" if evidence.engineering_pass is True else "验算结论待核实"
            self.results_risk.setText(f"{risk} {verdict}。控制设计组：{evidence.control_group or '待核实'}。")
            self.results_trace.setText("人工决定：" + (" → ".join(
                ACTION_LABELS.get(str(row.get("action")), str(row.get("action"))) for row in decisions[-4:]
            ) if decisions else "暂无记录"))
        if hasattr(self, "log_identity"):
            run_id = self.form.thread_id
            log_path = Path(self.form_edits["output_dir"].text().strip()) / "logs" / "agent_run_log.json"
            try:
                log_payload = json.loads(log_path.read_text(encoding="utf-8"))
                if isinstance(log_payload, dict) and log_payload.get("run_id"):
                    run_id = str(log_payload["run_id"])
            except (OSError, ValueError):
                pass
            self.log_identity.setText(
                f"任务：{self.task_request.toPlainText().strip() or '尚未填写'}\n"
                f"运行 ID：{run_id}\n结果：{self._display_status()}"
            )
            self.log_decisions.setPlainText("\n".join(
                f"{index:02d}  {row.get('decided_at') or '时间未知'}  "
                f"{ACTION_LABELS.get(str(row.get('action')), row.get('action'))}"
                for index, row in enumerate(decisions, 1)
            ) or "暂无人工复核决定。")
        self._refresh_confirm_submit()

    def _refresh_confirm_submit(self) -> None:
        if not hasattr(self, "confirm_submit"):
            return
        offered = (self.current_interrupt or {}).get("available_actions") or []
        enabled = (
            "accept_check_and_finish" in offered and bool(self.current_snapshot)
            and not self.busy and self._capacity().can_accept_modeling_risk
            and self.confirm_check_metrics.isChecked() and self.confirm_check_risk.isChecked()
        )
        self.confirm_submit.setText("记录风险并结束" if "accept_check_and_finish" in offered else "当前无待处理复核")
        self.confirm_submit.setEnabled(enabled)

    def _refresh_action_log(self) -> None:
        log_path = Path(self.form.output_dir) / "logs" / "agent_run_log.json"
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
            events = payload.get("events") or []
        except (OSError, ValueError, AttributeError):
            return
        if not isinstance(events, list):
            return
        for event in events[self._agent_log_offset:]:
            if not isinstance(event, dict):
                continue
            message = format_run_event(event)
            if message:
                moment = str(event.get("time") or "")
                stamp = moment[11:19] if len(moment) >= 19 else "时间未知"
                self.run_log.appendPlainText(f"[{stamp}] {message}")
        self._agent_log_offset = len(events)

    def _refresh_review_buttons(self) -> None:
        super()._refresh_review_buttons()
        self._refresh_confirm_submit()

    def _submit_risk_decision(self) -> None:
        self._refresh_confirm_submit()
        if not self.confirm_submit.isEnabled():
            return
        note = self.confirm_note.toPlainText().strip()
        if note:
            self.review_feedback.setPlainText(note)
        self.resume_task("accept_check_and_finish")

    def _continue_from_missing(self) -> None:
        offered = (self.current_interrupt or {}).get("available_actions") or []
        for action in ("continue_modeling_revision", "continue_revision", "retry_failed_tasks"):
            if action in offered:
                self.resume_task(action)
                return
        self.show_page("review")

    def start_task(self) -> None:
        super().start_task()
        if self.busy:
            self.show_page("overview")

    def resume_task(self, action: str) -> None:
        if self.current_interrupt and not self.busy:
            self._pending_review_action = action
        super().resume_task(action)

    def _handle_operation(self, result: Any) -> None:
        pending = self._pending_review_action
        super()._handle_operation(result)
        if result.name != "resume":
            return
        self._pending_review_action = ""
        if result.error or self.current_interrupt:
            return
        if pending in {"continue_modeling_revision", "continue_revision", "retry_failed_tasks"}:
            target = str((result.payload or {}).get("message") or "正在继续返修")
            self.rework_detail.setText(f"{target}\n追加轮次：{self.extra_rounds_spin.value()}\n运行 ID：{self.form.thread_id}")
            self.show_page("rework")
        elif pending == "accept_check_and_finish":
            self.show_page("results")

    def show_page(self, name: str) -> None:
        if name not in self.PAGE_NAMES:
            raise ValueError(f"unknown workbench page: {name}")
        if name == "confirm" and not self._capacity().can_accept_modeling_risk:
            name = "missing"
        self._switch_page(self.PAGE_NAMES[name])
        if name == "review":
            self._render_interrupt(self.current_interrupt)

    def _switch_page(self, index: int) -> None:
        if index in {2, 3, 5, 6, 7, 8}:
            self._refresh_context()
        super()._switch_page(index)
        if index == 5:
            self._refresh_action_log()
        if index >= 6:
            for button_index, button in enumerate(self.nav_buttons):
                button.setChecked(button_index == 2 if index in {6, 7} else button_index == 0)

