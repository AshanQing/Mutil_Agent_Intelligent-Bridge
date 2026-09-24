from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from bridge_agents.gui_controller import (
    GraphV2Controller, extract_interrupt, infer_stage_states, load_drawing_catalog,
    snapshot_run_progress,
)
from bridge_agents.gui_modern_chat import ChatSession, format_answer
from bridge_agents.gui_modern_chat_view import ChatTranscript
from bridge_agents.gui_modern_image_view import PanZoomImageView
from bridge_agents.gui_modern_model import DashboardModel, build_dashboard_model
from bridge_agents.gui_modern_results import ArtifactRow, build_artifact_rows
from bridge_agents.gui_modern_runtime import OperationResult, format_run_event, run_operation
from bridge_agents.gui_modern_session import (
    ModernGuiForm, find_latest_snapshot, load_form_from_yaml, new_thread_id,
    route_range_from_request, save_form_state,
)
from bridge_agents.gui_modern_theme import DEFAULT_THEME, build_stylesheet
from bridge_agents.gui_modern_widgets import BridgeStageRail, MetricCard, StatusPill, TimelineMarker
from bridge_agents.result_catalog import STAGE_TITLES, scan as scan_result_catalog


ACTION_LABELS = {
    "continue_revision": "继续布跨修正",
    "accept_and_continue": "接受布跨并继续",
    "retry_coordinator": "重新交给协调器",
    "retry_failed_tasks": "重新设计失败任务",
    "accept_partial_and_continue": "接受部分成果并继续",
    "continue_modeling_revision": "重新设计并复验",
    "accept_check_and_finish": "接受风险并结束",
    "abort": "终止任务",
}


class _OperationBus(QObject):
    finished = Signal(object)


class ChatInput(QPlainTextEdit):
    send_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.send_requested.emit()
            return
        super().keyPressEvent(event)


class ModernBridgeWindow(QMainWindow):
    def __init__(
        self,
        model: DashboardModel,
        *,
        paper_mode: bool = False,
        controller: GraphV2Controller | None = None,
        initial_form: ModernGuiForm | None = None,
        operation_dispatcher: Callable[[str, Callable[[], dict[str, Any]], Callable[[OperationResult], None]], None] | None = None,
        form_state_saver: Callable[[ModernGuiForm], Any] = save_form_state,
        catalog_view_builder: Callable[[str], dict[str, Any]] | None = None,
        route_range: str = "",
    ) -> None:
        super().__init__()
        self.model = model
        self.paper_mode = paper_mode
        self.controller = controller or GraphV2Controller()
        self.form = initial_form or ModernGuiForm(output_dir="output")
        # 显式传入的路线范围优先（--route-range）；否则每次刷新都从任务描述里现取。
        self._route_range_override = route_range.strip()
        self.operation_dispatcher = operation_dispatcher or self._dispatch_async
        self.form_state_saver = form_state_saver
        self.catalog_view_builder = catalog_view_builder or self._generate_catalog_views_impl
        self.current_result: dict[str, Any] = {}
        self.current_interrupt: dict[str, Any] | None = None
        snapshot = find_latest_snapshot(self.form.output_dir, self.form.thread_id) if self.form.output_dir else None
        self.current_snapshot = str(snapshot or "")
        self.busy = False
        self.stop_requested = False
        self.chat_session = ChatSession()
        self.artifact_rows: tuple[ArtifactRow, ...] = ()
        self.visible_artifacts: list[ArtifactRow] = []
        self.selected_artifact: ArtifactRow | None = None
        self._agent_log_offset = 0
        self._operation_bus = _OperationBus(self)
        self._operation_bus.finished.connect(self._handle_operation)

        self.setWindowTitle("Bridge AI Studio · 工程设计控制台")
        self.resize(1440 if paper_mode else 1360, 900 if paper_mode else 850)
        self.setMinimumSize(1180, 740)
        self.setStyleSheet(build_stylesheet())
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._build_sidebar())
        shell.addWidget(self._build_workspace(), 1)
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self.refresh_progress)
        self._progress_timer.start(2000)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
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
        self.nav_buttons: list[QPushButton] = []
        navigation = (
            ("01", "工程总览"), ("02", "任务启动"), ("03", "人工复核"),
            ("04", "成果目录"), ("05", "成果问答"), ("06", "运行信息"),
        )
        for index, (number, label) in enumerate(navigation):
            button = QPushButton(f"{number}    {label}")
            button.setObjectName("navButton")
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

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        outer = QVBoxLayout(workspace)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_topbar())
        self.pages = QStackedWidget()
        self.pages.addWidget(self._scroll_page(self._build_overview()))
        self.pages.addWidget(self._scroll_page(self._build_task_page()))
        self.pages.addWidget(self._scroll_page(self._build_review_page()))
        self.pages.addWidget(self._build_results_page())
        self.pages.addWidget(self._build_chat_page())
        self.pages.addWidget(self._scroll_page(self._build_info_page()))
        outer.addWidget(self.pages, 1)
        return workspace

    def _build_topbar(self) -> QWidget:
        topbar = QFrame()
        topbar.setStyleSheet("background: white; border-bottom: 1px solid #DCE4EA;")
        topbar.setFixedHeight(72)
        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(28, 12, 30, 12)
        title_box = QVBoxLayout()
        self.top_project_name = QLabel(self.model.project_name)
        self.top_project_name.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.top_route_range = QLabel(self.model.route_range)
        self.top_route_range.setObjectName("muted")
        self.top_route_range.setStyleSheet("font-size: 11px;")
        title_box.addWidget(self.top_project_name)
        title_box.addWidget(self.top_route_range)
        layout.addLayout(title_box)
        layout.addStretch()
        self.top_status_host = QHBoxLayout()
        self.top_status_host.addWidget(self._status_pill())
        layout.addLayout(self.top_status_host)
        return topbar

    def _status_pill(self) -> StatusPill:
        tone = "warning" if self.model.review.required else "success" if self.model.run_status == "已完成" else "primary"
        return StatusPill(f"●  {self.model.run_status}", tone)

    @staticmethod
    def _scroll_page(content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _page_container() -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 23, 28, 28)
        layout.setSpacing(18)
        return page, layout

    @staticmethod
    def _page_heading(eyebrow: str, title: str, subtitle: str) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        eye = QLabel(eyebrow.upper())
        eye.setObjectName("sectionEyebrow")
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        sub = QLabel(subtitle)
        sub.setObjectName("pageSubtitle")
        sub.setWordWrap(True)
        layout.addWidget(eye)
        layout.addWidget(heading)
        layout.addWidget(sub)
        return box

    def _build_overview(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("PROJECT OVERVIEW", "工程设计总览", "聚焦阶段推进、工程成果与需要人工判断的异常。"))
        rail_card = QFrame()
        rail_card.setObjectName("panelCard")
        rail_layout = QVBoxLayout(rail_card)
        rail_layout.setContentsMargins(20, 15, 20, 8)
        header = QHBoxLayout()
        title = QLabel("五阶段设计链路")
        title.setObjectName("cardTitle")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(StatusPill(f"总体完成度 {self.model.progress_percent}%", "accent"))
        rail_layout.addLayout(header)
        rail_layout.addWidget(BridgeStageRail(self.model.stages))
        layout.addWidget(rail_card)
        metrics = QHBoxLayout()
        metrics.setSpacing(13)
        for metric in self.model.metrics:
            metrics.addWidget(MetricCard(metric), 1)
        layout.addLayout(metrics)
        body = QHBoxLayout()
        body.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(14)
        left.addWidget(self._build_active_card())
        left.addWidget(self._build_timeline_card())
        body.addLayout(left, 7)
        body.addWidget(self._build_review_summary_card(), 4)
        layout.addLayout(body)
        layout.addStretch()
        return page

    def _build_active_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("heroCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 20)
        top = QHBoxLayout()
        caption = QLabel("CURRENT AGENT")
        caption.setStyleSheet("color: #69C9C6; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        top.addWidget(caption)
        top.addStretch()
        top.addWidget(StatusPill(self.model.run_status, "accent"))
        layout.addLayout(top)
        title = QLabel(self.model.active_agent)
        title.setObjectName("heroTitle")
        scope = QLabel(f"当前对象  ·  {self.model.current_scope}")
        scope.setObjectName("heroText")
        scope.setWordWrap(True)
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(self.model.progress_percent)
        progress.setTextVisible(False)
        layout.addWidget(title)
        layout.addWidget(scope)
        layout.addSpacing(7)
        layout.addWidget(progress)
        return card

    def _build_timeline_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("panelCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 17, 20, 18)
        title = QLabel("最近活动")
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        for item in self.model.timeline:
            row = QHBoxLayout()
            row.addWidget(TimelineMarker(item.tone), 0, Qt.AlignmentFlag.AlignTop)
            text_box = QVBoxLayout()
            event_title = QLabel(item.title)
            event_title.setStyleSheet("font-weight: 600;")
            detail = QLabel(item.detail)
            detail.setObjectName("muted")
            detail.setWordWrap(True)
            text_box.addWidget(event_title)
            text_box.addWidget(detail)
            row.addLayout(text_box, 1)
            layout.addLayout(row)
        return card

    def _build_review_summary_card(self) -> QWidget:
        review = self.model.review
        card = QFrame()
        card.setObjectName("panelCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 20)
        top = QHBoxLayout()
        title = QLabel("人工判断")
        title.setObjectName("cardTitle")
        top.addWidget(title)
        top.addStretch()
        top.addWidget(StatusPill("需要处理" if review.required else "无需介入", "warning" if review.required else "success"))
        layout.addLayout(top)
        heading = QLabel(review.title)
        heading.setStyleSheet("font-size: 16px; font-weight: 700;")
        message = QLabel(review.message)
        message.setWordWrap(True)
        message.setObjectName("muted")
        layout.addWidget(heading)
        layout.addWidget(message)
        layout.addStretch()
        if review.required:
            button = QPushButton("前往人工复核")
            button.setObjectName("primaryButton")
            button.clicked.connect(lambda: self._switch_page(2))
            layout.addWidget(button)
        return card

    def _build_task_page(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("NEW DESIGN RUN", "任务启动", "填写原始输入并启动 Graph V2；中间成果继续由输出目录自动发现。"))
        content = QHBoxLayout()
        content.setSpacing(16)
        form_card = QFrame()
        form_card.setObjectName("panelCard")
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(22, 19, 22, 20)
        form_layout.setSpacing(9)
        form_title = QLabel("工程输入")
        form_title.setObjectName("cardTitle")
        form_layout.addWidget(form_title)
        self.task_request = QPlainTextEdit(self.form.user_request)
        self.task_request.setPlaceholderText("例如：完成 K1+600—K3+100 全流程桥梁设计并输出配筋图")
        self.task_request.setMaximumHeight(92)
        form_layout.addWidget(self._field_label("任务描述"))
        form_layout.addWidget(self.task_request)
        self.form_edits: dict[str, QLineEdit] = {}
        path_fields = (
            ("base_config_path", "基础配置", "file"),
            ("data_path", "路线数据目录", "dir"),
            ("drawing_path", "输入 DXF / DWG 图纸", "drawing"),
            ("output_dir", "输出目录", "dir"),
        )
        for key, label, kind in path_fields:
            form_layout.addWidget(self._field_label(label))
            edit = QLineEdit(str(getattr(self.form, key)))
            self.form_edits[key] = edit
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            choose = QPushButton("选择")
            choose.setObjectName("secondaryButton")
            choose.clicked.connect(lambda checked=False, field=key, mode=kind: self._choose_path(field, mode))
            row.addWidget(choose)
            form_layout.addLayout(row)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.addWidget(self._field_label("线路文件前缀"), 0, 0)
        grid.addWidget(self._field_label("运行 ID（自动生成）"), 0, 1)
        self.prefix_edit = QLineEdit(self.form.file_prefix)
        self.thread_edit = QLineEdit(self.form.thread_id)
        self.thread_edit.setReadOnly(True)
        grid.addWidget(self.prefix_edit, 1, 0)
        grid.addWidget(self.thread_edit, 1, 1)
        grid.addWidget(self._field_label("布跨修正轮数"), 2, 0)
        grid.addWidget(self._field_label("验算返修轮数"), 2, 1)
        self.revision_spin = QSpinBox()
        self.revision_spin.setRange(1, 20)
        self.revision_spin.setValue(self.form.max_revision_rounds)
        self.check_revision_spin = QSpinBox()
        self.check_revision_spin.setRange(1, 20)
        self.check_revision_spin.setValue(self.form.max_check_revision_rounds)
        grid.addWidget(self.revision_spin, 3, 0)
        grid.addWidget(self.check_revision_spin, 3, 1)
        form_layout.addLayout(grid)
        self.rag_check = QCheckBox("启用规范 RAG（评估与问答）")
        self.rag_check.setChecked(self.form.code_rag_enabled)
        self.authorization_check = QCheckBox("允许向配置中的外部模型发送任务与成果摘要")
        self.authorization_check.setChecked(self.form.external_authorized)
        form_layout.addWidget(self.rag_check)
        form_layout.addWidget(self.authorization_check)
        actions = QHBoxLayout()
        self.start_button = QPushButton("启动 Graph V2")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_task)
        self.stop_button = QPushButton("安全终止")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.request_stop)
        open_button = QPushButton("打开输出目录")
        open_button.setObjectName("secondaryButton")
        open_button.clicked.connect(self.open_output_dir)
        actions.addWidget(self.start_button, 1)
        actions.addWidget(self.stop_button)
        actions.addWidget(open_button)
        form_layout.addLayout(actions)
        content.addWidget(form_card, 7)
        status_card = QFrame()
        status_card.setObjectName("panelCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(20, 18, 20, 20)
        status_title = QLabel("运行记录")
        status_title.setObjectName("cardTitle")
        self.run_status_label = QLabel("等待启动")
        self.run_status_label.setWordWrap(True)
        self.run_status_label.setStyleSheet(f"color: {DEFAULT_THEME.primary}; font-weight: 600;")
        self.run_log = QPlainTextEdit()
        self.run_log.setReadOnly(True)
        self.run_log.setPlaceholderText("启动后显示阶段结果和异常摘要。")
        status_layout.addWidget(status_title)
        status_layout.addWidget(self.run_status_label)
        status_layout.addWidget(self.run_log, 1)
        content.addWidget(status_card, 4)
        layout.addLayout(content)
        return page

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("muted")
        label.setStyleSheet("font-size: 11px; font-weight: 600;")
        return label

    def _build_review_page(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("HUMAN IN THE LOOP", "人工复核中心", "自动重试耗尽后，在这里选择重新设计、继续下一步或终止任务。"))
        card = QFrame()
        card.setObjectName("panelCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 20, 22, 22)
        self.review_title = QLabel("当前没有等待处理的人工复核")
        self.review_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.review_message = QLabel("任务运行到人工判断节点后，原因、失败对象和可用动作会显示在这里。")
        self.review_message.setObjectName("muted")
        self.review_message.setWordWrap(True)
        self.review_details = QPlainTextEdit()
        self.review_details.setReadOnly(True)
        self.review_details.setMaximumHeight(180)
        self.review_feedback = QPlainTextEdit()
        self.review_feedback.setPlaceholderText("可选：补充工程师判断、修改要求或风险接受理由")
        self.review_feedback.setMaximumHeight(90)
        rounds = QHBoxLayout()
        rounds.addWidget(QLabel("增加重试轮数"))
        self.extra_rounds_spin = QSpinBox()
        self.extra_rounds_spin.setRange(1, 20)
        rounds.addWidget(self.extra_rounds_spin)
        rounds.addStretch()
        self.review_actions_layout = QHBoxLayout()
        self.review_action_buttons: list[QPushButton] = []
        card_layout.addWidget(self.review_title)
        card_layout.addWidget(self.review_message)
        card_layout.addWidget(self.review_details)
        card_layout.addWidget(self._field_label("补充要求"))
        card_layout.addWidget(self.review_feedback)
        card_layout.addLayout(rounds)
        card_layout.addLayout(self.review_actions_layout)
        layout.addWidget(card)
        layout.addStretch()
        self._render_interrupt(self.current_interrupt)
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 23, 28, 28)
        layout.setSpacing(14)
        heading_row = QHBoxLayout()
        heading_row.addWidget(self._page_heading("DELIVERABLES", "成果目录", "按阶段查看数据、诊断和可视化成果。"), 1)
        self.stage_filter = QComboBox()
        self.stage_filter.addItem("全部阶段", "")
        for stage, title in STAGE_TITLES.items():
            self.stage_filter.addItem(title, stage)
        self.stage_filter.currentIndexChanged.connect(self._filter_artifacts)
        refresh = QPushButton("刷新成果")
        refresh.setObjectName("secondaryButton")
        refresh.clicked.connect(self.refresh_artifacts)
        generate = QPushButton("生成方案视图")
        generate.setObjectName("secondaryButton")
        generate.clicked.connect(self.generate_catalog_views)
        heading_row.addWidget(self.stage_filter)
        heading_row.addWidget(generate)
        heading_row.addWidget(refresh)
        layout.addLayout(heading_row)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        table_card = QFrame()
        table_card.setObjectName("panelCard")
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(12, 12, 12, 12)
        self.artifact_table = QTableWidget(0, 3)
        self.artifact_table.setHorizontalHeaderLabels(("阶段", "成果", "设计组"))
        self.artifact_table.horizontalHeader().setStretchLastSection(True)
        self.artifact_table.verticalHeader().setVisible(False)
        self.artifact_table.setAlternatingRowColors(True)
        self.artifact_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.artifact_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.artifact_table.itemSelectionChanged.connect(self._show_selected_artifact)
        table_layout.addWidget(self.artifact_table)
        splitter.addWidget(table_card)
        preview_card = QFrame()
        preview_card.setObjectName("panelCard")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 15, 16, 16)
        self.preview_title = QLabel("选择一项成果")
        self.preview_title.setObjectName("cardTitle")
        self.preview_view = PanZoomImageView()
        self.preview_view.setMinimumSize(420, 330)
        self.preview_view.zoom_changed.connect(self._on_preview_zoom_changed)
        self.preview_view.show_message("可预览的图件会显示在这里")

        self.artifact_info = QPlainTextEdit()
        self.artifact_info.setReadOnly(True)
        self.artifact_info.setMaximumHeight(130)
        artifact_actions = QHBoxLayout()
        open_file = QPushButton("打开成果")
        open_file.setObjectName("primaryButton")
        open_file.clicked.connect(self.open_selected_artifact)
        open_dir = QPushButton("打开所在目录")
        open_dir.setObjectName("secondaryButton")
        open_dir.clicked.connect(self.open_selected_artifact_dir)
        artifact_actions.addWidget(open_file)
        artifact_actions.addWidget(open_dir)
        artifact_actions.addStretch()
        preview_layout.addWidget(self.preview_title)
        preview_tools = QHBoxLayout()
        self.preview_zoom_label = QLabel("100%")
        self.preview_zoom_label.setObjectName("muted")
        zoom_hint = QLabel("滚轮缩放 · 拖拽平移 · 双击复位")
        zoom_hint.setObjectName("muted")
        preview_tools.addWidget(self.preview_zoom_label)
        preview_tools.addWidget(zoom_hint)
        preview_tools.addStretch()
        fit_button = QPushButton("适应窗口")
        fit_button.setObjectName("secondaryButton")
        fit_button.setToolTip("恢复为完整显示（也可双击图像）")
        fit_button.clicked.connect(self.fit_preview)
        preview_tools.addWidget(fit_button)
        preview_layout.addLayout(preview_tools)
        preview_layout.addWidget(self.preview_view, 1)
        preview_layout.addWidget(self.artifact_info)
        preview_layout.addLayout(artifact_actions)
        splitter.addWidget(preview_card)
        splitter.setSizes((430, 700))
        layout.addWidget(splitter, 1)
        QTimer.singleShot(0, self.refresh_artifacts)
        return page

    def _build_chat_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 23, 28, 28)
        layout.setSpacing(13)
        header = QHBoxLayout()
        header.addWidget(self._page_heading("DESIGN REVIEW", "成果问答", "基于当前工程成果和规范证据进行评估与多轮追问。"), 1)
        assess = QPushButton("整体评估")
        assess.setObjectName("secondaryButton")
        assess.clicked.connect(self.assess_results)
        clear = QPushButton("清空对话")
        clear.setObjectName("secondaryButton")
        clear.clicked.connect(self.clear_chat)
        header.addWidget(assess)
        header.addWidget(clear)
        layout.addLayout(header)
        self.chat_view = ChatTranscript()
        layout.addWidget(self.chat_view, 1)
        self._append_chat("system", "可连续追问工程成果。回答会列出使用的成果与规范证据编号。")
        input_row = QHBoxLayout()
        self.chat_input = ChatInput()
        self.chat_input.setPlaceholderText("输入问题，Enter 发送，Shift+Enter 换行")
        self.chat_input.setMaximumHeight(92)
        self.chat_input.send_requested.connect(self.send_question)
        self.chat_send_button = QPushButton("发送问题")
        self.chat_send_button.setObjectName("primaryButton")
        self.chat_send_button.clicked.connect(self.send_question)
        input_row.addWidget(self.chat_input, 1)
        input_row.addWidget(self.chat_send_button)
        layout.addLayout(input_row)
        return page

    def _build_info_page(self) -> QWidget:
        page, layout = self._page_container()
        layout.addWidget(self._page_heading("RUN INFORMATION", "运行信息", "展示当前任务的公开运行摘要；论文模式不显示本地敏感路径。"))
        card = QFrame()
        card.setObjectName("panelCard")
        grid = QGridLayout(card)
        grid.setContentsMargins(22, 20, 22, 20)
        for row, (name, value) in enumerate(self._info_fields()):
            label = QLabel(name)
            label.setObjectName("muted")
            data = QLabel(value)
            data.setWordWrap(True)
            data.setStyleSheet("font-weight: 600;")
            grid.addWidget(label, row, 0)
            grid.addWidget(data, row, 1)
        grid.setColumnStretch(1, 1)
        layout.addWidget(card)
        layout.addStretch()
        return page

    def _info_fields(self) -> tuple[tuple[str, str], ...]:
        fields = [
            ("项目", self.model.project_name), ("路线范围", self.model.route_range),
            ("运行状态", self.model.run_status), ("当前智能体", self.model.active_agent),
            ("当前设计对象", self.model.current_scope), ("运行 ID", self.form.thread_id),
            ("配置快照", Path(self.current_snapshot).name if self.current_snapshot else "尚未生成"),
        ]
        if not self.paper_mode:
            fields.append(("输出目录", self.form.output_dir or "尚未设置"))
        return tuple(fields)

    def _switch_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        if index == 3:
            self.refresh_artifacts()

    def show_page(self, name: str) -> None:
        pages = {"overview": 0, "task": 1, "review": 2, "results": 3, "qa": 4, "info": 5}
        if name not in pages:
            raise ValueError(f"unknown workbench page: {name}")
        self._switch_page(pages[name])

    def _read_form(self) -> ModernGuiForm:
        return ModernGuiForm(
            base_config_path=self.form_edits["base_config_path"].text().strip(),
            user_request=self.task_request.toPlainText().strip(),
            data_path=self.form_edits["data_path"].text().strip(),
            drawing_path=self.form_edits["drawing_path"].text().strip(),
            output_dir=self.form_edits["output_dir"].text().strip(),
            file_prefix=self.prefix_edit.text().strip(), thread_id=self.thread_edit.text().strip(),
            max_revision_rounds=self.revision_spin.value(),
            max_check_revision_rounds=self.check_revision_spin.value(),
            code_rag_enabled=self.rag_check.isChecked(),
            external_authorized=self.authorization_check.isChecked(),
        )

    def _apply_form(self, form: ModernGuiForm) -> None:
        self.form = form
        self.task_request.setPlainText(form.user_request)
        for key in ("base_config_path", "data_path", "drawing_path", "output_dir"):
            self.form_edits[key].setText(str(getattr(form, key)))
        self.prefix_edit.setText(form.file_prefix)
        self.thread_edit.setText(form.thread_id)
        self.revision_spin.setValue(form.max_revision_rounds)
        self.check_revision_spin.setValue(form.max_check_revision_rounds)
        self.rag_check.setChecked(form.code_rag_enabled)
        self.authorization_check.setChecked(form.external_authorized)

    def _choose_path(self, key: str, kind: str) -> None:
        current = self.form_edits[key].text().strip()
        if kind == "dir":
            path = QFileDialog.getExistingDirectory(self, "选择目录", current)
        else:
            file_filter = "CAD 图纸 (*.dxf *.dwg);;全部文件 (*)" if kind == "drawing" else "YAML (*.yaml *.yml);;全部文件 (*)"
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", current, file_filter)
        if not path:
            return
        self.form_edits[key].setText(path)
        if key == "base_config_path":
            try:
                loaded = load_form_from_yaml(path)
                loaded.thread_id = self.thread_edit.text() or new_thread_id()
                loaded.external_authorized = self.authorization_check.isChecked()
                self._apply_form(loaded)
                self._append_run_log("已从基础配置载入公开运行输入。")
            except Exception as exc:
                QMessageBox.warning(self, "配置读取失败", str(exc))

    def _require_authorization(self) -> bool:
        if self.authorization_check.isChecked():
            return True
        QMessageBox.warning(self, "需要授权", "请先勾选外部模型发送授权。任务描述或成果摘要可能发送到配置中的模型服务。")
        return False

    def start_task(self) -> None:
        if self.busy or not self._require_authorization():
            return
        form = self._read_form()
        if str(self.current_result.get("task_status") or "") in {"completed", "completed_with_accepted_risks"}:
            form.thread_id = new_thread_id()
            self.thread_edit.setText(form.thread_id)
        try:
            config = form.to_run_config()
            config.validate()
            self.form_state_saver(form)
        except Exception as exc:
            QMessageBox.warning(self, "配置不完整", str(exc))
            return
        self.form = form
        self.current_interrupt = None
        self.current_snapshot = ""
        self.stop_requested = False
        self._agent_log_offset = 0
        self._append_run_log(f"启动 Graph V2：{form.thread_id}")
        self._set_busy(True, "任务已提交，正在执行…")
        self.operation_dispatcher("start", lambda: self.controller.start(config, base_config_path=form.base_config_path), self._handle_operation)

    def resume_task(self, action: str) -> None:
        if self.busy:
            self._append_run_log(
                f"任务仍在后台运行，复核动作 {ACTION_LABELS.get(action, action)} 未提交："
                "请等待当前步骤返回后再操作。"
            )
            QMessageBox.information(self, "任务正在运行", "当前任务仍在后台运行，请等待其返回后再提交人工复核动作。")
            return
        if not self.current_interrupt:
            self._append_run_log(f"当前没有待处理的人工复核，已忽略动作 {ACTION_LABELS.get(action, action)}。")
            return
        offered = self.current_interrupt.get("available_actions") or []
        self._set_busy(True, f"正在提交：{ACTION_LABELS.get(action, action)}")
        self.operation_dispatcher(
            "resume",
            lambda: self.controller.resume(
                thread_id=str(self.current_result.get("thread_id") or self.form.thread_id),
                config_snapshot_path=self.current_snapshot, action=action,
                feedback=self.review_feedback.toPlainText().strip(),
                extra_rounds=self.extra_rounds_spin.value(), available_actions=offered,
            ),
            self._handle_operation,
        )

    def request_stop(self) -> None:
        if not self.busy and not self.current_interrupt:
            return
        answer = QMessageBox.question(self, "安全终止", "任务会在下一个安全点提交终止动作，已生成成果和 checkpoint 将保留。继续吗？")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.stop_requested = True
        self.stop_button.setEnabled(False)
        self._append_run_log("已请求安全终止，等待当前步骤结束。")
        if self.current_interrupt and not self.busy:
            self.stop_requested = False
            self.resume_task("abort")

    def _dispatch_async(self, name: str, operation: Callable[[], dict[str, Any]], callback: Callable[[OperationResult], None]) -> None:
        del callback
        def worker() -> None:
            self._operation_bus.finished.emit(run_operation(name, operation))
        threading.Thread(target=worker, name=f"modern-gui-{name}", daemon=True).start()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.busy = busy
        self.start_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy and not self.stop_requested)
        self.chat_send_button.setEnabled(not busy)
        self._refresh_review_buttons()
        if message:
            self.run_status_label.setText(message)

    def _refresh_review_buttons(self) -> None:
        """后台运行期间禁用复核按钮，避免"看起来可点、点了没反应"。"""
        enabled = bool(self.current_snapshot) and not self.busy
        for button in self.review_action_buttons:
            button.setEnabled(enabled)

    def _handle_operation(self, result: OperationResult) -> None:
        self._set_busy(False)
        self.chat_view.clear_pending()
        if result.error:
            message = f"{result.error_type}: {result.error}" if result.error_type else result.error
            self._append_run_log(f"{result.name} 失败：{message}")
            if result.name in {"answer", "assess"}:
                self._append_chat("system", f"本次操作未完成：{message}")
                self._switch_page(4)
            QMessageBox.critical(self, "操作失败", message)
            return
        payload = dict(result.payload)
        if result.name in {"start", "resume"}:
            self._handle_run_result(payload)
        elif result.name == "answer":
            self.chat_session.append("assistant", str(payload.get("answer") or "问答服务未返回正文。"))
            self._append_chat("assistant", format_answer(payload))
            self._switch_page(4)
        elif result.name == "assess":
            text = str(payload.get("display_text") or payload.get("overall_conclusion") or json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            self._append_chat("assessment", "整体评估\n\n" + text)
            self._switch_page(4)
        elif result.name == "gen_views":
            generated = [name for name, path in payload.items() if path]
            if generated:
                self._append_run_log(f"已生成 {len(generated)} 张目录视图：" + "、".join(generated))
            else:
                self._append_run_log("目录视图未生成：缺少布跨方案或坐标文件。")
            self.refresh_artifacts()
            self._switch_page(3)

    def _handle_run_result(self, result: dict[str, Any]) -> None:
        self.current_result = result
        self.current_snapshot = str(result.get("config_snapshot_path") or self.current_snapshot)
        self.current_interrupt = extract_interrupt(result)
        self.run_status_label.setText(str(result.get("message") or result.get("task_status") or "运行已返回"))
        summary = {key: result.get(key) for key in ("task_status", "message", "error", "thread_id", "checkpoint_path") if result.get(key) is not None}
        self._append_run_log(json.dumps(summary, ensure_ascii=False, indent=2))
        self._render_interrupt(self.current_interrupt)
        self._update_model_from_result(result)
        self.refresh_artifacts()
        if self.stop_requested and self.current_interrupt:
            self.stop_requested = False
            self.resume_task("abort")
        elif self.current_interrupt:
            self._switch_page(2)

    def _current_user_request(self) -> str:
        """任务描述以界面输入框为准，输入框还没建好时退回表单里的值。"""
        widget = getattr(self, "task_request", None)
        if widget is not None:
            return widget.toPlainText().strip() or self.form.user_request
        return self.form.user_request

    def _resolve_route_range(self) -> str:
        """顶栏路线范围。

        必须每次刷新都重算：模型层只在构造时收一次值，若沿用上一次的结果，
        一旦开局是占位文案，开始运行后也不会自己变过来。空串交给模型层写占位文案。
        """
        if self._route_range_override:
            return self._route_range_override
        return route_range_from_request(self._current_user_request())

    def _update_model_from_result(self, result: dict[str, Any]) -> None:
        stage_by_agent = {
            "InitialDesignAgent": "initial_design", "LayoutRevisionAgent": "layout_revision",
            "StructuralDesignAgent": "structural_design", "ModelingCheckAgent": "modeling_check",
        }
        progress = snapshot_run_progress(self.form.output_dir) if self.form.output_dir else {}
        progress.update(
            {
                "running": self.busy,
                "active_stage": stage_by_agent.get(str(result.get("active_agent") or "")),
                "stage_states": infer_stage_states(result),
                "current_scope": result.get("current_scope") or progress.get("current_scope"),
                "recent_log": result.get("message") or progress.get("recent_log") or "",
            }
        )
        self.model = build_dashboard_model(
            progress,
            drawings=load_drawing_catalog(self.form.output_dir) if self.form.output_dir else (),
            project_name=self.model.project_name, route_range=self._resolve_route_range(),
            review=self.current_interrupt,
        )
        self._refresh_overview_page()

    def refresh_progress(self) -> None:
        if not self.busy or not self.form.output_dir:
            return
        progress = snapshot_run_progress(self.form.output_dir)
        self.model = build_dashboard_model(
            progress, drawings=load_drawing_catalog(self.form.output_dir),
            project_name=self.model.project_name, route_range=self._resolve_route_range(),
            review=self.current_interrupt,
        )
        self._refresh_overview_page()
        recent = str(progress.get("recent_log") or "")
        if recent and recent not in self.run_log.toPlainText()[-600:]:
            self._append_run_log(recent)
        self._refresh_action_log()

    def _refresh_action_log(self) -> None:
        log_path = Path(self.form.output_dir) / "logs" / "agent_run_log.json"
        if not log_path.is_file():
            return
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
            events = payload.get("events") or []
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(events, list):
            return
        for event in events[self._agent_log_offset :]:
            if isinstance(event, dict):
                line = format_run_event(event)
                if line:
                    self._append_run_log(line)
        self._agent_log_offset = len(events)

    def _refresh_overview_page(self) -> None:
        current = self.pages.currentWidget()
        old = self.pages.widget(0)
        replacement = self._scroll_page(self._build_overview())
        self.pages.removeWidget(old)
        self.pages.insertWidget(0, replacement)
        old.deleteLater()
        # QStackedWidget 在移除当前页时会自动切到下一个页（任务启动页），
        # 运行中每 2 秒刷新会把停留在总览页的用户弹走；这里恢复刷新前的页面。
        if current is old:
            self.pages.setCurrentWidget(replacement)
        else:
            self.pages.setCurrentWidget(current)
        self.top_project_name.setText(self.model.project_name)
        self.top_route_range.setText(self.model.route_range)
        while self.top_status_host.count():
            item = self.top_status_host.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.top_status_host.addWidget(self._status_pill())

    def _render_interrupt(self, review: dict[str, Any] | None) -> None:
        while self.review_actions_layout.count():
            item = self.review_actions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.review_action_buttons = []
        if not review:
            self.review_title.setText("当前没有等待处理的人工复核")
            self.review_message.setText("任务运行到人工判断节点后，原因、失败对象和可用动作会显示在这里。")
            self.review_details.setPlainText("尚无复核上下文。")
            return
        self.review_title.setText("需要工程师判断")
        self.review_message.setText(str(review.get("message") or "当前阶段需要选择后续处理方式。"))
        self.review_details.setPlainText(json.dumps(review, ensure_ascii=False, indent=2, default=str))
        for action in review.get("available_actions") or []:
            button = QPushButton(ACTION_LABELS.get(str(action), str(action)))
            button.setObjectName("dangerButton" if action == "abort" else "secondaryButton" if str(action).startswith("accept") else "primaryButton")
            button.clicked.connect(lambda checked=False, value=str(action): self.resume_task(value))
            button.setEnabled(bool(self.current_snapshot) and not self.busy)
            self.review_actions_layout.addWidget(button)
            self.review_action_buttons.append(button)
        self.review_actions_layout.addStretch()

    def refresh_artifacts(self) -> None:
        output_dir = self.form_edits.get("output_dir").text().strip() if hasattr(self, "form_edits") else self.form.output_dir
        self.form.output_dir = output_dir
        try:
            self.artifact_rows = build_artifact_rows(scan_result_catalog(output_dir)) if output_dir else ()
        except Exception as exc:
            self.artifact_rows = ()
            self.artifact_info.setPlainText(f"成果扫描失败：{exc}")
        self._filter_artifacts()

    @staticmethod
    def _generate_catalog_views_impl(output_dir: str) -> dict[str, Any]:
        from bridge_agents.initial_visualizer import build_catalog_views

        return dict(build_catalog_views(output_dir))

    def generate_catalog_views(self) -> None:
        output_dir = self.form_edits["output_dir"].text().strip()
        if not output_dir or not Path(output_dir).is_dir():
            QMessageBox.warning(self, "缺少输出目录", "请先设置有效的输出目录。")
            return
        self._set_busy(True, "正在生成初始与最终方案视图…")
        self.operation_dispatcher(
            "gen_views",
            lambda: self.catalog_view_builder(output_dir),
            self._handle_operation,
        )

    def _filter_artifacts(self) -> None:
        stage = str(self.stage_filter.currentData() or "")
        self.visible_artifacts = [row for row in self.artifact_rows if not stage or row.stage == stage]
        self.artifact_table.setRowCount(len(self.visible_artifacts))
        for index, row in enumerate(self.visible_artifacts):
            self.artifact_table.setItem(index, 0, QTableWidgetItem(row.stage_title))
            self.artifact_table.setItem(index, 1, QTableWidgetItem(row.title))
            self.artifact_table.setItem(index, 2, QTableWidgetItem(row.design_group or "—"))
        self.artifact_table.resizeColumnsToContents()
        if self.visible_artifacts:
            self.artifact_table.selectRow(0)
        else:
            self.selected_artifact = None
            self.preview_title.setText("尚未发现成果")
            self.preview_view.show_message("检查输出目录，或等待当前阶段完成。")

    def _show_selected_artifact(self) -> None:
        index = self.artifact_table.currentRow()
        if index < 0 or index >= len(self.visible_artifacts):
            return
        row = self.visible_artifacts[index]
        self.selected_artifact = row
        self.preview_title.setText(row.title)
        info = [f"阶段：{row.stage_title}", f"来源：{row.source or row.file_path}"]
        if row.design_group:
            info.append(f"设计组：{row.design_group}")
        if row.diagnostics_text:
            info.append(row.diagnostics_text)
        self.artifact_info.setPlainText("\n".join(info))
        preview_path = row.preview_path or (row.file_path if Path(row.file_path).suffix.lower() == ".svg" else "")
        loaded = False
        if preview_path and Path(preview_path).is_file():
            loaded = self.preview_view.load_file(preview_path)
        if not loaded:
            self.preview_view.show_message("该成果没有窗口内预览，可使用“打开成果”。")

    def _on_preview_zoom_changed(self, zoom: float) -> None:
        if hasattr(self, "preview_zoom_label"):
            self.preview_zoom_label.setText(f"{zoom * 100:.0f}%")

    def fit_preview(self) -> None:
        if hasattr(self, "preview_view"):
            self.preview_view.fit_to_view()

    @staticmethod
    def _open_path(path: Path) -> None:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_selected_artifact(self) -> None:
        if not self.selected_artifact:
            return
        path = Path(self.selected_artifact.file_path)
        if path.exists():
            self._open_path(path)
        else:
            QMessageBox.warning(self, "成果不存在", str(path))

    def open_selected_artifact_dir(self) -> None:
        if self.selected_artifact:
            path = Path(self.selected_artifact.file_path).parent
            if path.is_dir():
                self._open_path(path)

    def open_output_dir(self) -> None:
        path = Path(self.form_edits["output_dir"].text().strip() or ".").resolve()
        if path.is_dir():
            self._open_path(path)
        else:
            QMessageBox.warning(self, "目录不存在", str(path))

    def _ensure_snapshot(self) -> bool:
        if self.current_snapshot and Path(self.current_snapshot).is_file():
            return True
        candidate = find_latest_snapshot(self.form_edits["output_dir"].text().strip(), self.form.thread_id)
        if candidate:
            self.current_snapshot = str(candidate)
            return True
        QMessageBox.warning(self, "缺少运行配置", "请先启动任务，或选择包含 run_configs 配置快照的输出目录。")
        return False

    def send_question(self) -> None:
        if self.busy or not self._require_authorization() or not self._ensure_snapshot():
            return
        question = self.chat_input.toPlainText().strip()
        if not question:
            return
        self.chat_input.clear()
        self.chat_session.append("user", question)
        self._append_chat("user", question)
        self._set_busy(True, "正在生成专业回答…")
        self.chat_view.show_pending("正在检索项目成果与规范证据…")
        self.operation_dispatcher(
            "answer",
            lambda: self.controller.answer(
                question, output_dir=self.form_edits["output_dir"].text().strip(),
                config_snapshot_path=self.current_snapshot, history=self.chat_session.context(),
            ),
            self._handle_operation,
        )

    def assess_results(self) -> None:
        if self.busy or not self._require_authorization() or not self._ensure_snapshot():
            return
        self._set_busy(True, "正在执行成果整体评估…")
        self.chat_view.show_pending("正在检索项目成果与规范证据，生成整体评估…")
        self.operation_dispatcher(
            "assess",
            lambda: self.controller.assess(
                output_dir=self.form_edits["output_dir"].text().strip(),
                config_snapshot_path=self.current_snapshot,
            ),
            self._handle_operation,
        )

    def clear_chat(self) -> None:
        self.chat_session.clear()
        self.chat_view.clear()
        self._append_chat("system", "对话已清空，后续问题不会携带此前上下文。")

    def _append_chat(self, role: str, text: str) -> None:
        self.chat_view.append_message(role, text)

    def _append_run_log(self, message: str) -> None:
        if hasattr(self, "run_log"):
            self.run_log.appendPlainText(f"[{datetime.now():%H:%M:%S}] {message}")
