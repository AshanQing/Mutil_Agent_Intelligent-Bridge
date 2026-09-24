from __future__ import annotations

import argparse
import json
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml

from bridge_agents.gui_controller import (
    HUMAN_REVIEW_ACTIONS,
    GraphV2Controller,
    GuiRunConfig,
    extract_interrupt,
    infer_stage_states,
    load_drawing_catalog,
    snapshot_run_progress,
)
from bridge_agents.gui_display import enable_windows_high_dpi
from bridge_agents.initial_visualizer import build_catalog_views
from bridge_agents.preview_images import PreviewUnavailableError, image_size
from bridge_agents.result_catalog import (
    STAGE_TITLES,
    ResultArtifact,
    display_action,
    group_by_stage,
    scan as scan_result_catalog,
)
from bridge_agents.svg_preview import render_svg_to_png
from PIL import Image, ImageTk


APP_TITLE = "桥梁设计控制台 · Graph V2"
GUI_STATE_VERSION = 1
GUI_STATE_KEYS = (
    "base_config",
    "data_path",
    "drawing_path",
    "output_dir",
    "file_prefix",
    "max_revision",
    "max_check_revision",
    "code_rag",
    "user_request",
)
ACTION_LABELS = {
    "continue_revision": "继续布跨修正",
    "accept_and_continue": "接受布跨并继续",
    "retry_coordinator": "重新交给协调器",
    "retry_failed_tasks": "重试失败任务",
    "accept_partial_and_continue": "接受部分成果",
    "continue_modeling_revision": "继续验算返修",
    "accept_check_and_finish": "接受验算风险并结束",
    "abort": "终止任务",
}
STAGE_LABELS = {
    "initial_design": "初始布跨",
    "layout_revision": "布跨修正",
    "structural_design": "结构设计",
    "modeling_check": "建模验算",
    "final_output": "最终交付",
}
STATE_COLORS = {
    "pending": ("#f2f2f7", "#8e8e93"),
    "running": ("#e0f0ff", "#0066cc"),
    "completed": ("#d9f5e0", "#1e8e3e"),
    "review": ("#fff1d6", "#c77800"),
    "failed": ("#fde3e2", "#d93025"),
}

# 运行过程时间线的中文命名（agent_run_log.json 的 agent / action 名）
AGENT_LABELS = {
    "DesignCoordinatorAgent": "协调器",
    "InitialDesignAgent": "初始布跨设计",
    "LayoutRevisionAgent": "布跨修正",
    "StructuralDesignAgent": "结构设计",
    "ModelingCheckAgent": "建模验算",
}
ACTION_LABELS_RUN = {
    "compute_pier_groups": "设计组归并计算",
    "dimension_design": "下部结构尺寸设计",
    "reinforcement_design": "盖梁/墩柱配筋设计",
    "run_capacity_check": "承载力批量验算",
    "generate_initial_layout": "生成初始布跨方案",
    "detect_collision": "碰撞检测",
    "layout_collision_review": "碰撞复核",
    "extract_obstacles": "障碍物语义提取",
    "drawing_mask_preprocess": "图纸预处理",
    "load_route_data": "加载路线资料",
    "run_capacity_reexamine": "承载力复验",
    "batch_modeling_check": "批量建模验算",
}


def blueprint_background_path() -> Path:
    """Return the packaged blueprint background independently of the working directory."""
    return Path(__file__).resolve().parent / "assets" / "gui" / "bridge_blueprint_background.png"


def gui_state_path() -> Path:
    """Return the project-local, git-ignored GUI form state file."""
    return Path(__file__).resolve().parent / "output" / ".graph_v2_gui_state.json"


def load_gui_form_state(path: str | Path | None = None) -> Dict[str, Any]:
    state_path = Path(path) if path is not None else gui_state_path()
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != GUI_STATE_VERSION:
        return {}
    values = payload.get("values")
    if not isinstance(values, dict):
        return {}
    return {key: values[key] for key in GUI_STATE_KEYS if key in values}


def save_gui_form_state(values: Dict[str, Any], path: str | Path | None = None) -> Path:
    state_path = Path(path) if path is not None else gui_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": GUI_STATE_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "values": {key: values[key] for key in GUI_STATE_KEYS if key in values},
    }
    temp_path = state_path.with_name(f"{state_path.name}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(state_path)
    return state_path


def _pick(settings: Dict[str, Any], section: str, key: str, default: Any = "") -> Any:
    value = settings.get(section)
    if isinstance(value, dict) and value.get(key) not in (None, ""):
        return value[key]
    return default


class BridgeDesignConsole(tk.Tk):
    def __init__(self, *, controller: GraphV2Controller | None = None) -> None:
        super().__init__()
        self.controller = controller or GraphV2Controller()
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.current_result: Dict[str, Any] = {}
        self.current_interrupt: Dict[str, Any] | None = None
        self.current_snapshot = ""
        self.busy = False
        self._stop_requested = False  # 用户请求终止运行：在下个安全点以 abort 停止
        self._last_progress_log = ""  # 上次已显示的进度日志，避免重复
        self._progress_polling = False  # 防止轮询定时器叠加
        self.review_buttons: List[Any] = []  # 人工复核动作按钮，busy 期间需要禁用

        self.title(APP_TITLE)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = max(1040, int(screen_width * 0.90))
        window_height = max(700, int(screen_height * 0.86))
        self.geometry(f"{window_width}x{window_height}")
        self.minsize(min(1120, int(screen_width * 0.82)), min(720, int(screen_height * 0.76)))
        self.configure(bg="#f2f2f7")
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._configure_style()
        self._build_header()
        self._build_body()
        self._restore_form_state()
        self._refresh_drawings()
        self._set_stage_states({stage: "pending" for stage in STAGE_LABELS})
        self.after(120, self._poll_events)
        self.after(2000, self._poll_progress)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        # ---- 基础层级：iOS 分组背景 + 白色面板/卡片 ----
        style.configure("App.TFrame", background="#f2f2f7")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Card.TFrame", background="#ffffff")
        # ---- 文字层级（近似 SF 中文回退：微软雅黑）----
        style.configure("TLabel", background="#ffffff", foreground="#1c1c1e",
                        font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background="#f2f2f7", foreground="#1c1c1e",
                        font=("Microsoft YaHei UI", 24, "bold"))
        style.configure("Sub.TLabel", background="#f2f2f7", foreground="#0a84ff",
                        font=("Consolas", 10, "bold"))
        style.configure("PanelTitle.TLabel", background="#ffffff", foreground="#1c1c1e",
                        font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#8e8e93",
                        font=("Microsoft YaHei UI", 9))
        # ---- 输入（白底 + 浅灰描边，聚焦变主蓝）----
        style.configure("TEntry", fieldbackground="#ffffff", foreground="#1c1c1e",
                        bordercolor="#d1d1d6", lightcolor="#d1d1d6", darkcolor="#d1d1d6",
                        padding=8, insertcolor="#007aff", readonlybackground="#f2f2f7")
        style.map("TEntry", bordercolor=[("focus", "#007aff")],
                  lightcolor=[("focus", "#007aff")], darkcolor=[("focus", "#007aff")])
        style.configure("TSpinbox", fieldbackground="#ffffff", foreground="#1c1c1e",
                        bordercolor="#d1d1d6", lightcolor="#d1d1d6", darkcolor="#d1d1d6",
                        padding=6, insertcolor="#007aff")
        style.configure("TCheckbutton", background="#ffffff", foreground="#1c1c1e",
                        font=("Microsoft YaHei UI", 10))
        style.map("TCheckbutton", background=[("active", "#ffffff")])
        # ---- 按钮：iOS 填充蓝 / 灰色按钮 / 警示橙 / 危险红，加大点按高度 ----
        style.configure("Primary.TButton", background="#007aff", foreground="#ffffff",
                        padding=(20, 11), font=("Microsoft YaHei UI", 10, "bold"),
                        borderwidth=0, focusthickness=0)
        style.map("Primary.TButton",
                  background=[("active", "#0066d6"), ("pressed", "#0062cc"), ("disabled", "#8ec5ff")])
        style.configure("Secondary.TButton", background="#e5e5ea", foreground="#2c2c2e",
                        padding=(15, 10), font=("Microsoft YaHei UI", 10), borderwidth=0, focusthickness=0)
        style.map("Secondary.TButton",
                  background=[("active", "#d9d9de"), ("pressed", "#cfcfd6")])
        style.configure("Warning.TButton", background="#ff9f0a", foreground="#ffffff",
                        padding=(15, 10), font=("Microsoft YaHei UI", 10, "bold"), borderwidth=0)
        style.map("Warning.TButton", background=[("active", "#f09100"), ("pressed", "#e08600")])
        style.configure("Danger.TButton", background="#ff3b30", foreground="#ffffff",
                        padding=(15, 10), font=("Microsoft YaHei UI", 10, "bold"), borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#f22d21"), ("pressed", "#e31f13")])
        # ---- 页签：iOS 分段式——白色背景上，选中蓝字加粗 ----
        style.configure("TNotebook", background="#f2f2f7", borderwidth=0)
        style.configure("TNotebook.Tab", background="#f2f2f7", foreground="#6c6c70",
                        padding=(22, 11), font=("Microsoft YaHei UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", "#ffffff"), ("active", "#eaeaf0")],
                  foreground=[("selected", "#007aff")],
                  font=[("selected", ("Microsoft YaHei UI", 10, "bold"))])
        # ---- 表格 ----
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff",
                        foreground="#2c2c2e", rowheight=34, borderwidth=0)
        style.configure("Treeview.Heading", background="#f2f2f7", foreground="#3a3a3c",
                        font=("Microsoft YaHei UI", 9, "bold"), padding=(8, 8))
        style.map("Treeview", background=[("selected", "#d6e9ff")],
                  foreground=[("selected", "#1c1c1e")])
        # ---- 滚动条：细圆杆 iOS 感 ----
        style.configure("Vertical.TScrollbar", background="#d1d1d6", troughcolor="#f2f2f7",
                        bordercolor="#f2f2f7", arrowcolor="#8e8e93", width=10, relief="flat")
        style.map("Vertical.TScrollbar", background=[("active", "#b8b8bf")])

    def _build_header(self) -> None:
        header = ttk.Frame(self, style="App.TFrame", padding=(24, 18, 24, 12))
        header.pack(fill="x")
        ttk.Label(header, text="桥梁设计控制台", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="GRAPH V2 · CONSOLE", style="Sub.TLabel").pack(side="left", padx=(16, 0), pady=(10, 0))
        self.connection_label = ttk.Label(header, text="● 就绪", style="Sub.TLabel")
        self.connection_label.pack(side="right", pady=(9, 0))

    def _build_body(self) -> None:
        body = ttk.Frame(self, style="App.TFrame", padding=(20, 0, 20, 20))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=380)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.config_panel = ttk.Frame(body, style="Panel.TFrame", padding=18)
        self.config_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._build_config_panel(self.config_panel)

        right = ttk.Frame(body, style="Panel.TFrame", padding=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.tabs = ttk.Notebook(right)
        self.tabs.grid(row=0, column=0, sticky="nsew")

        self.run_tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=18)
        self.review_tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=18)
        self.drawings_tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=18)
        self.qa_tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=18)
        self.result_tab = ttk.Frame(self.tabs, style="Panel.TFrame", padding=18)
        self.tabs.add(self.run_tab, text="运行轨道")
        self.tabs.add(self.review_tab, text="人工复核")
        self.tabs.add(self.drawings_tab, text="绘图成果")
        self.tabs.add(self.result_tab, text="成果中心")
        self.tabs.add(self.qa_tab, text="成果问答")
        self._build_run_tab()
        self._build_review_tab()
        self._build_drawings_tab()
        self._build_result_center_tab()
        self._build_qa_tab()

    def _build_config_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="任务启动", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(parent, text="仅填写原始输入；中间成果由输出目录自动恢复。", style="Hint.TLabel").pack(anchor="w", pady=(3, 15))

        self.vars = {
            "base_config": tk.StringVar(value="config/settings.yaml"),
            "data_path": tk.StringVar(),
            "drawing_path": tk.StringVar(),
            "output_dir": tk.StringVar(),
            "file_prefix": tk.StringVar(),
            "thread_id": tk.StringVar(value=f"bridge-{datetime.now():%Y%m%d-%H%M}-{uuid4().hex[:5]}"),
            "max_revision": tk.IntVar(value=3),
            "max_check_revision": tk.IntVar(value=2),
            "code_rag": tk.BooleanVar(value=False),
            "external_authorized": tk.BooleanVar(value=False),
        }
        self.request_text = tk.Text(parent, height=5, wrap="word", bg="#f6f6f8", fg="#1c1c1e", relief="flat", padx=8, pady=8, font=("Microsoft YaHei UI", 10))
        self._label("任务描述")
        self.request_text.pack(fill="x", pady=(0, 11))
        self._path_field("基础配置", "base_config", self._choose_base_config)
        self._path_field("路线数据目录", "data_path", lambda: self._choose_dir("data_path"))
        self._path_field("输入 DXF 图纸", "drawing_path", self._choose_drawing)
        self._path_field("输出目录", "output_dir", lambda: self._choose_dir("output_dir"))
        self._entry_field("线路文件前缀", "file_prefix")
        self._thread_field()

        rounds = ttk.Frame(parent, style="Panel.TFrame")
        rounds.pack(fill="x", pady=(2, 8))
        for column, (label, key) in enumerate((("布跨修正轮数", "max_revision"), ("验算返修轮数", "max_check_revision"))):
            box = ttk.Frame(rounds, style="Panel.TFrame")
            box.grid(row=0, column=column, sticky="ew", padx=(0, 8) if column == 0 else (8, 0))
            rounds.columnconfigure(column, weight=1)
            ttk.Label(box, text=label, style="Hint.TLabel").pack(anchor="w")
            ttk.Spinbox(box, from_=1, to=20, textvariable=self.vars[key], width=8).pack(fill="x", pady=(3, 0))

        ttk.Checkbutton(parent, text="启用规范 RAG（设计后评估/问答）", variable=self.vars["code_rag"]).pack(anchor="w", pady=(5, 2))
        ttk.Checkbutton(parent, text="允许向配置中的外部模型发送任务与成果摘要", variable=self.vars["external_authorized"]).pack(anchor="w", pady=(2, 12))

        buttons = ttk.Frame(parent, style="Panel.TFrame")
        buttons.pack(fill="x", pady=(6, 0))
        self.start_button = ttk.Button(buttons, text="启动 Graph V2", style="Primary.TButton", command=self._start)
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(buttons, text="终止运行", style="Danger.TButton",
                                      command=self._request_stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="打开输出", style="Secondary.TButton", command=self._open_output).pack(side="left", padx=(8, 0))

    def _label(self, text: str) -> None:
        ttk.Label(self.config_panel, text=text, style="Hint.TLabel").pack(anchor="w", pady=(0, 3))

    def _entry_field(self, label: str, key: str) -> None:
        self._label(label)
        ttk.Entry(self.config_panel, textvariable=self.vars[key]).pack(fill="x", pady=(0, 10))

    def _path_field(self, label: str, key: str, command: Callable[[], None]) -> None:
        self._label(label)
        row = ttk.Frame(self.config_panel, style="Panel.TFrame")
        row.pack(fill="x", pady=(0, 10))
        ttk.Entry(row, textvariable=self.vars[key]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="选择", style="Secondary.TButton", command=command).pack(side="left", padx=(6, 0))

    def _thread_field(self) -> None:
        """运行 ID 只读展示：自动生成、随运行状态推进，用户不直接编辑。"""
        self._label("运行 ID（自动生成 · 用于断点恢复）")
        row = ttk.Frame(self.config_panel, style="Panel.TFrame")
        row.pack(fill="x", pady=(0, 10))
        readonly_entry = ttk.Entry(row, textvariable=self.vars["thread_id"], state="readonly")
        readonly_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(
            row,
            text="复制",
            style="Secondary.TButton",
            command=lambda: (self.clipboard_clear(), self.clipboard_append(self.vars["thread_id"].get())),
        ).pack(side="left", padx=(6, 0))

    def _build_run_tab(self) -> None:
        hero = tk.Canvas(self.run_tab, height=250, bg="#ffffff", highlightthickness=1, highlightbackground="#e5e5ea")
        hero.pack(fill="x", pady=(0, 16))
        asset = blueprint_background_path()
        if asset.is_file():
            source = tk.PhotoImage(file=str(asset))
            self.blueprint_image = source.subsample(4, 4)
            blueprint_item = hero.create_image(1, 1, image=self.blueprint_image, anchor="ne")
            hero.bind("<Configure>", lambda event: hero.coords(blueprint_item, event.width - 1, 1))
        hero.create_text(24, 26, text="设计阶段轨道", anchor="nw", fill="#1c1c1e", font=("Microsoft YaHei UI", 17, "bold"))
        hero.create_text(25, 92, text="GRAPH V2  ·  COORDINATED BRIDGE DESIGN", anchor="nw", fill="#0a84ff", font=("Consolas", 10, "bold"))
        hero.create_text(25, 145, text="每个专业阶段完成后返回协调器决策；橙色节点表示等待人工接管。", anchor="nw", fill="#8e8e93", font=("Microsoft YaHei UI", 10))
        track = ttk.Frame(self.run_tab, style="Panel.TFrame")
        track.pack(fill="x")
        self.stage_cards: Dict[str, tk.Label] = {}
        for index, (stage, label) in enumerate(STAGE_LABELS.items()):
            if index:
                tk.Label(track, text="━━", bg="#ffffff", fg="#d1d1d6", font=("Consolas", 15, "bold")).pack(side="left", padx=2)
            card = tk.Label(track, text=f"{index + 1:02d}\n{label}\n待开始", width=13, height=4, bg="#f2f2f7", fg="#8e8e93", relief="flat", font=("Microsoft YaHei UI", 10, "bold"))
            card.pack(side="left", fill="x", expand=True)
            self.stage_cards[stage] = card

        meta = ttk.Frame(self.run_tab, style="Card.TFrame", padding=14)
        meta.pack(fill="x", pady=(18, 12))
        self.status_var = tk.StringVar(value="等待启动")
        self.agent_var = tk.StringVar(value="当前 Agent：—")
        self.thread_var = tk.StringVar(value="运行 ID：—")
        ttk.Label(meta, textvariable=self.status_var, style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(meta, textvariable=self.agent_var, style="Hint.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Label(meta, textvariable=self.thread_var, style="Hint.TLabel").pack(anchor="w")

        ttk.Label(self.run_tab, text="运行记录", style="PanelTitle.TLabel").pack(anchor="w", pady=(4, 6))
        log_wrap = ttk.Frame(self.run_tab, style="Panel.TFrame")
        log_wrap.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_wrap, wrap="word", bg="#f6f6f8", fg="#2c2c2e",
                                insertbackground="#2c2c2e", relief="flat", padx=12, pady=10,
                                font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_wrap, command=self.log_text.yview,
                                   style="Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def _build_review_tab(self) -> None:
        ttk.Label(self.review_tab, text="人工复核", style="PanelTitle.TLabel").pack(anchor="w")
        review_wrap = ttk.Frame(self.review_tab, style="Panel.TFrame")
        review_wrap.pack(fill="both", expand=True, pady=(10, 12))
        self.review_summary = tk.Text(review_wrap, height=15, wrap="word", bg="#f6f6f8",
                                      fg="#2c2c2e", relief="flat", padx=12, pady=10,
                                      font=("Microsoft YaHei UI", 10))
        review_scroll = ttk.Scrollbar(review_wrap, command=self.review_summary.yview,
                                      style="Vertical.TScrollbar")
        self.review_summary.configure(yscrollcommand=review_scroll.set)
        self.review_summary.pack(side="left", fill="both", expand=True)
        review_scroll.pack(side="right", fill="y")
        ttk.Label(self.review_tab, text="补充要求", style="Hint.TLabel").pack(anchor="w")
        self.feedback_text = tk.Text(self.review_tab, height=4, wrap="word", bg="#f6f6f8", fg="#1c1c1e", relief="flat", padx=8, pady=8)
        self.feedback_text.pack(fill="x", pady=(4, 8))
        option_row = ttk.Frame(self.review_tab, style="Panel.TFrame")
        option_row.pack(fill="x")
        ttk.Label(option_row, text="增加轮数", style="Hint.TLabel").pack(side="left")
        self.extra_rounds = tk.IntVar(value=1)
        ttk.Spinbox(option_row, from_=1, to=20, textvariable=self.extra_rounds, width=7).pack(side="left", padx=(8, 0))
        self.review_actions = ttk.Frame(self.review_tab, style="Panel.TFrame")
        self.review_actions.pack(fill="x", pady=(12, 0))
        ttk.Label(self.review_actions, text="流程尚未请求人工复核。", style="Hint.TLabel").pack(anchor="w")

    def _build_qa_tab(self) -> None:
        heading = ttk.Frame(self.qa_tab, style="Panel.TFrame")
        heading.pack(fill="x")
        heading.columnconfigure(0, weight=1)
        copy = ttk.Frame(heading, style="Panel.TFrame")
        copy.grid(row=0, column=0, sticky="nsew")
        ttk.Label(copy, text="成果问答（只读对话）", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            copy,
            text="基于当前输出目录的工程成果与本地规范索引；提问自动带上最近对话，检索仍以本轮问题为准。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        actions = ttk.Frame(heading, style="Panel.TFrame")
        actions.grid(row=0, column=1, sticky="e", padx=(12, 0))
        ttk.Button(actions, text="整体评估", style="Secondary.TButton",
                   command=self._assess).pack(side="left")
        ttk.Button(actions, text="清空对话", style="Secondary.TButton",
                   command=self._clear_chat).pack(side="left", padx=(6, 0))

        self.chat_history: list[Dict[str, str]] = []
        chat_wrap = ttk.Frame(self.qa_tab, style="Panel.TFrame")
        chat_wrap.pack(fill="both", expand=True, pady=(10, 0))
        self.answer_text = tk.Text(
            chat_wrap, wrap="word", state="disabled", bg="#f2f2f7", fg="#1c1c1e",
            relief="flat", padx=16, pady=14, font=("Microsoft YaHei UI", 10),
            spacing1=2, spacing3=4, highlightthickness=0,
        )
        # iOS 消息样式：自己=蓝色气泡右对齐；对方=浅灰气泡左对齐
        self.answer_text.tag_configure("user", background="#007aff", foreground="#ffffff",
                                       justify="right", lmargin1=150, rmargin=16,
                                       spacing1=7, spacing3=7, font=("Microsoft YaHei UI", 10))
        self.answer_text.tag_configure("assistant", background="#e9e9ef", foreground="#1c1c1e",
                                       lmargin1=16, rmargin=150, spacing1=7, spacing3=7)
        self.answer_text.tag_configure("system", foreground="#8e8e93", justify="center",
                                       font=("Microsoft YaHei UI", 9), spacing1=4, spacing3=4)
        self.answer_text.tag_configure("error", foreground="#ff3b30", justify="center")
        chat_scroll = ttk.Scrollbar(chat_wrap, command=self.answer_text.yview,
                                    style="Vertical.TScrollbar")
        self.answer_text.configure(yscrollcommand=chat_scroll.set)
        self.answer_text.pack(side="left", fill="both", expand=True)
        chat_scroll.pack(side="right", fill="y")
        self._append_chat("system", "只读问答区：可追问多轮（例如\"那盖梁呢？\"会沿用前文主题）。"
                                    "项目事实与规范结论仍需按本轮检索的证据引用，Enter 发送，Shift+Enter 换行。")

        input_frame = ttk.Frame(self.qa_tab, style="Panel.TFrame")
        input_frame.pack(fill="x", pady=(8, 0))
        input_frame.columnconfigure(0, weight=1)
        self.chat_input = tk.Text(input_frame, height=3, wrap="word", bg="#ffffff", fg="#1c1c1e",
                                  relief="solid", bd=1, padx=8, pady=8,
                                  font=("Microsoft YaHei UI", 10))
        self.chat_input.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.chat_input.bind("<Return>", self._on_chat_return)
        self.chat_input.bind("<KP_Enter>", self._on_chat_return)
        self.chat_send_button = ttk.Button(input_frame, text="发送", style="Primary.TButton",
                                           command=self._send_chat)
        self.chat_send_button.grid(row=0, column=1, sticky="ns")
        self.chat_input.focus_set()

    def _append_chat(self, kind: str, text: str) -> None:
        if not hasattr(self, "answer_text"):
            return
        self.answer_text.configure(state="normal")
        self.answer_text.insert("end", f"{text}\n\n", kind)
        self.answer_text.configure(state="disabled")
        self.answer_text.see("end")

    def _on_chat_return(self, event: Any) -> str:
        if event.state & 0x0001:  # Shift 按下 → 换行
            return None  # type: ignore[no-any-return]
        self._send_chat()
        return "break"

    def _clear_chat(self) -> None:
        self.chat_history = []
        self.answer_text.configure(state="normal")
        self.answer_text.delete("1.0", "end")
        self.answer_text.configure(state="disabled")
        self._append_chat("system", "对话已清空，新一轮问答不再携带之前的上下文。")
        self.chat_input.focus_set()

    def _send_chat(self) -> None:
        if self.busy or not self._require_external_authorization():
            return
        if not self.current_snapshot:
            messagebox.showwarning("缺少运行配置", "请先启动一次任务，生成本次运行的配置快照。")
            return
        question = self.chat_input.get("1.0", "end").strip()
        if not question:
            return
        self.chat_input.delete("1.0", "end")
        self.chat_history.append({"role": "user", "content": question})
        self._append_chat("user", question)
        # 上下文窗口控制：只带最近 12 轮
        history = self.chat_history[-24:]
        self._run_background(
            "answer",
            lambda: self.controller.answer(
                question,
                output_dir=self.vars["output_dir"].get(),
                config_snapshot_path=self.current_snapshot,
                history=history,
            ),
        )

    def _build_result_center_tab(self) -> None:
        """全阶段成果中心：左阶段成果树、中预览画布、右状态与来源信息。"""
        self.artifacts: list[ResultArtifact] = []
        self._preview_photo = None
        self._catalog_scanned_dir = ""

        heading = ttk.Frame(self.result_tab, style="Panel.TFrame")
        heading.pack(fill="x")
        heading.columnconfigure(0, weight=1)

        copy = ttk.Frame(heading, style="Panel.TFrame")
        copy.grid(row=0, column=0, sticky="nsew")
        ttk.Label(copy, text="全阶段成果中心", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            copy,
            text="按 初步设计 / 布跨修正 / 结构设计 / 建模验算 / 最终交付 浏览全部工程成果；"
            "PNG/JPG 窗口内预览，SVG/SCR 外部打开。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        actions = ttk.Frame(heading, style="Panel.TFrame")
        # 用 grid 固定按钮列宽度（不被左侧文本区挤压，避免按钮文字显示不全）
        actions.grid(row=0, column=1, sticky="e", padx=(12, 0))
        ttk.Button(actions, text="生成目录视图", style="Secondary.TButton",
                   command=self._generate_catalog_views).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="刷新目录", style="Primary.TButton",
                   command=self._refresh_result_center).pack(side="left")
        self.result_hint = ttk.Label(self.result_tab, text="尚未读取成果目录。",
                                     style="Hint.TLabel")
        self.result_hint.pack(anchor="w", pady=(8, 0))

        body = ttk.Frame(self.result_tab, style="Panel.TFrame")
        body.pack(fill="both", expand=True, pady=(8, 0))
        body.columnconfigure(0, weight=0, minsize=250)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=0)

        tree_frame = ttk.Frame(body, style="Card.TFrame")
        tree_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        self.result_tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse")
        self.result_tree.pack(fill="both", expand=True)
        self.result_tree.bind("<<TreeviewSelect>>", self._on_result_select)
        self.result_tree.bind("<Double-1>", lambda _event: self._open_artifact_external())

        # 上部大画布：图片预览 + 滚轮缩放 / 左键拖拽平移 / 双击切换 100%↔适应
        self.preview_canvas = tk.Canvas(
            body, bg="#0B1F3A", highlightthickness=0, cursor="fleur"
        )
        self.preview_canvas.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=(0, 8))
        self._preview_source: Optional[str] = None
        self._preview_zoom = 1.0  # 相对“适应视图”的倍率（1.0 = 全貌）
        self._fit_scale = 1.0
        self._preview_size = (0, 0)
        self.preview_canvas.bind("<MouseWheel>", self._on_canvas_wheel)
        self.preview_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.preview_canvas.bind("<Double-Button-1>", self._on_canvas_double)
        self._show_canvas_hint("选择左侧成果条目查看预览")

        # 底部信息横条：左侧滚动文字信息，右侧预览工具
        bottom = ttk.Frame(body, style="Card.TFrame")
        bottom.grid(row=1, column=1, sticky="nsew")
        bottom.columnconfigure(0, weight=1)
        self.info_text = tk.Text(
            bottom, wrap="word", state="disabled", bg="#f9f9fb", fg="#2c2c2e",
            relief="flat", padx=10, pady=6, font=("Microsoft YaHei UI", 9), height=4,
        )
        self.info_text.grid(row=0, column=0, sticky="nsew")
        info_scroll = ttk.Scrollbar(bottom, command=self.info_text.yview,
                                    style="Vertical.TScrollbar")
        self.info_text.configure(yscrollcommand=info_scroll.set)
        info_scroll.grid(row=0, column=1, sticky="ns")
        tools = ttk.Frame(bottom, style="Card.TFrame")
        tools.grid(row=0, column=2, sticky="ne", padx=(8, 6), pady=4)
        self._zoom_label = ttk.Label(tools, text="适应视图", style="Hint.TLabel")
        self._zoom_label.pack(anchor="e", pady=(0, 2))
        for text, cmd in (
            ("适应", self._preview_fit),
            ("100%", self._preview_actual),
            ("放大", lambda: self._zoom_by(1.5)),
            ("缩小", lambda: self._zoom_by(1 / 1.5)),
        ):
            ttk.Button(tools, text=text, style="Secondary.TButton",
                       command=cmd).pack(fill="x", pady=(1, 0))
        ttk.Button(tools, text="外部打开", style="Secondary.TButton",
                   command=self._open_artifact_external).pack(fill="x", pady=(3, 0))
        ttk.Button(tools, text="打开目录", style="Secondary.TButton",
                   command=self._open_artifact_dir).pack(fill="x", pady=(1, 0))

    def _refresh_result_center(self) -> None:
        if not hasattr(self, "result_tree"):
            return
        output_dir = self.vars["output_dir"].get().strip()
        self._catalog_scanned_dir = output_dir
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self.artifacts = []
        if not output_dir or not os.path.isdir(output_dir):
            self.result_hint.configure(text="请先在左侧设置有效的输出目录。")
            self._show_canvas_hint("尚无输出目录")
            return
        try:
            artifacts = scan_result_catalog(output_dir)
        except Exception as exc:
            self.result_hint.configure(text=f"成果扫描失败：{exc}")
            self._show_canvas_hint("成果扫描失败")
            return
        self.artifacts = artifacts
        grouped = group_by_stage(artifacts)
        total = len(artifacts)
        for stage in STAGE_TITLES:
            items = grouped.get(stage) or []
            stage_iid = self.result_tree.insert(
                "", "end", text=f"{STAGE_TITLES[stage]}（{len(items)}）", tags=("stage",)
            )
            for index, artifact in enumerate(artifacts):
                if artifact.stage != stage:
                    continue
                self.result_tree.insert(
                    stage_iid, "end", text=self._artifact_tree_text(artifact), tags=(str(index),)
                )
        if total:
            self.result_hint.configure(
                text=f"已收录 {total} 项成果。生成目录视图可补充初始/最终布跨平面与纵断示意。"
            )
        else:
            self.result_hint.configure(text="输出目录中暂未发现五阶段成果（可能需要先生成目录视图）。")
            self._show_canvas_hint("暂无可预览成果")
        self._refresh_output_dir_buttons()

    def _artifact_tree_text(self, artifact: ResultArtifact) -> str:
        def clip(text: str, limit: int) -> str:
            text = str(text or "").strip()
            return text if len(text) <= limit else text[: limit - 1] + "…"

        suffix = ""
        if artifact.design_group:
            suffix = f"  [{artifact.design_group}]"
        elif artifact.diagnostics:
            sample = next(iter(artifact.diagnostics.values()))
            suffix = f"  ·  {clip(sample, 24)}"
        mark = "◎ " if display_action(artifact) == "preview" else ("↗ " if artifact.external_only else "· ")
        return f"{mark}{clip(artifact.title, 40)}{suffix}"

    def _selected_artifact_index(self) -> int | None:
        selection = self.result_tree.selection()
        if not selection:
            return None
        tags = self.result_tree.item(selection[0], "tags")
        if not tags or tags[0] == "stage":
            return None
        try:
            return int(tags[0])
        except (TypeError, ValueError):
            return None

    def _selected_artifact(self) -> ResultArtifact | None:
        index = self._selected_artifact_index()
        if index is None or not (0 <= index < len(self.artifacts)):
            return None
        return self.artifacts[index]

    def _on_result_select(self, _event: Any) -> None:
        artifact = self._selected_artifact()
        if artifact is None:
            self._show_canvas_hint("阶段节点：展开查看各阶段成果")
            return
        self._render_artifact_info(artifact)
        action = display_action(artifact)
        if action == "preview":
            source = artifact.preview_path or artifact.file_path or ""
            self._preview_source = source or None
            self._preview_zoom = 1.0
            if self._preview_source:
                self._render_preview()
            else:
                self._show_canvas_hint("无可预览图像文件")
        elif action == "external":
            self._preview_source = None
            self._reset_zoom_label()
            self._show_canvas_hint("该成果为外部文件（SVG/SCR/表格等）：\n用底部按钮在外部程序打开")
        elif action == "missing":
            self._preview_source = None
            self._reset_zoom_label()
            self._show_canvas_hint("文件缺失：\n原路径已不存在或尚未生成")
        else:
            self._preview_source = None
            self._reset_zoom_label()
            self._show_canvas_hint("数据成果：\n双击可在外部打开，底部横条显示成果信息")

    def _render_artifact_info(self, artifact: ResultArtifact) -> None:
        from datetime import datetime

        lines = [
            artifact.title,
            f"阶段：{STAGE_TITLES.get(artifact.stage, artifact.stage)}"
            f" · 类型：{artifact.kind}"
            + (f" · 设计组：{artifact.design_group}" if artifact.design_group else ""),
            f"状态：{artifact.status}",
            f"来源：{artifact.source or artifact.file_path}",
        ]
        source_path = artifact.file_path or artifact.preview_path or ""
        if display_action(artifact) == "preview":
            size = image_size(artifact.preview_path or source_path)
            if size:
                lines.append(f"图像尺寸：{size[0]} × {size[1]} px")
        try:
            mtime = os.path.getmtime(source_path)
            lines.append(f"修改时间：{datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M:%S}")
        except OSError:
            pass
        if artifact.diagnostics:
            snippet = "; ".join(
                f"{key}={value}" for key, value in list(artifact.diagnostics.items())[:3]
            )
            lines.append(f"诊断：{snippet}")
        self._set_info_text("\n".join(lines))

    def _set_info_text(self, content: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", content)
        self.info_text.configure(state="disabled")

    # ------------------------------------------------------------------ #
    # 画布图像预览：适应 / 100% / 缩放 / 平移
    # ------------------------------------------------------------------ #
    def _reset_zoom_label(self) -> None:
        if not hasattr(self, "_zoom_label"):
            return
        self._fit_scale = 1.0
        self._preview_size = (0, 0)
        self._zoom_label.configure(text="适应视图")

    def _update_zoom_label(self) -> None:
        if not hasattr(self, "_zoom_label"):
            return
        zoom_percent = self._fit_scale * self._preview_zoom * 100
        width, height = self._preview_size
        self._zoom_label.configure(
            text=f"显示 {zoom_percent:.0f}%"
            + (f"（原图 {width}×{height}px）" if width and height else "")
        )

    def _render_preview(self) -> None:
        source = self._preview_source
        if not source:
            return
        index = self._selected_artifact_index()
        if index is None:
            return
        canvas_w = max(self.preview_canvas.winfo_width(), 300)
        canvas_h = max(self.preview_canvas.winfo_height(), 260)
        size = image_size(source)
        if size:
            width, height = size
        else:
            width, height = canvas_w, canvas_h
        fit_scale = min(canvas_w / width, canvas_h / height, 1.0)
        fit_scale = fit_scale if fit_scale > 0 else 1.0
        self._fit_scale = fit_scale
        self._preview_size = (width, height)
        zoom = max(1.0, float(self._preview_zoom))
        target_box = (
            max(2, int(width * fit_scale * zoom)),
            max(2, int(height * fit_scale * zoom)),
        )
        self._update_zoom_label()
        self._show_canvas_hint("正在载入预览…")

        def worker() -> None:
            try:
                from bridge_agents.preview_images import open_fit

                source_path = source
                # 本仓库绘图 SVG 受控（仅 rect/line/polyline/circle/text）：
                # 先轻量渲染为 PNG 再窗口内预览，无需外部程序
                if source_path.lower().endswith(".svg"):
                    cache_dir = os.path.join(
                        self.vars["output_dir"].get().strip() or str(Path(source_path).parent),
                        "catalog_views",
                        "_preview",
                    )
                    png_path = render_svg_to_png(source_path, cache_dir)
                    if not png_path:
                        raise PreviewUnavailableError(f"SVG 渲染失败：{source_path}")
                    source_path = png_path
                image = open_fit(source_path, max_box=target_box)
            except PreviewUnavailableError as exc:
                self.events.put(("preview_ready", {"index": index, "error": str(exc)}))
            except Exception as exc:  # noqa: BLE001
                self.events.put(("preview_ready", {"index": index, "error": str(exc)}))
            else:
                self.events.put(("preview_ready", {"index": index, "image": image}))

        threading.Thread(target=worker, name=f"result-preview-{index}", daemon=True).start()

    def _zoom_by(self, factor: float) -> None:
        if not self._preview_source:
            return
        self._preview_zoom = min(20.0, max(1.0, self._preview_zoom * factor))
        self._render_preview()

    def _preview_fit(self) -> None:
        if not self._preview_source:
            return
        self._preview_zoom = 1.0
        self._render_preview()

    def _preview_actual(self) -> None:
        """切到接近 1:1（原始像素）比例查看细节；超出画布时配合拖拽/滚轮平移。"""
        if not self._preview_source:
            return
        if self._fit_scale <= 0 or self._fit_scale >= 1.0:
            self._preview_zoom = 1.0
        else:
            self._preview_zoom = min(20.0, max(1.0, 1.0 / self._fit_scale))
        self._render_preview()

    def _on_canvas_wheel(self, event: Any) -> None:
        if not self._preview_source:
            return
        delta = 1 if event.delta > 0 else (-1 if event.delta < 0 else 0)
        if delta:
            self._zoom_by(1.25 ** delta)

    def _on_canvas_press(self, event: Any) -> None:
        self.preview_canvas.scan_mark(event.x, event.y)

    def _on_canvas_drag(self, event: Any) -> None:
        self.preview_canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_canvas_double(self, _event: Any) -> None:
        if not self._preview_source:
            return
        if self._fit_scale >= 1.0 or self._preview_zoom <= 1.0001:
            self._preview_actual()
        else:
            self._preview_fit()

    def _handle_preview_ready(self, payload: Dict[str, Any]) -> None:
        if self._selected_artifact_index() != payload.get("index"):
            return  # 用户已切换条目，丢弃过期预览
        error = payload.get("error")
        if error:
            self._show_canvas_hint(f"预览失败：\n{error}")
            return
        image = payload.get("image")
        if image is None:
            return
        try:
            photo = ImageTk.PhotoImage(image)
        except Exception as exc:  # noqa: BLE001
            self._show_canvas_hint(f"预览失败：{exc}")
            return
        self._preview_photo = photo
        self.preview_canvas.delete("all")
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        image_w, image_h = image.size
        scroll_w = max(canvas_w, image_w)
        scroll_h = max(canvas_h, image_h)
        self.preview_canvas.configure(scrollregion=(0, 0, scroll_w, scroll_h))
        offset_x = max(0, (canvas_w - image_w) // 2)
        offset_y = max(0, (canvas_h - image_h) // 2)
        self.preview_canvas.create_image(
            offset_x, offset_y, image=photo, anchor="nw"
        )
        if image_w > canvas_w or image_h > canvas_h:
            # 放大后图像大于画布：回到左上起点，便于拖拽 / 滚轮浏览
            self.preview_canvas.xview_moveto(0.0)
            self.preview_canvas.yview_moveto(0.0)

    def _show_canvas_hint(self, text: str) -> None:
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(
            max(self.preview_canvas.winfo_width() // 2, 40),
            max(self.preview_canvas.winfo_height() // 2, 40),
            text=text,
            fill="#8fa9c4",
            font=("Microsoft YaHei UI", 10),
            justify="center",
        )

    def _open_artifact_external(self) -> None:
        artifact = self._selected_artifact()
        if artifact is None:
            messagebox.showwarning("未选择成果", "请先在成果树中选择一个成果条目。")
            return
        path = artifact.file_path or artifact.preview_path
        if not path or not Path(path).is_file():
            messagebox.showwarning("文件不存在", str(path or "（无文件路径）"))
            return
        os.startfile(Path(path).resolve())  # type: ignore[attr-defined]

    def _open_artifact_dir(self) -> None:
        artifact = self._selected_artifact()
        if artifact is None:
            messagebox.showwarning("未选择成果", "请先在成果树中选择一个成果条目。")
            return
        path = Path(artifact.file_path or artifact.preview_path or "")
        directory = path.parent if path.name else Path(self.vars["output_dir"].get())
        if not directory.is_dir():
            messagebox.showwarning("目录不存在", str(directory))
            return
        os.startfile(directory.resolve())  # type: ignore[attr-defined]

    def _generate_catalog_views(self) -> None:
        output_dir = self.vars["output_dir"].get().strip()
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showwarning("缺少输出目录", "请先设置有效的输出目录。")
            return
        self.result_hint.configure(text="正在生成初始/最终方案目录视图…")
        self._run_background("gen_views", lambda: build_catalog_views(output_dir))

    def _handle_views_result(self, payload: Dict[str, Any]) -> None:
        generated = [name for name, path in payload.items() if path]
        if generated:
            self.result_hint.configure(
                text=f"已生成 {len(generated)} 张目录视图（初始/最终分目录保存）："
                + "、".join(generated)
            )
        else:
            self.result_hint.configure(
                text="目录视图未生成：缺少布跨方案或坐标文件（PGW/平曲线/地形栅格）。"
            )
        self._refresh_result_center()

    def _refresh_output_dir_buttons(self) -> None:
        """成果目录已切到新输出目录时，同步刷新绘图索引提示。"""
        output_dir = self.vars["output_dir"].get().strip()
        if hasattr(self, "drawing_hint") and output_dir != getattr(self, "_drawings_dir", ""):
            self._drawings_dir = output_dir
            self._refresh_drawings()

    def _build_drawings_tab(self) -> None:
        heading = ttk.Frame(self.drawings_tab, style="Panel.TFrame")
        heading.pack(fill="x")
        heading.columnconfigure(0, weight=1)
        copy = ttk.Frame(heading, style="Panel.TFrame")
        copy.grid(row=0, column=0, sticky="nsew")
        ttk.Label(copy, text="设计组绘图成果", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            copy,
            text="按设计组汇集盖梁配筋与墩柱配筋分页成果。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        actions = ttk.Frame(heading, style="Panel.TFrame")
        actions.grid(row=0, column=1, sticky="e", padx=(12, 0))
        ttk.Button(
            actions,
            text="刷新索引",
            style="Secondary.TButton",
            command=self._refresh_drawings,
        ).pack(side="left")

        columns = ("group", "sheet", "piers", "issue", "geometry")
        self.drawing_tree = ttk.Treeview(
            self.drawings_tab,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=15,
        )
        headings = {
            "group": "设计组",
            "sheet": "图纸",
            "piers": "关联桥墩",
            "issue": "签发状态",
            "geometry": "几何检查",
        }
        widths = {"group": 160, "sheet": 320, "piers": 200, "issue": 130, "geometry": 110}
        for name in columns:
            self.drawing_tree.heading(name, text=headings[name])
            self.drawing_tree.column(name, width=widths[name], minwidth=90, anchor="w",
                                     stretch=(name == "sheet"))
        self.drawing_tree.pack(fill="both", expand=True, pady=(16, 10))
        self.drawing_tree.bind("<Double-1>", lambda _event: self._open_selected_drawing("svg"))

        actions = ttk.Frame(self.drawings_tab, style="Panel.TFrame")
        actions.pack(fill="x")
        ttk.Button(
            actions,
            text="打开 SVG 预览",
            style="Primary.TButton",
            command=lambda: self._open_selected_drawing("svg"),
        ).pack(side="left")
        ttk.Button(
            actions,
            text="打开 AutoCAD SCR",
            style="Secondary.TButton",
            command=lambda: self._open_selected_drawing("scr"),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            actions,
            text="打开绘图目录",
            style="Secondary.TButton",
            command=self._open_drawings_dir,
        ).pack(side="left", padx=(8, 0))
        self.drawing_hint = ttk.Label(actions, text="尚未读取绘图索引。", style="Hint.TLabel")
        self.drawing_hint.pack(side="right")

    def _load_base_config(self) -> None:
        path = Path(self.vars["base_config"].get())
        if not path.is_file():
            return
        try:
            settings = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            self._log(f"基础配置读取失败：{exc}")
            return
        self.request_text.delete("1.0", "end")
        self.request_text.insert("1.0", _pick(settings, "task", "user_request"))
        self.vars["data_path"].set(_pick(settings, "paths", "data_path"))
        self.vars["drawing_path"].set(_pick(settings, "paths", "input_drawing_path"))
        self.vars["output_dir"].set(_pick(settings, "paths", "output_dir"))
        self.vars["file_prefix"].set(_pick(settings, "route_data", "file_prefix"))
        self.vars["max_revision"].set(int(_pick(settings, "agent", "max_revision_rounds", 3)))
        self.vars["max_check_revision"].set(int(_pick(settings, "modeling", "max_check_revision_rounds", 2)))
        self.vars["code_rag"].set(bool(_pick(settings, "code_rag", "enabled", False)))
        self._refresh_result_center()

    def _form_state_values(self) -> Dict[str, Any]:
        return {
            "base_config": self.vars["base_config"].get().strip(),
            "data_path": self.vars["data_path"].get().strip(),
            "drawing_path": self.vars["drawing_path"].get().strip(),
            "output_dir": self.vars["output_dir"].get().strip(),
            "file_prefix": self.vars["file_prefix"].get().strip(),
            "max_revision": int(self.vars["max_revision"].get()),
            "max_check_revision": int(self.vars["max_check_revision"].get()),
            "code_rag": bool(self.vars["code_rag"].get()),
            "user_request": self.request_text.get("1.0", "end").strip(),
        }

    def _save_form_state(self) -> None:
        try:
            save_gui_form_state(self._form_state_values())
        except (OSError, TypeError, ValueError) as exc:
            self._log(f"上次输入保存失败：{exc}")

    def _restore_form_state(self) -> None:
        values = load_gui_form_state()
        if not values:
            return
        for key in ("base_config", "data_path", "drawing_path", "output_dir", "file_prefix"):
            if key in values and isinstance(values[key], str):
                self.vars[key].set(values[key])
        for key in ("max_revision", "max_check_revision"):
            try:
                value = int(values.get(key))
            except (TypeError, ValueError):
                continue
            if 1 <= value <= 20:
                self.vars[key].set(value)
        if isinstance(values.get("code_rag"), bool):
            self.vars["code_rag"].set(values["code_rag"])
        if isinstance(values.get("user_request"), str):
            self.request_text.delete("1.0", "end")
            self.request_text.insert("1.0", values["user_request"])
        self._log("已恢复上次填写的任务配置；运行 ID 和外发授权未复用。")
        self._refresh_result_center()

    def _choose_base_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("全部文件", "*.*")])
        if path:
            self.vars["base_config"].set(path)
            self._load_base_config()

    def _choose_dir(self, key: str) -> None:
        path = filedialog.askdirectory(initialdir=self.vars[key].get() or None)
        if path:
            self.vars[key].set(path)

    def _choose_drawing(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CAD 图纸", "*.dxf *.dwg"), ("全部文件", "*.*")])
        if path:
            self.vars["drawing_path"].set(path)

    def _config(self) -> GuiRunConfig:
        return GuiRunConfig(
            user_request=self.request_text.get("1.0", "end").strip(),
            data_path=self.vars["data_path"].get(),
            input_drawing_path=self.vars["drawing_path"].get(),
            output_dir=self.vars["output_dir"].get(),
            file_prefix=self.vars["file_prefix"].get(),
            thread_id=self.vars["thread_id"].get(),
            max_revision_rounds=self.vars["max_revision"].get(),
            max_check_revision_rounds=self.vars["max_check_revision"].get(),
            code_rag_enabled=self.vars["code_rag"].get(),
        )

    def _require_external_authorization(self) -> bool:
        if self.vars["external_authorized"].get():
            return True
        messagebox.showwarning("需要授权", "该操作会把任务描述或检索到的工程成果摘要发送到配置中的外部模型。请先勾选外发授权。")
        return False

    def _request_stop(self) -> None:
        """请求终止当前运行：LangGraph 运行中无法安全强杀，标记后在下一个
        安全点（阶段完成 / 人工复核 / 任务结束）以 abort 动作停止并落盘 cancelled。

        运行记录会明确提示"已请求终止"，避免用户误以为界面失去响应。
        """
        if not self.busy or self._stop_requested:
            return
        if not messagebox.askyesno(
            "终止运行",
            "将终止当前任务：系统会在下一个安全点自动提交终止动作（abort），"
            "已完成的成果与 checkpoint 会保留。确定终止？",
        ):
            return
        self._stop_requested = True
        self.stop_button.configure(state="disabled")  # 避免重复请求
        self.connection_label.configure(text="● 终止请求中")
        self.status_var.set("终止请求已提交：等待当前步骤完成后安全停止…")
        self._log(
            "用户请求终止运行；将在下一个安全点（阶段完成 / 人工复核 / 任务结束）自动提交 abort。"
            "注意：若任务正处于自动返修循环，可能要等若干轮才会到达安全点；"
            "必要时可结束进程，已生成成果与 checkpoint 会保留，之后仍可 resume。"
        )

    def _auto_abort_at_safe_point(self, result: Dict[str, Any]) -> bool:
        """终止请求到达安全点时自动以 abort 停止（消费请求），返回是否已提交。"""
        if not self._stop_requested:
            return False
        self._stop_requested = False
        interrupt = extract_interrupt(result)
        if not interrupt:
            status = str(result.get("task_status") or "已完成")
            self._log(f"终止请求到达时任务已结束（{status}），无需继续终止。")
            return False
        offered = set(interrupt.get("available_actions") or HUMAN_REVIEW_ACTIONS)
        self._log("终止请求到达安全点：自动提交 abort，任务将取消（checkpoint 已保存）。")
        self._run_background(
            "resume",
            lambda: self.controller.resume(
                thread_id=str(self.current_result.get("thread_id") or self.vars["thread_id"].get()),
                config_snapshot_path=self.current_snapshot,
                action="abort",
                available_actions=offered,
            ),
        )
        return True

    def _start(self) -> None:
        if self.busy or not self._require_external_authorization():
            return
        self._stop_requested = False
        # 运行 ID 只读自动管理：上一任务已完成（或尚无记录）时轮换新 ID；
        # 任务处于失败/待复核状态时保留当前 ID，便于同一线程重试或恢复。
        previous_status = str(self.current_result.get("task_status") or "")
        if previous_status in {"", "completed", "completed_with_accepted_risks"}:
            self.vars["thread_id"].set(
                f"bridge-{datetime.now():%Y%m%d-%H%M}-{uuid4().hex[:5]}"
            )
        try:
            config = self._config()
            config.validate()
        except Exception as exc:
            messagebox.showerror("配置不完整", str(exc))
            return
        self._save_form_state()
        self.current_snapshot = ""
        self.current_interrupt = None
        self._log(f"启动 Graph V2：{config.thread_id}")
        self._run_background("start", lambda: self.controller.start(config, base_config_path=self.vars["base_config"].get()))

    def _resume(self, action: str) -> None:
        if self.busy:
            self._log(
                "任务仍在后台运行，复核动作未提交：请等待当前步骤返回，"
                "或在“安全终止”后重试（若长时间无进展，可结束进程，成果与 checkpoint 会保留）。"
            )
            messagebox.showinfo("任务正在运行", "当前任务仍在后台运行，请等待其返回后再提交人工复核动作。")
            return
        if not self.current_interrupt:
            self._log("当前没有待处理的人工复核，已忽略该动作。")
            return
        if self._stop_requested:
            self._log("已请求终止：忽略手动复核动作，等待安全点自动 abort。")
            return
        feedback = self.feedback_text.get("1.0", "end").strip()
        offered = self.current_interrupt.get("available_actions") or []
        self._log(f"提交人工复核动作：{action}")
        self._run_background(
            "resume",
            lambda: self.controller.resume(
                thread_id=str(self.current_result.get("thread_id") or self.vars["thread_id"].get()),
                config_snapshot_path=self.current_snapshot,
                action=action,
                feedback=feedback,
                extra_rounds=self.extra_rounds.get(),
                available_actions=offered,
            ),
        )

    def _assess(self) -> None:
        if self.busy or not self._require_external_authorization():
            return
        if not self.current_snapshot:
            messagebox.showwarning("缺少运行配置", "请先启动一次任务，生成本次运行的配置快照。")
            return
        self._append_chat("user", "（执行整体评估）")
        self._run_background("assess", lambda: self.controller.assess(output_dir=self.vars["output_dir"].get(), config_snapshot_path=self.current_snapshot))

    def _run_background(self, operation: str, func: Callable[[], Dict[str, Any]]) -> None:
        self.busy = True
        self.start_button.configure(state="disabled")
        if hasattr(self, "stop_button"):
            self.stop_button.configure(state="normal")
        if hasattr(self, "chat_send_button"):
            self.chat_send_button.configure(state="disabled")
        self.connection_label.configure(text="● 运行中")
        self._refresh_review_buttons()

        def worker() -> None:
            try:
                self.events.put((operation, func()))
            except Exception as exc:
                self.events.put(("error", {"operation": operation, "error": str(exc), "type": type(exc).__name__}))

        threading.Thread(target=worker, name=f"graph-v2-{operation}", daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                operation, payload = self.events.get_nowait()
                self.busy = False
                self.start_button.configure(state="normal")
                if hasattr(self, "stop_button"):
                    self.stop_button.configure(state="disabled")
                if hasattr(self, "chat_send_button"):
                    self.chat_send_button.configure(state="normal")
                self.connection_label.configure(text="● 就绪")
                self._refresh_review_buttons()
                if operation == "error":
                    self._log(f"{payload['operation']} 失败：{payload['type']}: {payload['error']}")
                    messagebox.showerror("操作失败", payload["error"])
                elif operation in {"start", "resume"}:
                    self._handle_run_result(payload)
                elif operation == "preview_ready":
                    self._handle_preview_ready(payload)
                elif operation == "gen_views":
                    self._handle_views_result(payload)
                else:
                    self._handle_review_result(operation, payload)
        except queue.Empty:
            pass
        self.after(120, self._poll_events)

    def _event_line(self, event: Dict[str, Any]) -> str | None:
        """把 agent_run_log 的一条事件翻译为中文时间线行（不关心的返回 None）。"""
        event_type = str(event.get("event_type") or "")
        agent = str(event.get("agent") or "")
        name = str(event.get("name") or "")
        status = str(event.get("status") or "")
        agent_cn = AGENT_LABELS.get(agent, agent.replace("Agent", ""))
        action_cn = ACTION_LABELS_RUN.get(name) or name.replace("_", " ")
        if event_type == "stage_plan":
            return f"[阶段规划] {agent_cn}：{action_cn}"
        if event_type == "action_start":
            return f"[开始] {agent_cn} · {action_cn}"
        if event_type == "action_end":
            if status == "completed":
                return f"[完成] {agent_cn} · {action_cn}"
            error = str(event.get("error") or "")
            detail = error[:160] if error else f"状态 {status}"
            return f"[失败] {agent_cn} · {action_cn}：{detail}"
        if event_type == "agent_skill_loaded":
            return f"[准备] {agent_cn} 已加载专业说明"
        return None

    def _refresh_run_timeline(self, output_dir: str) -> None:
        """增量读取 agent_run_log.json，把新增事件以中文时间线追加到运行记录。

        比只读 prompt_trace 最新一条更能精确反映"当前运行到哪一步"：
        阶段规划 / 动作开始 / 完成 / 失败（含错误摘要）。
        """
        if not output_dir or not hasattr(self, "log_text"):
            return
        log_path = Path(output_dir) / "logs" / "agent_run_log.json"
        if not log_path.is_file():
            return
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
            events = payload.get("events") or []
        except Exception:
            return
        if not isinstance(events, list):
            return
        offset = getattr(self, "_agent_log_offset", 0)
        if len(events) <= offset:
            return
        for event in events[offset:]:
            if not isinstance(event, dict):
                continue
            line = self._event_line(event)
            if line:
                self._log(f"[运行] {line}")
        self._agent_log_offset = len(events)

    def _poll_progress(self) -> None:
        """定时扫描输出目录，实时刷新运行轨道/当前位置/运行记录。

        任务在后台线程同步执行 run_agent，期间 run_agent 不返回结果；
        本方法读取 agent_run_log.json（动作级事件时间线）与文件完成度推断
        当前阶段/动作/设计组，避免 UI 全程空白。仅刷新显示，不触碰任务线程。
        """
        # 输出目录切换后自动刷新成果中心（不打断用户当前选中）
        try:
            poll_output = self.vars["output_dir"].get().strip()
            if (
                poll_output
                and hasattr(self, "result_tree")
                and poll_output != getattr(self, "_catalog_scanned_dir", "")
            ):
                self._refresh_result_center()
        except Exception:
            pass
        try:
            output_dir = self.vars["output_dir"].get().strip()
            if not output_dir or self.busy and not self._progress_polling:
                pass
            progress = snapshot_run_progress(output_dir) if output_dir else {
                "running": False, "stage_states": {s: "pending" for s in STAGE_LABELS},
                "current_scope": None, "completed_groups": 0, "total_groups": 0, "recent_log": "",
            }
            # 动作级运行时间线（增量；无 agent_run_log 时回退 prompt_trace 最新行）
            run_log_path = Path(output_dir) / "logs" / "agent_run_log.json" if output_dir else None
            has_run_log = bool(run_log_path and run_log_path.is_file())
            if has_run_log:
                self._refresh_run_timeline(output_dir)
            # 运行轨道：仅在没有任务结果覆盖时刷新（busy 或已启动）
            if self.busy or progress.get("running"):
                states = progress.get("stage_states") or {}
                if states:
                    self._set_stage_states(states)
                active = progress.get("active_stage")
                if active:
                    self.agent_var.set(f"当前 Agent：{STAGE_LABELS.get(active, active)}（后台运行）")
                self.thread_var.set(f"运行 ID：{self.vars['thread_id'].get()}")
                scope = progress.get("current_scope")
                completed = progress.get("completed_groups") or 0
                total = progress.get("total_groups") or 0
                if active or scope or total:
                    position = []
                    if active:
                        position.append(f"阶段：{STAGE_LABELS.get(active, active)}")
                    if total:
                        position.append(f"设计组 {completed}/{total}")
                    if scope:
                        position.append(f"当前 {scope}")
                    self.status_var.set(" · ".join(position) if position else "后台运行中")
                log_line = progress.get("recent_log") or ""
                if log_line and log_line != self._last_progress_log and not has_run_log:
                    self._last_progress_log = log_line
                    self._log(f"[进度] {log_line}")
        except Exception:
            # 进度轮询失败不影响主流程
            pass
        self.after(2000, self._poll_progress)

    def _handle_run_result(self, result: Dict[str, Any]) -> None:
        self.current_result = result
        self.current_snapshot = str(result.get("config_snapshot_path") or self.current_snapshot)
        # 终止请求：到达安全点则自动提交 abort，不再向用户展示继续复核选项
        if self._auto_abort_at_safe_point(result):
            return
        self.current_interrupt = extract_interrupt(result)
        self._set_stage_states(infer_stage_states(result))
        status = str(result.get("task_status") or "运行暂停")
        self.status_var.set(str(result.get("message") or status))
        self.agent_var.set(f"当前 Agent：{result.get('active_agent') or '—'}")
        self.thread_var.set(f"运行 ID：{result.get('thread_id') or self.vars['thread_id'].get()}")
        self._log(json.dumps({key: result.get(key) for key in ("task_status", "message", "error", "thread_id", "checkpoint_path") if result.get(key) is not None}, ensure_ascii=False, indent=2))
        self._render_review(self.current_interrupt)
        self._refresh_drawings()
        self._refresh_result_center()
        if self.current_interrupt:
            self.tabs.select(self.review_tab)

    def _handle_review_result(self, operation: str, result: Dict[str, Any]) -> None:
        if operation == "answer":
            content = result.get("answer") or json.dumps(result, ensure_ascii=False, indent=2, default=str)
            self.chat_history.append({"role": "assistant", "content": content})
            cited = result.get("cited_artifact_ids") or []
            code_cited = result.get("cited_code_evidence_ids") or []
            if cited or code_cited:
                content = content + "\n\n引用：" + "、".join(list(cited)[:10] + list(code_cited)[:10])
            self._append_chat("assistant", content)
        else:  # assess：整体评估也追加为消息，但不进入后续问答的上下文（避免上下文膨胀）
            content = result.get("display_text") or result.get("overall_conclusion") or json.dumps(result, ensure_ascii=False, indent=2, default=str)
            prefix = "—— 整体评估 ——\n\n"
            self._append_chat("assistant", prefix + content)
        self._log(f"{operation} 已完成，输出：{result.get('answer_path') or result.get('assessment_path') or '已返回'}")
        self.tabs.select(self.qa_tab)

    def _render_review(self, review: Dict[str, Any] | None) -> None:
        self.review_summary.delete("1.0", "end")
        for child in self.review_actions.winfo_children():
            child.destroy()
        if not review:
            self.review_summary.insert("1.0", "当前没有等待处理的人工复核。")
            self.review_buttons = []
            ttk.Label(self.review_actions, text="流程尚未请求人工复核。", style="Hint.TLabel").pack(anchor="w")
            return
        self.review_summary.insert("1.0", json.dumps(review, ensure_ascii=False, indent=2, default=str))
        self.review_buttons = []
        for action in review.get("available_actions") or []:
            style = "Danger.TButton" if action == "abort" else "Warning.TButton" if action.startswith("accept") else "Secondary.TButton"
            button = ttk.Button(self.review_actions, text=ACTION_LABELS.get(action, action), style=style, command=lambda value=action: self._resume(value))
            button.pack(side="left", padx=(0, 8), pady=4)
            self.review_buttons.append(button)
        self._refresh_review_buttons()

    def _refresh_review_buttons(self) -> None:
        """任务在后台运行时禁用复核按钮：否则按钮看起来可点，实际被 _resume 静默忽略。"""
        state = "disabled" if self.busy else "normal"
        for button in getattr(self, "review_buttons", []):
            try:
                button.configure(state=state)
            except tk.TclError:
                continue

    def _set_stage_states(self, states: Dict[str, str]) -> None:
        labels = {"pending": "待开始", "running": "运行中", "completed": "已完成", "review": "待复核", "failed": "失败"}
        for index, (stage, title) in enumerate(STAGE_LABELS.items()):
            state = states.get(stage, "pending")
            bg, fg = STATE_COLORS[state]
            self.stage_cards[stage].configure(text=f"{index + 1:02d}\n{title}\n{labels[state]}", bg=bg, fg=fg)

    def _log(self, message: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.insert("end", f"[{datetime.now():%H:%M:%S}] {message}\n")
        self.log_text.see("end")

    def _open_output(self) -> None:
        path = Path(self.vars["output_dir"].get() or ".").resolve()
        if not path.exists():
            messagebox.showwarning("目录不存在", str(path))
            return
        os.startfile(path)  # type: ignore[attr-defined]

    def _refresh_drawings(self) -> None:
        if not hasattr(self, "drawing_tree"):
            return
        for item in self.drawing_tree.get_children():
            self.drawing_tree.delete(item)
        output_dir = self.vars["output_dir"].get().strip()
        catalog = load_drawing_catalog(output_dir) if output_dir else []
        sheet_count = 0
        for group in catalog:
            piers = ", ".join(str(value) for value in group.get("member_piers") or []) or "—"
            issue = str(group.get("issue_status") or "unknown")
            geometry = "通过" if group.get("geometry_valid") is True else "待复核"
            for sheet in group.get("sheets") or []:
                item_id = self.drawing_tree.insert(
                    "",
                    "end",
                    values=(
                        group.get("design_group_id") or "—",
                        sheet.get("title") or sheet.get("sheet_id") or "—",
                        piers,
                        issue,
                        geometry,
                    ),
                )
                self.drawing_tree.set(item_id, "group", group.get("design_group_id") or "—")
                self.drawing_tree.item(
                    item_id,
                    tags=(str(sheet.get("svg_path") or ""), str(sheet.get("scr_path") or "")),
                )
                sheet_count += 1
        self.drawing_hint.configure(
            text=f"{len(catalog)} 个设计组 / {sheet_count} 张图纸" if catalog else "当前输出目录没有绘图索引。"
        )

    def _selected_drawing_paths(self) -> tuple[str, str] | None:
        selected = self.drawing_tree.selection()
        if not selected:
            messagebox.showwarning("未选择图纸", "请先选择一张图纸。")
            return None
        tags = self.drawing_tree.item(selected[0], "tags")
        return (str(tags[0]) if len(tags) > 0 else "", str(tags[1]) if len(tags) > 1 else "")

    def _open_selected_drawing(self, kind: str) -> None:
        paths = self._selected_drawing_paths()
        if not paths:
            return
        path = Path(paths[0] if kind == "svg" else paths[1])
        if not path.is_file():
            messagebox.showwarning("文件不存在", str(path))
            return
        os.startfile(path)  # type: ignore[attr-defined]

    def _open_drawings_dir(self) -> None:
        path = Path(self.vars["output_dir"].get()) / "deliverables" / "drawings"
        if not path.is_dir():
            messagebox.showwarning("目录不存在", str(path))
            return
        os.startfile(path.resolve())  # type: ignore[attr-defined]

    def _close(self) -> None:
        if self.busy and not messagebox.askyesno("任务仍在运行", "关闭窗口不会清理已生成成果，但后台线程将随进程结束。确定关闭？"):
            return
        self._save_form_state()
        self.destroy()


def check_environment() -> int:
    import tkinter

    controller = GraphV2Controller(run_agent_fn=lambda *_args, **_kwargs: {})
    print(json.dumps({"tkinter": tkinter.TkVersion, "controller": type(controller).__name__, "graph_mode": "v2", "dpi_awareness": enable_windows_high_dpi()}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 Graph V2 桥梁设计桌面控制台。")
    parser.add_argument("--check", action="store_true", help="只检查 Tkinter 与控制器导入，不打开窗口。")
    args = parser.parse_args()
    if args.check:
        return check_environment()
    enable_windows_high_dpi()
    app = BridgeDesignConsole()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
