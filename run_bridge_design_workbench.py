"""Launch the user-facing Bridge Design Workbench for Graph V2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="桥梁设计工作台（Graph V2）")
    parser.add_argument("--output-dir", default="", help="工程成果输出目录")
    parser.add_argument("--base-config", default="", help="基础 YAML 运行配置")
    parser.add_argument("--project-name", default="桥梁设计工程", help="项目显示名称")
    parser.add_argument("--route-range", default="", help="顶栏桩号范围")
    parser.add_argument("--demo", action="store_true", help="显示明确标注的演示数据")
    parser.add_argument(
        "--page", choices=("overview", "task", "review", "confirm", "missing", "rework", "results", "qa", "log"),
        default="task", help="启动时显示的页面",
    )
    parser.add_argument("--screenshot", default="", help="导出窗口 PNG 后退出")
    return parser


def load_initial_form(args: argparse.Namespace):
    from bridge_agents.gui_modern_session import load_form_from_yaml, load_form_state

    form = load_form_state()
    base_config = args.base_config or form.base_config_path
    should_prefill = bool(args.base_config) or not any(
        (form.user_request, form.data_path, form.drawing_path)
    )
    if should_prefill and Path(base_config).is_file():
        form = load_form_from_yaml(base_config)
        form.external_authorized = False
    if args.output_dir:
        form.output_dir = args.output_dir
    if args.base_config:
        form.base_config_path = args.base_config
    return form


def _load_model(args: argparse.Namespace, form=None):
    from bridge_agents.gui_controller import load_drawing_catalog, snapshot_run_progress
    from bridge_agents.gui_modern_model import build_dashboard_model, build_demo_dashboard_model
    from bridge_agents.gui_modern_session import route_range_from_request

    if args.demo:
        return build_demo_dashboard_model()
    output_dir = Path(args.output_dir) if args.output_dir else Path("output")
    route_range = args.route_range or route_range_from_request(
        str(getattr(form, "user_request", "") or "")
    )
    return build_dashboard_model(
        snapshot_run_progress(output_dir),
        drawings=load_drawing_catalog(output_dir),
        project_name=args.project_name,
        route_range=route_range,
    )


def load_workbench_form(args: argparse.Namespace):
    """Use the chosen output's saved run identity when browsing an existing run."""
    import yaml

    from bridge_agents.gui_modern_session import find_latest_snapshot
    form = load_initial_form(args)
    if args.output_dir:
        snapshot = find_latest_snapshot(args.output_dir)
        if snapshot:
            form.thread_id = snapshot.stem
            try:
                payload = yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                payload = {}
            if isinstance(payload, dict):
                task = payload.get("task") or {}
                paths = payload.get("paths") or {}
                if isinstance(task, dict) and task.get("user_request"):
                    form.user_request = str(task["user_request"])
                if isinstance(paths, dict):
                    form.data_path = str(paths.get("data_path") or form.data_path)
                    form.drawing_path = str(paths.get("input_drawing_path") or form.drawing_path)
    return form


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit("桥梁设计工作台需要 PySide6，请先安装项目依赖。") from exc

    from bridge_agents.bridge_design_workbench import BridgeDesignWorkbenchWindow
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Bridge Design Workbench")
    font_family = "Microsoft YaHei UI"
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if candidate.is_file():
            font_id = QFontDatabase.addApplicationFont(str(candidate))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                font_family = families[0]
                break
    app.setFont(QFont(font_family, 9))
    form = load_workbench_form(args)
    if not args.output_dir and form.output_dir:
        args.output_dir = form.output_dir
    window = BridgeDesignWorkbenchWindow(
        _load_model(args, form), paper_mode=False,
        initial_form=form, route_range=args.route_range,
    )
    window.show_page(args.page)
    window.show()
    if args.screenshot:
        target = Path(args.screenshot).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        def capture() -> None:
            window.grab().save(str(target), "PNG")
            app.quit()

        QTimer.singleShot(500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
