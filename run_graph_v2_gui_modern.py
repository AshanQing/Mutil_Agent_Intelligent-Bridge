from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="桥梁设计 Graph V2 现代可视化控制台")
    parser.add_argument("--output-dir", default="", help="读取运行成果的输出目录")
    parser.add_argument("--base-config", default="", help="任务启动使用的基础 YAML 配置")
    parser.add_argument("--project-name", default="桥梁设计工程", help="界面显示的项目名称")
    parser.add_argument("--route-range", default="", help="界面显示的路线桩号范围")
    parser.add_argument("--paper", action="store_true", help="启用论文展示布局（1440×900）")
    parser.add_argument("--demo", action="store_true", help="使用明确标注的演示数据")
    parser.add_argument(
        "--page",
        choices=("overview", "task", "review", "results", "qa", "info"),
        default="overview",
        help="启动后显示的工作台页面",
    )
    parser.add_argument("--screenshot", default="", help="启动后将窗口保存为 PNG 并退出")
    return parser


def load_initial_form(args: argparse.Namespace):
    from bridge_agents.gui_modern_session import load_form_from_yaml, load_form_state

    form = load_form_state()
    base_config = args.base_config or form.base_config_path
    should_prefill = bool(args.base_config) or not any(
        (form.user_request, form.data_path, form.drawing_path, form.file_prefix)
    )
    if should_prefill and Path(base_config).is_file():
        loaded = load_form_from_yaml(base_config)
        loaded.external_authorized = False
        form = loaded
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
    # 没给 --route-range 时从任务描述里取桩号，否则顶栏会一直是"路线范围未设置"。
    route_range = args.route_range or route_range_from_request(
        str(getattr(form, "user_request", "") or "")
    )
    return build_dashboard_model(
        snapshot_run_progress(output_dir),
        drawings=load_drawing_catalog(output_dir),
        project_name=args.project_name,
        route_range=route_range,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit("现代界面需要 PySide6，请先执行：pip install -r requirements.txt") from exc

    from bridge_agents.gui_modern_window import ModernBridgeWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Bridge AI Studio")
    # Qt 的 offscreen 平台插件在部分 Windows 环境不会自动枚举系统字体。
    # 显式注册微软雅黑，保证论文截图与正常桌面启动都能稳定显示中文。
    font_family = "Microsoft YaHei UI"
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if not candidate.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(candidate))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            font_family = families[0]
            break
    app.setFont(QFont(font_family, 9))
    initial_form = load_initial_form(args)
    if not args.output_dir and initial_form.output_dir:
        args.output_dir = initial_form.output_dir
    window = ModernBridgeWindow(
        _load_model(args, initial_form),
        paper_mode=args.paper,
        initial_form=initial_form,
        route_range=args.route_range,
    )
    window.show_page(args.page)
    window.show()
    if args.screenshot:
        target = Path(args.screenshot).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        def capture() -> None:
            window.grab().save(str(target), "PNG")
            app.quit()

        QTimer.singleShot(350, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
