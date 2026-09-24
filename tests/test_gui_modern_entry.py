from pathlib import Path

from run_graph_v2_gui_modern import build_parser, load_initial_form


def test_modern_gui_entry_supports_paper_and_screenshot_modes() -> None:
    args = build_parser().parse_args(
        [
            "--output-dir",
            "output/示例项目9.8",
            "--project-name",
            "示例项目9.8",
            "--paper",
            "--page",
            "results",
            "--screenshot",
            "paper.png",
        ]
    )

    assert args.paper is True
    assert args.page == "results"
    assert args.project_name == "示例项目9.8"
    assert Path(args.screenshot).name == "paper.png"


def test_cli_output_and_base_config_override_saved_form(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--output-dir", str(tmp_path / "run"), "--base-config", "config/custom.yaml"]
    )

    form = load_initial_form(args)

    assert form.output_dir == str(tmp_path / "run")
    assert form.base_config_path == "config/custom.yaml"


def test_existing_explicit_base_config_prefills_task_form(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "task:\n  user_request: 执行全流程设计\npaths:\n  data_path: route\n  input_drawing_path: input.dxf\n  output_dir: old\nroute_data:\n  file_prefix: K\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["--base-config", str(config), "--output-dir", str(tmp_path / "new-output")]
    )

    form = load_initial_form(args)

    assert form.user_request == "执行全流程设计"
    assert form.data_path == "route"
    assert form.output_dir == str(tmp_path / "new-output")
