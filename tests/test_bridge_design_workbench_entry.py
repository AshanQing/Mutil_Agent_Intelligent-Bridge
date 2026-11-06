from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from run_bridge_design_workbench import build_parser, load_workbench_form, main


def test_default_entry_shows_task_workbench_screenshot(tmp_path: Path) -> None:
    target = tmp_path / "workbench.png"
    assert main(["--output-dir", str(tmp_path), "--screenshot", str(target)]) == 0
    assert target.is_file()
    assert target.stat().st_size > 10_000


def test_entry_supports_all_contextual_pages() -> None:
    parser = build_parser()
    for page in ("overview", "task", "review", "confirm", "missing", "rework", "results", "qa", "log"):
        assert parser.parse_args(["--page", page]).page == page


def test_explicit_output_recovers_existing_run_identity(tmp_path: Path) -> None:
    snapshot = tmp_path / "run_configs/existing-thread.yaml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("task:\n  user_request: 桥梁设计任务\n", encoding="utf-8")
    form = load_workbench_form(build_parser().parse_args(["--output-dir", str(tmp_path)]))
    assert form.thread_id == "existing-thread"
    assert form.user_request == "桥梁设计任务"
