from __future__ import annotations

import json
import locale
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_selected_examples_render_two_traceable_sheet_pairs(tmp_path: Path) -> None:
    from bridge_agents.drawing.sample_rendering import (
        render_reinforcement_sample_set,
    )

    result = render_reinforcement_sample_set(
        output_dir=tmp_path,
        repo_root=REPO_ROOT,
    )

    assert result.sheet_ids == [
        "cap_reinforcement_detail",
        "column_reinforcement_detail",
    ]
    assert len(result.svg_paths) == 2
    assert len(result.scr_paths) == 2
    assert all(Path(path).is_file() for path in result.svg_paths + result.scr_paths)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["source_examples"] == {
        "cap": {"example_name": "示例1", "drawing_id": 12},
        "column": {"example_name": "示例1", "drawing_id": 14},
    }
    assert set(manifest["sheets"]) == set(result.sheet_ids)


def test_sample_rendering_records_source_diagnostics_without_silently_repairing(
    tmp_path: Path,
) -> None:
    from bridge_agents.drawing.sample_rendering import (
        render_reinforcement_sample_set,
    )

    result = render_reinforcement_sample_set(
        output_dir=tmp_path,
        repo_root=REPO_ROOT,
    )

    diagnostics = json.loads(Path(result.diagnostics_path).read_text(encoding="utf-8"))
    source_codes = [item["code"] for item in diagnostics["source_diagnostics"]]
    # count 语义修正后区段按 count×spacing 累计，示例1 箍筋数据自洽（0 mismatch）
    assert source_codes.count("stirrup_segment_length_mismatch") == 0
    assert diagnostics["semantic_review_items"] == [
        "cap:N3/N4 路径局部超出名义盖梁轮廓",
        "column:N4 环向加强筋细部语义待原图复核",
    ]


def test_each_svg_contains_only_its_sheet_views_and_scr_has_command_terminator(
    tmp_path: Path,
) -> None:
    from bridge_agents.drawing.sample_rendering import (
        render_reinforcement_sample_set,
    )

    result = render_reinforcement_sample_set(
        output_dir=tmp_path,
        repo_root=REPO_ROOT,
    )

    cap_svg = (tmp_path / "cap_reinforcement_detail.svg").read_text(encoding="utf-8")
    column_svg = (tmp_path / "column_reinforcement_detail.svg").read_text(
        encoding="utf-8"
    )
    assert "主骨架" in cap_svg
    assert "盖梁钢筋表" in cap_svg
    assert "代表性单柱配筋立面" not in cap_svg
    assert "代表性单柱配筋立面" in column_svg
    assert "墩柱钢筋表" in column_svg

    for path in result.scr_paths:
        # SCR 供 AutoCAD 中文版执行，按系统代码页（GBK）写入
        script = Path(path).read_text(encoding="gbk")
        assert script.endswith("_.ZOOM\n_E\n")
        assert "_.-LAYER" in script


def test_sample_renderer_cli_generates_manifest_without_online_model(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "render_reinforcement_sample.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False),
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "drawing_manifest.json").is_file()
    assert "未调用在线模型" in completed.stdout
