from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .config import DrawingConfig
from .detail_sheets import build_reinforcement_sheets, sheet_layout_config
from .layout import build_combined_layout
from .reinforcement_resolution import (
    load_reinforcement_example,
    resolve_reinforcement_examples,
)
from .scr_exporter import render_autocad_script
from .svg_exporter import render_svg
from .validation import validate_geometry


class SampleDrawingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheet_ids: list[str]
    svg_paths: list[str]
    scr_paths: list[str]
    drawing_ir_paths: list[str]
    manifest_path: str
    diagnostics_path: str


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _layout_config(sheet_id: str) -> DrawingConfig:
    return sheet_layout_config(sheet_id)


def render_reinforcement_sample_set(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
) -> SampleDrawingResult:
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cap_example = load_reinforcement_example(
        root / "samples" / "reinforcement_design_samples.yaml",
        example_name="示例1",
        expected_drawing_id=12,
    )
    column_example = load_reinforcement_example(
        root / "samples" / "reinforcement_column.yaml",
        example_name="示例1",
        expected_drawing_id=14,
    )
    resolved = resolve_reinforcement_examples(cap_example, column_example)
    sheets = build_reinforcement_sheets(
        resolved,
        design_group_id="sample-cap12-column14",
        member_piers=["样例同类墩柱-1", "样例同类墩柱-2"],
    )

    manifest_sheets: dict[str, Any] = {}
    geometry_diagnostics: dict[str, Any] = {}
    svg_paths: list[str] = []
    scr_paths: list[str] = []
    drawing_ir_paths: list[str] = []
    for sheet in sheets:
        document = build_combined_layout(
            sheet.document,
            _layout_config(sheet.sheet_id),
        )
        svg_path = output / f"{sheet.sheet_id}.svg"
        scr_path = output / f"{sheet.sheet_id}.scr"
        ir_path = output / f"{sheet.sheet_id}.drawing_ir.json"
        svg_path.write_text(render_svg(document), encoding="utf-8")
        # SCR 供 AutoCAD 中文版 SCRIPT 执行：按系统代码页（GBK）写入，避免中文乱码
        scr_path.write_text(
            render_autocad_script(document), encoding="gbk", errors="replace"
        )
        ir_path.write_text(document.to_stable_json() + "\n", encoding="utf-8")
        diagnostics = [
            validate_geometry(section).model_dump(mode="json")
            for view in document.views
            if view.view_id == "cad-layout"
            for section in view.sections
        ]
        geometry_diagnostics[sheet.sheet_id] = diagnostics
        manifest_sheets[sheet.sheet_id] = {
            "title": sheet.title,
            "svg": svg_path.name,
            "scr": scr_path.name,
            "drawing_ir": ir_path.name,
            "view_ids": [
                view.view_id
                for view in document.views
                if view.view_id != "cad-layout"
            ],
            "geometry_valid": all(item["valid"] for item in diagnostics),
            "sha256": {
                "svg": _sha256(svg_path),
                "scr": _sha256(scr_path),
                "drawing_ir": _sha256(ir_path),
            },
        }
        svg_paths.append(str(svg_path))
        scr_paths.append(str(scr_path))
        drawing_ir_paths.append(str(ir_path))

    diagnostics_path = output / "diagnostics.json"
    _write_json(
        diagnostics_path,
        {
            "source_diagnostics": [
                item.model_dump(mode="json") for item in resolved.diagnostics
            ],
            "semantic_review_items": [
                "cap:N3/N4 路径局部超出名义盖梁轮廓",
                "column:N4 环向加强筋细部语义待原图复核",
            ],
            "geometry_diagnostics": geometry_diagnostics,
        },
    )
    manifest_path = output / "drawing_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "reinforcement-sample-drawing-v1",
            "source_examples": {
                "cap": {"example_name": "示例1", "drawing_id": 12},
                "column": {"example_name": "示例1", "drawing_id": 14},
            },
            "representative_design_rule": (
                "同一设计组内同类墩柱只绘制一套代表性配筋详图"
            ),
            "sheets": manifest_sheets,
            "diagnostics": diagnostics_path.name,
        },
    )
    return SampleDrawingResult(
        sheet_ids=[sheet.sheet_id for sheet in sheets],
        svg_paths=svg_paths,
        scr_paths=scr_paths,
        drawing_ir_paths=drawing_ir_paths,
        manifest_path=str(manifest_path),
        diagnostics_path=str(diagnostics_path),
    )
