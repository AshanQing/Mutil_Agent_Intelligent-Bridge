from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .detail_sheets import (
    DrawingSheetSpec,
    build_reinforcement_sheets,
    sheet_layout_config,
)
from .layout import build_combined_layout
from .group_sources import collect_drawing_group_sources
from .models import DrawingDocument, DrawingGroupSource, TextEntity
from .reinforcement_adapter import build_bar_schedule
from .reinforcement_resolution import resolve_drawing_source
from .scr_exporter import render_autocad_script
from .svg_exporter import render_svg
from .validation import validate_geometry


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class UnverifiedDrawingSourceError(ValueError):
    pass


class DrawingPackageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    generated_group_ids: list[str] = Field(default_factory=list)
    skipped_group_ids: list[str] = Field(default_factory=list)
    failed_groups: list[dict[str, str]] = Field(default_factory=list)
    drawing_index_path: str
    design_manifest_path: str
    cad_script_paths: list[str] = Field(default_factory=list)
    drawing_preview_paths: list[str] = Field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _add_notice(document: DrawingDocument, notice: str) -> DrawingDocument:
    payload = document.model_dump(mode="python")
    first_section = payload["views"][0]["sections"][0]
    first_section["entities"].append(
        TextEntity(
            entity_id="drawing-issue-notice-001",
            layer="A-TITLE",
            insertion=(0.0, 2600.0),
            text=notice,
            height=220.0,
            metadata={"role": "issue_notice"},
        ).model_dump(mode="python")
    )
    payload["metadata"]["issue_notice"] = notice
    return DrawingDocument.model_validate(payload)


def _write_schedule(path: Path, source: DrawingGroupSource) -> None:
    rows = build_bar_schedule(source)
    fieldnames = [
        "mark",
        "member",
        "category",
        "diameter_mm",
        "count",
        "spacing_mm",
        "geometry_description",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def _replace_group_directory(staging: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    try:
        if target.exists():
            os.replace(target, backup)
        os.replace(staging, target)
    except Exception:
        if not target.exists() and backup.exists():
            os.replace(backup, target)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


"""人工风险条目的 scope 说明（用于图纸注记与放行判定）。"""
_RISK_SCOPE_LABELS = {
    "modeling_check": "建模验算",
    "structural_design": "结构设计",
    "layout_revision": "布跨修正",
}


def _risk_covers_group(
    risk: Mapping[str, Any],
    *,
    design_group_id: str,
    task_id: str = "",
) -> bool:
    """判断一条人工接受的风险是否覆盖该设计组的"未通过自动验算"。

    粒度规则：
    - scope=modeling_check：人工接受的是验算结论本身，属于全局决定，覆盖所有组；
    - scope=structural_design：只覆盖该风险记录里列出的失败配筋任务号，
      或其归属的尺寸设计单元（按"2-2 → 2-2-1-G1"前缀归属）；
    - 其它 scope（如 layout_revision 布跨修正）：属于布跨/上游风险，
      不能替代某个组的承载力验算结论，因此不放行。
    """
    if not isinstance(risk, Mapping):
        return False
    scope = str(risk.get("scope") or "").strip()
    if scope == "modeling_check":
        return True
    if scope != "structural_design":
        return False

    identifiers = {value for value in (design_group_id, task_id) if value}
    failed_tasks = {
        str(item).strip() for item in (risk.get("failed_task_ids") or []) if str(item).strip()
    }
    if identifiers & failed_tasks:
        return True
    for raw_unit in risk.get("failed_dimension_unit_ids") or []:
        unit = str(raw_unit).strip()
        if unit and any(value == unit or value.startswith(f"{unit}-") for value in identifiers):
            return True
    return False


def _covering_risks(
    accepted_risks: Sequence[Mapping[str, Any]],
    *,
    design_group_id: str,
    task_id: str = "",
) -> list[Mapping[str, Any]]:
    return [
        risk
        for risk in accepted_risks
        if _risk_covers_group(risk, design_group_id=design_group_id, task_id=task_id)
    ]


def _risk_notice(covering: Sequence[Mapping[str, Any]]) -> str:
    labels: list[str] = []
    for risk in covering:
        scope = str(risk.get("scope") or "").strip()
        label = _RISK_SCOPE_LABELS.get(scope, scope or "人工判断")
        if label not in labels:
            labels.append(label)
    return "含人工接受风险：" + "、".join(labels) if labels else "含人工接受风险"


def _issue_status(
    source: DrawingGroupSource,
    *,
    allow_unverified: bool,
    accepted_risks: Sequence[Mapping[str, Any]],
) -> tuple[str, str | None]:
    covering = _covering_risks(
        accepted_risks,
        design_group_id=source.design_group_id,
        task_id=getattr(source, "task_id", "") or "",
    )
    if source.check_status == "passed":
        # 该组自身验算通过：状态保持 verified（不再被"任意一条人工风险"整体改成
        # accepted_risk，避免 verified 组被误标）。覆盖该组的人工风险仍会记录进成果索引。
        return "verified", None
    if covering:
        return "accepted_risk", _risk_notice(covering)
    if source.check_status == "failed":
        # 验算"失败"是明确的否定结论，只能由覆盖该组的人工风险放行；
        # 不能因为人工在别处接受了风险（例如只接受布跨碰撞风险）就把失败组也放出去。
        raise UnverifiedDrawingSourceError(
            f"设计组 {source.design_group_id} 的自动验算未通过（check_status=failed），"
            "且没有覆盖该组的人工接受风险。"
        )
    if not allow_unverified:
        raise UnverifiedDrawingSourceError(
            f"设计组 {source.design_group_id} 的自动验算状态为 {source.check_status}。"
        )
    return "draft_unverified", "自动验算未完成，仅供复核"


def generate_drawing_package(
    *,
    sources: Sequence[DrawingGroupSource],
    output_dir: str | Path,
    allow_unverified: bool = False,
    accepted_risks: Sequence[Mapping[str, Any]] = (),
) -> DrawingPackageResult:
    root = Path(output_dir).resolve()
    drawings_root = root / "deliverables" / "drawings"
    drawings_root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(sources, key=lambda item: item.design_group_id)
    if not ordered:
        raise ValueError("没有可生成绘图成果的设计组。")

    # 出图放行状态必须逐组解析，且不能因为个别组不可放行就中断整包：
    # 少数组验算未通过（且没有覆盖它的人工接受风险）时，只应跳过这些组，
    # 其余已成功设计出的尺寸/配筋仍要出图
    # （2026-09-16 示例项目K31 少数组验算失败即导致一张图纸都没出）。
    statuses: dict[str, tuple[str, str | None]] = {}
    unverified: dict[str, UnverifiedDrawingSourceError] = {}
    for source in ordered:
        try:
            statuses[source.design_group_id] = _issue_status(
                source,
                allow_unverified=allow_unverified,
                accepted_risks=accepted_risks,
            )
        except UnverifiedDrawingSourceError as exc:
            unverified[source.design_group_id] = exc
    if unverified and not statuses:
        # 一组都放不出来：维持"整包拒绝"的既有契约，交由上层判为失败
        raise next(iter(unverified.values()))

    entries: list[dict[str, Any]] = []
    generated: list[str] = []
    failed: list[dict[str, str]] = []
    cad_paths: list[str] = []
    preview_paths: list[str] = []

    for source in ordered:
        group_id = source.design_group_id
        if group_id in unverified:
            failed.append({"design_group_id": group_id, "error": str(unverified[group_id])})
            continue
        if not _SAFE_ID.fullmatch(group_id):
            failed.append({"design_group_id": group_id, "error": "设计组 ID 不能用于安全目录名。"})
            continue
        staging = Path(tempfile.mkdtemp(prefix=f".{group_id}-", dir=drawings_root))
        try:
            issue_status, notice = statuses[group_id]
            # 解析为工程语义模型，再生成两张分页图纸（盖梁配筋详图/墩柱配筋详图）
            resolved = resolve_drawing_source(source)
            source_diagnostics = [
                item.model_dump(mode="json") for item in resolved.diagnostics
            ]
            source_has_errors = any(
                item["severity"] == "error" for item in source_diagnostics
            )
            sheets = build_reinforcement_sheets(
                resolved,
                design_group_id=group_id,
                member_piers=list(source.member_piers),
            )
            if notice:
                sheets = [
                    DrawingSheetSpec(
                        sheet_id=sheet.sheet_id,
                        title=sheet.title,
                        document=_add_notice(sheet.document, notice),
                    )
                    for sheet in sheets
                ]
            sheet_names = {
                "pier_general_arrangement": "pier_general_arrangement",
                "cap_reinforcement_detail": "cap_reinforcement_detail",
                "column_reinforcement_detail": "column_reinforcement_detail",
            }
            cad_paths_for_group: list[Path] = []
            preview_paths_for_group: list[Path] = []
            sheet_entries: list[dict[str, Any]] = []
            all_geometry_valid = True
            for sheet in sheets:
                sheet_id = sheet.sheet_id
                document = build_combined_layout(sheet.document, sheet_layout_config(sheet_id))
                diagnostics = [
                    validate_geometry(section).model_dump(mode="json")
                    for view in document.views
                    if view.view_id == "cad-layout"
                    for section in view.sections
                ]
                geometry_valid = (
                    all(item["valid"] for item in diagnostics)
                    and not source_has_errors
                )
                diagnostics.append(
                    {
                        "valid": not source_has_errors,
                        "diagnostics": source_diagnostics,
                        "region_areas": {},
                        "source": "reinforcement_resolution",
                    }
                )
                all_geometry_valid = all_geometry_valid and geometry_valid
                stem = sheet_names.get(sheet_id, sheet_id)
                names = {
                    "scr": f"{stem}.scr",
                    "svg": f"{stem}.svg",
                    "drawing_ir": f"{stem}.drawing_ir.json",
                    "geometry_diagnostics": f"{stem}.geometry_diagnostics.json",
                }
                # SCR 供 AutoCAD 中文版 SCRIPT 执行：按系统代码页（GBK）写入，避免中文乱码
                (staging / names["scr"]).write_text(
                    render_autocad_script(document), encoding="gbk", errors="replace"
                )
                (staging / names["svg"]).write_text(render_svg(document), encoding="utf-8")
                (staging / names["drawing_ir"]).write_text(
                    document.to_stable_json() + "\n", encoding="utf-8"
                )
                _write_json(staging / names["geometry_diagnostics"], diagnostics)
                paths = {key: staging / name for key, name in names.items()}
                cad_paths_for_group.append(paths["scr"])
                preview_paths_for_group.append(paths["svg"])
                sheet_entries.append(
                    {
                        "sheet_id": sheet_id,
                        "title": sheet.title,
                        "view_ids": [
                            view.view_id
                            for view in document.views
                            if view.view_id != "cad-layout"
                        ],
                        "geometry_valid": geometry_valid,
                        **{
                            f"{key}_name": name
                            for key, name in names.items()
                        },
                        "file_hashes": {key: _sha256(path) for key, path in paths.items()},
                    }
                )
            _write_schedule(staging / "bar_schedule.csv", source)
            target = drawings_root / group_id
            _replace_group_directory(staging, target)
            # staging 已移动为 target，路径重新映射
            target_cad_paths = [
                target / path.name for path in cad_paths_for_group
            ]
            target_preview_paths = [
                target / path.name for path in preview_paths_for_group
            ]
            entry = {
                "design_group_id": group_id,
                "task_id": source.task_id,
                "member_piers": source.member_piers,
                "issue_status": issue_status,
                "geometry_valid": all_geometry_valid,
                "source_hashes": source.source_hashes,
                # 只记录覆盖该组的人工风险，避免把无关风险挂到每个组上（人工风险全量清单在 design_manifest.json）
                "accepted_risks": list(
                    _covering_risks(
                        accepted_risks,
                        design_group_id=group_id,
                        task_id=source.task_id,
                    )
                ),
                "sheets": sheet_entries,
                "scr_paths": [
                    _relative(path, root) for path in target_cad_paths
                ],
                "svg_paths": [
                    _relative(path, root) for path in target_preview_paths
                ],
                "bar_schedule_path": _relative(target / "bar_schedule.csv", root),
            }
            entries.append(entry)
            generated.append(group_id)
            cad_paths.extend(str(path) for path in target_cad_paths)
            preview_paths.extend(str(path) for path in target_preview_paths)
        except Exception as exc:
            if staging.exists():
                shutil.rmtree(staging)
            failed.append({"design_group_id": group_id, "error": str(exc)})

    index_path = drawings_root / "drawing_index.json"
    manifest_path = root / "deliverables" / "design_manifest.json"
    index = {"schema_version": "drawing-index-v1", "groups": entries}
    _write_json(index_path, index)
    _write_json(
        manifest_path,
        {
            "schema_version": "design-manifest-v1",
            "drawing_index_path": _relative(index_path, root),
            "drawing_group_count": len(entries),
            "accepted_risks": list(accepted_risks),
            # 未出图的设计组要落盘备案：少数组验算不过只应排除该组，
            # 不能因为"索引不完整"就让其余已成功分组的图纸一起作废，
            # 也不能让恢复逻辑把这种合法部分交付永久判为未完成。
            "excluded_groups": list(failed),
        },
    )
    return DrawingPackageResult(
        success=not failed and bool(generated),
        generated_group_ids=generated,
        skipped_group_ids=[],
        failed_groups=failed,
        drawing_index_path=str(index_path),
        design_manifest_path=str(manifest_path),
        cad_script_paths=cad_paths,
        drawing_preview_paths=preview_paths,
    )


def build_final_deliverables(state: Mapping[str, Any]) -> dict[str, Any]:
    """从最终设计状态生成或复用确定性绘图成果。"""
    invalidated = set(state.get("invalidated_artifacts") or [])
    existing = state.get("drawing_package_result")
    if (
        isinstance(existing, Mapping)
        and existing.get("success") is True
        and "drawing_package" not in invalidated
        and "final_deliverables" not in invalidated
    ):
        return {
            "drawing_package_result": dict(existing),
            "drawing_index_path": state.get("drawing_index_path"),
            "design_manifest_path": state.get("design_manifest_path"),
            "cad_script_paths": list(state.get("cad_script_paths") or []),
            "drawing_preview_paths": list(state.get("drawing_preview_paths") or []),
        }

    required = (
        "pier_group_result",
        "dimension_design_result",
        "reinforcement_design_result",
    )
    if any(not state.get(key) for key in required):
        remaining = [
            name
            for name in state.get("invalidated_artifacts") or []
            if name not in {"drawing_package", "final_deliverables"}
        ]
        if remaining == list(state.get("invalidated_artifacts") or []):
            return {}
        return {
            "invalidated_artifacts": remaining,
            "minimal_rework_path": remaining,
        }
    sources = collect_drawing_group_sources(
        pier_group_result=state["pier_group_result"],
        dimension_design_result=state["dimension_design_result"],
        reinforcement_design_result=state["reinforcement_design_result"],
        capacity_check_result=state.get("capacity_check_result") or state.get("check_result"),
    )
    risks = list(state.get("accepted_risks") or [])
    result = generate_drawing_package(
        sources=sources,
        output_dir=state.get("output_dir") or "outputs",
        allow_unverified=bool(risks),
        accepted_risks=risks,
    )
    # 少数组未出图（例如该组验算未通过且无覆盖它的人工接受风险）不应作废整包：
    # 只要还有组成功出图，就按"部分交付"返回，失败组已记入 design_manifest.json
    # 的 excluded_groups。只有"一组都没出"才是真正的绘图失败。
    if not result.generated_group_ids:
        details = "; ".join(
            f"{item.get('design_group_id')}: {item.get('error')}"
            for item in result.failed_groups
        )
        raise RuntimeError(f"绘图成果包生成失败: {details or '没有成功生成的设计组'}")
    remaining = [
        name
        for name in state.get("invalidated_artifacts") or []
        if name not in {"drawing_package", "final_deliverables"}
    ]
    return {
        "drawing_package_result": result.model_dump(mode="json"),
        "drawing_index_path": result.drawing_index_path,
        "design_manifest_path": result.design_manifest_path,
        "cad_script_paths": result.cad_script_paths,
        "drawing_preview_paths": result.drawing_preview_paths,
        "invalidated_artifacts": remaining,
        "minimal_rework_path": remaining,
    }
