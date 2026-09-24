from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import bridge_agents.drawing as drawing
from bridge_agents.drawing.package import (
    DrawingPackageResult,
    UnverifiedDrawingSourceError,
    generate_drawing_package,
)


def test_drawing_package_exposes_group_package_contract() -> None:
    assert drawing.DrawingPackageResult is DrawingPackageResult
    assert drawing.generate_drawing_package is generate_drawing_package


def test_package_refuses_unverified_group_by_default(
    tmp_path: Path,
    drawing_source,
) -> None:
    source = drawing_source.model_copy(update={"check_status": "failed"})

    with pytest.raises(UnverifiedDrawingSourceError):
        generate_drawing_package(sources=[source], output_dir=tmp_path)


def test_package_writes_three_paged_sheets_per_design_group(
    tmp_path: Path,
    drawing_source,
) -> None:
    result = generate_drawing_package(
        sources=[drawing_source],
        output_dir=tmp_path,
    )

    assert result.success is True, f"failed_groups={result.failed_groups}"
    assert result.generated_group_ids == ["B1-U1-1"]
    index = json.loads(Path(result.drawing_index_path).read_text(encoding="utf-8"))
    assert len(index["groups"]) == 1
    entry = index["groups"][0]
    assert entry["member_piers"] == ["P1", "P2"]
    assert entry["issue_status"] == "verified"
    assert entry["geometry_valid"] is True

    # 分页链路：每设计组两张图纸（盖梁详图/墩柱详图），各含一个 SCR
    assert {sheet["sheet_id"] for sheet in entry["sheets"]} == {
        "cap_reinforcement_detail",
        "column_reinforcement_detail",
    }
    assert len(entry["scr_paths"]) == 2
    assert len(entry["svg_paths"]) == 2
    for rel in entry["scr_paths"]:
        scr_path = tmp_path / rel
        assert scr_path.is_file()
        # SCR 按系统代码页（GBK）写入，不含 BOM
        scr_bytes = scr_path.read_bytes()
        assert not scr_bytes.startswith(b"\xef\xbb\xbf")
        assert "N1" in scr_bytes.decode("gbk")

    with (tmp_path / entry["bar_schedule_path"]).open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mark"] for row in rows] == [
        "N1",
        "N2",
        "N3",
        "N5",
        "N6",
        "C1",
        "C2",
    ]


def test_allowed_unverified_preview_contains_visible_notice(
    tmp_path: Path,
    drawing_source,
) -> None:
    source = drawing_source.model_copy(update={"check_status": "missing"})

    result = generate_drawing_package(
        sources=[source],
        output_dir=tmp_path,
        allow_unverified=True,
    )

    index = json.loads(Path(result.drawing_index_path).read_text(encoding="utf-8"))
    entry = index["groups"][0]
    assert entry["issue_status"] == "draft_unverified"
    # 至少一张图纸的 SVG 出现醒目风险注记
    notices = []
    for rel in entry["svg_paths"]:
        svg = (tmp_path / rel).read_text(encoding="utf-8")
        notices.append("自动验算未完成，仅供复核" in svg)
    assert any(notices)


def test_accepted_risk_package_records_risk_and_visible_notice(
    tmp_path: Path,
    drawing_source,
) -> None:
    source = drawing_source.model_copy(update={"check_status": "failed"})
    risks = [{"scope": "modeling_check", "reason": "人工接受当前结果"}]

    result = generate_drawing_package(
        sources=[source],
        output_dir=tmp_path,
        accepted_risks=risks,
    )

    index = json.loads(Path(result.drawing_index_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result.design_manifest_path).read_text(encoding="utf-8"))
    entry = index["groups"][0]
    assert entry["issue_status"] == "accepted_risk"
    assert entry["accepted_risks"] == risks
    assert manifest["accepted_risks"] == risks
    notices = []
    for rel in entry["svg_paths"]:
        svg = (tmp_path / rel).read_text(encoding="utf-8")
        notices.append("含人工接受风险" in svg)
    assert any(notices)


# --------------------------------------------------------------------------- #
# 出图准入细化：风险放行按"是否覆盖该组"判定，不再"有任意风险就整体放行"
# --------------------------------------------------------------------------- #
def test_passed_group_stays_verified_with_unrelated_layout_risk(
    tmp_path: Path,
    drawing_source,
) -> None:
    """验算通过的组不因"只接受了布跨风险"被标成 accepted_risk（复现 示例项目K12 的标注失真）。"""
    source = drawing_source.model_copy(update={"check_status": "passed"})
    risks = [{"scope": "layout_revision", "reason": "人工接受剩余碰撞风险", "metrics": {}}]

    result = generate_drawing_package(
        sources=[source],
        output_dir=tmp_path,
        accepted_risks=risks,
    )

    index = json.loads(Path(result.drawing_index_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result.design_manifest_path).read_text(encoding="utf-8"))
    entry = index["groups"][0]
    assert entry["issue_status"] == "verified"
    # 布跨风险不覆盖该组的验算维度：不挂到组上，但仍完整记录在交付清单里
    assert entry["accepted_risks"] == []
    assert manifest["accepted_risks"] == risks


def test_failed_group_is_not_released_by_unrelated_risk(
    tmp_path: Path,
    drawing_source,
) -> None:
    """验算失败的组不能被"别处接受的风险"放行，必须由覆盖该组的人工风险兜底。"""
    source = drawing_source.model_copy(update={"check_status": "failed"})
    risks = [{"scope": "layout_revision", "reason": "人工接受剩余碰撞风险"}]

    with pytest.raises(UnverifiedDrawingSourceError, match="没有覆盖该组的人工接受风险"):
        generate_drawing_package(
            sources=[source],
            output_dir=tmp_path,
            accepted_risks=risks,
            allow_unverified=True,
        )


def test_structural_risk_releases_only_its_own_groups(drawing_source) -> None:
    from bridge_agents.drawing.package import _issue_status

    covered = drawing_source.model_copy(
        update={"design_group_id": "2-2-3-G2", "task_id": "2-2-3-G2", "check_status": "failed"}
    )
    other = drawing_source.model_copy(
        update={"design_group_id": "1-1-1-G1", "task_id": "1-1-1-G1", "check_status": "failed"}
    )
    risks = [
        {
            "scope": "structural_design",
            "failed_task_ids": ["2-2-3-G2"],
            "reason": "人工接受该组部分成果",
        }
    ]

    status, notice = _issue_status(covered, allow_unverified=False, accepted_risks=risks)
    assert status == "accepted_risk"
    assert "结构设计" in notice

    with pytest.raises(UnverifiedDrawingSourceError):
        _issue_status(other, allow_unverified=False, accepted_risks=risks)


def test_dimension_unit_risk_covers_its_reinforcement_groups(drawing_source) -> None:
    from bridge_agents.drawing.package import _issue_status

    group = drawing_source.model_copy(
        update={"design_group_id": "2-2-1-G1", "task_id": "2-2-1-G1", "check_status": "failed"}
    )
    unrelated = drawing_source.model_copy(
        update={"design_group_id": "2-6-1-G1", "task_id": "2-6-1-G1", "check_status": "failed"}
    )
    risks = [
        {
            "scope": "structural_design",
            "failed_dimension_unit_ids": ["2-2"],
            "reason": "人工接受尺寸设计部分成果",
        }
    ]

    status, _ = _issue_status(group, allow_unverified=False, accepted_risks=risks)
    assert status == "accepted_risk"

    with pytest.raises(UnverifiedDrawingSourceError):
        _issue_status(unrelated, allow_unverified=False, accepted_risks=risks)
