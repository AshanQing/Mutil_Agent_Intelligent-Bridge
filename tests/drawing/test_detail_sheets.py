from __future__ import annotations

from pathlib import Path

from bridge_agents.drawing.models import CircleEntity, DimensionEntity, LineEntity
from bridge_agents.drawing.reinforcement_resolution import (
    load_reinforcement_example,
    resolve_reinforcement_examples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolved():
    cap = load_reinforcement_example(
        REPO_ROOT / "samples" / "reinforcement_design_samples.yaml",
        example_name="示例1",
        expected_drawing_id=12,
    )
    column = load_reinforcement_example(
        REPO_ROOT / "samples" / "reinforcement_column.yaml",
        example_name="示例1",
        expected_drawing_id=14,
    )
    return resolve_reinforcement_examples(cap, column)


def _entities(document, view_id: str):
    view = next(view for view in document.views if view.view_id == view_id)
    return [entity for section in view.sections for entity in section.entities]


def test_builds_two_member_scoped_drawing_sheets() -> None:
    from bridge_agents.drawing.detail_sheets import build_reinforcement_sheets

    sheets = build_reinforcement_sheets(
        _resolved(), design_group_id="sample-12-14", member_piers=["示例墩1", "示例墩2"]
    )

    assert [sheet.sheet_id for sheet in sheets] == [
        "cap_reinforcement_detail",
        "column_reinforcement_detail",
    ]
    assert all(sheet.document.units == "mm" for sheet in sheets)


def test_general_arrangement_relates_two_columns_to_cap_and_height() -> None:
    from bridge_agents.drawing.detail_sheets import _general_arrangement

    view = _general_arrangement(_resolved())
    entities = [
        entity for section in view.sections for entity in section.entities
    ]

    outlines = [
        entity
        for entity in entities
        if isinstance(entity, LineEntity)
        and entity.metadata.get("role") == "column_outline"
    ]
    assert {entity.metadata["column_index"] for entity in outlines} == {1, 2}
    assert max(point.y for line in outlines for point in (line.start, line.end)) == 11769
    assert any(
        isinstance(entity, DimensionEntity)
        and entity.metadata.get("dimension_role") == "column_spacing"
        for entity in entities
    )
    assert any(
        isinstance(entity, DimensionEntity)
        and entity.metadata.get("dimension_role") == "column_height"
        for entity in entities
    )


def test_cap_sheet_separates_skeletons_and_builds_station_sections() -> None:
    from bridge_agents.drawing.detail_sheets import build_reinforcement_sheets

    sheet = build_reinforcement_sheets(
        _resolved(), design_group_id="sample-12-14", member_piers=["P1", "P2"]
    )[0]
    view_ids = [view.view_id for view in sheet.document.views]

    assert "cap-skeleton-1" in view_ids
    assert "cap-skeleton-2" in view_ids
    assert {"cap-section-end", "cap-section-support", "cap-section-mid"}.issubset(
        view_ids
    )
    skeleton_1_marks = {
        entity.metadata.get("bar_mark")
        for entity in _entities(sheet.document, "cap-skeleton-1")
        if entity.metadata.get("bar_mark")
    }
    skeleton_2_marks = {
        entity.metadata.get("bar_mark")
        for entity in _entities(sheet.document, "cap-skeleton-2")
        if entity.metadata.get("bar_mark")
    }
    assert skeleton_1_marks == {"N1", "N2", "N3", "N4", "N5", "N6"}
    assert skeleton_2_marks == {"N1", "N2", "N7"}

    mid_bars = [
        entity
        for entity in _entities(sheet.document, "cap-section-mid")
        if isinstance(entity, CircleEntity)
        and entity.metadata.get("bar_role") == "cap_longitudinal_section"
    ]
    assert "N7" not in {entity.metadata["bar_mark"] for entity in mid_bars}
    assert len(
        {
            round(entity.center.x, 3)
            for entity in mid_bars
            if entity.metadata["bar_mark"] == "N1"
        }
    ) == 23


def test_cap_section_contains_four_cage_components() -> None:
    from bridge_agents.drawing.detail_sheets import build_reinforcement_sheets

    sheet = build_reinforcement_sheets(
        _resolved(), design_group_id="sample-12-14", member_piers=["P1", "P2"]
    )[0]
    support = _entities(sheet.document, "cap-section-support")
    cage_ids = {
        entity.metadata.get("cage_component_id")
        for entity in support
        if entity.metadata.get("cage_component_id")
    }
    assert cage_ids == {"N10_left", "N10_right", "N11_left", "N11_right"}


def test_column_sheet_draws_one_representative_elevation_and_44_bar_sections() -> None:
    from bridge_agents.drawing.detail_sheets import build_reinforcement_sheets

    sheet = build_reinforcement_sheets(
        _resolved(), design_group_id="sample-12-14", member_piers=["P1", "P2"]
    )[1]
    elevation = _entities(sheet.document, "column-representative-elevation")
    outlines = [
        entity
        for entity in elevation
        if isinstance(entity, LineEntity)
        and entity.metadata.get("role") == "column_outline"
    ]
    assert {entity.metadata.get("representative_column") for entity in outlines} == {
        True
    }
    assert {entity.metadata.get("bar_mark") for entity in elevation} >= {
        "N1", "N2", "N3", "N4"
    }

    body_bars = [
        entity
        for entity in _entities(sheet.document, "column-section-body")
        if isinstance(entity, CircleEntity)
        and entity.metadata.get("bar_role") == "column_longitudinal"
    ]
    embed_bars = [
        entity
        for entity in _entities(sheet.document, "column-section-embed")
        if isinstance(entity, CircleEntity)
        and entity.metadata.get("bar_role") == "column_longitudinal"
    ]
    assert len(body_bars) == 44
    assert len(embed_bars) == 44


def test_column_elevation_spiral_is_one_continuous_zigzag_with_true_pitch_spacing() -> None:
    """N2 螺旋箍立面必须是连续斜向 Z 形折线（不是 X 交叉），且同侧相邻折点间距 = 分区 pitch。

    示例1 分区 100×27 + 150×42 + 69×1 + 100×27 = 97 圈，每圈两段斜线，共 194 段，
    段与段首尾相连（gap=0），柱底至柱顶贯通。
    """
    from bridge_agents.drawing.detail_sheets import build_reinforcement_sheets

    sheet = build_reinforcement_sheets(
        _resolved(), design_group_id="sample-12-14", member_piers=["P1", "P2"]
    )[1]
    elevation = _entities(sheet.document, "column-representative-elevation")
    helix = [
        entity
        for entity in elevation
        if isinstance(entity, LineEntity)
        and entity.metadata.get("continuous_helix") is True
    ]
    # 每圈两段斜线：97 圈 = 194 段
    assert len(helix) == 2 * (27 + 42 + 1 + 27)

    def key(entity):
        import re

        match = re.search(r"(\d+)$", entity.entity_id)
        return int(match.group(1)) if match else 0

    helix.sort(key=key)
    # 段与段首尾相接：上段 end 恰好等于下段 start
    for prev, nxt in zip(helix, helix[1:]):
        assert abs(nxt.start.x - prev.end.x) < 1e-6
        assert abs(nxt.start.y - prev.end.y) < 1e-6
    # 横向只在螺旋中心线半径两侧（±860）交替
    x_values = {round(entity.start.x, 3) for entity in helix}
    x_values |= {round(entity.end.x, 3) for entity in helix}
    assert x_values == {-860.0, 860.0}
    # 同侧相邻折点（回到同一相位）间距 = pitch ∈ {69, 100, 150}
    vertices = [entity.start for entity in helix] + [helix[-1].end]
    left_y = sorted(point.y for point in vertices if abs(point.x + 860.0) < 1e-6)
    same_side_spacings = {
        round(b - a, 3) for a, b in zip(left_y, left_y[1:])
    }
    assert same_side_spacings <= {69.0, 100.0, 150.0}
    # 柱底到柱顶贯通
    assert min(helix[0].start.y, helix[-1].end.y) == 0.0
    assert max(helix[0].start.y, helix[-1].end.y) == 11769.0


def test_column_schedule_uses_resolved_values_instead_of_sample_constants() -> None:
    from bridge_agents.drawing.detail_sheets import build_reinforcement_sheets
    from bridge_agents.drawing.models import TextEntity

    sheet = build_reinforcement_sheets(
        _resolved(), design_group_id="sample-12-14", member_piers=["P1", "P2"]
    )[1]
    texts = [
        entity.text
        for entity in _entities(sheet.document, "column-bar-schedule")
        if isinstance(entity, TextEntity)
    ]
    joined = "\n".join(texts)

    assert "N2  Φ10" in joined
    assert "9069-11769@100" in joined
    assert "N4  Φ28  6道  环径1615" in joined
