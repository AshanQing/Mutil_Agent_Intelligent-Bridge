from __future__ import annotations

import math
from pathlib import Path

import pytest

from bridge_agents.drawing.reinforcement_resolution import (
    load_reinforcement_example,
    resolve_reinforcement_examples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolved_column():
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
    return resolve_reinforcement_examples(cap, column).column


def test_column_section_places_all_44_main_bars_on_declared_circle() -> None:
    from bridge_agents.drawing.column_reinforcement import (
        compile_column_reinforcement,
    )

    detail = compile_column_reinforcement(_resolved_column())

    assert len(detail.longitudinal_centers) == 44
    assert detail.longitudinal_centers[0] == pytest.approx((841, 0))
    assert detail.longitudinal_centers[22] == pytest.approx((-841, 0))
    assert all(
        math.hypot(x, y) == pytest.approx(841)
        for x, y in detail.longitudinal_centers
    )
    assert detail.angular_spacing_deg == pytest.approx(360 / 44)


def test_column_detail_preserves_n2_n3_and_n4_geometry() -> None:
    from bridge_agents.drawing.column_reinforcement import (
        compile_column_reinforcement,
    )

    detail = compile_column_reinforcement(_resolved_column())

    assert detail.spiral_centerline_radius_mm == 860
    assert [(zone.y_from_mm, zone.y_to_mm, zone.pitch_mm) for zone in detail.spiral_zones] == [
        (0, 2700, 100),
        (2700, 9000, 150),
        (9000, 9069, 69),
        (9069, 11769, 100),
    ]
    assert detail.ordinary_hoop_positions_mm == [
        11969, 12169, 12369, 12569, 12769, 12969, 13169
    ]
    assert detail.strengthening_radius_mm == 807.5
    assert detail.strengthening_positions_mm[-1] == 10885
    assert detail.strengthening_semantic_review_required is True


def test_column_detail_represents_one_member_of_identical_pair() -> None:
    from bridge_agents.drawing.column_reinforcement import (
        compile_column_reinforcement,
    )

    detail = compile_column_reinforcement(_resolved_column())

    assert detail.source_column_count == 2
    assert detail.representative_member_count == 1

