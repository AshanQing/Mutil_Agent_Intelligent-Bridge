from __future__ import annotations

from pathlib import Path

import pytest

from bridge_agents.drawing.reinforcement_resolution import (
    load_reinforcement_example,
    resolve_reinforcement_examples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolved_cap():
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
    return resolve_reinforcement_examples(cap, column).cap


def _path(mark: str):
    from bridge_agents.drawing.cap_reinforcement import compile_cap_bar_paths

    return compile_cap_bar_paths(_resolved_cap())[mark]


def test_n1_top_full_length_bar_is_horizontal_at_cover_y() -> None:
    n1 = _path("N1")
    assert len(n1.components) == 1
    start, end = n1.components[0].segments[0].start, n1.components[0].segments[0].end
    assert start == pytest.approx((-5790, 1740))
    assert end == pytest.approx((5790, 1740))


def test_n2_bottom_full_length_bar_follows_bottom_slope() -> None:
    n2 = _path("N2")
    assert len(n2.components) == 1
    segments = n2.components[0].segments
    assert segments[0].start == pytest.approx((-5790, 865.6522), abs=0.01)
    ends = [segment.end for segment in segments]
    assert len(ends) == 3
    assert ends[0] == pytest.approx((-4700, 60), abs=0.01)
    assert ends[1] == pytest.approx((4700, 60), abs=0.01)
    assert ends[2] == pytest.approx((5790, 865.6522), abs=0.01)


def test_n3_path_intersects_cantilever_bottom_and_reaches_centerline() -> None:
    n3 = _path("N3")
    source = n3.components[0]
    segments = source.segments
    # 左斜端：45° 与悬臂段斜面保护层线求交（弯点已改 pier±300）
    assert segments[0].start == pytest.approx((-5100.2, 383.8), abs=0.5)
    # 顶水平段两弯点（pier±300 = -3800/-3200）
    assert segments[1].start == pytest.approx((-3800, 1684))
    assert segments[1].end == pytest.approx((-3200, 1684))
    # 右斜端 + step_2 到跨中
    assert segments[-1].end == pytest.approx((0, 88))
    # 镜像组件与 source 在跨中拼接（连续不断开）
    mirrored = n3.components[1]
    assert mirrored.segments[0].start == pytest.approx((0, 88))
    assert mirrored.segments[-1].end == pytest.approx((5100.2, 383.8), abs=0.5)


def test_n4_path_without_step_two_ends_at_bottom_cover() -> None:
    n4 = _path("N4")
    source = n4.components[0]
    segments = source.segments
    # 弯点 pier-(800+300)=-4600 / pier+(300+650+650)=-1900，y=1800-60-φ=1712
    assert segments[0].start == pytest.approx((-5576.3, 735.7), abs=0.5)
    assert segments[1].start == pytest.approx((-4600, 1712))
    assert segments[1].end == pytest.approx((-1900, 1712))
    assert segments[-1].end == pytest.approx((-276, 88))


def test_n5_independent_diagonal_ends_at_bottom_cover() -> None:
    n5 = _path("N5")
    source = n5.components[0]
    assert source.segments[0].start == pytest.approx((-2550, 1684))
    assert source.segments[-1].end == pytest.approx((-982, 116))


def test_n6_arc_is_90_degrees_tangent_to_45_degree_branches() -> None:
    n6 = _path("N6")
    source = n6.components[0]
    arc = next(segment for segment in source.segments if segment.kind == "arc")
    # 弧角 90°：圆心在 pier 上，两端点相对圆心角度 45°/135°
    assert arc.center == pytest.approx((-3500, 893.0), abs=0.1)
    assert arc.radius_mm == pytest.approx(350)
    assert arc.start == pytest.approx((-3747.5, 1140.5), abs=0.1)
    assert arc.end == pytest.approx((-3252.5, 1140.5), abs=0.1)
    assert arc.arc_bulge == "upper"
    # 左斜段与底面保护层线求交
    assert source.segments[0].start == pytest.approx((-4757.5, 130.5), abs=0.5)
    # 右斜段：弧右端 -> right_bottom_end
    assert source.segments[-1].end == pytest.approx((-2200, 88))


def test_n7_local_top_bar_two_components_around_pier() -> None:
    n7 = _path("N7")
    assert len(n7.components) == 2
    first = n7.components[0]
    second = n7.components[1]
    assert first.segments[0].start == pytest.approx((-5600, 1712))
    assert first.segments[0].end == pytest.approx((-1400, 1712))
    assert second.segments[0].start == pytest.approx((1400, 1712))
    assert second.segments[0].end == pytest.approx((5600, 1712))


def test_all_bar_endpoints_stay_inside_cap_outline() -> None:
    from bridge_agents.drawing.cap_reinforcement import (
        bottom_y,
        compile_cap_bar_paths,
    )

    cap = _resolved_cap()
    compiled = compile_cap_bar_paths(cap)
    half = cap.length_mm / 2
    for mark in ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]:
        for component in compiled[mark].components:
            for segment in component.segments:
                for point in (segment.start, segment.end):
                    x, y = point
                    assert abs(x) <= half + 1e-6, f"{mark} 端点 x={x:g} 越出盖梁"
                    assert y >= bottom_y(cap, x) - 1e-6, (
                        f"{mark} 端点 ({x:g},{y:g}) 低于底面 {bottom_y(cap, x):g}"
                    )
                    assert y <= cap.height_mid_mm + 1e-6, (
                        f"{mark} 端点 y={y:g} 高于顶面"
                    )


def test_section_intersections_use_compiled_longitudinal_path() -> None:
    from bridge_agents.drawing.cap_reinforcement import (
        compile_cap_bar_paths,
        intersections_at_x,
    )

    paths = compile_cap_bar_paths(_resolved_cap())

    assert intersections_at_x(paths["N3"], -3500) == pytest.approx([1684])
    assert intersections_at_x(paths["N7"], 0) == []
    assert intersections_at_x(paths["N7"], -3500) == pytest.approx([1712])
