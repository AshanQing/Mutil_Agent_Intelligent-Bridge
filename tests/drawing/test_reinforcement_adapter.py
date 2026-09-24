from __future__ import annotations

import pytest

import bridge_agents.drawing as drawing
from bridge_agents.drawing.reinforcement_adapter import (
    BarScheduleRow,
    DrawingExpressionError,
    adapt_cap_geometry,
    adapt_column_geometry,
    build_bar_schedule,
    evaluate_drawing_expression,
)
from bridge_agents.drawing.views import build_reinforcement_drawing


def test_drawing_package_exposes_reinforcement_translation_contract() -> None:
    assert drawing.BarScheduleRow is BarScheduleRow
    assert drawing.evaluate_drawing_expression is evaluate_drawing_expression
    assert drawing.build_bar_schedule is build_bar_schedule
    assert drawing.build_reinforcement_drawing is build_reinforcement_drawing


def test_expression_evaluator_resolves_declared_geometry_symbols() -> None:
    value = evaluate_drawing_expression(
        "column_diameter - 2 * concrete_cover - dia",
        {"column_diameter": 1600, "concrete_cover": 50, "dia": 28},
    )
    assert value == 1472


def test_expression_evaluator_supports_only_approved_math_functions() -> None:
    value = evaluate_drawing_expression(
        "radius * math.tan(math.radians(bend_angle))",
        {"radius": 350, "bend_angle": 45},
    )
    assert value == pytest.approx(350)


def test_expression_evaluator_validates_equation_and_returns_resolved_value() -> None:
    value = evaluate_drawing_expression(
        "100 * 27 + 150 * 42 + 69 + 100 * 27 = column_height",
        {"column_height": 11769},
    )

    assert value == 11769


def test_expression_evaluator_rejects_an_inconsistent_equation() -> None:
    with pytest.raises(DrawingExpressionError, match="等式校验失败"):
        evaluate_drawing_expression(
            "100 * 27 + 150 * 42 + 69 + 100 * 27 = column_height",
            {"column_height": 11770},
        )


@pytest.mark.parametrize(
    "expression",
    [
        "unknown_symbol + 1",
        "__import__('os').system('echo unsafe')",
        "column_diameter / 0",
    ],
)
def test_expression_evaluator_rejects_unknown_calls_and_division_by_zero(
    expression: str,
) -> None:
    with pytest.raises(DrawingExpressionError):
        evaluate_drawing_expression(expression, {"column_diameter": 1600})


def test_geometry_adapter_converts_task_metres_to_drawing_millimetres(
    drawing_source,
) -> None:
    cap = adapt_cap_geometry(drawing_source)
    column = adapt_column_geometry(drawing_source)

    assert cap.length_mm == 12000
    assert cap.width_mm == 2200
    assert cap.height_mid_mm == 1800
    assert column.count == 2
    assert column.diameter_mm == 1600
    assert column.spacing_mm == 6500
    assert column.height_mm == 10500
    assert column.controlling_pier_id == "P2"


def test_column_geometry_uses_group_height_and_cover_x_fallbacks(
    drawing_source,
) -> None:
    source = drawing_source.model_copy(deep=True)
    task_input = source.design_context["reinforcement_task"]["桥墩尺寸信息"]
    del task_input["墩柱几何信息"]["column_height"]
    del task_input["保护层与控制参数"]["concrete_cover"]

    column = adapt_column_geometry(source)

    assert column.height_mm == 10500
    assert column.concrete_cover_mm == 50


def test_bar_schedule_has_unique_marks_for_cap_and_column(drawing_source) -> None:
    rows = build_bar_schedule(drawing_source)

    assert [row.mark for row in rows] == [
        "N1",
        "N2",
        "N3",
        "N5",
        "N6",
        "C1",
        "C2",
    ]
    assert len({row.mark for row in rows}) == len(rows)
    column_main = next(row for row in rows if row.mark == "C1")
    assert column_main.member == "pier_column"
    assert column_main.diameter_mm == 28
    assert column_main.count == 16
    column_spiral = next(row for row in rows if row.mark == "C2")
    assert column_spiral.spacing_mm == 100
    assert "100/200" in column_spiral.geometry_description


def test_bar_schedule_scopes_duplicate_source_marks_by_member(drawing_source) -> None:
    source = drawing_source.model_copy(deep=True)
    column = source.reinforcement["reinforcement"]["pier_column"]
    column["longitudinal_bars"][0]["id"] = "N1"
    column["spiral_stirrups"][0]["id"] = "N2"

    rows = build_bar_schedule(source)

    assert len(rows) == 7
    assert len({row.key for row in rows}) == 7
    assert next(row for row in rows if row.key == "pier_cap:N1").mark == "N1"
    assert next(row for row in rows if row.key == "pier_column:N1").mark == "N1"
