from __future__ import annotations

import ast
import math
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .models import DrawingGroupSource


def resolve_z_positions(z_patterns: Mapping[str, Any], name: Any) -> list[int]:
    """解析 z 方向布置位置，支持 "pattern_1+pattern_3" 这类组合写法（取位置并集）。

    配筋 LLM 会用"A+B"表达某个骨架/钢筋同时落在两套分组布置位置上；解析器原先只认
    z_patterns 里登记的单名，遇到组合名直接 KeyError，导致整组图纸解析失败
    （2026-09-16 示例项目K29 运行中 8/13 组即因此无法出图）。
    """
    key = str(name or "").strip()
    if key in z_patterns:
        return [int(position) for position in z_patterns[key]]
    if "+" in key:
        positions: list[int] = []
        for part in (item.strip() for item in key.split("+")):
            if part in z_patterns:
                positions.extend(int(position) for position in z_patterns[part])
        if positions:
            return sorted(set(positions))
    raise DrawingExpressionError(
        f"未知 z_pattern: {key!r}；已登记: {sorted(str(item) for item in z_patterns)}"
    )


ALLOWED_SYMBOLS = frozenset(
    {
        "cap_length",
        "cap_width",
        "cap_height_mid",
        "cap_height_end",
        "cantilever_length",
        "column_count",
        "column_diameter",
        "column_height",
        "column_spacing",
        "concrete_cover",
        "cover_x",
        "cover_y",
        "cover_z",
        "dia",
        "pier_centerline_x",
        "cap_centerline_x",
        "first_y",
        "last_y",
        "count",
        "radius",
        "bend_angle",
    }
)

ALLOWED_FUNCTIONS = {
    "abs": abs,
    "cos": math.cos,
    "radians": math.radians,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}


class DrawingExpressionError(ValueError):
    pass


class _AdapterModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CapGeometry(_AdapterModel):
    length_mm: float = Field(gt=0)
    width_mm: float = Field(gt=0)
    height_mid_mm: float = Field(gt=0)
    height_end_mm: float = Field(gt=0)
    cantilever_mm: float = Field(ge=0)
    cover_x_mm: float = Field(ge=0)
    cover_y_mm: float = Field(ge=0)
    cover_z_mm: float = Field(ge=0)


class ColumnGeometry(_AdapterModel):
    count: int = Field(gt=0)
    diameter_mm: float = Field(gt=0)
    spacing_mm: float = Field(ge=0)
    height_mm: float = Field(gt=0)
    concrete_cover_mm: float = Field(ge=0)
    controlling_pier_id: str | None = None


class BarScheduleRow(_AdapterModel):
    mark: str
    member: Literal["pier_cap", "pier_column"]
    category: str
    diameter_mm: float = Field(gt=0)
    count: int | None = Field(default=None, gt=0)
    spacing_mm: float | None = Field(default=None, gt=0)
    geometry_description: str

    @property
    def key(self) -> str:
        return f"{self.member}:{self.mark}"


def _require_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise DrawingExpressionError(f"{name} 不能使用布尔值。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DrawingExpressionError(f"{name} 不是有效数值: {value!r}") from exc
    if not math.isfinite(number):
        raise DrawingExpressionError(f"{name} 必须是有限数值。")
    return number


def _evaluate_node(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        return _require_number(node.value, name="表达式常量")
    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_SYMBOLS or node.id not in values:
            raise DrawingExpressionError(f"表达式包含未声明变量: {node.id}")
        return _require_number(values[node.id], name=node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _evaluate_node(node.operand, values)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.Call):
        if node.keywords:
            raise DrawingExpressionError("受控数学函数不接受关键字参数。")
        function_name: str | None = None
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
        ):
            function_name = node.func.attr
        function = ALLOWED_FUNCTIONS.get(function_name or "")
        if function is None:
            raise DrawingExpressionError("表达式包含未批准的函数调用。")
        arguments = [_evaluate_node(argument, values) for argument in node.args]
        try:
            return _require_number(function(*arguments), name=function_name or "函数")
        except (TypeError, ValueError, OverflowError) as exc:
            raise DrawingExpressionError(
                f"受控数学函数计算失败: {function_name}"
            ) from exc
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, values)
        right = _evaluate_node(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if math.isclose(right, 0.0, abs_tol=1e-12):
                raise DrawingExpressionError("表达式发生除零。")
            return left / right
    raise DrawingExpressionError(
        f"表达式包含不允许的语法: {type(node).__name__}"
    )


def evaluate_drawing_expression(
    expression: str | float | int,
    values: Mapping[str, float],
) -> float:
    if isinstance(expression, (int, float)) and not isinstance(expression, bool):
        return _require_number(expression, name="表达式")
    if not isinstance(expression, str) or not expression.strip():
        raise DrawingExpressionError("表达式必须是非空字符串或数值。")
    normalized = expression.strip()
    if normalized.count("=") == 1:
        left_expression, right_expression = (
            part.strip() for part in normalized.split("=", maxsplit=1)
        )
        if not left_expression or not right_expression:
            raise DrawingExpressionError(f"表达式等式缺少一侧: {expression!r}")
        left = evaluate_drawing_expression(left_expression, values)
        right = evaluate_drawing_expression(right_expression, values)
        if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6):
            raise DrawingExpressionError(
                f"等式校验失败: {left_expression}={left:g}, "
                f"{right_expression}={right:g}"
            )
        return right
    try:
        tree = ast.parse(normalized, mode="eval")
        result = _evaluate_node(tree.body, values)
    except (SyntaxError, RecursionError) as exc:
        raise DrawingExpressionError(f"表达式语法非法: {expression!r}") from exc
    return _require_number(result, name="表达式结果")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _task_dimension_parts(
    source: DrawingGroupSource,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    task = _mapping(source.design_context.get("reinforcement_task"))
    task_input = _mapping(task.get("桥墩尺寸信息"))
    cap = _mapping(task_input.get("盖梁几何信息"))
    column = _mapping(task_input.get("墩柱几何信息"))
    cover = _mapping(task_input.get("保护层与控制参数"))
    if cap and column:
        return cap, column, cover

    dimension = _mapping(source.dimension)
    cap_cn = _mapping(dimension.get("盖梁尺寸"))
    column_cn = _mapping(dimension.get("墩柱尺寸"))
    return (
        {
            "cap_length": cap_cn.get("长度"),
            "cap_width": cap_cn.get("宽度"),
            "cap_height_mid": cap_cn.get("中高"),
            "cap_height_end": cap_cn.get("端高"),
            "cantilever_length": cap_cn.get("悬臂"),
        },
        {
            "column_count": column_cn.get("数量"),
            "column_diameter": column_cn.get("直径"),
            "column_spacing": column_cn.get("中心间距"),
        },
        {},
    )


def _metres_to_mm(value: Any, *, name: str) -> float:
    return 1000.0 * _require_number(value, name=name)


def adapt_cap_geometry(source: DrawingGroupSource) -> CapGeometry:
    cap, _, cover = _task_dimension_parts(source)
    return CapGeometry(
        length_mm=_metres_to_mm(cap.get("cap_length"), name="cap_length"),
        width_mm=_metres_to_mm(cap.get("cap_width"), name="cap_width"),
        height_mid_mm=_metres_to_mm(
            cap.get("cap_height_mid"), name="cap_height_mid"
        ),
        height_end_mm=_metres_to_mm(
            cap.get("cap_height_end"), name="cap_height_end"
        ),
        cantilever_mm=_metres_to_mm(
            cap.get("cantilever_length") or 0.0,
            name="cantilever_length",
        ),
        cover_x_mm=_require_number(cover.get("cover_x", 50.0), name="cover_x"),
        cover_y_mm=_require_number(cover.get("cover_y", 50.0), name="cover_y"),
        cover_z_mm=_require_number(cover.get("cover_z", 60.0), name="cover_z"),
    )


def adapt_column_geometry(source: DrawingGroupSource) -> ColumnGeometry:
    _, column, cover = _task_dimension_parts(source)
    pier_group = _mapping(source.design_context.get("pier_group"))
    height_m = column.get("column_height")
    if height_m is None:
        height_m = pier_group.get("controlling_net_height_m")
    controlling_pier = pier_group.get("controlling_pier_id")
    return ColumnGeometry(
        count=int(_require_number(column.get("column_count"), name="column_count")),
        diameter_mm=_metres_to_mm(
            column.get("column_diameter"), name="column_diameter"
        ),
        spacing_mm=_metres_to_mm(
            column.get("column_spacing") or 0.0,
            name="column_spacing",
        ),
        height_mm=_metres_to_mm(height_m, name="column_height"),
        concrete_cover_mm=_require_number(
            cover.get("concrete_cover", cover.get("cover_x", 50.0)),
            name="concrete_cover",
        ),
        controlling_pier_id=(
            str(controlling_pier).strip() if controlling_pier is not None else None
        ),
    )


def _reinforcement_parts(
    source: DrawingGroupSource,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    root = _mapping(source.reinforcement.get("reinforcement"))
    return _mapping(root.get("pier_cap")), _mapping(root.get("pier_column"))


def _bar_row(
    bar: Mapping[str, Any],
    *,
    member: Literal["pier_cap", "pier_column"],
    count: int | None = None,
    spacing: float | None = None,
    description: str,
) -> BarScheduleRow:
    mark = str(bar.get("id") or "").strip()
    if not mark:
        raise DrawingExpressionError(f"{member} 钢筋缺少 id。")
    return BarScheduleRow(
        mark=mark,
        member=member,
        category=str(bar.get("category") or bar.get("subtype") or "钢筋"),
        diameter_mm=_require_number(bar.get("dia"), name=f"{mark}.dia"),
        count=count,
        spacing_mm=spacing,
        geometry_description=description,
    )


def build_bar_schedule(source: DrawingGroupSource) -> list[BarScheduleRow]:
    cap, column = _reinforcement_parts(source)
    rows: list[BarScheduleRow] = []
    patterns = _mapping(cap.get("z_patterns"))
    for bar in cap.get("longitudinal_bars", []) or []:
        if not isinstance(bar, Mapping):
            continue
        try:
            pattern = resolve_z_positions(patterns, bar.get("z_pattern") or "")
        except DrawingExpressionError:
            # 未登记的 z_pattern 名称：保持原有宽松行为（数量留空），由图纸诊断提示。
            pattern = None
        count = len(pattern) if isinstance(pattern, list) and pattern else None
        rows.append(
            _bar_row(
                bar,
                member="pier_cap",
                count=count,
                description=str(bar.get("subtype") or "盖梁纵向钢筋"),
            )
        )

    for bar in column.get("longitudinal_bars", []) or []:
        if not isinstance(bar, Mapping):
            continue
        rows.append(
            _bar_row(
                bar,
                member="pier_column",
                count=int(_require_number(bar.get("count"), name="纵筋数量")),
                description=str(bar.get("subtype") or "墩柱纵向钢筋"),
            )
        )

    for bar in column.get("spiral_stirrups", []) or []:
        if not isinstance(bar, Mapping):
            continue
        pitches = [
            _require_number(zone.get("pitch"), name="螺旋箍筋螺距")
            for zone in bar.get("vertical_distribution", []) or []
            if isinstance(zone, Mapping) and zone.get("pitch") is not None
        ]
        distinct = sorted(set(pitches))
        description = str(bar.get("subtype") or "墩柱螺旋箍筋")
        if distinct:
            pitch_text = "/".join(f"{value:g}" for value in distinct)
            description = f"{description}，pitch={pitch_text} mm"
        rows.append(
            _bar_row(
                bar,
                member="pier_column",
                spacing=min(distinct) if distinct else None,
                description=description,
            )
        )

    keys = [row.key for row in rows]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise DrawingExpressionError(f"同一构件内钢筋编号重复: {duplicates}")
    return rows
