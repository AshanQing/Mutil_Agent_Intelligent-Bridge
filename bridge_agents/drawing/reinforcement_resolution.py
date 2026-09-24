from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, List, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .reinforcement_adapter import (
    DrawingExpressionError,
    evaluate_drawing_expression,
    resolve_z_positions,
)


class _ResolvedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReinforcementDiagnostic(_ResolvedModel):
    severity: Literal["warning", "error"]
    code: str
    field_path: str
    message: str


class ResolvedBar(_ResolvedModel):
    key: str
    member: Literal["cap", "column"]
    mark: str
    category: str
    subtype: str
    diameter_mm: float = Field(gt=0)
    count: int | None = Field(default=None, gt=0)
    z_pattern: str | None = None
    centerline_diameter_mm: float | None = Field(default=None, gt=0)
    y_from_mm: float | None = None
    y_to_mm: float | None = None
    source: dict[str, Any]


class ResolvedCapSkeleton(_ResolvedModel):
    skeleton_id: str
    name: str
    bar_marks: list[str]
    z_positions: list[int]


class ResolvedCapStirrupSegment(_ResolvedModel):
    name: str
    x_from_mm: float
    x_to_mm: float
    spacing_mm: float = Field(gt=0)
    declared_interval_count: int | None = Field(default=None, ge=0)


class ResolvedCap(_ResolvedModel):
    length_mm: float = Field(gt=0)
    width_mm: float = Field(gt=0)
    height_mid_mm: float = Field(gt=0)
    height_end_mm: float = Field(gt=0)
    cantilever_mm: float = Field(ge=0)
    cover_x_mm: float = Field(ge=0)
    cover_y_mm: float = Field(ge=0)
    cover_z_mm: float = Field(ge=0)
    column_count: int = Field(gt=0)
    column_diameter_mm: float = Field(gt=0)
    column_spacing_mm: float = Field(ge=0)
    z_patterns: dict[str, list[int]]
    skeletons: dict[str, ResolvedCapSkeleton]
    bars: dict[str, ResolvedBar]
    stirrup_segments: list[ResolvedCapStirrupSegment]
    transverse_cage: dict[str, Any]


class ResolvedColumnZone(_ResolvedModel):
    name: str
    y_from_mm: float
    y_to_mm: float
    pitch_mm: float = Field(gt=0)


class ResolvedOrdinaryHoop(_ResolvedModel):
    bar: ResolvedBar
    count: int = Field(gt=0)
    spacing_mm: float = Field(gt=0)
    y_positions_mm: list[float]


class ResolvedStrengthening(_ResolvedModel):
    bar: ResolvedBar
    count: int = Field(gt=0)
    hoop_diameter_mm: float = Field(gt=0)
    spacing_mm: float = Field(gt=0)
    y_positions_mm: list[float]
    semantic_review_required: bool = True


class ResolvedColumn(_ResolvedModel):
    count: int = Field(gt=0)
    diameter_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)
    spacing_mm: float = Field(ge=0)
    concrete_cover_mm: float = Field(ge=0)
    longitudinal: ResolvedBar
    spiral: ResolvedBar
    spiral_zones: list[ResolvedColumnZone]
    ordinary_hoop: ResolvedOrdinaryHoop
    strengthening: ResolvedStrengthening


class ResolvedPierReinforcement(_ResolvedModel):
    cap: ResolvedCap
    column: ResolvedColumn
    diagnostics: list[ReinforcementDiagnostic] = Field(default_factory=list)

    def bar(self, member: Literal["cap", "column"], mark: str) -> ResolvedBar:
        if member == "cap":
            return self.cap.bars[mark]
        column_bars = {
            self.column.longitudinal.mark: self.column.longitudinal,
            self.column.spiral.mark: self.column.spiral,
            self.column.ordinary_hoop.bar.mark: self.column.ordinary_hoop.bar,
            self.column.strengthening.bar.mark: self.column.strengthening.bar,
        }
        return column_bars[mark]


_MM_GEOMETRY_VARIABLES = (
    "cap_length",
    "cap_width",
    "cap_height_mid",
    "cap_height_end",
    "cantilever_length",
    "column_diameter",
    "column_height",
    "column_spacing",
)
_REDUNDANT_MM_PATTERNS = tuple(
    (
        variable,
        re.compile(
            rf"(?<![A-Za-z0-9_]){variable}\s*\*\s*1000(?:\.0)?(?![A-Za-z0-9_])"
        ),
        re.compile(
            rf"(?<![A-Za-z0-9_])1000(?:\.0)?\s*\*\s*{variable}(?![A-Za-z0-9_])"
        ),
    )
    for variable in _MM_GEOMETRY_VARIABLES
)


def _normalize_runtime_expressions(
    value: Any,
    *,
    field_path: str,
    diagnostics: list[ReinforcementDiagnostic],
) -> Any:
    """兼容旧成果：绘图运行时几何变量已统一为 mm，移除再次乘 1000。"""
    if isinstance(value, Mapping):
        return {
            key: _normalize_runtime_expressions(
                item,
                field_path=f"{field_path}.{key}",
                diagnostics=diagnostics,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_runtime_expressions(
                item,
                field_path=f"{field_path}[{index}]",
                diagnostics=diagnostics,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    normalized = re.sub(
        r"(?<![A-Za-z0-9_])controlling_net_height_m\s*\*\s*1000(?:\.0)?(?![A-Za-z0-9_])",
        "column_height",
        value,
    )
    replaced_variables: list[str] = []
    if normalized != value:
        replaced_variables.append("controlling_net_height_m")
    for variable, right_pattern, left_pattern in _REDUNDANT_MM_PATTERNS:
        updated = right_pattern.sub(variable, normalized)
        updated = left_pattern.sub(variable, updated)
        if updated != normalized:
            replaced_variables.append(variable)
            normalized = updated
    if replaced_variables:
        diagnostics.append(
            ReinforcementDiagnostic(
                severity="warning",
                code="redundant_mm_conversion_normalized",
                field_path=field_path,
                message=(
                    "绘图变量已是 mm，已移除重复的 ×1000："
                    + ", ".join(sorted(set(replaced_variables)))
                ),
            )
        )
    return normalized


def _engineering_diagnostics(
    resolved: "ResolvedPierReinforcement",
) -> list[ReinforcementDiagnostic]:
    diagnostics: list[ReinforcementDiagnostic] = []
    cap = resolved.cap
    column = resolved.column
    cap_limit = cap.length_mm / 2.0 + max(cap.cover_x_mm, 1.0)
    for index, segment in enumerate(cap.stirrup_segments):
        if max(abs(segment.x_from_mm), abs(segment.x_to_mm)) > cap_limit:
            diagnostics.append(
                ReinforcementDiagnostic(
                    severity="error",
                    code="cap_reinforcement_outside_member",
                    field_path=f"cap.stirrups[{index}]",
                    message="盖梁箍筋坐标超出构件纵向边界。",
                )
            )
    if (column.longitudinal.centerline_diameter_mm or 0.0) >= column.diameter_mm:
        diagnostics.append(
            ReinforcementDiagnostic(
                severity="error",
                code="column_bar_centerline_outside_section",
                field_path="column.longitudinal.section_definition",
                message="墩柱纵筋中心圆直径不小于墩柱直径。",
            )
        )
    # N4 环向加强箍应位于纵筋 N1 的内侧（环径小于 N1 中心圆），
    # 避免把 N4 按 N1/N2 外圈公式推算到与纵筋同环或外侧。
    if (
        column.strengthening.bar.diameter_mm > 1
        and (column.longitudinal.centerline_diameter_mm or 0.0) > 0
        and column.strengthening.hoop_diameter_mm
        >= column.longitudinal.centerline_diameter_mm - 1e-6
    ):
        diagnostics.append(
            ReinforcementDiagnostic(
                severity="warning",
                code="column_strengthening_not_inside_longitudinal",
                field_path="column.strengthening.section_definition",
                message=(
                    "墩柱加强箍(N4)环径不小于纵筋(N1)中心圆直径：N4 应位于纵筋"
                    "内侧，环径必须小于 N1 的 bar_centerline_diameter。"
                ),
            )
        )
    for index, zone in enumerate(column.spiral_zones):
        if (
            zone.y_from_mm < 0
            or zone.y_to_mm < zone.y_from_mm
            or zone.y_to_mm > column.height_mm + 1e-6
        ):
            diagnostics.append(
                ReinforcementDiagnostic(
                    severity="error",
                    code="column_spiral_zone_outside_member",
                    field_path=f"column.spiral_zones[{index}]",
                    message="墩柱螺旋箍筋分区超出柱高或起终点倒置。",
                )
            )
        elif (zone.y_to_mm - zone.y_from_mm) / zone.pitch_mm > 10000:
            diagnostics.append(
                ReinforcementDiagnostic(
                    severity="error",
                    code="column_spiral_entity_count_excessive",
                    field_path=f"column.spiral_zones[{index}]",
                    message="单个螺旋箍筋分区超过10000个节距，疑似单位错误。",
                )
            )
    return diagnostics


def _mapping(value: Any, *, field_path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DrawingExpressionError(f"{field_path} 必须是映射。")
    return value


def _number(value: Any, *, field_path: str) -> float:
    if isinstance(value, bool):
        raise DrawingExpressionError(f"{field_path} 不能使用布尔值。")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DrawingExpressionError(f"{field_path} 不是有效数值: {value!r}") from exc
    if not math.isfinite(result):
        raise DrawingExpressionError(f"{field_path} 必须是有限数值。")
    return result


def load_reinforcement_example(
    path: str | Path,
    *,
    example_name: str,
    expected_drawing_id: int,
) -> dict[str, Any]:
    source_path = Path(path)
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    root = _mapping(payload, field_path=str(source_path))
    example = _mapping(root.get(example_name), field_path=example_name)
    inputs = _mapping(example.get("输入"), field_path=f"{example_name}.输入")
    basic = _mapping(inputs.get("基本信息"), field_path=f"{example_name}.输入.基本信息")
    actual_drawing_id = int(_number(basic.get("drawing_id"), field_path="drawing_id"))
    if actual_drawing_id != expected_drawing_id:
        raise DrawingExpressionError(
            f"{example_name} drawing_id={actual_drawing_id}，"
            f"预期为 {expected_drawing_id}。"
        )
    return dict(example)


def _resolved_bar(
    bar: Mapping[str, Any],
    *,
    member: Literal["cap", "column"],
    count: int | None = None,
    z_pattern: str | None = None,
    centerline_diameter_mm: float | None = None,
    y_from_mm: float | None = None,
    y_to_mm: float | None = None,
) -> ResolvedBar:
    mark = str(bar.get("id") or "").strip()
    if not mark:
        raise DrawingExpressionError(f"{member} 钢筋缺少 id。")
    return ResolvedBar(
        key=f"{member}:{mark}",
        member=member,
        mark=mark,
        category=str(bar.get("category") or "").strip(),
        subtype=str(bar.get("subtype") or "").strip(),
        diameter_mm=_number(bar.get("dia"), field_path=f"{member}.{mark}.dia"),
        count=count,
        z_pattern=z_pattern,
        centerline_diameter_mm=centerline_diameter_mm,
        y_from_mm=y_from_mm,
        y_to_mm=y_to_mm,
        source=dict(bar),
    )


def _resolve_cap(
    example: Mapping[str, Any],
) -> tuple[ResolvedCap, list[ReinforcementDiagnostic]]:
    inputs = _mapping(example.get("输入"), field_path="cap.输入")
    geometry = _mapping(inputs.get("盖梁几何信息"), field_path="cap.盖梁几何信息")
    columns = _mapping(inputs.get("墩柱几何信息"), field_path="cap.墩柱几何信息")
    controls = _mapping(
        inputs.get("保护层与控制参数"), field_path="cap.保护层与控制参数"
    )
    output = _mapping(example.get("输出"), field_path="cap.输出")
    reinforcement = _mapping(
        output.get("reinforcement"), field_path="cap.输出.reinforcement"
    )
    pier_cap = _mapping(reinforcement.get("pier_cap"), field_path="cap.pier_cap")

    length_mm = 1000 * _number(geometry.get("cap_length"), field_path="cap_length")
    width_mm = 1000 * _number(geometry.get("cap_width"), field_path="cap_width")
    height_mid_mm = 1000 * _number(
        geometry.get("cap_height_mid"), field_path="cap_height_mid"
    )
    height_end_mm = 1000 * _number(
        geometry.get("cap_height_end"), field_path="cap_height_end"
    )
    cantilever_mm = 1000 * _number(
        geometry.get("cantilever_length"), field_path="cantilever_length"
    )
    column_spacing_mm = 1000 * _number(
        columns.get("column_spacing"), field_path="column_spacing"
    )
    values = {
        "cap_length": length_mm,
        "cap_width": width_mm,
        "cap_height_mid": height_mid_mm,
        "cap_height_end": height_end_mm,
        "cantilever_length": cantilever_mm,
        "column_count": _number(columns.get("column_count"), field_path="column_count"),
        "column_diameter": 1000
        * _number(columns.get("column_diameter"), field_path="column_diameter"),
        "column_spacing": column_spacing_mm,
        "cover_x": _number(controls.get("cover_x"), field_path="cover_x"),
        "cover_y": _number(controls.get("cover_y"), field_path="cover_y"),
        "cover_z": _number(controls.get("cover_z"), field_path="cover_z"),
        "cap_centerline_x": _number(
            controls.get("cap_centerline_x", 0), field_path="cap_centerline_x"
        ),
    }
    values["pier_centerline_x"] = evaluate_drawing_expression(
        controls.get("pier_centerline_x", "-column_spacing / 2"), values
    )

    z_patterns = {
        str(name): [int(position) for position in positions]
        for name, positions in _mapping(
            pier_cap.get("z_patterns"), field_path="cap.z_patterns"
        ).items()
    }
    skeletons: dict[str, ResolvedCapSkeleton] = {}
    for skeleton_id, raw in _mapping(
        pier_cap.get("skeleton_definitions"), field_path="cap.skeleton_definitions"
    ).items():
        skeleton = _mapping(raw, field_path=f"cap.skeletons.{skeleton_id}")
        pattern_name = str(skeleton.get("z_pattern") or "")
        skeletons[str(skeleton_id)] = ResolvedCapSkeleton(
            skeleton_id=str(skeleton_id),
            name=str(skeleton.get("name") or skeleton_id),
            bar_marks=[str(mark) for mark in skeleton.get("rebar_groups", [])],
            z_positions=resolve_z_positions(z_patterns, pattern_name),
        )

    bars: dict[str, ResolvedBar] = {}
    for raw in pier_cap.get("longitudinal_bars", []) or []:
        bar = _mapping(raw, field_path="cap.longitudinal_bars")
        resolved = _resolved_bar(
            bar,
            member="cap",
            z_pattern=str(bar.get("z_pattern") or "") or None,
        )
        bars[resolved.mark] = resolved

    stirrups = _mapping(pier_cap.get("stirrups"), field_path="cap.stirrups")
    longitudinal_distribution = stirrups.get("longitudinal_distribution", []) or []
    raw_overview = stirrups.get("distribution_overview")
    if isinstance(raw_overview, Mapping):
        overview = raw_overview
    elif longitudinal_distribution:
        first_segment = _mapping(
            longitudinal_distribution[0], field_path="cap.stirrups.segment[0]"
        )
        last_segment = _mapping(
            longitudinal_distribution[-1], field_path="cap.stirrups.segment[-1]"
        )
        overview = {
            "x_start_expr": first_segment.get("x_from"),
            "x_end_expr": last_segment.get("x_to"),
            "segment_sequence": [],
        }
    else:
        raise DrawingExpressionError(
            "cap.stirrups 缺少 distribution_overview 和 longitudinal_distribution。"
        )
    sequence = [
        item
        for item in overview.get("segment_sequence", []) or []
        if isinstance(item, Mapping)
    ]
    # count 是"间距个数"，不是钢筋条数：区段长度 = count × spacing，
    # 箍筋位置 = 起点 + i*spacing（i=0..count），共 count+1 个位置。
    declared_counts = {
        str(item.get("name")): int(_number(item.get("count"), field_path="count"))
        for item in sequence
        if item.get("count") is not None
    }
    segments: list[ResolvedCapStirrupSegment] = []
    diagnostics: list[ReinforcementDiagnostic] = []
    # 权威排布：从 distribution_overview 的 x_start 出发，按 segment_sequence 的
    # count×spacing 逐段累计端点（count 为间距个数），保证总长 = x_end - x_start。
    x_cursor = evaluate_drawing_expression(overview.get("x_start_expr"), values)
    x_end_target = evaluate_drawing_expression(overview.get("x_end_expr"), values)
    if sequence:
        for item_index, item in enumerate(sequence):
            name = str(item.get("name") or "")
            spacing = evaluate_drawing_expression(item.get("spacing"), values)
            count = declared_counts.get(name)
            if count is None:
                diagnostics.append(
                    ReinforcementDiagnostic(
                        severity="warning",
                        code="stirrup_segment_missing_count",
                        field_path=f"cap.stirrups.{name}",
                        message=f"区段 {name} 缺少 count（间距个数）。",
                    )
                )
                continue
            segment_x_from = x_cursor
            declared_x_to = x_cursor + count * spacing
            segment_x_to = declared_x_to
            is_last = item_index == len(sequence) - 1
            residual = x_end_target - declared_x_to
            if is_last and not math.isclose(residual, 0.0, abs_tol=1e-6):
                # 分区总长不能被末段标准间距整除时，末个间距采用不大于标准值的
                # 收口间距，保证保护层端点准确；超过一个标准间距则不自动修补。
                if abs(residual) < spacing:
                    segment_x_to = x_end_target
                    diagnostics.append(
                        ReinforcementDiagnostic(
                            severity="warning",
                            code="terminal_stirrup_spacing_adjusted",
                            field_path=f"cap.stirrups.{name}",
                            message=(
                                f"末段以 {abs(residual):g} mm 余量收口，"
                                "端点已对齐盖梁保护层边界。"
                            ),
                        )
                    )
            segments.append(
                ResolvedCapStirrupSegment(
                    name=name,
                    x_from_mm=segment_x_from,
                    x_to_mm=segment_x_to,
                    spacing_mm=spacing,
                    declared_interval_count=count,
                )
            )
            x_cursor = segment_x_to
        if not math.isclose(x_cursor, x_end_target, rel_tol=0.0, abs_tol=1e-6):
            diagnostics.append(
                ReinforcementDiagnostic(
                    severity="error",
                    code="stirrup_distribution_length_mismatch",
                    field_path="cap.stirrups.distribution_overview",
                    message=(
                        f"箍筋分段累计终点 {x_cursor:g} mm 与 "
                        f"x_end_expr={x_end_target:g} mm 不一致。"
                    ),
                )
            )
    else:
        # 兼容旧数据：无 segment_sequence 时回退用 longitudinal_distribution 的 x_from/x_to
        for raw in longitudinal_distribution:
            segment = _mapping(raw, field_path="cap.stirrups.segment")
            name = str(segment.get("name") or "")
            x_from = evaluate_drawing_expression(segment.get("x_from"), values)
            x_to = evaluate_drawing_expression(segment.get("x_to"), values)
            spacing = evaluate_drawing_expression(segment.get("spacing"), values)
            declared_count = declared_counts.get(name)
            segments.append(
                ResolvedCapStirrupSegment(
                    name=name,
                    x_from_mm=x_from,
                    x_to_mm=x_to,
                    spacing_mm=spacing,
                    declared_interval_count=declared_count,
                )
            )
            if declared_count is not None and not math.isclose(
                abs(x_to - x_from),
                spacing * declared_count,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                diagnostics.append(
                    ReinforcementDiagnostic(
                        severity="warning",
                        code="stirrup_segment_length_mismatch",
                        field_path=f"cap.stirrups.{name}",
                        message=(
                            f"区段长度 {abs(x_to - x_from):g} mm 与 "
                            f"spacing×count={spacing * declared_count:g} mm 不一致。"
                        ),
                    )
                )

    transverse_cage = dict(
        _mapping(
            _mapping(pier_cap.get("stirrups"), field_path="cap.stirrups").get(
                "transverse_cage_definition"
            ),
            field_path="cap.stirrups.transverse_cage_definition",
        )
    )
    return (
        ResolvedCap(
            length_mm=length_mm,
            width_mm=width_mm,
            height_mid_mm=height_mid_mm,
            height_end_mm=height_end_mm,
            cantilever_mm=cantilever_mm,
            cover_x_mm=values["cover_x"],
            cover_y_mm=values["cover_y"],
            cover_z_mm=values["cover_z"],
            column_count=int(values["column_count"]),
            column_diameter_mm=values["column_diameter"],
            column_spacing_mm=column_spacing_mm,
            z_patterns=z_patterns,
            skeletons=skeletons,
            bars=bars,
            stirrup_segments=segments,
            transverse_cage=transverse_cage,
        ),
        diagnostics,
    )


def _resolve_column(
    example: Mapping[str, Any],
) -> tuple[ResolvedColumn, list[ReinforcementDiagnostic]]:
    diagnostics: list[ReinforcementDiagnostic] = []
    inputs = _mapping(example.get("输入"), field_path="column.输入")
    geometry = _mapping(inputs.get("墩柱几何信息"), field_path="column.墩柱几何信息")
    controls = _mapping(
        inputs.get("保护层与控制参数"), field_path="column.保护层与控制参数"
    )
    output = _mapping(example.get("输出"), field_path="column.输出")
    reinforcement = _mapping(
        output.get("reinforcement"), field_path="column.输出.reinforcement"
    )
    pier_column = _mapping(
        reinforcement.get("pier_column"), field_path="column.pier_column"
    )
    values = {
        "column_count": _number(geometry.get("column_count"), field_path="column_count"),
        "column_diameter": _number(
            geometry.get("column_diameter"), field_path="column_diameter"
        ),
        "column_height": _number(
            geometry.get("column_height"), field_path="column_height"
        ),
        "column_spacing": _number(
            geometry.get("column_spacing"), field_path="column_spacing"
        ),
        "concrete_cover": _number(
            controls.get("concrete_cover", controls.get("cover_x")),
            field_path="concrete_cover",
        ),
    }
    # 保护层别名：cover_x/cover_y/cover_z 供表达式引用（回退到 concrete_cover）
    if values.get("concrete_cover") is not None:
        values.setdefault("cover_x", values["concrete_cover"])
        values.setdefault("cover_y", values["concrete_cover"])
        values.setdefault("cover_z", values["concrete_cover"])
    if controls.get("cover_x") is not None:
        values["cover_x"] = _number(controls.get("cover_x"), field_path="cover_x")
    if controls.get("cover_y") is not None:
        values["cover_y"] = _number(controls.get("cover_y"), field_path="cover_y")
    if controls.get("cover_z") is not None:
        values["cover_z"] = _number(controls.get("cover_z"), field_path="cover_z")

    raw_longitudinal = _mapping(
        (pier_column.get("longitudinal_bars") or [None])[0],
        field_path="column.longitudinal_bars[0]",
    )
    longitudinal_values = {**values, "dia": _number(raw_longitudinal.get("dia"), field_path="N1.dia")}
    longitudinal_section = _mapping(
        raw_longitudinal.get("section_definition"), field_path="column.N1.section_definition"
    )
    longitudinal_range = _mapping(
        raw_longitudinal.get("range_definition"), field_path="column.N1.range_definition"
    )
    longitudinal = _resolved_bar(
        raw_longitudinal,
        member="column",
        count=int(_number(raw_longitudinal.get("count"), field_path="N1.count")),
        centerline_diameter_mm=evaluate_drawing_expression(
            longitudinal_section.get("bar_centerline_diameter"), longitudinal_values
        ),
        y_from_mm=evaluate_drawing_expression(
            longitudinal_range.get("y_start"), longitudinal_values
        ),
        y_to_mm=evaluate_drawing_expression(
            longitudinal_range.get("y_end"), longitudinal_values
        ),
    )

    raw_spiral = _mapping(
        (pier_column.get("spiral_stirrups") or [None])[0],
        field_path="column.spiral_stirrups[0]",
    )
    spiral_values = {**values, "dia": _number(raw_spiral.get("dia"), field_path="N2.dia")}
    spiral_section = _mapping(
        raw_spiral.get("section_definition"), field_path="column.N2.section_definition"
    )
    spiral = _resolved_bar(
        raw_spiral,
        member="column",
        centerline_diameter_mm=evaluate_drawing_expression(
            spiral_section.get("hoop_diameter"), spiral_values
        ),
    )
    spiral_zones = [
        ResolvedColumnZone(
            name=str(zone.get("name") or ""),
            y_from_mm=evaluate_drawing_expression(zone.get("y_from"), spiral_values),
            y_to_mm=evaluate_drawing_expression(zone.get("y_to"), spiral_values),
            pitch_mm=evaluate_drawing_expression(zone.get("pitch"), spiral_values),
        )
        for raw in raw_spiral.get("vertical_distribution", []) or []
        for zone in [_mapping(raw, field_path="column.N2.vertical_distribution")]
    ]

    ordinary_hoops = pier_column.get("ordinary_hoops") or []
    if ordinary_hoops:
        raw_ordinary = _mapping(
            ordinary_hoops[0],
            field_path="column.ordinary_hoops[0]",
        )
        ordinary_overview = _mapping(
            raw_ordinary.get("distribution_overview"),
            field_path="column.N3.distribution_overview",
        )
        ordinary_values = {**values, "dia": _number(raw_ordinary.get("dia"), field_path="N3.dia")}
        ordinary_count = int(_number(ordinary_overview.get("count"), field_path="N3.count"))
        ordinary_spacing = evaluate_drawing_expression(
            ordinary_overview.get("spacing"), ordinary_values
        )
        ordinary_start = evaluate_drawing_expression(
            ordinary_overview.get("y_start"), ordinary_values
        )
        ordinary_positions = [
            ordinary_start + index * ordinary_spacing for index in range(ordinary_count)
        ]
        ordinary_end = evaluate_drawing_expression(
            ordinary_overview.get("y_end"), ordinary_values
        )
        if not math.isclose(ordinary_positions[-1], ordinary_end, abs_tol=1e-6):
            diagnostics.append(
                ReinforcementDiagnostic(
                    severity="warning",
                    code="ordinary_hoop_end_normalized",
                    field_path="column.N3.distribution_overview",
                    message=(
                        f"N3 声明终点 {ordinary_end:g} mm 与 count/spacing 计算终点 "
                        f"{ordinary_positions[-1]:g} mm 不一致；绘图按 count/spacing 执行。"
                    ),
                )
            )
        ordinary_bar = _resolved_bar(raw_ordinary, member="column", count=ordinary_count)
        ordinary_hoop_review = False
    else:
        # 墩柱无普通箍（N3）时用最小占位，标记待复核（detail_sheets 不再画 N3）
        ordinary_count = 1
        ordinary_spacing = 1.0
        ordinary_positions = [0.0]
        ordinary_bar = _resolved_bar(
            {"id": "N3", "category": "箍筋", "subtype": "普通箍筋（无）", "dia": 1},
            member="column",
            count=1,
        )
        ordinary_hoop_review = True

    strengthening_bars = pier_column.get("strengthening_bars") or []
    if strengthening_bars:
        raw_strengthening = _mapping(
            strengthening_bars[0],
            field_path="column.strengthening_bars[0]",
        )
        strengthening_section = _mapping(
            raw_strengthening.get("section_definition"), field_path="column.N4.section_definition"
        )
        strengthening_distribution = _mapping(
            raw_strengthening.get("vertical_distribution"), field_path="column.N4.vertical_distribution"
        )
        strengthening_count = int(
            _number(strengthening_distribution.get("count"), field_path="N4.count")
        )
        strengthening_values = {
            **values,
            "dia": _number(raw_strengthening.get("dia"), field_path="N4.dia"),
            "first_y": evaluate_drawing_expression(
                strengthening_distribution.get("first_y"), values
            ),
            "count": strengthening_count,
        }
        strengthening_values["last_y"] = evaluate_drawing_expression(
            strengthening_distribution.get("last_y"), strengthening_values
        )
        strengthening_spacing = evaluate_drawing_expression(
            strengthening_distribution.get("spacing"), strengthening_values
        )
        if strengthening_spacing <= 0:
            # 柱高过矮时 last_y=column_height-first_y 可能小于 first_y，分布区间倒挂、
            # 间距非正（如柱高 1117mm、first_y=800 时 last_y=317、spacing=-161）。
            # 该加强筋无法布置，降级为占位并记录诊断，避免整组图纸因此作废。
            diagnostics.append(
                ReinforcementDiagnostic(
                    severity="warning",
                    code="strengthening_distribution_invalid",
                    field_path="column.N4.vertical_distribution",
                    message=(
                        f"N4 加强筋间距 {strengthening_spacing:g} mm 非正"
                        f"（柱高 {values['column_height']:g} mm、first_y "
                        f"{strengthening_values['first_y']:g} mm），按无环向加强筋出图。"
                    ),
                )
            )
            strengthening_count = 1
            strengthening_spacing = 1.0
            strengthening_positions = [0.0]
            strengthening_bar = _resolved_bar(
                {"id": "N4", "category": "加强钢筋", "subtype": "环向加强筋（未布置）", "dia": 1},
                member="column",
                count=1,
            )
            strengthening_hoop_diameter = 1.0
        else:
            strengthening_positions = [
                strengthening_values["first_y"] + index * strengthening_spacing
                for index in range(strengthening_count)
            ]
            strengthening_bar = _resolved_bar(
                raw_strengthening, member="column", count=strengthening_count
            )
            # hoop_diameter 可能是表达式（如 column_diameter - 2*concrete_cover - dia*2），
            # 也可能是数值；统一走表达式求值（数值也会原样返回）
            strengthening_hoop_diameter = evaluate_drawing_expression(
                strengthening_section.get("hoop_diameter"), strengthening_values
            )
        # N4 环向加强筋细部语义需对照原图复核（样例固有标记）
        strengthening_semantic_review = True
    else:
        # 墩柱无环向加强筋（N4）时用最小占位，标记待复核
        strengthening_count = 1
        strengthening_spacing = 1.0
        strengthening_positions = [0.0]
        strengthening_bar = _resolved_bar(
            {"id": "N4", "category": "加强钢筋", "subtype": "环向加强筋（无）", "dia": 1},
            member="column",
            count=1,
        )
        strengthening_hoop_diameter = 1.0
        strengthening_semantic_review = True

    return ResolvedColumn(
        count=int(values["column_count"]),
        diameter_mm=values["column_diameter"],
        height_mm=values["column_height"],
        spacing_mm=values["column_spacing"],
        concrete_cover_mm=values["concrete_cover"],
        longitudinal=longitudinal,
        spiral=spiral,
        spiral_zones=spiral_zones,
        ordinary_hoop=ResolvedOrdinaryHoop(
            bar=ordinary_bar,
            count=ordinary_count,
            spacing_mm=ordinary_spacing,
            y_positions_mm=ordinary_positions,
        ),
        strengthening=ResolvedStrengthening(
            bar=strengthening_bar,
            count=strengthening_count,
            hoop_diameter_mm=max(strengthening_hoop_diameter, 1.0),
            spacing_mm=strengthening_spacing,
            y_positions_mm=strengthening_positions,
            semantic_review_required=strengthening_semantic_review,
        ),
    ), diagnostics


def resolve_reinforcement_examples(
    cap_example: Mapping[str, Any],
    column_example: Mapping[str, Any],
) -> ResolvedPierReinforcement:
    cap, diagnostics = _resolve_cap(cap_example)
    column, column_diagnostics = _resolve_column(column_example)
    return ResolvedPierReinforcement(
        cap=cap,
        column=column,
        diagnostics=[*diagnostics, *column_diagnostics],
    )


def resolve_drawing_source(
    source: "DrawingGroupSource",
) -> ResolvedPierReinforcement:
    """把设计组绘图输入（DrawingGroupSource）解析为工程语义模型。

    DrawingGroupSource 携带尺寸设计、联合配筋与验算状态；几何信息位于
    design_context.reinforcement_task.桥墩尺寸信息，配筋位于 reinforcement。
    本函数将其包装为 _resolve_cap/_resolve_column 需要的 example 结构复用解析。
    """
    from .models import DrawingGroupSource as _DGS

    if not isinstance(source, _DGS):
        raise DrawingExpressionError("resolve_drawing_source 需要 DrawingGroupSource。")
    task = _mapping(
        source.design_context.get("reinforcement_task"),
        field_path="design_context.reinforcement_task",
    )
    geometry = _mapping(task.get("桥墩尺寸信息"), field_path="reinforcement_task.桥墩尺寸信息")
    # DrawingGroupSource.reinforcement 形如 {"reinforcement": {"pier_cap":..., "pier_column":...}}
    reinforcement_root = _mapping(
        source.reinforcement, field_path="source.reinforcement"
    )
    reinforcement_payload = reinforcement_root.get("reinforcement", reinforcement_root)
    normalization_diagnostics: list[ReinforcementDiagnostic] = []
    reinforcement_payload = _normalize_runtime_expressions(
        reinforcement_payload,
        field_path="source.reinforcement",
        diagnostics=normalization_diagnostics,
    )

    # 单位适配：_resolve_cap 假定盖梁/墩柱几何为米（内部 ×1000 换算毫米）；
    # _resolve_column 假定墩柱几何已为毫米。真实 DrawingGroupSource 全为米，
    # 因此分别为 cap 与 column 构造 example：cap 用米制墩柱几何，column 用毫米制。
    cap_geometry = dict(
        _mapping(geometry.get("盖梁几何信息"), field_path="盖梁几何信息")
    )
    column_geometry_raw = _mapping(
        geometry.get("墩柱几何信息"), field_path="墩柱几何信息"
    )
    column_geometry_mm = {
        key: (
            1000 * value
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else value
        )
        for key, value in column_geometry_raw.items()
    }
    controls = dict(
        _mapping(geometry.get("保护层与控制参数"), field_path="保护层与控制参数")
    )
    cap_example = {
        "输入": {
            "盖梁几何信息": cap_geometry,
            "墩柱几何信息": dict(column_geometry_raw),
            "保护层与控制参数": controls,
        },
        "输出": {
            "reinforcement": dict(
                _mapping(reinforcement_payload, field_path="source.reinforcement")
            )
        },
    }
    column_example = {
        "输入": {
            "盖梁几何信息": cap_geometry,
            "墩柱几何信息": column_geometry_mm,
            "保护层与控制参数": controls,
        },
        "输出": {
            "reinforcement": dict(
                _mapping(reinforcement_payload, field_path="source.reinforcement")
            )
        },
    }
    resolved = resolve_reinforcement_examples(cap_example, column_example)
    return resolved.model_copy(
        update={
            "diagnostics": [
                *normalization_diagnostics,
                *resolved.diagnostics,
                *_engineering_diagnostics(resolved),
            ]
        }
    )

