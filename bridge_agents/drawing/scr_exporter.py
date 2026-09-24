from __future__ import annotations

import math
import re

from .models import (
    ArcEntity,
    CircleEntity,
    DimensionEntity,
    DrawingDocument,
    LeaderEntity,
    LineEntity,
    Point2D,
    TextEntity,
)


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")


def _number(value: float) -> str:
    normalized = 0.0 if math.isclose(value, 0.0, abs_tol=0.0005) else value
    return f"{normalized:.3f}"


def format_point(point: Point2D) -> str:
    return f"{_number(point.x)},{_number(point.y)}"


def _text(value: str) -> str:
    return _CONTROL_RE.sub(" ", value).strip()


def _layer_commands(document: DrawingDocument) -> list[str]:
    lines: list[str] = []
    line_types = sorted(
        {
            layer.line_type
            for layer in document.layers
            if layer.line_type.lower() != "continuous"
        }
    )
    for line_type in line_types:
        lines.extend(["_.-LINETYPE", "_L", line_type, "", ""])
    for layer in document.layers:
        lines.extend(["_.-LAYER", "_M", layer.name, ""])
        lines.extend(
            [
                "_.-LAYER",
                "_LW",
                f"{layer.lineweight_mm:.2f}",
                layer.name,
                "",
            ]
        )
        if layer.line_type.lower() not in {"continuous", "bylayer"}:
            lines.extend(["_.-LAYER", "_LT", layer.line_type, layer.name, ""])
    return lines


def _dimension_commands(entity: DimensionEntity) -> list[str]:
    dx = entity.end.x - entity.start.x
    dy = entity.end.y - entity.start.y
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    first = Point2D(x=entity.start.x + nx * entity.offset, y=entity.start.y + ny * entity.offset)
    second = Point2D(x=entity.end.x + nx * entity.offset, y=entity.end.y + ny * entity.offset)
    midpoint = Point2D(x=(first.x + second.x) / 2, y=(first.y + second.y) / 2)
    label = entity.text_override or _number(length)
    return [
        "_.LINE", format_point(entity.start), format_point(first), "",
        "_.LINE", format_point(entity.end), format_point(second), "",
        "_.LINE", format_point(first), format_point(second), "",
        "_.TEXT", "_J", "_MC", format_point(midpoint), "100.000", "0.000", _text(label),
    ]


def _entity_commands(entity) -> list[str]:
    if isinstance(entity, LineEntity):
        return ["_.LINE", format_point(entity.start), format_point(entity.end), ""]
    if isinstance(entity, CircleEntity):
        return ["_.CIRCLE", format_point(entity.center), _number(entity.radius)]
    if isinstance(entity, ArcEntity):
        start = Point2D(
            x=entity.center.x + entity.radius * math.cos(math.radians(entity.start_angle_deg)),
            y=entity.center.y + entity.radius * math.sin(math.radians(entity.start_angle_deg)),
        )
        end = Point2D(
            x=entity.center.x + entity.radius * math.cos(math.radians(entity.end_angle_deg)),
            y=entity.center.y + entity.radius * math.sin(math.radians(entity.end_angle_deg)),
        )
        if entity.clockwise:
            start, end = end, start
        return ["_.ARC", "_C", format_point(entity.center), format_point(start), format_point(end)]
    if isinstance(entity, TextEntity):
        return [
            "_.TEXT", "_J", "_ML", format_point(entity.insertion),
            _number(entity.height), _number(entity.rotation_deg), _text(entity.text),
        ]
    if isinstance(entity, LeaderEntity):
        commands: list[str] = []
        for start, end in zip(entity.points, entity.points[1:]):
            commands.extend(["_.LINE", format_point(start), format_point(end), ""])
        commands.extend(
            ["_.TEXT", "_J", "_ML", format_point(entity.points[-1]),
             _number(entity.text_height), "0.000", _text(entity.text)]
        )
        return commands
    if isinstance(entity, DimensionEntity):
        return _dimension_commands(entity)
    raise TypeError(f"不支持的图元类型: {type(entity).__name__}")


def render_autocad_script(document: DrawingDocument) -> str:
    try:
        layout = next(view for view in document.views if view.view_id == "cad-layout")
    except StopIteration as exc:
        raise ValueError("SCR 导出要求 DrawingDocument 包含 cad-layout 视图。") from exc
    commands = _layer_commands(document)
    colors = {
        layer.name: ",".join(str(item) for item in layer.color_rgb)
        for layer in document.layers
    }
    current_layer: str | None = None
    for section in layout.sections:
        for entity in section.entities:
            if entity.layer != current_layer:
                commands.extend(
                    [
                        "_.-LAYER",
                        "_S",
                        entity.layer,
                        "",
                        "_.CECOLOR",
                        colors[entity.layer],
                    ]
                )
                current_layer = entity.layer
            commands.extend(_entity_commands(entity))
    commands.extend(["_.ZOOM", "_E"])
    return "\n".join(commands) + "\n"
