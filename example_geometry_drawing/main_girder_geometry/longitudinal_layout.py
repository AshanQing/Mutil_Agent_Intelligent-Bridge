from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .geometry_models import DesignParameters


THICKENED_ZERO_BLOCK = "thickened"
TRANSITION_ZERO_BLOCK = "transition"
BOXED_ZERO_BLOCK = "boxed"


@dataclass(frozen=True)
class LongitudinalLayout:
    kind: str
    wall_half: float
    curve_start: float
    sequence_start: float
    curve_end: float
    zero_block_end: float
    box_start: float | None = None
    box_end: float | None = None
    wall_start: float | None = None
    wall_end: float | None = None


def resolve_longitudinal_layout(
    parameters: DesignParameters,
) -> LongitudinalLayout:
    longitudinal = parameters.longitudinal
    zero_block_end = _positive(longitudinal, "L_0_half")
    variable_length = _positive(longitudinal, "L_H_var")
    box_count = _non_negative_integer(longitudinal, "n_box_0_side")

    if box_count > 0:
        if box_count != 1:
            raise ValueError("当前有箱室0号块仅支持单侧一个箱室。")
        wall_width = _positive(longitudinal, "L_wall_0_side")
        box_half = _positive(longitudinal, "L_box_0_half")
        curve_start = _positive(longitudinal, "x_b_lin_end_0")
        box_start = 0.0
        box_end = box_half
        wall_start = box_end
        wall_end = wall_start + wall_width
        if wall_end >= zero_block_end:
            raise ValueError(
                "有箱室0号块的箱室半长与外侧横墙必须位于L_0_half以内。"
            )
        if curve_start < wall_end or curve_start > zero_block_end:
            raise ValueError(
                "有箱室0号块的x_b_lin_end_0必须位于外侧横墙之后、"
                "L_0_half以内。"
            )
        return LongitudinalLayout(
            kind=BOXED_ZERO_BLOCK,
            wall_half=wall_width,
            curve_start=curve_start,
            sequence_start=zero_block_end,
            curve_end=curve_start + variable_length,
            zero_block_end=zero_block_end,
            box_start=box_start,
            box_end=box_end,
            wall_start=wall_start,
            wall_end=wall_end,
        )

    wall_half = _positive(longitudinal, "L_wall_0")
    if _has_positive_values(
        longitudinal,
        (
            "t_t_ch_0_s",
            "t_t_ch_0_e",
            "t_b_ch_0_s",
            "t_b_ch_0_m",
            "t_w_ch_0_s",
            "t_w_ch_1_s",
            "t_w_ch_2_s",
            "x_b_lin_end_0",
            "L_trans_0",
        ),
    ):
        curve_start = _positive(longitudinal, "x_b_lin_end_0")
        sequence_start = curve_start + _positive(longitudinal, "L_trans_0")
        return LongitudinalLayout(
            kind=THICKENED_ZERO_BLOCK,
            wall_half=wall_half,
            curve_start=curve_start,
            sequence_start=sequence_start,
            curve_end=curve_start + variable_length,
            zero_block_end=zero_block_end,
        )

    top_chamfer = _positive_pair(longitudinal, "ch_t_0_v")
    curve_start = wall_half + top_chamfer[0]
    sequence_start = curve_start + _positive(longitudinal, "L_trans_0")
    if not math.isclose(sequence_start, zero_block_end, abs_tol=1.0):
        raise ValueError(
            "倒角过渡型0号块必须满足L_0_half="
            "L_wall_0+ch_t_0_v纵向宽度+L_trans_0。"
        )
    return LongitudinalLayout(
        kind=TRANSITION_ZERO_BLOCK,
        wall_half=wall_half,
        curve_start=curve_start,
        sequence_start=sequence_start,
        curve_end=curve_start + variable_length,
        zero_block_end=zero_block_end,
    )


def _has_positive_values(data: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return all(_is_positive(data.get(key)) for key in keys)


def _positive(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not _is_positive(value):
        raise ValueError(f"参数{key}必须是有限正数。")
    return float(value)


def _positive_pair(data: dict[str, Any], key: str) -> tuple[float, float]:
    value = data.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(_is_positive(item) for item in value)
    ):
        raise ValueError(f"参数{key}必须是两个有限正数组成的尺寸对。")
    return float(value[0]), float(value[1])


def _non_negative_integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        or not float(value).is_integer()
    ):
        raise ValueError(f"参数{key}必须是非负整数。")
    return int(value)


def _is_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )
