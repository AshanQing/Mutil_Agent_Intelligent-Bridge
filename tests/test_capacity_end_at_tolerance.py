"""回归：验算脚本对盖梁弯起筋 step_2.end_at 的 str/dict 两种写法都应容忍。

背景：配筋 LLM 曾把 step_2.end_at 输出为裸字符串（如 cap_centerline_x），
而脚本原实现按 dict {x: ...} 读取导致 'str' object has no attribute 'get'，
整组验算崩溃、结果为空。修复后两种写法都能正确求值。
"""
from __future__ import annotations

from tools.cap_beam_capacity_envelope_overlay_refined import (
    Segment,
    build_control_segments,
)

_CTX = {
    "cap_length": 12450.0,
    "cap_width": 2200.0,
    "cap_height_mid": 1600.0,
    "cap_height_end": 850.0,
    "cantilever_length": 1300.0,
    "column_spacing": 7750.0,
    "column_diameter": 1600.0,
    "pier_centerline_x": 3875.0,
    "cap_centerline_x": 0.0,
    "dia": 28.0,
    "top_cover": 50.0,
    "bottom_cover": 50.0,
    "cover_x": 50.0,
    "cover_y": 50.0,
}


def _bar(end_at: object) -> dict:
    return {
        "id": "N3",
        "category": "弯起钢筋",
        "control_definition": {
            "bend_angle": 45,
            "key_points": {
                "right_bend_point": {
                    "x_expr": "pier_centerline_x + 350",
                    "y_ref": "top_cover",
                    "y_offset": "dia * 2",
                }
            },
            "branch_rules": {
                "right_branch": {
                    "step_1": {
                        "start_from": "right_bend_point",
                        "direction": "right_down",
                        "until": {
                            "y_ref": "bottom_cover",
                            "y_offset": "dia * 1",
                        },
                    },
                    "step_2": {
                        "direction": "horizontal",
                        "end_at": end_at,
                        "y_ref": "bottom_cover",
                        "y_offset": "dia * 1",
                    },
                }
            },
        },
    }


def test_dict_end_at_with_cap_centerline_x_is_evaluated() -> None:
    segs = build_control_segments(_bar({"x": "cap_centerline_x"}), h=2000.0, ctx=_CTX)
    tail = segs[-1]
    assert isinstance(tail, Segment)
    assert tail.kind == "constant"
    assert tail.x2 == 0.0  # cap_centerline_x = 0 作为 step_2 终点


def test_bare_string_end_at_is_tolerated_like_dict() -> None:
    # 复刻故障：end_at 直接写裸字符串 cap_centerline_x，旧实现在此崩溃。
    segs = build_control_segments(_bar("cap_centerline_x"), h=2000.0, ctx=_CTX)
    tail = segs[-1]
    assert isinstance(tail, Segment)
    assert tail.kind == "constant"
    assert tail.x2 == 0.0
