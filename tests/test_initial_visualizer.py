"""initial_visualizer 可视化器测试（纯函数 + 渲染产出）。"""

from __future__ import annotations

import json

from PIL import Image

from bridge_agents.initial_visualizer import (
    _render_zdm_profile,
    build_catalog_views,
    build_vertical_profile,
    format_k,
    parse_k,
    pier_rows,
    plane_segments,
    render_plan_overlay,
    render_profile,
    station_to_xy,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _plane_json():
    return {
        "平曲线结构": [
            {
                "曲线类型": "直线",
                "起点": {"桩号": "K41+000.000", "X": 3195400.0, "Y": 481000.0},
                "终点": {"桩号": "K41+600.000", "X": 3195300.0, "Y": 481100.0},
            },
            {
                "曲线类型": "直线",
                "起点": {"桩号": "K41+600.000", "X": 3195300.0, "Y": 481100.0},
                "终点": {"桩号": "K42+400.000", "X": 3195100.0, "Y": 481300.0},
            },
        ]
    }


def _layout_json():
    return {
        "设桥总览": {"是否设桥": "是", "桥位数量": 1},
        "桥位列表": [
            {
                "桥位编号": "1",
                "线路类型": "整体式",
                "统一布跨方案": {
                    "设桥信息": {"跨径组合": "40 + 60", "中心桩号": "K41+150"},
                    "墩位与墩高": [
                        {
                            "墩号": "0 (桥台)",
                            "桩号": "K41+080.000",
                            "墩位类型": "桥台",
                            "左幅墩高(m)": 0.0,
                            "右幅墩高(m)": 0.0,
                            "墩位校核说明": "桥台位置。设计高约244.33m，地面高约231.52m",
                        },
                        {
                            "墩号": "1",
                            "桩号": "K41+140.000",
                            "墩位类型": "桥墩",
                            "左幅墩高(m)": 12.8,
                            "右幅墩高(m)": 12.8,
                            "墩位校核说明": "桥墩。",
                        },
                        {
                            "墩号": "2",
                            "桩号": "K41+200.000",
                            "墩位类型": "桥墩",
                            "左幅墩高(m)": 15.5,
                            "右幅墩高(m)": 15.5,
                            "墩位校核说明": "桥墩。",
                        },
                        {
                            "墩号": "3 (桥台)",
                            "桩号": "K41+260.000",
                            "墩位类型": "桥台",
                            "左幅墩高(m)": 0.0,
                            "右幅墩高(m)": 0.0,
                            "墩位校核说明": "桥台位置。设计高约243.03m，地面高约246.40m",
                        },
                    ],
                },
            }
        ],
    }


def _terrain_with_pgw(dir_path):
    dir_path.mkdir(parents=True, exist_ok=True)
    png = dir_path / "pred_AB.png"
    image = Image.new("L", (500, 300), 120)
    image.save(png)
    pgw = dir_path / "pred_AB.pgw"
    # A D B E C F：cell=2，extent x 3194800..3195800, y 480700..481300
    pgw.write_text("2\n0\n0\n-2\n3194800\n481300\n", encoding="utf-8")
    return png, pgw


# ---- 纯函数 ----


def test_parse_k_variants():
    assert parse_k("K41+080.000") == 41080.0
    assert parse_k("K12+360") == 12360.0
    assert parse_k("K42+480") == 42480.0
    assert parse_k(12345) == 12345.0
    assert parse_k("980") == 980000.0  # 小于 1000 视为千米数
    assert parse_k("K41+80") == 41080.0


def test_format_k_roundtrip():
    assert format_k(41080.0).startswith("K41+080")
    assert parse_k(format_k(42480.0)) == 42480.0


def test_plane_segments_sorted():
    segments = plane_segments(_plane_json())
    assert len(segments) == 2
    assert segments[0]["k0"] < segments[1]["k0"]


def test_station_to_xy_midpoint():
    xy = station_to_xy(_plane_json(), 41300.0)  # K41+300 落在段1中点
    assert xy == (3195350.0, 481050.0)
    assert station_to_xy(_plane_json(), 41000.0) is not None  # 前伸
    assert station_to_xy(_plane_json(), 42600.0) is not None  # 后延


def test_pier_rows_extraction():
    bridges = pier_rows(_layout_json())
    assert len(bridges) == 1
    piers = bridges[0]["piers"]
    assert len(piers) == 4
    assert piers[0]["station"] == 41080.0
    assert piers[0]["design_h"] == 244.33
    assert piers[0]["ground_h"] == 231.52
    assert piers[1]["height_l"] == 12.8
    assert piers[3]["type"] == "桥台"
    assert bridges[0]["span_text"] == "40 + 60"


# ---- 渲染产出 ----


def test_render_plan_overlay_writes_png(tmp_path):
    terrain, pgw = _terrain_with_pgw(tmp_path)
    plane = tmp_path / "K_plane.json"
    layout = tmp_path / "layout.json"
    _write_json(plane, _plane_json())
    _write_json(layout, _layout_json())
    out = tmp_path / "views" / "plan_overlay.png"
    result = render_plan_overlay(
        terrain_png=str(terrain),
        pgw_path=str(pgw),
        plane_json_path=str(plane),
        layout_json_path=str(layout),
        out_png=str(out),
        variant="initial",
    )
    assert result == str(out)
    with Image.open(out) as image:
        assert image.size[0] > 400 and image.size[1] > 200


def test_render_plan_overlay_missing_input_returns_none(tmp_path):
    terrain, pgw = _terrain_with_pgw(tmp_path)
    layout = tmp_path / "layout.json"
    _write_json(layout, _layout_json())
    assert (
        render_plan_overlay(
            terrain_png=str(terrain),
            pgw_path=str(pgw),
            plane_json_path=str(tmp_path / "missing.json"),
            layout_json_path=str(layout),
            out_png=str(tmp_path / "out.png"),
        )
        is None
    )


def test_render_profile_writes_png(tmp_path):
    layout = tmp_path / "layout.json"
    _write_json(layout, _layout_json())
    out = tmp_path / "profile.png"
    result = render_profile(layout_json_path=str(layout), out_png=str(out))
    assert result == str(out)
    with Image.open(out) as image:
        assert image.size[0] > 400


def test_render_profile_empty_returns_none(tmp_path):
    layout = tmp_path / "layout.json"
    _write_json(layout, {"设桥总览": {}, "桥位列表": []})
    assert render_profile(layout_json_path=str(layout), out_png=str(tmp_path / "p.png")) is None


# ---- 顶层批量 ----


def test_build_catalog_views_full(tmp_path):
    run = tmp_path / "run"
    terrain, pgw = _terrain_with_pgw(run)
    del terrain, pgw
    _write_json(run / "plane_from_loader" / "K_plane.json", _plane_json())
    _write_json(run / "design_run_1" / "design_result.json", _layout_json())
    _write_json(run / "layout_revision" / "final_layout_result.json", _layout_json())

    result = build_catalog_views(run)
    for key, path in result.items():
        assert key in {"initial_plan", "initial_profile", "final_plan", "final_profile"}
        assert path and __import__("os").path.isfile(path), key

    # 初始与最终必须分目录保存、互不覆盖
    from pathlib import Path

    assert result["initial_plan"] != result["final_plan"]
    assert "initial" in Path(result["initial_plan"]).parts
    assert "final" in Path(result["final_plan"]).parts

    # 幂等：重复调用路径一致
    again = build_catalog_views(run)
    assert again == result


def test_build_catalog_views_without_plane_still_profiles(tmp_path):
    run = tmp_path / "run"
    _write_json(run / "layout_revision" / "final_layout_result.json", _layout_json())
    result = build_catalog_views(run)
    assert result["final_plan"] is None  # 缺 PGW/平面，不强行叠加
    assert result["final_profile"] is not None


# ---- zdm 纵断面移植 ----
def _vertical_payload():
    return {
        "设计线高程序列": [
            {"桩号": 41000.0, "高程": 100.0, "R": 0.0, "来源": "原始纵断面控制点"},
            {"桩号": 41200.0, "高程": 200.0, "R": 0.0, "来源": "原始纵断面控制点"},
            {"桩号": 41400.0, "高程": 100.0, "R": 10000.0, "来源": "原始纵断面控制点"},
        ],
        "地形线高程序列": [
            {"桩号": 40900.0, "高程": 95.0},
            {"桩号": 41100.0, "高程": 120.0},
            {"桩号": 41300.0, "高程": 130.0},
            {"桩号": 41500.0, "高程": 105.0},
        ],
    }


def test_build_vertical_profile_straight_grade():
    profile = build_vertical_profile(
        [{"桩号": 41000.0, "高程": 100.0, "R": 0.0}, {"桩号": 41400.0, "高程": 120.0, "R": 0.0}]
    )
    assert profile(41200.0) == 110.0  # 直线段中点
    assert profile(41000.0) == 100.0


def test_build_vertical_profile_applies_vertical_curve():
    # 竖曲线半径只作用于内部点（PVI）
    profile = build_vertical_profile(
        [
            {"桩号": 41000.0, "高程": 100.0, "R": 0.0},
            {"桩号": 41200.0, "高程": 200.0, "R": 10000.0},
            {"桩号": 41400.0, "高程": 100.0, "R": 0.0},
        ]
    )
    mid = float(profile(41200.0))
    # 凸竖曲线中点低于折线交点 200（约 150，按 zdm 抛物线语义）
    assert abs(mid - 150.0) < 3.0, mid
    assert mid < 200.0
    # 端点仍落在边界直坡
    assert abs(float(profile(41000.0)) - 100.0) < 0.01


def test_render_profile_with_vertical_data_renders_zdm(tmp_path):
    layout = tmp_path / "layout.json"
    _write_json(layout, _layout_json())
    out = tmp_path / "profile_zdm.png"
    result = render_profile(
        layout_json_path=str(layout),
        out_png=str(out),
        variant="final",
        vertical=_vertical_payload(),
    )
    assert result == str(out)
    with Image.open(out) as image:
        assert image.size[0] > 500 and image.size[1] > 250


def test_render_zdm_profile_missing_parts_returns_none(tmp_path):
    layout = tmp_path / "layout.json"
    _write_json(layout, _layout_json())
    vertical = {"设计线高程序列": [], "地形线高程序列": []}
    assert (
        _render_zdm_profile(
            layout_json_path=str(layout), vertical=vertical,
            out_png=str(tmp_path / "x.png"), variant="final",
        )
        is None
    )


def test_build_catalog_views_picks_vertical_from_input_data(tmp_path):
    run = tmp_path / "run"
    _write_json(run / "layout_revision" / "final_layout_result.json", _layout_json())
    _write_json(
        run / "design_run_1" / "input_data.json",
        {"线路类型": "整体式", "K": {"平曲线结构": [], "纵断面结构": _vertical_payload()}},
    )
    result = build_catalog_views(run)
    assert result["final_profile"] is not None
