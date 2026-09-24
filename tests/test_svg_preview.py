"""svg_preview 轻量 SVG→PNG 渲染测试（本仓库受控 SVG 子集）。"""

from __future__ import annotations

from PIL import Image

from bridge_agents.svg_preview import render_svg_to_png

SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
<rect width="100%" height="100%" fill="#111827" />
<line x1="10" y1="10" x2="190" y2="90" stroke="#00E5FF" stroke-width="2" />
<circle cx="100" cy="50" r="20" fill="#FFD54A" />
<polyline points="0,0 200,100" stroke="#FFFFFF" />
<text x="100" y="30" text-anchor="middle" fill="#FFFFFF" font-size="14">盖梁配筋</text>
</svg>
"""


def test_render_svg_to_png_basic(tmp_path):
    svg = tmp_path / "sheet.svg"
    svg.write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / "out"
    png_path = render_svg_to_png(str(svg), out)
    assert png_path and png_path.endswith(".png")
    with Image.open(png_path) as image:
        assert image.size == (200, 100)
        # 背景深蓝；圆内（避开与白色对角线交叠的圆心）
        assert image.getpixel((5, 5)) == (17, 24, 39)
        assert image.getpixel((95, 45)) == (255, 213, 74)


def test_render_is_deterministic_cached(tmp_path):
    svg = tmp_path / "a.svg"
    svg.write_text(SAMPLE, encoding="utf-8")
    first = render_svg_to_png(str(svg), tmp_path / "c1")
    second = render_svg_to_png(str(svg), tmp_path / "c2")
    assert first is not None and second is not None
    from pathlib import Path

    assert Path(first).read_bytes() == Path(second).read_bytes()


def test_render_invalid_inputs_return_none(tmp_path):
    assert render_svg_to_png(str(tmp_path / "missing.svg"), tmp_path) is None
    bad = tmp_path / "bad.svg"
    bad.write_text("not xml at all <<<", encoding="utf-8")
    assert render_svg_to_png(str(bad), tmp_path) is None
    wrong = tmp_path / "wrong.svg"
    wrong.write_text("<circle cx='1' cy='1' r='2'/>", encoding="utf-8")  # 根不是 svg
    assert render_svg_to_png(str(wrong), tmp_path) is None


def test_render_with_text_anchor_and_font(tmp_path):
    svg = tmp_path / "t.svg"
    svg.write_text(
        "<svg width='300' height='80'>"
        "<rect width='100%' height='100%' fill='white'/>"
        "<text x='150' y='40' text-anchor='middle' fill='#000000' font-size='16'>桥梁设计</text>"
        "</svg>",
        encoding="utf-8",
    )
    png_path = render_svg_to_png(str(svg), tmp_path / "out2")
    assert png_path is not None
    with Image.open(png_path) as image:
        assert image.size == (300, 80)
