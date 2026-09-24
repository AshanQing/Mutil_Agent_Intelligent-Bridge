"""轻量 SVG → PNG 窗口内预览渲染器。

本仓库的绘图 SVG（bridge_agents/drawing/svg_exporter.py 输出）元素集合受控，
仅含：svg / rect / line / polyline / circle / text。因此不需要引入 Cairo，
直接用 xml 解析 + Pillow 即可把图纸渲染为 PNG 供成果中心窗口内预览。

- 画布尺寸取自 <svg width/height>；
- 整幅 <rect width="100%"> 作为背景色；
- 文本按 SVG baseline 语义（text-anchor start/middle/end）定位；
- 中文与数字统一使用系统微软雅黑（msyh.ttc），回退 Arial。
"""

from __future__ import annotations

import hashlib
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
)
_AVAILABLE_FONT: Optional[str] = None
_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}
_FONT_CACHE_LIMIT = 32
_MAX_CANVAS = 4096

_COLOR_RE = re.compile(r"^#([0-9a-fA-F]{6})$|^#([0-9a-fA-F]{3})$")
_RGB_RE = re.compile(r"^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)")

_LOCAL = "{http://www.w3.org/2000/svg}"


def _resolve_font() -> Optional[str]:
    global _AVAILABLE_FONT
    if _AVAILABLE_FONT is None:
        for candidate in _FONT_CANDIDATES:
            if Path(candidate).is_file():
                _AVAILABLE_FONT = candidate
                break
        else:
            _AVAILABLE_FONT = ""  # 空串表示已探测且未找到
    return _AVAILABLE_FONT or None


def _font(size: float) -> ImageFont.FreeTypeFont:
    key = int(size)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    font_path = _resolve_font()
    try:
        font = ImageFont.truetype(font_path, key) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    if len(_FONT_CACHE) >= _FONT_CACHE_LIMIT:
        _FONT_CACHE.clear()
    _FONT_CACHE[key] = font
    return font


def _color(value: Any) -> Optional[Tuple[int, int, int]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "transparent"}:
        return None
    match = _COLOR_RE.match(text)
    if match:
        if match.group(1):
            hex_value = match.group(1)
        else:
            hex_value = "".join(ch * 2 for ch in match.group(2))
        return (int(hex_value[0:2], 16), int(hex_value[2:4], 16), int(hex_value[4:6], 16))
    match = _RGB_RE.match(text)
    if match:
        try:
            return tuple(int(round(float(match.group(i)))) for i in (1, 2, 3))  # type: ignore[misc]
        except (TypeError, ValueError):
            return None
    return None


def _number(value: Any, default: float = 1.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _local_tag(tag: str) -> str:
    if tag.startswith(_LOCAL) or tag.startswith("{"):
        return tag.split("}", 1)[-1]
    return tag


def _parse_points(value: Any) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    try:
        numbers = [float(part) for part in re.split(r"[,\s]+", str(value).strip()) if part]
    except ValueError:
        return points
    for i in range(0, len(numbers) - 1, 2):
        points.append((numbers[i], numbers[i + 1]))
    return points


def render_svg_to_png(svg_path: str, out_dir: str | Path) -> Optional[str]:
    """把受控 SVG 渲染为 PNG 并写入 out_dir，返回 PNG 路径；失败返回 None。

    输出文件名由 SVG 内容哈希决定（同源同内容复用缓存，不重复渲染）。
    """
    svg_path = str(svg_path)
    if not os.path.isfile(svg_path):
        return None
    try:
        raw = Path(svg_path).read_bytes()
        tree = ET.fromstring(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None

    root_tag = _local_tag(tree.tag)
    if root_tag != "svg":
        return None

    width = _number(tree.get("width"), 1600)
    height = _number(tree.get("height"), 1200)
    if width <= 0 or height <= 0 or width > _MAX_CANVAS or height > _MAX_CANVAS:
        return None
    canvas_w, canvas_h = int(width), int(height)

    background = (255, 255, 255)
    try:
        image = Image.new("RGB", (canvas_w, canvas_h), background)
        draw = ImageDraw.Draw(image)
    except Exception:
        return None

    def draw_children(element: ET.Element) -> None:
        for child in element:
            tag = _local_tag(child.tag)
            try:
                if tag == "rect":
                    w_attr = child.get("width")
                    h_attr = child.get("height")
                    if str(w_attr).strip() in {"100%", "100"} and str(h_attr).strip() in {"100%", "100"}:
                        bg = _color(child.get("fill"))
                        if bg:
                            draw.rectangle((0, 0, canvas_w, canvas_h), fill=bg)
                        continue
                    x = _number(child.get("x"), 0)
                    y = _number(child.get("y"), 0)
                    w = _number(w_attr, 0)
                    h = _number(h_attr, 0)
                    fill = _color(child.get("fill"))
                    stroke = _color(child.get("stroke"))
                    if fill is not None:
                        draw.rectangle((x, y, x + w, y + h), fill=fill)
                    if stroke is not None:
                        draw.rectangle((x, y, x + w, y + h), outline=stroke,
                                       width=max(1, int(_number(child.get("stroke-width"), 1))))
                elif tag == "line":
                    stroke = _color(child.get("stroke"))
                    if stroke is not None:
                        draw.line(
                            (_number(child.get("x1")), _number(child.get("y1")),
                             _number(child.get("x2")), _number(child.get("y2"))),
                            fill=stroke,
                            width=max(1, int(_number(child.get("stroke-width"), 1))),
                        )
                elif tag == "polyline":
                    points = _parse_points(child.get("points"))
                    stroke = _color(child.get("stroke"))
                    if len(points) >= 2 and stroke is not None:
                        draw.line(points, fill=stroke,
                                  width=max(1, int(_number(child.get("stroke-width"), 1))),
                                  joint="curve")
                elif tag == "circle":
                    cx = _number(child.get("cx"), 0)
                    cy = _number(child.get("cy"), 0)
                    radius = _number(child.get("r"), 0)
                    if radius <= 0:
                        continue
                    box = (cx - radius, cy - radius, cx + radius, cy + radius)
                    fill = _color(child.get("fill"))
                    stroke = _color(child.get("stroke"))
                    if fill is not None:
                        draw.ellipse(box, fill=fill)
                    if stroke is not None:
                        draw.ellipse(box, outline=stroke,
                                     width=max(1, int(_number(child.get("stroke-width"), 1))))
                elif tag == "text":
                    x = _number(child.get("x"), 0)
                    y = _number(child.get("y"), 0)
                    size = _number(child.get("font-size"), 14)
                    if size <= 0:
                        size = 14
                    anchor = str(child.get("text-anchor") or "start")
                    color = _color(child.get("fill")) or (0, 0, 0)
                    pil_anchor = {"middle": "ms", "end": "rs"}.get(anchor, "ls")
                    text = "".join(child.itertext())
                    if text:
                        draw.text((x, y), text, font=_font(size), fill=color,
                                  anchor=pil_anchor)
            except Exception:
                continue
            draw_children(child)

    try:
        draw_children(tree)
    except Exception:
        return None

    digest = hashlib.sha256(raw).hexdigest()[:12]
    out_dir_path = Path(out_dir)
    try:
        out_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    png_path = out_dir_path / f"{Path(svg_path).stem}_{digest}.png"
    try:
        image.save(png_path, format="PNG")
    except Exception:
        return None
    return str(png_path)
