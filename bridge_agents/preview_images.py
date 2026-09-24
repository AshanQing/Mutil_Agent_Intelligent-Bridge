"""窗口内图片预览管线（Pillow）。

职责：
- 判定某文件能否窗口内预览（PNG/JPG/JPEG；SVG/SCR/表格等一律外部打开）；
- 解码并适配到目标视口（缩略图金字塔：按 (path, mtime, box) 缓存缩放结果）；
- JPEG/高 DPI/EXIF 方向统一处理，坏文件抛可恢复的 PreviewUnavailableError。

本模块不依赖 Tkinter，可独立测试。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps

PREVIEW_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})

# (path, mtime_ns, max_box) -> 适配后 Image（RGB）
_cache: dict = {}
_CACHE_LIMIT = 64


class PreviewUnavailableError(RuntimeError):
    """文件不存在、非图片或解码失败时抛出，界面据此回退到提示。"""


def supports_preview(path) -> bool:
    """按扩展名判断是否可窗口内预览（PNG/JPG 等位图）。"""
    try:
        return Path(str(path)).suffix.lower() in PREVIEW_EXTENSIONS
    except Exception:
        return False


def image_size(path) -> Optional[Tuple[int, int]]:
    """快速读取图片原始宽高（不载入像素）；失败返回 None。"""
    try:
        with Image.open(str(path)) as img:
            return img.size
    except Exception:
        return None


def open_fit(path, max_box: Tuple[int, int] = (1600, 1200)) -> Image.Image:
    """解码图片并等比缩放到 max_box 内（不放大原图），返回独立 RGB 副本。

    结果按 (path, mtime, box) 缓存，多次选中同一文件不会重复解码大图。
    """
    target = str(path)
    if not os.path.isfile(target):
        raise PreviewUnavailableError(f"文件不存在：{target}")
    try:
        mtime_ns = os.stat(target).st_mtime_ns
    except OSError as exc:
        raise PreviewUnavailableError(f"无法读取文件信息：{target}") from exc

    box = (int(max_box[0]), int(max_box[1]))
    key = (target, mtime_ns, box)
    cached = _cache.get(key)
    if cached is not None:
        return cached.copy()

    if not supports_preview(target):
        raise PreviewUnavailableError(f"该类型不支持窗口内预览：{target}")

    try:
        with Image.open(target) as raw:
            image = ImageOps.exif_transpose(raw)
            image = image.convert("RGB")
            width, height = image.size
            scale = min(box[0] / width, box[1] / height, 1.0)
            if scale < 1.0:
                image = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.LANCZOS,
                )
    except PreviewUnavailableError:
        raise
    except Exception as exc:
        raise PreviewUnavailableError(f"图片解码失败：{target}") from exc

    if len(_cache) >= _CACHE_LIMIT:
        _cache.clear()
    _cache[key] = image.copy()
    return image.copy()


def clear_cache() -> None:
    _cache.clear()
