"""preview_images 预览管线测试。"""

from __future__ import annotations

import pytest
from PIL import Image

from bridge_agents.preview_images import (
    PreviewUnavailableError,
    clear_cache,
    image_size,
    open_fit,
    supports_preview,
)


def _make_image(path, size=(4000, 2000), color=(20, 40, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    image.save(path)
    return path


def test_supports_preview_extensions(tmp_path):
    png = tmp_path / "a.png"
    jpg = tmp_path / "a.jpg"
    jpeg = tmp_path / "a.jpeg"
    for p in (png, jpg, jpeg):
        p.write_bytes(b"x")
        assert supports_preview(p)
    assert not supports_preview(tmp_path / "a.svg")
    assert not supports_preview(tmp_path / "a.scr")
    assert not supports_preview(tmp_path / "a.json")
    assert not supports_preview(tmp_path / "no_ext")
    # 扩展名判定不依赖文件存在性；打开时才报错（见 test_open_fit_missing_file_raises）
    assert supports_preview(tmp_path / "missing.png")


def test_open_fit_downscales_large_image(tmp_path):
    path = _make_image(tmp_path / "large.png", size=(4000, 2000))
    image = open_fit(path, max_box=(400, 400))
    assert image.size == (400, 200)
    assert image.mode == "RGB"
    assert image.getpixel((10, 10)) == (20, 40, 80)


def test_open_fit_never_upscales_small_image(tmp_path):
    path = _make_image(tmp_path / "small.jpg", size=(120, 80))
    image = open_fit(path, max_box=(1000, 1000))
    assert image.size == (120, 80)


def test_open_fit_caches_by_mtime(tmp_path):
    path = _make_image(tmp_path / "cached.png", size=(2000, 1000))
    first = open_fit(path, max_box=(200, 200))
    assert first.size == (200, 100)

    # 重写文件（mtime 变化）后应重新解码为新内容
    _make_image(path, size=(800, 800), color=(200, 30, 30))
    second = open_fit(path, max_box=(200, 200))
    assert second.size == (200, 200)


def test_open_fit_missing_file_raises(tmp_path):
    with pytest.raises(PreviewUnavailableError):
        open_fit(tmp_path / "nope.png")


def test_open_fit_bad_content_raises(tmp_path):
    path = tmp_path / "fake.png"
    path.write_bytes(b"this is not an image")
    with pytest.raises(PreviewUnavailableError):
        open_fit(path)


def test_open_fit_svg_rejected(tmp_path):
    path = tmp_path / "a.svg"
    path.write_bytes(b"<svg/>")
    with pytest.raises(PreviewUnavailableError):
        open_fit(path)


def test_image_size_quick_read(tmp_path):
    path = _make_image(tmp_path / "size.png", size=(640, 480))
    assert image_size(path) == (640, 480)
    assert image_size(tmp_path / "missing.png") is None


def test_clear_cache(tmp_path):
    path = _make_image(tmp_path / "clear.png")
    open_fit(path)
    clear_cache()  # 不应抛错，后续打开仍正常（默认视口 1600x1200 下 2000x1000 缩放为 1600x800）
    assert open_fit(path).size == (1600, 800)
