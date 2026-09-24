"""成果图像查看器（QGraphicsView 实现）。

替代原先固定缩放一次的 QLabel 预览，解决三件事：
- 清晰度：位图按原始分辨率加载；受控 SVG 用 QSvgRenderer 矢量超采样渲染，
  不再依赖 QPixmap 加载 SVG 的默认低分辨率；
- 交互：滚轮以鼠标为锚点自由缩放，左键拖拽平移，双击（或工具栏按钮）复位适配窗口；
- 提示：无预览时在画布中央显示原因，不再用 QLabel 文本。

依赖 PySide6；本模块不引入 PIL，SVG 渲染回退时才会延迟导入 svg_preview。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QLabel,
)

_BITMAP_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
# 适配视图下的缩放倍数下限/上限（1.0 = 恰好适配窗口）
_MIN_SCALE = 0.02
_MAX_SCALE = 32.0
# SVG 矢量渲染的目标长边（像素），超采样保证初始显示与放大查看都足够清晰
_SVG_RENDER_LONG_EDGE = 4096
_SVG_MAX_SCALE = 8.0  # 对小尺寸 SVG 的超采样上限（避免渲染超大位图）
_BG_COLOR = "#F5F8FA"


class PanZoomImageView(QGraphicsView):
    """可缩放 / 可拖动的成果图像查看器。

    对外接口：
    - load_file(path) -> bool：加载 PNG/JPG/BMP/WebP/SVG，成功后自动适配窗口；
    - show_message(text)：清空图像并居中显示提示；
    - clear_image()：清空画布；
    - fit_to_view()：复位为完整显示；
    - zoom_changed(float) 信号：1.0 表示恰好适配窗口，其余为相对倍数。
    """

    zoom_changed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self._svg_source: str | None = None
        self._zoom_factor = 1.0
        self._dragging = False
        self._drag_start = QPointF()

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(_BG_COLOR))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "QGraphicsView { background: #F5F8FA; border: 1px dashed #C8D4DD;"
            " border-radius: 10px; }"
        )

        self._hint = QLabel(self)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(
            "background: transparent; color: #778997; font-size: 12px;"
        )
        self._hint.hide()

    # ------------------------------------------------------------------ 加载
    def load_file(self, path: str) -> bool:
        """加载图像文件；成功返回 True 并自动适配窗口。"""
        self.clear_image()
        suffix = Path(str(path)).suffix.lower()
        try:
            if suffix in _BITMAP_SUFFIXES:
                pixmap = QPixmap(str(path))
            elif suffix == ".svg":
                pixmap = self._render_svg(str(path))
            else:
                return False
        except Exception:
            return False
        if pixmap.isNull():
            return False
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self._svg_source = str(path) if suffix == ".svg" else None
        self._hint.hide()
        self.fit_to_view()
        return True

    def show_message(self, text: str) -> None:
        """清空图像并居中显示提示文字。"""
        self.clear_image()
        self._hint.setText(text)
        self._hint.show()

    def clear_image(self) -> None:
        self._item.setPixmap(QPixmap())
        self._scene.setSceneRect(0, 0, 0, 0)
        self._svg_source = None
        self.resetTransform()
        self._zoom_factor = 1.0

    # ------------------------------------------------------------------ 视图
    def fit_to_view(self) -> None:
        if self._item.pixmap().isNull():
            return
        self.resetTransform()
        self._zoom_factor = 1.0
        self.fitInView(self._item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # fitInView 会整体替换 transform；把"恰好适配"的缩放读回来作为基准
        m11 = self.transform().m11()
        self._zoom_factor = m11 if m11 > 0 else 1.0
        self.zoom_changed.emit(1.0)

    # ------------------------------------------------------------------ 交互
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._item.pixmap().isNull():
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        factor = 1.25 ** steps
        new_factor = self._zoom_factor * factor
        new_factor = max(_MIN_SCALE, min(_MAX_SCALE, new_factor))
        applied = new_factor / self._zoom_factor
        if abs(applied - 1.0) < 1e-6:
            return
        self.scale(applied, applied)
        self._zoom_factor = new_factor
        self.zoom_changed.emit(self._zoom_factor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._item.pixmap().isNull()
        ):
            self._dragging = True
            self._drag_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._dragging:
            delta = event.position() - self._drag_start
            self._drag_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - round(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - round(delta.y())
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_to_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        viewport_rect = self.viewport().geometry()
        self._hint.setGeometry(viewport_rect)
        self._hint.raise_()

    # ------------------------------------------------------------------ SVG
    @staticmethod
    def _render_svg(path: str) -> QPixmap:
        """用 QSvgRenderer 以高分辨率渲染受控 SVG；失败时回退项目自研渲染器。"""
        try:
            from PySide6.QtSvg import QSvgRenderer
        except ImportError:
            return PanZoomImageView._render_svg_with_pillow(path)
        renderer = QSvgRenderer(path)
        if not renderer.isValid():
            return PanZoomImageView._render_svg_with_pillow(path)
        size = renderer.defaultSize()
        width = max(size.width(), 1)
        height = max(size.height(), 1)
        scale = min(
            max(_SVG_RENDER_LONG_EDGE / max(width, height), 1.0),
            _SVG_MAX_SCALE,
        )
        image = QImage(
            int(width * scale), int(height * scale), QImage.Format.Format_ARGB32
        )
        image.fill(Qt.GlobalColor.white)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        renderer.render(painter)
        painter.end()
        return QPixmap.fromImage(image)

    @staticmethod
    def _render_svg_with_pillow(path: str) -> QPixmap:
        """回退到项目自带 svg_preview 渲染器（Pillow + 微软雅黑）。"""
        from bridge_agents.svg_preview import render_svg_to_png

        out_dir = Path(tempfile.gettempdir()) / "bridge_gui_preview"
        png_path = render_svg_to_png(path, out_dir)
        if not png_path:
            return QPixmap()
        return QPixmap(png_path)
