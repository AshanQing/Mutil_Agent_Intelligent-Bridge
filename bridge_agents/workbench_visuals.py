"""桥梁设计工作台的矢量导航图标与蓝图背景。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QFrame, QWidget


_BLUEPRINT = Path(__file__).parent / "assets" / "bridge_blueprint.svg"
_ICON_PATHS = {
    "overview": '<path d="M3 10.5 12 3l9 7.5V21H3z"/><path d="M8 21v-7h8v7"/>',
    "task": '<path d="M5 3h10l4 4v14H5z"/><path d="M15 3v5h4M9 12h6M9 16h4"/><path d="m16 15 3 2-3 2z"/>',
    "review": '<path d="M7 4H4v17h16V4h-3M9 3h6v3H9z"/><path d="m8 14 3 3 5-6"/>',
    "results": '<path d="M3 7h7l2 2h9v11H3z"/><path d="M3 7V4h7l2 3M7 14h10M7 17h7"/>',
    "qa": '<path d="M4 4h16v13H9l-5 4z"/><path d="M8 9h8M8 13h5"/>',
    "log": '<path d="M5 3h14v18H5z"/><path d="M9 8h6M9 12h6M9 16h4"/>',
}


def navigation_icon(name: str) -> QIcon:
    path = _ICON_PATHS[name]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
        '<g fill="none" stroke="#C6DBEA" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f"{path}</g></svg>"
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(QSize(40, 40))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


class _BlueprintMixin:
    def _paint_blueprint(self, painter: QPainter, *, dark: bool = False) -> None:
        renderer = QSvgRenderer(str(_BLUEPRINT))
        if not renderer.isValid():
            return
        width = self.width()
        height = self.height()
        target_width = max(width * (1.25 if dark else 0.88), 500)
        target_height = target_width * 620 / 1200
        x = (width - target_width) / 2 if dark else width - target_width + 55
        y = height - target_height - 170 if dark else height - target_height + 12
        painter.save()
        painter.setOpacity(0.28 if dark else 0.62)
        renderer.render(painter, QRectF(x, y, target_width, target_height))
        painter.restore()


class BlueprintPage(QWidget, _BlueprintMixin):
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#F3F6F9"))
        self._paint_blueprint(painter)
        painter.end()


class BlueprintSidebar(QFrame, _BlueprintMixin):
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().paintEvent(event)
        painter = QPainter(self)
        self._paint_blueprint(painter, dark=True)
        painter.end()
