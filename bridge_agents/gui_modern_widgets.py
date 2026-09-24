from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from bridge_agents.gui_modern_model import MetricModel, StageCardModel
from bridge_agents.gui_modern_theme import DEFAULT_THEME


STATE_COLORS = {
    "pending": "#AAB7C2",
    "running": DEFAULT_THEME.primary,
    "completed": DEFAULT_THEME.success,
    "review": DEFAULT_THEME.warning,
    "failed": DEFAULT_THEME.danger,
}
TONE_COLORS = {
    "primary": DEFAULT_THEME.primary,
    "accent": DEFAULT_THEME.accent,
    "warning": DEFAULT_THEME.warning,
    "success": DEFAULT_THEME.success,
    "neutral": "#7B8C99",
}


class MetricCard(QFrame):
    def __init__(self, metric: MetricModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setMinimumHeight(116)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 15, 18, 15)
        layout.setSpacing(4)

        heading = QHBoxLayout()
        label = QLabel(metric.label)
        label.setObjectName("muted")
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {TONE_COLORS.get(metric.tone, TONE_COLORS['neutral'])}; font-size: 11px;")
        heading.addWidget(label)
        heading.addStretch()
        heading.addWidget(dot)
        layout.addLayout(heading)

        value = QLabel(metric.value)
        value.setObjectName("metricValue")
        caption = QLabel(metric.caption)
        caption.setObjectName("muted")
        caption.setStyleSheet("font-size: 11px;")
        layout.addWidget(value)
        layout.addWidget(caption)


class StatusPill(QLabel):
    def __init__(self, text: str, tone: str = "primary", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        color = TONE_COLORS.get(tone, TONE_COLORS["neutral"])
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background: {color}1F; color: {color}; border: 1px solid {color}45; "
            "border-radius: 10px; padding: 4px 10px; font-weight: 600;"
        )


class BridgeStageRail(QWidget):
    """以桥梁纵断面表达五阶段状态，作为现代界面的视觉识别元素。"""

    def __init__(self, stages: tuple[StageCardModel, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stages = stages
        self.setMinimumHeight(178)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.width()
        left, right = 54.0, width - 54.0
        deck_y = 62.0
        gap = (right - left) / max(len(self.stages) - 1, 1)

        painter.setPen(QPen(QColor("#C7D5DF"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(left, deck_y), QPointF(right, deck_y))

        for index, stage in enumerate(self.stages):
            x = left + index * gap
            pier_top = deck_y + 14
            pier_bottom = 105.0 + (7 if index % 2 else 0)
            painter.setPen(QPen(QColor("#D4DEE6"), 4))
            painter.drawLine(QPointF(x, pier_top), QPointF(x, pier_bottom))
            painter.drawLine(QPointF(x - 15, pier_bottom), QPointF(x + 15, pier_bottom))

            color = QColor(STATE_COLORS.get(stage.state, STATE_COLORS["pending"]))
            painter.setPen(QPen(QColor("#FFFFFF"), 4))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, deck_y), 12, 12)

            painter.setPen(QColor("#FFFFFF"))
            number_font = QFont("Segoe UI", 8)
            number_font.setBold(True)
            painter.setFont(number_font)
            painter.drawText(QRectF(x - 11, deck_y - 11, 22, 22), Qt.AlignmentFlag.AlignCenter, str(stage.index))

            painter.setPen(QColor("#203849"))
            label_font = QFont("Microsoft YaHei UI", 9)
            label_font.setBold(True)
            painter.setFont(label_font)
            painter.drawText(QRectF(x - gap * 0.45, 122, gap * 0.9, 22), Qt.AlignmentFlag.AlignCenter, stage.title)

            painter.setPen(color)
            status_font = QFont("Microsoft YaHei UI", 8)
            painter.setFont(status_font)
            painter.drawText(QRectF(x - gap * 0.45, 145, gap * 0.9, 18), Qt.AlignmentFlag.AlignCenter, stage.state_label)


class TimelineMarker(QWidget):
    def __init__(self, tone: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.color = QColor(TONE_COLORS.get(tone, TONE_COLORS["neutral"]))
        self.setFixedSize(20, 44)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#D6E0E7"), 2))
        painter.drawLine(QPointF(10, 19), QPointF(10, 44))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.color)
        painter.drawEllipse(QPointF(10, 10), 5, 5)
