from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class _RoleStyle:
    """一种发言角色在对话区中的呈现方式。"""

    name: str
    avatar: str
    avatar_object: str
    bubble_object: str
    text_object: str
    side: str
    markdown: bool
    kind: str = "bubble"


ROLE_STYLES: dict[str, _RoleStyle] = {
    "user": _RoleStyle(
        name="你",
        avatar="你",
        avatar_object="chatAvatarUser",
        bubble_object="chatBubbleUser",
        text_object="chatBubbleTextUser",
        side="right",
        markdown=False,
    ),
    "assistant": _RoleStyle(
        name="设计复核智能体",
        avatar="AI",
        avatar_object="chatAvatarAgent",
        bubble_object="chatBubbleAgent",
        text_object="chatBubbleTextAgent",
        side="left",
        markdown=True,
    ),
    "assessment": _RoleStyle(
        name="整体评估",
        avatar="评",
        avatar_object="chatAvatarAgent",
        bubble_object="chatBubbleAssessment",
        text_object="chatBubbleTextAssessment",
        side="left",
        markdown=False,
    ),
    "system": _RoleStyle(
        name="系统",
        avatar="",
        avatar_object="",
        bubble_object="",
        text_object="chatSystemNote",
        side="center",
        markdown=False,
        kind="note",
    ),
}

_AVATAR_SIZE = 30
_BUBBLE_WIDTH_RATIO = 0.74
_BUBBLE_MIN_WIDTH = 300
# 气泡左右内边距 14+14、边框、QLabel 富文本文档默认外边距（4+4）的合计，
# 再加上一点余量，避免正文在理想宽度处被提前折行。
_BUBBLE_H_MARGINS = 40
_BUBBLE_SLACK = 4


@dataclass
class _BubbleEntry:
    bubble: QFrame
    label: QLabel
    style: _RoleStyle
    body: str
    natural_width: float


class ChatTranscript(QWidget):
    """问答页对话记录。

    每一轮问答渲染成带发言人和气泡的对话行：用户消息右对齐、智能体消息
    左对齐，智能体与整体评估正文按 Markdown 渲染，使 `## 标题`、列表、
    引用等结构可以直接呈现，而不是把原始标记文本罗列出来。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chatTranscript")
        self._messages: list[tuple[str, str, QLabel]] = []
        self._bubbles: list[_BubbleEntry] = []
        self._pending: QWidget | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.card = QFrame(self)
        self.card.setObjectName("chatCard")
        outer.addWidget(self.card)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(1, 1, 1, 1)
        self.scroll = QScrollArea(self.card)
        self.scroll.setObjectName("chatScrollArea")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 注意：不要在这里给 viewport 设置局部样式表。无选择器的局部样式表会
        # 覆盖整棵子树的 background，气泡底色会被一并冲掉。画布自身不透明即可。
        self.canvas = QFrame()
        self.canvas.setObjectName("chatCanvas")
        self.column = QVBoxLayout(self.canvas)
        self.column.setContentsMargins(18, 16, 18, 16)
        self.column.setSpacing(12)
        self.column.addStretch(1)
        self.scroll.setWidget(self.canvas)
        # 视口宽度决定气泡上限；视口尺寸变化时必须重算，否则会一直用首次的窄宽度。
        self.scroll.viewport().installEventFilter(self)
        card_layout.addWidget(self.scroll)

    # --- 对外接口 -----------------------------------------------------
    def append_message(self, role: str, text: str) -> None:
        style = ROLE_STYLES.get(role, ROLE_STYLES["system"])
        body = str(text or "")
        if style.kind == "note":
            label = QLabel(body)
            label.setObjectName(style.text_object)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._insert(label)
            self._messages.append((role, body, label))
        else:
            row, entry = self._build_bubble(style, body)
            self._insert(row)
            self._bubbles.append(entry)
            self._messages.append((role, body, entry.label))
        self._layout_bubbles()
        self._scroll_to_bottom()

    def show_pending(self, text: str) -> None:
        """展示"正在生成"的临时提示，结果到达后由 clear_pending 移除。"""
        self.clear_pending()
        note = QLabel(str(text))
        note.setObjectName("chatPendingNote")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._insert(note)
        self._pending = note
        self._scroll_to_bottom()

    def clear_pending(self) -> None:
        if self._pending is None:
            return
        self.column.removeWidget(self._pending)
        self._pending.deleteLater()
        self._pending = None

    def clear(self) -> None:
        pending = self._pending
        self._pending = None
        while self.column.count() > 1:
            widget = self.column.takeAt(0).widget()
            if widget is not None and widget is not pending:
                widget.deleteLater()
        if pending is not None:
            pending.deleteLater()
        self._messages.clear()
        self._bubbles.clear()

    def message_count(self) -> int:
        return len(self._messages)

    def toPlainText(self) -> str:
        """按追加顺序返回所有消息正文，便于断言与复制。"""
        return "\n".join(text for _, text, _ in self._messages)

    # --- 内部实现 -----------------------------------------------------
    def _insert(self, widget: QWidget) -> None:
        """插到末尾（保留最后一项的伸缩量）。"""
        self.column.insertWidget(self.column.count() - 1, widget)

    def _build_bubble(self, style: _RoleStyle, body: str) -> tuple[QWidget, _BubbleEntry]:
        row = QWidget()
        row.setObjectName("chatRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        avatar = QLabel(style.avatar)
        avatar.setObjectName(style.avatar_object)
        avatar.setFixedSize(_AVATAR_SIZE, _AVATAR_SIZE)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        bubble = QFrame()
        bubble.setObjectName(style.bubble_object)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(14, 10, 14, 12)
        bubble_layout.setSpacing(5)

        speaker = QLabel(style.name)
        speaker.setObjectName("chatRoleName")
        speaker.setAlignment(
            Qt.AlignmentFlag.AlignRight if style.side == "right" else Qt.AlignmentFlag.AlignLeft
        )
        bubble_layout.addWidget(speaker)

        label = QLabel(body)
        label.setObjectName(style.text_object)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        label.setTextFormat(
            Qt.TextFormat.MarkdownText if style.markdown else Qt.TextFormat.PlainText
        )
        self._enable_height_for_width(label)
        # 必须先让样式表落地，否则量到的是样式表生效前的字号，气泡会算窄。
        label.ensurePolished()
        bubble_layout.addWidget(label)
        self._enable_height_for_width(bubble)
        self._enable_height_for_width(row)

        if style.side == "right":
            layout.addStretch(1)
            layout.addWidget(bubble)
            layout.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
        else:
            layout.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(bubble)
            layout.addStretch(1)
        entry = _BubbleEntry(
            bubble=bubble,
            label=label,
            style=style,
            body=body,
            natural_width=self._natural_width(style, body, label.font()),
        )
        return row, entry

    @staticmethod
    def _enable_height_for_width(widget: QWidget) -> None:
        """让嵌套布局按给定宽度通报高度，气泡才能随内容正确增高。"""
        policy = widget.sizePolicy()
        policy.setHeightForWidth(True)
        policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        widget.setSizePolicy(policy)

    @staticmethod
    def _natural_width(style: _RoleStyle, body: str, font: Any) -> float:
        """按实际渲染方式计算正文的"理想宽度"，短消息据此贴合内容。"""
        document = QTextDocument()
        document.setDefaultFont(font)
        if style.markdown:
            document.setMarkdown(body)
        else:
            document.setPlainText(body)
        document.setTextWidth(-1.0)
        return float(document.idealWidth())

    def _layout_bubbles(self) -> None:
        """气泡宽度取"内容理想宽度"与视口上限的较小值。"""
        limit = max(_BUBBLE_MIN_WIDTH, int(self.scroll.viewport().width() * _BUBBLE_WIDTH_RATIO))
        for entry in self._bubbles:
            content_width = min(entry.natural_width, limit - _BUBBLE_H_MARGINS)
            entry.bubble.setFixedWidth(int(content_width) + _BUBBLE_H_MARGINS + _BUBBLE_SLACK)

    def _scroll_to_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._layout_bubbles()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if watched is self.scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._layout_bubbles()
        return super().eventFilter(watched, event)
