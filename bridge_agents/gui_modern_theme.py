from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeTokens:
    primary: str = "#1769AA"
    primary_dark: str = "#0F4C81"
    sidebar: str = "#102A43"
    sidebar_muted: str = "#8FA7BA"
    accent: str = "#0AA6A6"
    warning: str = "#E58A17"
    danger: str = "#D64545"
    success: str = "#238A62"
    canvas: str = "#F3F6F9"
    surface: str = "#FFFFFF"
    text: str = "#172B3A"
    text_muted: str = "#66788A"
    border: str = "#DCE4EA"


DEFAULT_THEME = ThemeTokens()


def build_stylesheet(theme: ThemeTokens = DEFAULT_THEME) -> str:
    return f"""
QWidget {{
    color: {theme.text};
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}}
QMainWindow, QWidget#appRoot {{ background: {theme.canvas}; }}
QFrame#sidebar {{ background: {theme.sidebar}; border: 0; }}
QLabel#brand {{ color: white; font-size: 20px; font-weight: 700; letter-spacing: 2px; }}
QLabel#brandCaption {{ color: {theme.sidebar_muted}; font-size: 11px; }}
QLabel#sectionEyebrow {{ color: {theme.primary}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel#pageTitle {{ color: {theme.text}; font-size: 25px; font-weight: 700; }}
QLabel#pageSubtitle, QLabel#muted {{ color: {theme.text_muted}; }}
QPushButton#navButton {{
    background: transparent; color: #BCD0DF; border: 0; border-radius: 9px;
    text-align: left; padding: 11px 14px; font-size: 14px;
}}
QPushButton#navButton:hover {{ background: #173A57; color: white; }}
QPushButton#navButton:checked {{ background: {theme.primary}; color: white; font-weight: 600; }}
QFrame#metricCard, QFrame#panelCard {{
    background: {theme.surface}; border: 1px solid {theme.border}; border-radius: 14px;
}}
QFrame#heroCard {{
    background: {theme.sidebar}; border: 1px solid #183C5A; border-radius: 16px;
}}
QLabel#metricValue {{ font-size: 25px; font-weight: 700; color: {theme.text}; }}
QLabel#cardTitle {{ font-size: 15px; font-weight: 700; color: {theme.text}; }}
QLabel#heroTitle {{ font-size: 19px; font-weight: 700; color: white; }}
QLabel#heroText {{ color: #C4D5E2; }}
QProgressBar {{
    background: #E7EDF2; border: 0; border-radius: 4px; height: 8px; text-align: center;
}}
QProgressBar::chunk {{ background: {theme.accent}; border-radius: 4px; }}
QPushButton#primaryButton {{
    background: {theme.primary}; color: white; border: 0; border-radius: 9px;
    padding: 9px 16px; font-weight: 600;
}}
QPushButton#primaryButton:hover {{ background: {theme.primary_dark}; }}
QPushButton#secondaryButton {{
    background: white; color: {theme.primary}; border: 1px solid {theme.primary};
    border-radius: 9px; padding: 8px 15px; font-weight: 600;
}}
QPushButton#dangerButton {{
    background: #FFF1F1; color: {theme.danger}; border: 1px solid #F0BABA;
    border-radius: 9px; padding: 8px 15px; font-weight: 600;
}}
QPushButton:disabled {{ background: #E9EEF2; color: #98A7B2; border-color: #D7E0E6; }}
QLineEdit, QPlainTextEdit, QTextBrowser, QComboBox, QSpinBox {{
    background: white; color: {theme.text}; border: 1px solid #CAD6DF;
    border-radius: 8px; padding: 7px; selection-background-color: #CFE7F8;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus, QComboBox:focus, QSpinBox:focus {{
    border: 1px solid {theme.primary};
}}
QCheckBox {{ spacing: 8px; color: #425D70; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid #AEBEC9; border-radius: 4px; background: white; }}
QCheckBox::indicator:checked {{ background: {theme.primary}; border-color: {theme.primary}; }}
QTableWidget {{
    background: white; alternate-background-color: #F7FAFC; border: 1px solid {theme.border};
    border-radius: 12px; gridline-color: #EDF1F4; selection-background-color: #DCEFFD;
}}
QHeaderView::section {{
    background: #EDF3F7; color: #486274; border: 0; border-bottom: 1px solid {theme.border};
    padding: 9px; font-weight: 600;
}}
QFrame#chatCard {{
    background: #F7FAFC; border: 1px solid {theme.border}; border-radius: 12px;
}}
QScrollArea#chatScrollArea, QWidget#chatTranscript, QWidget#chatRow {{
    background: transparent; border: 0;
}}
QFrame#chatCanvas {{ background: #F7FAFC; border: 0; }}
QFrame#chatBubbleUser {{ background: {theme.primary}; border: 1px solid {theme.primary_dark}; border-radius: 14px; }}
QFrame#chatBubbleAgent {{
    background: {theme.surface}; border: 1px solid {theme.border}; border-radius: 14px;
}}
QFrame#chatBubbleAssessment {{
    background: #E8F6F4; border: 1px solid #BFE5E0; border-radius: 14px;
}}
QLabel#chatRoleName {{ color: {theme.text_muted}; font-size: 11px; font-weight: 600; }}
QLabel#chatBubbleTextUser {{ color: #FFFFFF; }}
QLabel#chatBubbleTextAgent {{ color: {theme.text}; }}
QLabel#chatBubbleTextAssessment {{ color: #145E5E; }}
QLabel#chatAvatarUser, QLabel#chatAvatarAgent {{
    border-radius: 15px; color: white; font-size: 12px; font-weight: 700;
}}
QLabel#chatAvatarUser {{ background: {theme.sidebar}; }}
QLabel#chatAvatarAgent {{ background: {theme.accent}; }}
QLabel#chatSystemNote {{ color: {theme.text_muted}; font-size: 11px; }}
QLabel#chatPendingNote {{ color: {theme.primary}; font-size: 11px; font-weight: 600; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #B8C6D1; border-radius: 4px; min-height: 28px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""
