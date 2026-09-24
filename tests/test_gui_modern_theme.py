from bridge_agents.gui_modern_theme import DEFAULT_THEME, build_stylesheet


def test_modern_theme_uses_engineering_dashboard_palette() -> None:
    assert DEFAULT_THEME.primary == "#1769AA"
    assert DEFAULT_THEME.sidebar == "#102A43"
    assert DEFAULT_THEME.accent == "#0AA6A6"


def test_stylesheet_has_modern_cards_navigation_and_table_treatment() -> None:
    stylesheet = build_stylesheet()

    assert "QFrame#metricCard" in stylesheet
    assert "QPushButton#navButton:checked" in stylesheet
    assert "QTableWidget" in stylesheet
    assert "border-radius: 14px" in stylesheet


def test_stylesheet_covers_interactive_workbench_controls() -> None:
    stylesheet = build_stylesheet()

    assert "QLineEdit" in stylesheet
    assert "QTextBrowser" in stylesheet
    assert "QPushButton#dangerButton" in stylesheet
    assert "QCheckBox::indicator:checked" in stylesheet
