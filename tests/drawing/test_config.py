from __future__ import annotations

import json

import pytest

from bridge_agents.drawing.config import DrawingConfigError, load_drawing_config


def test_missing_config_uses_versioned_builtin_default(tmp_path) -> None:
    config = load_drawing_config(tmp_path / "missing.json")
    assert config.schema_version == "drawing-config-v1"
    assert config.config_source == "builtin_default"
    assert config.units == "mm"


def test_invalid_existing_config_fails_without_default_fallback(tmp_path) -> None:
    path = tmp_path / "drawing.json"
    path.write_text(json.dumps({"units": "m"}), encoding="utf-8")
    with pytest.raises(DrawingConfigError):
        load_drawing_config(path)
