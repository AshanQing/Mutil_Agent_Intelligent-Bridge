from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class DrawingConfigError(ValueError):
    pass


class DrawingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["drawing-config-v1"] = "drawing-config-v1"
    units: Literal["mm"] = "mm"
    view_rows: list[list[str]] = Field(
        default_factory=lambda: [
            ["cap-elevation", "cap-section"],
            ["column-elevation", "column-section"],
        ]
    )
    column_gap_mm: float = Field(default=1500.0, gt=0)
    row_gap_mm: float = Field(default=1800.0, gt=0)
    preview_margin_mm: float = Field(default=500.0, ge=0)
    config_source: str = "builtin_default"


def load_drawing_config(path: str | Path | None = None) -> DrawingConfig:
    if path is None:
        return DrawingConfig()
    config_path = Path(path)
    if not config_path.exists():
        return DrawingConfig()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise DrawingConfigError("绘图配置根节点必须是 JSON 对象。")
        raw["config_source"] = str(config_path.resolve())
        return DrawingConfig.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise DrawingConfigError(f"绘图配置无效: {config_path}: {exc}") from exc
