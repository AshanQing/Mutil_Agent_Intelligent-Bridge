from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import yaml

from bridge_agents.gui_controller import GuiRunConfig


STATE_VERSION = 1
PERSISTED_FIELDS = (
    "base_config_path",
    "user_request",
    "data_path",
    "drawing_path",
    "output_dir",
    "file_prefix",
    "max_revision_rounds",
    "max_check_revision_rounds",
    "code_rag_enabled",
)


def new_thread_id() -> str:
    return f"bridge-{datetime.now():%Y%m%d-%H%M}-{uuid4().hex[:5]}"


@dataclass
class ModernGuiForm:
    base_config_path: str = "config/settings.yaml"
    user_request: str = ""
    data_path: str = ""
    drawing_path: str = ""
    output_dir: str = ""
    file_prefix: str = ""
    thread_id: str = field(default_factory=new_thread_id)
    max_revision_rounds: int = 3
    max_check_revision_rounds: int = 2
    code_rag_enabled: bool = False
    external_authorized: bool = False

    def to_run_config(self) -> GuiRunConfig:
        return GuiRunConfig(
            user_request=self.user_request,
            data_path=self.data_path,
            input_drawing_path=self.drawing_path,
            output_dir=self.output_dir,
            file_prefix=self.file_prefix,
            thread_id=self.thread_id,
            max_revision_rounds=int(self.max_revision_rounds),
            max_check_revision_rounds=int(self.max_check_revision_rounds),
            code_rag_enabled=bool(self.code_rag_enabled),
        )


def default_state_path() -> Path:
    return Path(__file__).resolve().parents[1] / "output" / ".graph_v2_gui_modern_state.json"


def save_form_state(form: ModernGuiForm, path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else default_state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    values = asdict(form)
    payload = {
        "version": STATE_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "values": {key: values[key] for key in PERSISTED_FIELDS},
    }
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def load_form_state(path: str | Path | None = None) -> ModernGuiForm:
    source = Path(path) if path is not None else default_state_path()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return ModernGuiForm()
    if not isinstance(payload, Mapping) or payload.get("version") != STATE_VERSION:
        return ModernGuiForm()
    values = payload.get("values")
    if not isinstance(values, Mapping):
        return ModernGuiForm()
    safe_values = {key: values[key] for key in PERSISTED_FIELDS if key in values}
    try:
        safe_values["max_revision_rounds"] = max(1, int(safe_values.get("max_revision_rounds", 3)))
        safe_values["max_check_revision_rounds"] = max(1, int(safe_values.get("max_check_revision_rounds", 2)))
        return ModernGuiForm(**safe_values)
    except (TypeError, ValueError):
        return ModernGuiForm()


def _pick(payload: Mapping[str, Any], section: str, key: str, default: Any = "") -> Any:
    section_value = payload.get(section)
    if isinstance(section_value, Mapping) and section_value.get(key) not in (None, ""):
        return section_value[key]
    return default


def load_form_from_yaml(path: str | Path) -> ModernGuiForm:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"基础配置文件不存在：{source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("基础配置必须是 YAML 对象。")
    return ModernGuiForm(
        base_config_path=str(source),
        user_request=str(_pick(payload, "task", "user_request")),
        data_path=str(_pick(payload, "paths", "data_path")),
        drawing_path=str(_pick(payload, "paths", "input_drawing_path")),
        output_dir=str(_pick(payload, "paths", "output_dir")),
        file_prefix=str(_pick(payload, "route_data", "file_prefix")),
        max_revision_rounds=int(_pick(payload, "agent", "max_revision_rounds", 3)),
        max_check_revision_rounds=int(_pick(payload, "modeling", "max_check_revision_rounds", 2)),
        code_rag_enabled=bool(_pick(payload, "code_rag", "enabled", False)),
    )


def find_latest_snapshot(output_dir: str | Path, thread_id: str = "") -> Path | None:
    root = Path(output_dir) / "run_configs"
    if not root.is_dir():
        return None
    if thread_id:
        exact = root / f"{thread_id}.yaml"
        if exact.is_file():
            return exact
    candidates = [path for path in root.glob("*.yaml") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def route_range_from_request(user_request: str) -> str:
    """从任务描述里取起终点桩号，拼成顶栏要显示的一行；取不到返回空串。

    刻意复用运行时解析桩号用的同一个函数（`agent.py` 启动时也是拿
    `regex_extract_stations` 从任务描述里取范围），这样界面显示的就是本次真正
    要设计的范围，而不是另算一套口径。缺失时返回空串，由展示层写成占位文案。
    """
    from bridge_agents.utils import regex_extract_stations

    stations = regex_extract_stations(user_request)
    start = str(stations.get("start_station") or "").strip()
    end = str(stations.get("end_station") or "").strip()
    if not start or not end:
        return ""
    return f"{start} — {end}"
