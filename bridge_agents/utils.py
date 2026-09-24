from __future__ import annotations

import json
import logging
import os
import re
import inspect
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .llm_usage import LLMUsageCallback

load_dotenv()
logger = logging.getLogger(__name__)
_CONTROLLER_LLM: Optional[ChatOpenAI] = None
_REVISION_LLM: Optional[ChatOpenAI] = None
_STRUCTURAL_LLM: Optional[ChatOpenAI] = None


def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pick(settings: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """从 settings 中宽松读取配置，兼容 flat / nested 两种写法。"""
    for key in keys:
        if key in settings and settings[key] not in [None, ""]:
            return settings[key]

    sections = [
        "paths", "data", "layout", "model", "mask", "collision", "prompts",
        "llm", "task", "route_data", "retrieval", "segmentation",
        "drawing_crop_and_mask", "obstacle_semantic_extractor",
        "collision_detection", "agent", "structure", "modeling",
    ]
    for section in sections:
        value = settings.get(section)
        if isinstance(value, dict):
            for key in keys:
                if key in value and value[key] not in [None, ""]:
                    return value[key]
    return default


def resolve_api_key(api_config: Dict[str, Any]) -> str:
    api_key = api_config.get("api_key")
    if api_key == "ENV":
        env_name = api_config.get("api_key_env", "DEEPSEEK_API_KEY")
        api_key = os.getenv(env_name)
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未找到 LLM API Key，请检查 settings.yaml 或环境变量。")
    return api_key


def build_llm(config_path: str, role: str = "controller") -> ChatOpenAI:
    settings = load_settings(config_path)
    llm_settings = settings.get("llm", {})
    role_config = llm_settings.get(role)
    if not isinstance(role_config, dict):
        role_config = llm_settings.get("controller", llm_settings)

    model = role_config.get("model", "deepseek-chat")
    output_dir = pick(settings, "output_dir", default="output")
    usage_callback = LLMUsageCallback(output_dir=output_dir, role=role, model=model)

    return ChatOpenAI(
        model=model,
        openai_api_key=resolve_api_key(role_config),
        openai_api_base=role_config.get("base_url", "https://api.deepseek.com"),
        temperature=role_config.get("temperature", 0),
        callbacks=[usage_callback],
    )


def get_controller_llm(config_path: str = "config/settings.yaml") -> ChatOpenAI:
    global _CONTROLLER_LLM
    if _CONTROLLER_LLM is None:
        _CONTROLLER_LLM = build_llm(config_path, "controller")
    return _CONTROLLER_LLM


def get_revision_llm(config_path: str = "config/settings.yaml") -> ChatOpenAI:
    global _REVISION_LLM
    if _REVISION_LLM is None:
        _REVISION_LLM = build_llm(config_path, "revision")
    return _REVISION_LLM


def get_structural_llm(config_path: str = "config/settings.yaml") -> ChatOpenAI:
    """结构设计阶段 LLM。

    settings.yaml 中可以配置 llm.structural；若未配置，则自动回退到 llm.controller。
    """
    global _STRUCTURAL_LLM
    if _STRUCTURAL_LLM is None:
        _STRUCTURAL_LLM = build_llm(config_path, "structural")
    return _STRUCTURAL_LLM


def json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except TypeError:
        pass

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return str(obj)


def state_json(state: Dict[str, Any]) -> str:
    return json.dumps(json_safe(state), ensure_ascii=False)


def _strip_raw_control_chars(text: str) -> str:
    """把 JSON 文本中裸露的控制字符替换为空格（保留换行/回车/制表符）。

    LLM 常在字符串值里直接输出换行、\x0b 等裸控制字符，JSON 规范不允许，
    会导致 json.loads 抛 "Invalid control character"。这里做兜底清洗。
    """
    return "".join(
        ch if (ch >= " " or ch in "\n\r\t") else " " for ch in text
    )


def _loads_lenient(text: str) -> Any:
    """宽松 JSON 解析：先严格，再 strict=False，最后清洗控制字符后重试。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        # strict=False 允许字符串内部出现未转义的控制字符（LLM 输出常见）
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    return json.loads(_strip_raw_control_chars(text), strict=False)


def extract_json_object(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        data = _loads_lenient(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        data = _loads_lenient(content[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("LLM 输出 JSON 顶层必须是对象。")
    return data


def tool_failed(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, dict):
        return bool(result.get("success") is False or result.get("error"))
    return False


def invoke_tool(tool: Any, payload: Dict[str, Any]) -> Any:
    """Invoke LangChain tools or plain Python callables with explicit payload checks."""
    if tool is None:
        raise ImportError("工具未成功导入。")
    # Filter None values to avoid Pydantic validation errors with LangChain @tool wrappers.
    payload = {k: v for k, v in payload.items() if v is not None}
    if hasattr(tool, "invoke"):
        return tool.invoke(payload)
    if callable(tool):
        try:
            signature = inspect.signature(tool)
        except (TypeError, ValueError):
            return tool(**payload)

        parameters = list(signature.parameters.values())
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
        keyword_names = {
            p.name
            for p in parameters
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        payload_keys = set(payload)

        if accepts_kwargs or payload_keys.issubset(keyword_names):
            return tool(**payload)

        positional = [
            p
            for p in parameters
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) == 1 and not keyword_names.intersection(payload_keys):
            return tool(payload)

        unexpected = sorted(payload_keys - keyword_names)
        raise TypeError(
            f"Tool {getattr(tool, '__name__', type(tool).__name__)} received unexpected payload keys: {unexpected}"
        )
    raise TypeError(f"不支持的工具类型: {type(tool)}")


def read_text_file(path: Optional[str], default: str = "") -> str:
    if not path:
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


def _normalize_station_text(station: str) -> str:
    """把提取到的桩号文本规范成 K{km}+{m}（如 K1_000 -> K41+000）。"""
    text = station.strip().upper().replace(" ", "")
    match = re.match(r"^([A-Z]*K?[-+]?\d+)[_+]+(\d+(?:\.\d+)?)$", text)
    if match:
        km, m = match.groups()
        return f"{km}+{m}"
    # 已是 K41+000 / ZK12+200 等形式时原样返回。
    return text


def regex_extract_stations(text: str) -> Dict[str, Optional[str]]:
    """从用户请求中提取起终点桩号，兼容 K41+000 与 K1_000/K41 000 等写法。

    提取后统一规范成 K{km}+{m} 标准格式（如 K1_000 -> K41+000），供
    load_data 等下游按规范桩号解析。
    """
    pattern = r"[A-Z]*K?[-+]?\d+\s*[_+]\s*\d+(?:\.\d+)?"
    stations = re.findall(pattern, text or "", flags=re.IGNORECASE)
    cleaned = [_normalize_station_text(s) for s in stations]
    return {
        "start_station": cleaned[0] if len(cleaned) >= 1 else None,
        "end_station": cleaned[1] if len(cleaned) >= 2 else None,
    }


def metric_float(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = metrics.get(key, default)
        if value in [None, ""]:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
