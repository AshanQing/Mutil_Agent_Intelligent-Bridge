from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from langchain_core.callbacks import BaseCallbackHandler


def _message_role(message: Any) -> str:
    role = str(getattr(message, "type", None) or getattr(message, "role", None) or "message")
    return {"human": "user", "ai": "assistant"}.get(role, role)


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)


def _serialize_chat_batches(message_batches: Iterable[Iterable[Any]]) -> str:
    batches = []
    for messages in message_batches:
        parts = []
        for message in messages:
            parts.extend([_message_role(message), _message_content(message)])
        batches.append("\n".join(parts))
    return "\n\n--batch--\n\n".join(batches)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_token_usage(response: Any) -> Dict[str, Any]:
    llm_output = getattr(response, "llm_output", None)
    if not isinstance(llm_output, dict):
        llm_output = {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}

    generations = getattr(response, "generations", None) or []
    if generations and generations[0]:
        message = getattr(generations[0][0], "message", None)
        response_metadata = getattr(message, "response_metadata", None)
        if isinstance(response_metadata, dict):
            nested_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
            if isinstance(nested_usage, dict):
                usage = {**nested_usage, **usage}
        usage_metadata = getattr(message, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            usage = {**usage_metadata, **usage}
    return usage


class LLMUsageCallback(BaseCallbackHandler):
    """将 LLM token/cache 统计写入 JSONL，不保存 Prompt 或模型输出正文。"""

    def __init__(self, output_dir: str | Path, role: str, model: str) -> None:
        self.log_path = Path(output_dir) / "logs" / "llm_usage.jsonl"
        self.role = role
        self.model = model
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _remember_prompt(self, run_id: Any, prompt_text: str) -> None:
        prompt_bytes = prompt_text.encode("utf-8")
        prefix_bytes = prompt_text[:16384].encode("utf-8")
        self._pending[str(run_id)] = {
            "prompt_chars": len(prompt_text),
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "prefix_16k_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
        }

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._remember_prompt(run_id, _serialize_chat_batches(messages))

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._remember_prompt(run_id, "\n\n--batch--\n\n".join(prompts))

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        usage = _extract_token_usage(response)
        hit_tokens = _optional_int(usage.get("prompt_cache_hit_tokens"))
        miss_tokens = _optional_int(usage.get("prompt_cache_miss_tokens"))
        cache_total = None
        if hit_tokens is not None and miss_tokens is not None:
            cache_total = hit_tokens + miss_tokens
        hit_rate = (hit_tokens / cache_total) if cache_total else None

        llm_output = getattr(response, "llm_output", None)
        model_name = llm_output.get("model_name") if isinstance(llm_output, dict) else None
        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "run_id": str(run_id),
            "role": self.role,
            "model": str(model_name or self.model),
            **self._pending.pop(str(run_id), {}),
            "prompt_tokens": _optional_int(usage.get("prompt_tokens") or usage.get("input_tokens")),
            "completion_tokens": _optional_int(usage.get("completion_tokens") or usage.get("output_tokens")),
            "total_tokens": _optional_int(usage.get("total_tokens")),
            "prompt_cache_hit_tokens": hit_tokens,
            "prompt_cache_miss_tokens": miss_tokens,
            "cache_hit_rate": round(hit_rate, 6) if hit_rate is not None else None,
        }

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
