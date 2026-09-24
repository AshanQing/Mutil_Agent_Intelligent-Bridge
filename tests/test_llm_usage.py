from __future__ import annotations

import json
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult

from bridge_agents.llm_usage import LLMUsageCallback
from bridge_agents.utils import build_llm


def test_usage_callback_writes_cache_metrics_without_prompt_content(tmp_path) -> None:
    callback = LLMUsageCallback(
        output_dir=tmp_path,
        role="revision",
        model="deepseek-v4-pro",
    )
    run_id = uuid4()
    callback.on_chat_model_start(
        {"name": "ChatOpenAI"},
        [[SystemMessage(content="STATIC_SYSTEM"), HumanMessage(content="PRIVATE_PROMPT")]],
        run_id=run_id,
    )
    callback.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={
                "model_name": "deepseek-v4-pro",
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_cache_hit_tokens": 60,
                    "prompt_cache_miss_tokens": 40,
                },
            },
        ),
        run_id=run_id,
    )

    log_path = tmp_path / "logs" / "llm_usage.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8").strip())

    assert record["role"] == "revision"
    assert record["model"] == "deepseek-v4-pro"
    assert record["prompt_tokens"] == 100
    assert record["prompt_cache_hit_tokens"] == 60
    assert record["prompt_cache_miss_tokens"] == 40
    assert record["cache_hit_rate"] == 0.6
    assert record["prompt_chars"] == len("system\nSTATIC_SYSTEM\nuser\nPRIVATE_PROMPT")
    assert len(record["prompt_sha256"]) == 64
    assert len(record["prefix_16k_sha256"]) == 64
    assert "PRIVATE_PROMPT" not in json.dumps(record, ensure_ascii=False)
    assert "prompt_text" not in record


def test_usage_callback_records_unavailable_cache_metrics_as_null(tmp_path) -> None:
    callback = LLMUsageCallback(
        output_dir=tmp_path,
        role="generation",
        model="deepseek-v4-pro",
    )
    run_id = uuid4()
    callback.on_llm_start(
        {"name": "ChatOpenAI"},
        ["same prompt"],
        run_id=run_id,
    )
    callback.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={"token_usage": {"prompt_tokens": 12}},
        ),
        run_id=run_id,
    )

    record = json.loads(
        (tmp_path / "logs" / "llm_usage.jsonl").read_text(encoding="utf-8").strip()
    )

    assert record["prompt_tokens"] == 12
    assert record["prompt_cache_hit_tokens"] is None
    assert record["prompt_cache_miss_tokens"] is None
    assert record["cache_hit_rate"] is None


def test_build_llm_registers_usage_callback_for_configured_output_dir(tmp_path) -> None:
    output_dir = tmp_path / "case-output"
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "\n".join([
            "llm:",
            "  revision:",
            "    api_key: test-key",
            "    base_url: https://api.deepseek.com",
            "    model: deepseek-v4-pro",
            "paths:",
            f"  output_dir: '{output_dir.as_posix()}'",
        ]),
        encoding="utf-8",
    )

    llm = build_llm(str(config_path), "revision")

    callbacks = list(llm.callbacks or [])
    usage_callbacks = [item for item in callbacks if isinstance(item, LLMUsageCallback)]
    assert len(usage_callbacks) == 1
    assert usage_callbacks[0].log_path == output_dir / "logs" / "llm_usage.jsonl"
