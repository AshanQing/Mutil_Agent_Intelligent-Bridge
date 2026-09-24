from __future__ import annotations

import json

import pytest

from bridge_agents.llm_generation_guard import (
    StructuredGenerationError,
    generate_structured_output,
)


def _parse(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("top level must be object")
    return value


def test_format_repair_can_recover_invalid_output() -> None:
    prompts: list[str] = []
    responses = iter(["not-json", '{"ok": true}'])

    result = generate_structured_output(
        invoke=lambda prompt: prompts.append(prompt) or next(responses),
        parse=_parse,
        initial_prompt="design",
        repair_prompt=lambda raw, error: f"repair: {raw}: {error}",
        regeneration_prompt=lambda error: f"regenerate: {error}",
        max_format_repairs=1,
        max_regenerations=0,
    )

    assert result.value == {"ok": True}
    assert [item.kind for item in result.attempts] == ["initial", "format_repair"]
    assert prompts[1].startswith("repair:")


def test_regeneration_runs_after_format_repair_is_exhausted() -> None:
    responses = iter(["bad", "still bad", '{"unit": "1-3"}'])

    result = generate_structured_output(
        invoke=lambda _prompt: next(responses),
        parse=_parse,
        initial_prompt="design",
        repair_prompt=lambda raw, error: "repair",
        regeneration_prompt=lambda error: "regenerate",
        max_format_repairs=1,
        max_regenerations=1,
    )

    assert result.value == {"unit": "1-3"}
    assert [item.kind for item in result.attempts] == [
        "initial",
        "format_repair",
        "regeneration",
    ]


def test_exhaustion_exposes_attempt_history() -> None:
    with pytest.raises(StructuredGenerationError) as exc_info:
        generate_structured_output(
            invoke=lambda _prompt: "bad",
            parse=_parse,
            initial_prompt="design",
            repair_prompt=lambda raw, error: "repair",
            regeneration_prompt=lambda error: "regenerate",
            max_format_repairs=1,
            max_regenerations=1,
        )

    error = exc_info.value
    assert error.attempt_count == 3
    assert [item.kind for item in error.attempts] == [
        "initial",
        "format_repair",
        "regeneration",
    ]
    assert all(item.error for item in error.attempts)
