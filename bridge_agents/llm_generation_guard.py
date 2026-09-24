"""Bounded recovery for LLM outputs that must satisfy a structured parser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class GenerationAttempt:
    kind: str
    index: int
    raw_response: str
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "index": self.index,
            "raw_response": self.raw_response,
            "error": self.error,
        }


@dataclass(frozen=True)
class StructuredGenerationResult(Generic[ValueT]):
    value: ValueT
    attempts: tuple[GenerationAttempt, ...]

    @property
    def raw_responses(self) -> tuple[str, ...]:
        return tuple(item.raw_response for item in self.attempts)


class StructuredGenerationError(ValueError):
    def __init__(self, attempts: list[GenerationAttempt]):
        self.attempts = tuple(attempts)
        self.attempt_count = len(attempts)
        last_error = attempts[-1].error if attempts else "未调用 LLM"
        super().__init__(
            f"结构化输出经过 {self.attempt_count} 次尝试后仍无法解析: {last_error}"
        )


def generate_structured_output(
    *,
    invoke: Callable[[str], str],
    parse: Callable[[str], ValueT],
    initial_prompt: str,
    repair_prompt: Callable[[str, str], str],
    regeneration_prompt: Callable[[str], str],
    max_format_repairs: int = 1,
    max_regenerations: int = 1,
) -> StructuredGenerationResult[ValueT]:
    """Parse an LLM response with bounded repair and full-regeneration attempts."""
    format_repairs = max(0, int(max_format_repairs))
    regenerations = max(0, int(max_regenerations))
    attempts: list[GenerationAttempt] = []
    prompt = initial_prompt
    kinds = ["initial"] + ["format_repair"] * format_repairs + [
        "regeneration"
    ] * regenerations

    for index, kind in enumerate(kinds, start=1):
        raw = str(invoke(prompt) or "")
        try:
            value = parse(raw)
        except Exception as exc:  # noqa: BLE001 - parser failures drive recovery.
            error = f"{type(exc).__name__}: {exc}"
            attempts.append(GenerationAttempt(kind, index, raw, error))
            if index == len(kinds):
                break
            next_kind = kinds[index]
            prompt = (
                repair_prompt(raw, error)
                if next_kind == "format_repair"
                else regeneration_prompt(error)
            )
            continue
        attempts.append(GenerationAttempt(kind, index, raw, None))
        return StructuredGenerationResult(value=value, attempts=tuple(attempts))

    raise StructuredGenerationError(attempts)
