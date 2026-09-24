from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class ChatSession:
    messages: list[dict[str, str]] = field(default_factory=list)
    context_limit: int = 24

    def append(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported chat role: {role}")
        self.messages.append({"role": role, "content": str(content)})

    def context(self) -> list[dict[str, str]]:
        return [dict(item) for item in self.messages[-self.context_limit :]]

    def clear(self) -> None:
        self.messages.clear()


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values or ():
        text = str(value)
        if text and text not in result:
            result.append(text)
    return result


def format_answer(payload: Mapping[str, Any]) -> str:
    content = str(payload.get("answer") or "问答服务未返回正文。")
    citations = _unique(payload.get("cited_artifact_ids")) + _unique(payload.get("cited_code_evidence_ids"))
    citations = _unique(citations)
    if citations:
        content += "\n\n引用：" + "、".join(citations[:10])
    return content
