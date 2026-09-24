from __future__ import annotations

import json
from typing import Any, List


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def append_unique(left: Any, right: Any) -> List[Any]:
    """Append LangGraph state values while dropping exact duplicate records."""
    merged = _as_list(left) + _as_list(right)
    result: List[Any] = []
    seen = set()

    for item in merged:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)

    return result
