from __future__ import annotations

from typing import Any


def build_graph(*args: Any, **kwargs: Any) -> Any:
    from .agent import build_graph as _build_graph

    return _build_graph(*args, **kwargs)


def run_agent(*args: Any, **kwargs: Any) -> Any:
    from .agent import run_agent as _run_agent

    return _run_agent(*args, **kwargs)

__all__ = ["build_graph", "run_agent"]
