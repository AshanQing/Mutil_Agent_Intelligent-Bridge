"""Stable bounded execution helpers for independent batch tasks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar


ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class BatchItemResult(Generic[ResultT]):
    """Result of one indexed task without propagating worker exceptions."""

    index: int
    value: ResultT | None = None
    error: str | None = None


def run_indexed_batch(
    items: Iterable[ItemT],
    worker: Callable[[ItemT], ResultT],
    *,
    max_workers: int,
) -> list[BatchItemResult[ResultT]]:
    """Run independent tasks with bounded concurrency and stable output order."""

    indexed_items = list(enumerate(items))
    if not indexed_items:
        return []

    worker_count = max(1, min(int(max_workers), len(indexed_items)))
    collected: dict[int, BatchItemResult[ResultT]] = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(worker, item): index for index, item in indexed_items
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                collected[index] = BatchItemResult(index=index, value=future.result())
            except Exception as exc:  # noqa: BLE001 - isolation is this helper's contract.
                collected[index] = BatchItemResult(
                    index=index,
                    error=f"{type(exc).__name__}: {exc}",
                )

    return [collected[index] for index, _ in indexed_items]
