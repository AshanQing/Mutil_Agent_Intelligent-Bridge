from __future__ import annotations

import threading
import time

from bridge_agents.batch_execution import run_indexed_batch


def test_run_indexed_batch_executes_tasks_concurrently() -> None:
    barrier = threading.Barrier(2, timeout=2.0)

    def worker(item: int) -> int:
        barrier.wait()
        return item * 2

    results = run_indexed_batch([1, 2], worker, max_workers=2)

    assert [result.value for result in results] == [2, 4]
    assert all(result.error is None for result in results)


def test_run_indexed_batch_preserves_input_order() -> None:
    delays = {"first": 0.03, "second": 0.02, "third": 0.01}

    def worker(item: str) -> str:
        time.sleep(delays[item])
        return item.upper()

    results = run_indexed_batch(
        ["first", "second", "third"], worker, max_workers=3
    )

    assert [result.index for result in results] == [0, 1, 2]
    assert [result.value for result in results] == ["FIRST", "SECOND", "THIRD"]


def test_run_indexed_batch_isolates_worker_errors() -> None:
    def worker(item: int) -> int:
        if item == 2:
            raise ValueError("invalid item")
        return item * 10

    results = run_indexed_batch([1, 2, 3], worker, max_workers=3)

    assert [result.value for result in results] == [10, None, 30]
    assert results[1].error == "ValueError: invalid item"
    assert results[0].error is None
    assert results[2].error is None


def test_run_indexed_batch_handles_empty_input() -> None:
    assert run_indexed_batch([], lambda item: item, max_workers=4) == []
