from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from langgraph.checkpoint.sqlite import SqliteSaver


CHECKPOINT_RELATIVE_PATH = Path("checkpoints") / "graph_v2.sqlite"
# langgraph 在待提交写入里用这两个保留通道记录"本次恢复值"与"任务异常"。
STALE_RESUME_CHANNELS = ("__resume__", "__error__")
# 全局输入（含 Command(update=...) 补丁与 Command(resume=...) 取值）都挂在这个哨兵 task 上。
NULL_TASK_ID = "00000000-0000-0000-0000-000000000000"

# 连接按数据库路径复用并保持到进程结束：langgraph 会在节点线程里写 checkpoint
# （interrupt 的写入就发生在节点执行线程内），若 invoke 返回后立刻 close，
# 就可能与仍在进行的 put_writes 竞争，触发 Windows 访问违例（0xC0000005），
# 把真实结果和异常一起吞掉。SQLite 的提交本身是持久的（WAL + 崩溃恢复），
# 因此这里不做即时关闭，只在显式调用 close_all_checkpointers() 时统一收尾。
_CONNECTIONS: Dict[str, sqlite3.Connection] = {}
_CONNECTIONS_LOCK = threading.Lock()


def _shared_connection(checkpoint_path: Path) -> sqlite3.Connection:
    key = str(checkpoint_path.resolve())
    with _CONNECTIONS_LOCK:
        connection = _CONNECTIONS.get(key)
        if connection is None:
            connection = sqlite3.connect(key, check_same_thread=False)
            _CONNECTIONS[key] = connection
        return connection


def close_all_checkpointers() -> None:
    """关闭本进程打开的全部 checkpoint 连接（测试与显式收尾使用）。"""
    with _CONNECTIONS_LOCK:
        connections = list(_CONNECTIONS.values())
        _CONNECTIONS.clear()
    for connection in connections:
        try:
            connection.close()
        except Exception:  # pragma: no cover - 关闭失败不影响退出
            pass


@contextmanager
def open_sqlite_checkpointer(output_dir: str | Path) -> Iterator[Tuple[SqliteSaver, str]]:
    """打开 graph-v2 的本地持久化 checkpointer。

    连接按路径复用且不在退出时立即关闭，原因见 _CONNECTIONS 注释。
    """
    checkpoint_path = Path(output_dir) / CHECKPOINT_RELATIVE_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver(_shared_connection(checkpoint_path))
    checkpointer.setup()
    yield checkpointer, str(checkpoint_path)


def _connect(checkpoint_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def stale_resume_writes(output_dir: str | Path, thread_id: str) -> List[Dict[str, Any]]:
    """列出线程最新 checkpoint 上残留的待提交写入，需要清理的都在这里。

    两类残值都会让后续恢复失真：
    1. `__resume__` / `__error__`：上一次恢复在人工复核节点抛错时留下，之后
       `interrupt()` 会**优先重放这条任务级旧值**，新提交的决策被忽略
       （表现为"点了没反应"或反复抛同一个错）；
    2. 全局输入补丁（挂在哨兵 task 上的普通 channel 写入）：上次恢复中断在这次
       写入提交之前时，它们会残留并在下次恢复时二次写入同一 channel，
       触发 `InvalidUpdateError: Can receive only one value per step`。
    """
    checkpoint_path = Path(output_dir) / CHECKPOINT_RELATIVE_PATH
    if not checkpoint_path.is_file():
        return []
    connection = _connect(checkpoint_path)
    try:
        row = connection.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE thread_id=? AND checkpoint_ns='' "
            "ORDER BY rowid DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            return []
        checkpoint_id = row["checkpoint_id"]
        placeholders = ",".join("?" for _ in STALE_RESUME_CHANNELS)
        rows = connection.execute(
            f"SELECT rowid, task_id, channel, type, value FROM writes "
            f"WHERE thread_id=? AND checkpoint_id=? AND ("
            f"channel IN ({placeholders}) OR (task_id=? AND channel NOT LIKE '\\_\\_%')) "
            "ORDER BY rowid",
            (thread_id, checkpoint_id, *STALE_RESUME_CHANNELS, NULL_TASK_ID),
        ).fetchall()
    finally:
        connection.close()

    result: List[Dict[str, Any]] = []
    for item in rows:
        value: Any = item["value"]
        if isinstance(value, (bytes, bytearray)):
            value = _decode_checkpoint_value(bytes(value))
        result.append(
            {
                "rowid": item["rowid"],
                "task_id": item["task_id"],
                "channel": item["channel"],
                "value": value,
            }
        )
    return result


def _decode_checkpoint_value(raw: bytes) -> Any:
    """尽量把 checkpoint 里的二进制写入解成可读值（json → msgpack → repr）。"""
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        pass
    try:
        import msgpack  # langgraph-checkpoint 的序列化依赖

        return msgpack.unpackb(raw, raw=False)
    except Exception:
        return repr(raw[:160])


def clear_stale_resume_writes(output_dir: str | Path, thread_id: str) -> List[Dict[str, Any]]:
    """清除线程最新 checkpoint 上的残留待提交写入（恢复残值与全局输入补丁）。"""
    checkpoint_path = Path(output_dir) / CHECKPOINT_RELATIVE_PATH
    if not checkpoint_path.is_file():
        return []
    stale = stale_resume_writes(output_dir, thread_id)
    if not stale:
        return []
    connection = _connect(checkpoint_path)
    try:
        placeholders = ",".join("?" for _ in STALE_RESUME_CHANNELS)
        connection.execute(
            f"DELETE FROM writes WHERE thread_id=? AND checkpoint_id=("
            "SELECT checkpoint_id FROM checkpoints WHERE thread_id=? AND checkpoint_ns='' "
            "ORDER BY rowid DESC LIMIT 1"
            f") AND (channel IN ({placeholders}) OR (task_id=? AND channel NOT LIKE '\\_\\_%'))",
            (thread_id, thread_id, *STALE_RESUME_CHANNELS, NULL_TASK_ID),
        )
        connection.commit()
    finally:
        connection.close()
    return stale
