from __future__ import annotations

import numpy as np

from bridge_agents.code_rag.embedding import (
    BGE_QUERY_INSTRUCTION,
    SentenceTransformerBackend,
    resolve_cached_model_path,
)


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append((texts, kwargs))
        return np.ones((len(texts), 2), dtype=np.float32)


def test_bge_backend_adds_instruction_only_to_queries() -> None:
    model = RecordingModel()
    backend = SentenceTransformerBackend(
        model_name="BAAI/bge-small-zh-v1.5",
        model=model,
    )

    backend.encode_documents(["规范证据"])
    backend.encode_queries(["汽车荷载组合"])

    assert model.calls[0][0] == ["规范证据"]
    assert model.calls[1][0] == [f"{BGE_QUERY_INSTRUCTION}汽车荷载组合"]
    assert all(call[1]["normalize_embeddings"] is True for call in model.calls)
    assert all(call[1]["convert_to_numpy"] is True for call in model.calls)


def test_resolve_cached_model_path_uses_huggingface_snapshot(tmp_path) -> None:
    model_root = tmp_path / "models--BAAI--bge-small-zh-v1.5"
    (model_root / "refs").mkdir(parents=True)
    (model_root / "refs" / "main").write_text("revision-123\n", encoding="utf-8")
    snapshot = model_root / "snapshots" / "revision-123"
    snapshot.mkdir(parents=True)
    (snapshot / "modules.json").write_text("{}", encoding="utf-8")

    assert resolve_cached_model_path(
        "BAAI/bge-small-zh-v1.5",
        tmp_path,
    ) == snapshot
