from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bridge_agents.code_rag.vector_index import (
    VectorIndexError,
    build_vector_index,
    query_vector_index,
)


class FakeEmbeddingBackend:
    model_name = "fake-embedding-v1"

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._encode(text) for text in texts], dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._encode(text) for text in texts], dtype=np.float32)

    @staticmethod
    def _encode(text: str) -> list[float]:
        return [
            4.0 if "clearance" in text else 0.0,
            3.0 if "vehicle" in text or "automobile" in text else 0.0,
            2.0 if "reinforcement" in text else 0.0,
        ]


def test_build_and_query_vector_index(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence_documents.jsonl"
    documents = [
        _document("EV_CLEARANCE", "bridge clearance requirement"),
        _document("EV_VEHICLE", "vehicle load combination"),
        _document("EV_REBAR", "reinforcement anchorage"),
    ]
    _write_jsonl(evidence_path, documents)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    _write_index_manifest(index_dir, evidence_path)

    result = build_vector_index(
        evidence_path=evidence_path,
        index_dir=index_dir,
        backend=FakeEmbeddingBackend(),
    )

    vectors = np.load(index_dir / "embeddings.npy")
    ids_payload = json.loads((index_dir / "embedding_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
    assert vectors.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert ids_payload["evidence_ids"] == ["EV_CLEARANCE", "EV_VEHICLE", "EV_REBAR"]
    assert manifest["embedding_model"] == "fake-embedding-v1"
    assert manifest["embedding_dimension"] == 3
    assert manifest["embedding_count"] == 3
    assert result["embedding_sha256"] == manifest["embedding_sha256"]

    rows = query_vector_index(
        query_text="automobile load",
        evidence_path=evidence_path,
        index_dir=index_dir,
        backend=FakeEmbeddingBackend(),
        top_k=2,
    )

    assert rows[0]["evidence_id"] == "EV_VEHICLE"
    assert rows[0]["rank"] == 1
    assert rows[0]["score"] == pytest.approx(1.0)


def test_vector_index_rejects_stale_evidence_file(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence_documents.jsonl"
    _write_jsonl(evidence_path, [_document("EV_ONE", "vehicle load")])
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    _write_index_manifest(index_dir, evidence_path)
    backend = FakeEmbeddingBackend()
    build_vector_index(evidence_path=evidence_path, index_dir=index_dir, backend=backend)
    evidence_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(VectorIndexError, match="evidence hash"):
        query_vector_index(
            query_text="vehicle",
            evidence_path=evidence_path,
            index_dir=index_dir,
            backend=backend,
        )


def _document(evidence_id: str, search_text: str) -> dict[str, str]:
    return {"evidence_id": evidence_id, "search_text": search_text}


def _write_jsonl(path: Path, documents: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(document) + "\n" for document in documents),
        encoding="utf-8",
    )


def _write_index_manifest(index_dir: Path, evidence_path: Path) -> None:
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    (index_dir / "index_manifest.json").write_text(
        json.dumps({"evidence_file_sha256": digest}),
        encoding="utf-8",
    )
