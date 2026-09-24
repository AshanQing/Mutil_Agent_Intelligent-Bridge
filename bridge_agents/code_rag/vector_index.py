from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from .paths import DEFAULT_EVIDENCE_DOCUMENTS_PATH, DEFAULT_INDEX_DIR


VECTOR_SCHEMA_VERSION = "code_rag_vector_v0.1"


class VectorIndexError(RuntimeError):
    pass


class EmbeddingBackend(Protocol):
    model_name: str

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


def build_vector_index(
    *,
    evidence_path: Path = DEFAULT_EVIDENCE_DOCUMENTS_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    backend: EmbeddingBackend,
) -> dict[str, Any]:
    evidence_path = Path(evidence_path)
    index_dir = Path(index_dir)
    documents = list(_read_jsonl(evidence_path))
    if not documents:
        raise VectorIndexError("Cannot build a vector index without evidence documents.")

    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise VectorIndexError(f"Keyword index manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_hash = _sha256_file(evidence_path)
    if manifest.get("evidence_file_sha256") != evidence_hash:
        raise VectorIndexError("Keyword index and evidence hash do not match.")

    evidence_ids = [str(document["evidence_id"]) for document in documents]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise VectorIndexError("Evidence IDs must be unique before embedding.")
    texts = [str(document.get("search_text") or "") for document in documents]
    vectors = _normalize_rows(backend.encode_documents(texts))
    if vectors.shape[0] != len(documents):
        raise VectorIndexError("Embedding count does not match evidence count.")

    index_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = index_dir / "embeddings.npy"
    ids_path = index_dir / "embedding_ids.json"
    np.save(embedding_path, vectors, allow_pickle=False)
    ids_payload = {
        "schema_version": VECTOR_SCHEMA_VERSION,
        "embedding_model": backend.model_name,
        "embedding_dimension": int(vectors.shape[1]),
        "normalized": True,
        "evidence_file_sha256": evidence_hash,
        "evidence_ids": evidence_ids,
    }
    ids_path.write_text(
        json.dumps(ids_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest.update(
        {
            "embedding_model": backend.model_name,
            "embedding_dimension": int(vectors.shape[1]),
            "embedding_count": len(evidence_ids),
            "embedding_normalized": True,
            "embedding_sha256": _sha256_file(embedding_path),
            "embedding_ids_sha256": _sha256_file(ids_path),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "embedding_path": str(embedding_path),
        "embedding_ids_path": str(ids_path),
        "embedding_model": backend.model_name,
        "embedding_dimension": int(vectors.shape[1]),
        "embedding_count": len(evidence_ids),
        "embedding_sha256": manifest["embedding_sha256"],
    }


def query_vector_index(
    *,
    query_text: str,
    evidence_path: Path = DEFAULT_EVIDENCE_DOCUMENTS_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    backend: EmbeddingBackend,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    index_dir = Path(index_dir)
    ids_payload = json.loads(
        (index_dir / "embedding_ids.json").read_text(encoding="utf-8")
    )
    if ids_payload.get("evidence_file_sha256") != _sha256_file(Path(evidence_path)):
        raise VectorIndexError("Vector index evidence hash does not match the current evidence hash.")
    if ids_payload.get("embedding_model") != backend.model_name:
        raise VectorIndexError("Embedding backend does not match the vector index model.")

    vectors = np.load(index_dir / "embeddings.npy", allow_pickle=False)
    evidence_ids = list(ids_payload.get("evidence_ids") or [])
    if vectors.ndim != 2 or vectors.shape[0] != len(evidence_ids):
        raise VectorIndexError("Embedding matrix and evidence IDs are inconsistent.")
    query_vector = _normalize_rows(backend.encode_queries([query_text]))[0]
    scores = vectors @ query_vector
    ordered = sorted(range(len(evidence_ids)), key=lambda index: (-scores[index], evidence_ids[index]))
    return [
        {
            "evidence_id": evidence_ids[index],
            "score": float(scores[index]),
            "rank": rank,
        }
        for rank, index in enumerate(ordered[: max(0, int(top_k))], start=1)
    ]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise VectorIndexError("Embedding backend must return a two-dimensional matrix.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise VectorIndexError("Embedding backend returned a zero vector.")
    return vectors / norms


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
