from __future__ import annotations

import numpy as np

from bridge_agents.code_rag.build_index import build_keyword_index
from bridge_agents.code_rag.dataset import build_chunk_dataset
from bridge_agents.code_rag.query import query_keyword_index
from bridge_agents.code_rag.vector_index import build_vector_index


class ConstantEmbeddingBackend:
    model_name = "constant-fake-v1"

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


def test_build_keyword_index_and_query(tmp_path) -> None:
    build_chunk_dataset(artifact_root=tmp_path)
    result = build_keyword_index(
        evidence_path=tmp_path / "evidence" / "evidence_documents.jsonl",
        index_dir=tmp_path / "index",
    )

    assert 0 < result["evidence_document_count"] < 1908
    assert result["relation_count"] > 0
    assert result["source_file_count"] == 12
    rows = query_keyword_index(
        db_path=tmp_path / "index" / "code_knowledge.db",
        text="盖梁 斜截面 抗剪 承载力",
        standards=["JTG 3362-2018"],
        top_k=5,
    )

    assert rows
    assert all("raw_payload" not in row for row in rows)
    assert all(row["prompt_text"] for row in rows)
    assert all("manual_review" not in row["prompt_text"] for row in rows)

    combination_rows = query_keyword_index(
        db_path=tmp_path / "index" / "code_knowledge.db",
        text="汽车荷载作用组合",
        standards=["JTG D60-2015"],
        top_k=3,
    )
    assert "F_D60_L3_ULS_BASIC_COMBINATION_DESIGN" in {
        row["primary_entity_id"] for row in combination_rows
    }

    span_rows = query_keyword_index(
        db_path=tmp_path / "index" / "code_knowledge.db",
        text="标准化跨径",
        standards=["JTG 3362-2018"],
        top_k=1,
    )
    assert span_rows[0]["primary_entity_id"] == "R_3362_CH04_003"

    rule_rows = query_keyword_index(
        db_path=tmp_path / "index" / "code_knowledge.db",
        text="承载能力极限状态作用基本组合",
        standards=["JTG D60-2015"],
        library_types=["rule"],
        top_k=10,
    )
    assert "R_D60_CH04_005" in {row["primary_entity_id"] for row in rule_rows}


def test_rebuilding_keyword_index_removes_stale_vector_files(tmp_path) -> None:
    build_chunk_dataset(artifact_root=tmp_path)
    evidence_path = tmp_path / "evidence" / "evidence_documents.jsonl"
    index_dir = tmp_path / "index"
    build_keyword_index(evidence_path=evidence_path, index_dir=index_dir)
    build_vector_index(
        evidence_path=evidence_path,
        index_dir=index_dir,
        backend=ConstantEmbeddingBackend(),
    )
    assert (index_dir / "embeddings.npy").exists()
    assert (index_dir / "embedding_ids.json").exists()

    build_keyword_index(evidence_path=evidence_path, index_dir=index_dir)

    assert not (index_dir / "embeddings.npy").exists()
    assert not (index_dir / "embedding_ids.json").exists()
