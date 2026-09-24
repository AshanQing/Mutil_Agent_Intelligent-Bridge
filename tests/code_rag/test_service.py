from __future__ import annotations

from pathlib import Path

import numpy as np

from bridge_agents.code_rag.build_index import build_keyword_index
from bridge_agents.code_rag.dataset import build_chunk_dataset
from bridge_agents.code_rag.evidence import build_evidence_documents
from bridge_agents.code_rag.evaluate import load_evaluation_dataset
from bridge_agents.code_rag.loader import load_code_library
from bridge_agents.code_rag.paths import DEFAULT_EVALUATION_DATASET
from bridge_agents.code_rag.schemas import CodeQuery
from bridge_agents.code_rag.service import CodeRAGService, render_evidence_summary
from bridge_agents.code_rag.vector_index import build_vector_index


class SemanticOnlyFakeBackend:
    model_name = "semantic-only-fake-v1"

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [1.0, 0.0]
                if "承载能力极限状态作用组合规则" in text
                else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if "semantic-only" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def test_evaluation_dataset_contains_30_content_level_cases() -> None:
    dataset = load_evaluation_dataset(DEFAULT_EVALUATION_DATASET)

    assert dataset["schema_version"] == "code_rag_eval_v0.2"
    assert len(dataset["queries"]) == 30
    entities = [
        entity
        for path in Path("data/code").glob("*.yaml")
        for entity in load_code_library(path).entities
    ]
    known_ids = {
        document.primary_entity_id
        for document in build_evidence_documents(entities).documents
    }

    required_ids = {
        evidence["primary_entity_id"]
        for query in dataset["queries"]
        for evidence in query.get("required_evidence", [])
    }
    assert required_ids <= known_ids
    assert all(
        evidence["must_contain"]
        for query in dataset["queries"]
        for evidence in query.get("required_evidence", [])
    )
    assert {query["expected_status"] for query in dataset["queries"]} == {
        "ok",
        "no_valid_evidence",
    }
    assert {
        query["design_stage"] for query in dataset["queries"]
    } >= {
        "initial_layout",
        "layout_revision",
        "dimension_design",
        "reinforcement_design",
        "modeling",
        "verification",
    }


def test_service_returns_prompt_ready_evidence_without_raw_yaml(tmp_path) -> None:
    build_chunk_dataset(artifact_root=tmp_path)
    build_keyword_index(
        evidence_path=tmp_path / "evidence" / "evidence_documents.jsonl",
        index_dir=tmp_path / "index",
    )
    service = CodeRAGService(tmp_path / "index" / "code_knowledge.db")
    assert service.validate_index()["ok"]
    bundle = service.retrieve(
        CodeQuery(
            query_text="钢筋混凝土上部结构跨径限制",
            standard_codes=["JTG 3362-2018"],
            top_k=3,
        )
    )

    assert bundle.retrieval_status == "ok"
    assert bundle.evidence_documents
    assert all("raw_payload" not in item.model_dump() for item in bundle.evidence_documents)
    assert all("人工复核" not in item.prompt_text for item in bundle.evidence_documents)
    summary = render_evidence_summary(bundle)
    assert "规范证据" in summary
    assert "装配式板桥不大于10m" in summary


def test_service_returns_no_valid_evidence_for_title_only_clause(tmp_path) -> None:
    build_chunk_dataset(artifact_root=tmp_path)
    build_keyword_index(
        evidence_path=tmp_path / "evidence" / "evidence_documents.jsonl",
        index_dir=tmp_path / "index",
    )
    bundle = CodeRAGService(tmp_path / "index" / "code_knowledge.db").retrieve(
        CodeQuery(
            query_text="8.4.2",
            standard_codes=["JTG 3362-2018"],
            top_k=5,
        )
    )

    assert all(item.clause != "8.4.2" for item in bundle.evidence_documents)


def test_service_does_not_treat_related_cap_beam_checks_as_dimension_guidance(tmp_path) -> None:
    build_chunk_dataset(artifact_root=tmp_path)
    build_keyword_index(
        evidence_path=tmp_path / "evidence" / "evidence_documents.jsonl",
        index_dir=tmp_path / "index",
    )
    bundle = CodeRAGService(tmp_path / "index" / "code_knowledge.db").retrieve(
        CodeQuery(
            query_text="盖梁尺寸",
            standard_codes=["JTG 3362-2018"],
            top_k=5,
        )
    )

    assert bundle.retrieval_status == "no_valid_evidence"
    assert bundle.evidence_documents == []


def test_service_uses_vector_candidates_in_hybrid_mode(tmp_path, monkeypatch) -> None:
    build_chunk_dataset(artifact_root=tmp_path)
    evidence_path = tmp_path / "evidence" / "evidence_documents.jsonl"
    index_dir = tmp_path / "index"
    build_keyword_index(evidence_path=evidence_path, index_dir=index_dir)
    backend = SemanticOnlyFakeBackend()
    build_vector_index(evidence_path=evidence_path, index_dir=index_dir, backend=backend)
    monkeypatch.setattr(
        "bridge_agents.code_rag.service.SentenceTransformerBackend",
        lambda model_name, **kwargs: backend,
    )

    bundle = CodeRAGService(
        index_dir / "code_knowledge.db",
        evidence_path=evidence_path,
        index_dir=index_dir,
    ).retrieve(
        CodeQuery(
            query_text="semantic-only",
            standard_codes=["JTG D60-2015"],
            retrieval_mode="auto",
            top_k=3,
        )
    )

    assert bundle.retrieval_status == "ok"
    assert bundle.retrieval_trace.retrieval_mode == "hybrid"
    assert "EVID_R_D60_CH04_005" in bundle.retrieval_trace.vector_result_ids
    assert bundle.evidence_documents[0].primary_entity_id == "R_D60_CH04_005"
    assert bundle.evidence_documents[0].vector_rank == 1
    assert bundle.evidence_documents[0].fused_rank == 1


def test_validate_index_rejects_corrupted_vector_file(tmp_path) -> None:
    build_chunk_dataset(artifact_root=tmp_path)
    evidence_path = tmp_path / "evidence" / "evidence_documents.jsonl"
    index_dir = tmp_path / "index"
    build_keyword_index(evidence_path=evidence_path, index_dir=index_dir)
    build_vector_index(
        evidence_path=evidence_path,
        index_dir=index_dir,
        backend=SemanticOnlyFakeBackend(),
    )
    (index_dir / "embeddings.npy").write_bytes(b"corrupted")

    validation = CodeRAGService(
        index_dir / "code_knowledge.db",
        evidence_path=evidence_path,
        index_dir=index_dir,
    ).validate_index()

    assert validation["ok"] is False
    assert validation["reason"] == "invalid_vector_index"
    assert validation["vector_errors"]


def test_custom_database_uses_manifest_from_its_own_index_directory(tmp_path) -> None:
    build_chunk_dataset(artifact_root=tmp_path)
    evidence_path = tmp_path / "evidence" / "evidence_documents.jsonl"
    index_dir = tmp_path / "index"
    db_path = index_dir / "code_knowledge.db"
    build_keyword_index(evidence_path=evidence_path, index_dir=index_dir)
    (index_dir / "index_manifest.json").write_text("not-json", encoding="utf-8")

    validation = CodeRAGService(db_path).validate_index()

    assert validation["ok"] is False
    assert validation["reason"] == "invalid_vector_index"
