from __future__ import annotations

from pathlib import Path

import numpy as np

from bridge_agents.code_rag.build_index import build_keyword_index
from bridge_agents.code_rag.dataset import build_chunk_dataset
from bridge_agents.code_rag.evaluate import evaluate_retrieval
from bridge_agents.code_rag.vector_index import build_vector_index


class EvaluationFakeBackend:
    model_name = "evaluation-fake-v1"

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
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_evaluate_retrieval_reports_the_requested_mode(tmp_path: Path) -> None:
    build_chunk_dataset(artifact_root=tmp_path)
    evidence_path = tmp_path / "evidence" / "evidence_documents.jsonl"
    index_dir = tmp_path / "index"
    db_path = index_dir / "code_knowledge.db"
    build_keyword_index(evidence_path=evidence_path, index_dir=index_dir)
    backend = EvaluationFakeBackend()
    build_vector_index(evidence_path=evidence_path, index_dir=index_dir, backend=backend)
    dataset_path = tmp_path / "evaluation.yaml"
    dataset_path.write_text(
        """schema_version: test_v0.1
default_top_k: 3
queries:
  - id: HYBRID_ONE
    design_stage: modeling
    query: semantic-only
    standard_codes: [JTG D60-2015]
    expected_status: ok
    required_evidence:
      - primary_entity_id: R_D60_CH04_005
        must_contain: [S_ud]
""",
        encoding="utf-8",
    )

    result = evaluate_retrieval(
        db_path=db_path,
        evidence_path=evidence_path,
        index_dir=index_dir,
        dataset_path=dataset_path,
        output_dir=tmp_path / "evaluation",
        retrieval_mode="hybrid",
        embedding_backend=backend,
    )

    assert result["summary"]["retrieval_mode"] == "hybrid"
    assert result["summary"]["passed_query_count"] == 1
    assert (tmp_path / "evaluation" / "hybrid_evaluation.json").exists()
