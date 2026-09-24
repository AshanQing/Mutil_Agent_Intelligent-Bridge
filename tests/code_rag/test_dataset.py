from __future__ import annotations

import json

from bridge_agents.code_rag.dataset import build_chunk_dataset


def test_build_chunk_dataset_outputs_source_and_evidence_jsonl(tmp_path) -> None:
    result = build_chunk_dataset(artifact_root=tmp_path)

    assert result["chunk_count"] == 1908
    assert result["evidence_document_count"] > 0
    chunks_path = tmp_path / "source" / "source_entities.jsonl"
    evidence_path = tmp_path / "evidence" / "evidence_documents.jsonl"
    exclusions_path = tmp_path / "review" / "excluded_entities.jsonl"
    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "audit" / "audit_summary.json"

    assert chunks_path.exists()
    assert evidence_path.exists()
    assert exclusions_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()

    first = json.loads(chunks_path.read_text(encoding="utf-8").splitlines()[0])
    assert set(first) == {
        "chunk_id",
        "ordinal",
        "entity",
        "raw_payload",
        "search_text",
        "semantic_text",
    }
    assert first["entity"]["library_type"] in {"rule", "formula", "table", "variable"}
    assert "raw_payload" not in first["entity"]
    assert first["raw_payload"]["id"] == first["entity"]["entity_id"]
    assert first["entity"]["source_file"].endswith(".yaml")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source_file_count"] == 12
    assert summary["counts_by_library_type"] == {
        "formula": 323,
        "rule": 463,
        "table": 104,
        "variable": 1018,
    }
    assert summary["evidence_completeness"]["raw_payload_entities"] == 1908
    assert summary["evidence_completeness"]["tables_with_data"] == 104
    assert summary["evidence_completeness"]["formulas_with_expression"] == 323
    assert summary["blocking_error_count"] == 0

    evidence = json.loads(evidence_path.read_text(encoding="utf-8").splitlines()[0])
    assert "raw_payload" not in evidence
    assert "related_entities" not in evidence
    assert evidence["prompt_text"]
