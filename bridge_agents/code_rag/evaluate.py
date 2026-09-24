from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .paths import (
    DEFAULT_DB_PATH,
    DEFAULT_EVALUATION_DATASET,
    DEFAULT_EVALUATION_DIR,
    DEFAULT_EVIDENCE_DOCUMENTS_PATH,
    DEFAULT_INDEX_DIR,
)
from .schemas import CodeQuery
from .service import CodeRAGService
from .vector_index import EmbeddingBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a code RAG retrieval mode.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to code_knowledge.db.")
    parser.add_argument("--dataset", default=str(DEFAULT_EVALUATION_DATASET), help="Evaluation YAML path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_EVALUATION_DIR), help="Evaluation report directory.")
    parser.add_argument(
        "--mode",
        choices=["keyword_only", "vector_only", "hybrid"],
        default="keyword_only",
        help="Retrieval mode to evaluate.",
    )
    args = parser.parse_args()

    result = evaluate_retrieval(
        db_path=Path(args.db),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        retrieval_mode=args.mode,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def load_evaluation_dataset(path: Path = DEFAULT_EVALUATION_DATASET) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError(f"Invalid evaluation dataset: {path}")

    ids = [str(item.get("id") or "") for item in payload["queries"]]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Evaluation query IDs must be non-empty and unique.")
    for item in payload["queries"]:
        if not item.get("query") or item.get("expected_status") not in {"ok", "no_valid_evidence"}:
            raise ValueError(f"Evaluation query is incomplete: {item.get('id')}")
        required = item.get("required_evidence")
        if not isinstance(required, list):
            raise ValueError(f"Evaluation required_evidence must be a list: {item.get('id')}")
        if item["expected_status"] == "ok" and not required:
            raise ValueError(f"Positive evaluation query has no required evidence: {item.get('id')}")
        if item["expected_status"] == "no_valid_evidence" and required:
            raise ValueError(f"Negative evaluation query cannot require evidence: {item.get('id')}")
        for evidence in required:
            if not evidence.get("primary_entity_id") or not evidence.get("must_contain"):
                raise ValueError(f"Invalid evidence assertion in query: {item.get('id')}")
    return payload


def evaluate_retrieval(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    evidence_path: Path = DEFAULT_EVIDENCE_DOCUMENTS_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    dataset_path: Path = DEFAULT_EVALUATION_DATASET,
    output_dir: Path = DEFAULT_EVALUATION_DIR,
    retrieval_mode: str = "keyword_only",
    embedding_backend: EmbeddingBackend | None = None,
) -> dict[str, Any]:
    dataset = load_evaluation_dataset(dataset_path)
    default_top_k = int(dataset.get("default_top_k") or 10)
    results = []
    total_required = 0
    total_found = 0
    total_content_assertions = 0
    passed_content_assertions = 0
    service = CodeRAGService(
        db_path,
        evidence_path=evidence_path,
        index_dir=index_dir,
        embedding_backend=embedding_backend,
    )

    for item in dataset["queries"]:
        top_k = int(item.get("top_k") or default_top_k)
        bundle = service.retrieve(
            CodeQuery(
                query_text=str(item["query"]),
                standard_codes=list(item.get("standard_codes") or []),
                stage=item.get("stage"),
                member_types=list(item.get("member_types") or []),
                materials=list(item.get("materials") or []),
                limit_states=list(item.get("limit_states") or []),
                library_types=list(item.get("library_types") or []),
                retrieval_mode=retrieval_mode,
                top_k=top_k,
            )
        )
        returned = {document.primary_entity_id: document for document in bundle.evidence_documents}
        returned_ids = list(returned)
        required_items = list(item.get("required_evidence") or [])
        required_ids = [str(evidence["primary_entity_id"]) for evidence in required_items]
        found_ids = [entity_id for entity_id in required_ids if entity_id in returned_ids]
        missing_ids = [entity_id for entity_id in required_ids if entity_id not in returned_ids]
        failed_content: dict[str, list[str]] = {}
        for evidence in required_items:
            evidence_id = str(evidence["primary_entity_id"])
            assertions = [str(value) for value in evidence["must_contain"]]
            total_content_assertions += len(assertions)
            document = returned.get(evidence_id)
            missing_content = [
                assertion
                for assertion in assertions
                if document is None or assertion not in document.prompt_text
            ]
            passed_content_assertions += len(assertions) - len(missing_content)
            if missing_content:
                failed_content[evidence_id] = missing_content
        total_required += len(required_ids)
        total_found += len(found_ids)
        status_matches = bundle.retrieval_status == item["expected_status"]
        passed = status_matches and not missing_ids and not failed_content
        results.append(
            {
                "id": item["id"],
                "design_stage": item.get("design_stage"),
                "query": item["query"],
                "top_k": top_k,
                "expected_status": item["expected_status"],
                "actual_status": bundle.retrieval_status,
                "required_evidence": required_ids,
                "returned_evidence": returned_ids,
                "found_evidence": found_ids,
                "missing_evidence": missing_ids,
                "failed_content_assertions": failed_content,
                "recall": len(found_ids) / len(required_ids) if required_ids else 1.0,
                "passed": passed,
            }
        )

    query_count = len(results)
    passed_count = sum(1 for result in results if result["passed"])
    summary = {
        "schema_version": "code_rag_prompt_evidence_eval_v0.4",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "query_count": query_count,
        "passed_query_count": passed_count,
        "query_pass_rate": passed_count / query_count if query_count else 0.0,
        "required_evidence_recall": total_found / total_required if total_required else 0.0,
        "content_assertion_pass_rate": (
            passed_content_assertions / total_content_assertions
            if total_content_assertions
            else 1.0
        ),
        "positive_query_count": sum(item["expected_status"] == "ok" for item in dataset["queries"]),
        "negative_query_count": sum(
            item["expected_status"] == "no_valid_evidence" for item in dataset["queries"]
        ),
        "target_top10_recall": 0.9,
        "target_met": (total_found / total_required) >= 0.9 if total_required else False,
        "retrieval_mode": retrieval_mode,
    }
    payload = {"summary": summary, "results": results}

    output_dir.mkdir(parents=True, exist_ok=True)
    report_stem = f"{retrieval_mode}_evaluation"
    (output_dir / f"{report_stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{report_stem}.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    return payload


def evaluate_keyword_retrieval(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    dataset_path: Path = DEFAULT_EVALUATION_DATASET,
    output_dir: Path = DEFAULT_EVALUATION_DIR,
) -> dict[str, Any]:
    return evaluate_retrieval(
        db_path=db_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        retrieval_mode="keyword_only",
    )


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        f"# Code RAG {summary['retrieval_mode']} Evaluation",
        "",
        f"- Queries: `{summary['query_count']}`",
        f"- Passed queries: `{summary['passed_query_count']}`",
        f"- Query pass rate: `{summary['query_pass_rate']:.1%}`",
        f"- Required evidence recall: `{summary['required_evidence_recall']:.1%}`",
        f"- Content assertion pass rate: `{summary['content_assertion_pass_rate']:.1%}`",
        f"- Target: `{summary['target_top10_recall']:.1%}`",
        "",
        "| ID | Status | Recall | Missing evidence/content |",
        "| --- | --- | ---: | --- |",
    ]
    for result in payload["results"]:
        missing = list(result["missing_evidence"])
        missing.extend(
            f"{evidence_id}:{','.join(values)}"
            for evidence_id, values in result["failed_content_assertions"].items()
        )
        detail = ", ".join(missing) or "-"
        status = f"{result['actual_status']} / {result['expected_status']}"
        lines.append(f"| {result['id']} | {status} | {result['recall']:.0%} | {detail} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
