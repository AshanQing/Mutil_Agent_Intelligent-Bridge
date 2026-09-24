from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .audit import audit_code_library
from .evidence import EvidenceBuildResult, build_evidence_documents
from .loader import load_code_library
from .paths import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_AUDIT_DIR,
    DEFAULT_CODE_DIR,
    DEFAULT_EVIDENCE_DOCUMENTS_PATH,
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_EXCLUSIONS_PATH,
    DEFAULT_REVIEW_DIR,
    DEFAULT_SOURCE_DIR,
    DEFAULT_SOURCE_ENTITIES_PATH,
)
from .schemas import AuditIssue, CodeEntity


SCHEMA_VERSION = "code_rag_evidence_v0.3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full entity-level chunk dataset for code RAG.")
    parser.add_argument("--code-dir", default=str(DEFAULT_CODE_DIR), help="Directory containing data/code/*.yaml files.")
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Root directory for generated code RAG artifacts.",
    )
    args = parser.parse_args()

    result = build_chunk_dataset(
        code_dir=Path(args.code_dir),
        artifact_root=Path(args.artifact_root),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_chunk_dataset(
    *,
    code_dir: Path = DEFAULT_CODE_DIR,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Dict[str, Any]:
    source_dir = artifact_root / DEFAULT_SOURCE_DIR.name
    evidence_dir = artifact_root / DEFAULT_EVIDENCE_DIR.name
    review_dir = artifact_root / DEFAULT_REVIEW_DIR.name
    audit_dir = artifact_root / DEFAULT_AUDIT_DIR.name
    source_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    yaml_paths = sorted(code_dir.glob("*.yaml"))
    libraries = [load_code_library(path) for path in yaml_paths]
    library_reports = [audit_code_library(library) for library in libraries]
    entities = [entity for library in libraries for entity in library.entities]

    source_path = source_dir / DEFAULT_SOURCE_ENTITIES_PATH.name
    evidence_path = evidence_dir / DEFAULT_EVIDENCE_DOCUMENTS_PATH.name
    exclusions_path = review_dir / DEFAULT_EXCLUSIONS_PATH.name
    manifest_path = artifact_root / "manifest.json"
    summary_path = audit_dir / "audit_summary.json"
    report_path = audit_dir / "audit_report.md"

    chunk_count = _write_source_entities(source_path, entities)
    evidence_result = build_evidence_documents(entities)
    _write_evidence_documents(evidence_path, evidence_result)
    _write_exclusions(exclusions_path, evidence_result)
    global_issues = _global_audit(entities)
    summary = _build_summary(
        yaml_paths=yaml_paths,
        libraries=libraries,
        library_reports=library_reports,
        global_issues=global_issues,
        source_path=source_path,
        chunk_count=chunk_count,
        evidence_result=evidence_result,
    )
    manifest = _build_manifest(
        yaml_paths=yaml_paths,
        entities=entities,
        source_path=source_path,
        evidence_path=evidence_path,
        exclusions_path=exclusions_path,
        evidence_result=evidence_result,
    )

    _write_json(manifest_path, manifest)
    _write_json(summary_path, summary)
    report_path.write_text(_render_markdown_report(summary), encoding="utf-8")

    return {
        "chunk_count": chunk_count,
        "source_entity_count": chunk_count,
        "evidence_document_count": len(evidence_result.documents),
        "excluded_entity_count": len(evidence_result.exclusions),
        "source_entities_path": str(source_path),
        "evidence_documents_path": str(evidence_path),
        "excluded_entities_path": str(exclusions_path),
        "manifest_path": str(manifest_path),
        "audit_summary_path": str(summary_path),
        "audit_report_path": str(report_path),
        "blocking_error_count": summary["blocking_error_count"],
        "warning_count": summary["warning_count"],
    }


def _write_source_entities(path: Path, entities: List[CodeEntity]) -> int:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, entity in enumerate(entities, start=1):
            payload = {
                "chunk_id": _chunk_id(entity),
                "ordinal": index,
                "entity": entity.model_dump(exclude={"raw_payload"}),
                "raw_payload": entity.raw_payload,
                "search_text": entity.search_text(),
                "semantic_text": entity.semantic_text(),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return len(entities)


def _write_evidence_documents(path: Path, result: EvidenceBuildResult) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in result.documents:
            handle.write(document.model_dump_json() + "\n")


def _write_exclusions(path: Path, result: EvidenceBuildResult) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for exclusion in result.exclusions:
            handle.write(exclusion.model_dump_json() + "\n")


def _global_audit(entities: List[CodeEntity]) -> List[AuditIssue]:
    issues: list[AuditIssue] = []
    ids = [entity.entity_id for entity in entities if entity.entity_id]
    id_counts = Counter(ids)
    for entity_id, count in sorted(id_counts.items()):
        if count > 1:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="global_duplicate_id",
                    message=f"全局实体 ID 重复 {count} 次。",
                    entity_id=entity_id,
                )
            )

    all_ids = set(ids)
    for entity in entities:
        issues.extend(_evidence_completeness_issues(entity))
        for relation_type, targets in entity.related_entities.items():
            for target in targets:
                if target and target not in all_ids:
                    issues.append(
                        AuditIssue(
                            severity="error",
                            code="dangling_relation",
                            message=f"{relation_type} 引用不存在的实体 {target}。",
                            entity_id=entity.entity_id,
                        )
                    )
    return issues


def _evidence_completeness_issues(entity: CodeEntity) -> List[AuditIssue]:
    issues: list[AuditIssue] = []
    raw = entity.raw_payload
    if not raw:
        return [
            AuditIssue(
                severity="error",
                code="missing_raw_payload",
                message="实体没有完整 YAML 条目负载。",
                entity_id=entity.entity_id,
            )
        ]
    if str(raw.get("id") or "") != entity.entity_id:
        issues.append(
            AuditIssue(
                severity="error",
                code="raw_payload_id_mismatch",
                message="完整 YAML 条目的 ID 与标准化实体 ID 不一致。",
                entity_id=entity.entity_id,
            )
        )
    if entity.library_type == "rule" and not (entity.original_text or entity.description):
        issues.append(
            AuditIssue(
                severity="error",
                code="rule_content_missing",
                message="规则缺少可检索的正文或描述。",
                entity_id=entity.entity_id,
            )
        )
    if entity.library_type == "formula":
        expression = (raw.get("math") or {}).get("expression") or (raw.get("execution") or {}).get("expression")
        if not expression:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="formula_expression_missing",
                    message="公式缺少表达式。",
                    entity_id=entity.entity_id,
                )
            )
    if entity.library_type == "table":
        data = raw.get("data")
        if not isinstance(data, list) or not data:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="table_data_missing",
                    message="表格缺少完整数据行。",
                    entity_id=entity.entity_id,
                )
            )
    if entity.library_type == "variable":
        definition = raw.get("definition") or raw.get("description") or entity.execution.get("definition")
        if not definition:
            issues.append(
                AuditIssue(
                    severity="warning",
                    code="variable_definition_missing",
                    message="变量缺少独立定义字段，使用名称和上下文检索。",
                    entity_id=entity.entity_id,
                )
            )
    return issues


def _build_summary(
    *,
    yaml_paths: List[Path],
    libraries: List[Any],
    library_reports: List[Any],
    global_issues: List[AuditIssue],
    source_path: Path,
    chunk_count: int,
    evidence_result: EvidenceBuildResult,
) -> Dict[str, Any]:
    file_summaries = []
    blocking_errors: list[Dict[str, Any]] = []
    warnings: list[Dict[str, Any]] = []
    by_standard: Dict[str, int] = defaultdict(int)
    by_type: Dict[str, int] = defaultdict(int)
    entities = [entity for library in libraries for entity in library.entities]

    for path, library, report in zip(yaml_paths, libraries, library_reports):
        for entity in library.entities:
            by_standard[entity.standard_code] += 1
            by_type[entity.library_type] += 1

        file_errors = [issue.model_dump() for issue in report.errors]
        file_warnings = [issue.model_dump() for issue in report.warnings]
        blocking_errors.extend({"file": path.name, **issue} for issue in file_errors)
        warnings.extend({"file": path.name, **issue} for issue in file_warnings)
        file_summaries.append(
            {
                "file": path.name,
                "standard_code": library.metadata.standard_code,
                "library_type": library.entities[0].library_type if library.entities else None,
                "metadata_item_count": library.metadata.item_count,
                "actual_entity_count": len(library.entities),
                "error_count": len(report.errors),
                "warning_count": len(report.warnings),
            }
        )

    global_issue_payloads = [issue.model_dump() for issue in global_issues]
    blocking_errors.extend({"file": None, **issue} for issue in global_issue_payloads if issue["severity"] == "error")
    warnings.extend({"file": None, **issue} for issue in global_issue_payloads if issue["severity"] == "warning")

    return {
        "schema_version": SCHEMA_VERSION,
        "built_at": _utc_now(),
        "source_file_count": len(yaml_paths),
        "chunk_count": chunk_count,
        "source_entity_count": chunk_count,
        "source_entities_path": str(source_path),
        "evidence_document_count": len(evidence_result.documents),
        "excluded_entity_count": len(evidence_result.exclusions),
        "exclusions_by_reason": dict(
            sorted(Counter(item.reason for item in evidence_result.exclusions).items())
        ),
        "counts_by_standard": dict(sorted(by_standard.items())),
        "counts_by_library_type": dict(sorted(by_type.items())),
        "evidence_completeness": _evidence_completeness_summary(entities),
        "files": file_summaries,
        "blocking_error_count": len(blocking_errors),
        "warning_count": len(warnings),
        "blocking_errors": blocking_errors,
        "warnings": warnings,
    }


def _evidence_completeness_summary(entities: List[CodeEntity]) -> Dict[str, int]:
    tables = [entity for entity in entities if entity.library_type == "table"]
    formulas = [entity for entity in entities if entity.library_type == "formula"]
    rules = [entity for entity in entities if entity.library_type == "rule"]
    variables = [entity for entity in entities if entity.library_type == "variable"]
    return {
        "raw_payload_entities": sum(bool(entity.raw_payload) for entity in entities),
        "rules_with_content": sum(bool(entity.original_text or entity.description) for entity in rules),
        "formulas_with_expression": sum(
            bool(
                (entity.raw_payload.get("math") or {}).get("expression")
                or (entity.raw_payload.get("execution") or {}).get("expression")
            )
            for entity in formulas
        ),
        "tables_with_data": sum(
            isinstance(entity.raw_payload.get("data"), list) and bool(entity.raw_payload.get("data"))
            for entity in tables
        ),
        "table_data_rows": sum(len(entity.raw_payload.get("data") or []) for entity in tables),
        "variables_with_definition": sum(
            bool(
                entity.raw_payload.get("definition")
                or entity.raw_payload.get("description")
                or entity.execution.get("definition")
            )
            for entity in variables
        ),
    }


def _build_manifest(
    *,
    yaml_paths: List[Path],
    entities: List[CodeEntity],
    source_path: Path,
    evidence_path: Path,
    exclusions_path: Path,
    evidence_result: EvidenceBuildResult,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "built_at": _utc_now(),
        "source_entity_count": len(entities),
        "evidence_document_count": len(evidence_result.documents),
        "excluded_entity_count": len(evidence_result.exclusions),
        "source_entities_file": str(source_path),
        "source_entities_sha256": _sha256_file(source_path),
        "evidence_documents_file": str(evidence_path),
        "evidence_documents_sha256": _sha256_file(evidence_path),
        "excluded_entities_file": str(exclusions_path),
        "excluded_entities_sha256": _sha256_file(exclusions_path),
        "evidence_includes_raw_payload": False,
        "source_verification": "not_verified_against_official_text",
        "source_file_hashes": {
            str(path): _sha256_file(path)
            for path in yaml_paths
        },
        "embedding_model": None,
        "embedding_dimension": None,
    }


def _render_markdown_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Code RAG Chunk Audit Report",
        "",
        f"- Built at: `{summary['built_at']}`",
        f"- Source files: `{summary['source_file_count']}`",
        f"- Source entities: `{summary['source_entity_count']}`",
        f"- Prompt-ready evidence documents: `{summary['evidence_document_count']}`",
        f"- Excluded entities: `{summary['excluded_entity_count']}`",
        f"- Blocking errors: `{summary['blocking_error_count']}`",
        f"- Warnings: `{summary['warning_count']}`",
        "",
        "## Evidence Completeness",
        "",
    ]
    for name, count in summary["evidence_completeness"].items():
        lines.append(f"- `{name}`: {count}")

    lines.extend(
        [
        "",
        "## Counts By Standard",
        "",
        ]
    )
    for standard, count in summary["counts_by_standard"].items():
        lines.append(f"- `{standard}`: {count}")

    lines.extend(["", "## Counts By Library Type", ""])
    for library_type, count in summary["counts_by_library_type"].items():
        lines.append(f"- `{library_type}`: {count}")

    lines.extend(["", "## Source Files", "", "| File | Standard | Type | Metadata Count | Actual Count | Errors | Warnings |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for item in summary["files"]:
        lines.append(
            f"| {item['file']} | {item['standard_code']} | {item['library_type']} | "
            f"{item['metadata_item_count']} | {item['actual_entity_count']} | "
            f"{item['error_count']} | {item['warning_count']} |"
        )

    if summary["blocking_errors"]:
        lines.extend(["", "## Blocking Errors", ""])
        for issue in summary["blocking_errors"][:200]:
            file_name = issue.get("file") or "global"
            entity_id = issue.get("entity_id") or "-"
            lines.append(f"- `{file_name}` `{issue['code']}` `{entity_id}`: {issue['message']}")
        if len(summary["blocking_errors"]) > 200:
            lines.append(f"- ... {len(summary['blocking_errors']) - 200} more errors omitted in markdown.")

    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for issue in summary["warnings"][:200]:
            file_name = issue.get("file") or "global"
            entity_id = issue.get("entity_id") or "-"
            lines.append(f"- `{file_name}` `{issue['code']}` `{entity_id}`: {issue['message']}")
        if len(summary["warnings"]) > 200:
            lines.append(f"- ... {len(summary['warnings']) - 200} more warnings omitted in markdown.")

    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _chunk_id(entity: CodeEntity) -> str:
    return f"{entity.standard_code}|{entity.library_type}|{entity.entity_id}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
