from __future__ import annotations

from collections import Counter

from .schemas import AuditIssue, AuditReport, CodeLibrary


def audit_code_library(library: CodeLibrary) -> AuditReport:
    errors: list[AuditIssue] = []
    warnings: list[AuditIssue] = []
    ids = [entity.entity_id for entity in library.entities]

    for entity in library.entities:
        if not entity.entity_id:
            errors.append(_issue("error", "missing_id", "实体缺少 id。"))
        if not entity.standard_code:
            errors.append(_issue("error", "missing_standard", "实体缺少规范编号。", entity.entity_id))
        if not entity.source:
            warnings.append(_issue("warning", "missing_source", "实体缺少规范来源 source。", entity.entity_id))
        if not entity.name and not entity.description:
            warnings.append(_issue("warning", "missing_name", "实体缺少名称和描述。", entity.entity_id))

    for entity_id, count in Counter(ids).items():
        if entity_id and count > 1:
            errors.append(_issue("error", "duplicate_id", f"实体 id 重复 {count} 次。", entity_id))

    expected = library.metadata.item_count
    if expected is not None and expected != len(library.entities):
        errors.append(
            _issue(
                "error",
                "item_count_mismatch",
                f"metadata.item_count={expected}，实际实体数={len(library.entities)}。",
            )
        )

    return AuditReport(
        source_path=library.metadata.source_path,
        entity_count=len(library.entities),
        errors=errors,
        warnings=warnings,
    )


def _issue(severity: str, code: str, message: str, entity_id: str | None = None) -> AuditIssue:
    return AuditIssue(severity=severity, code=code, message=message, entity_id=entity_id)  # type: ignore[arg-type]
