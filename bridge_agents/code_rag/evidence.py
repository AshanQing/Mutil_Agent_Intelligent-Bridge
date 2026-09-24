from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

from pydantic import BaseModel

from .schemas import (
    CodeEntity,
    EvidenceDocument,
    EvidenceFormula,
    EvidenceTable,
    EvidenceVariable,
)


_NORMATIVE_FORMULA_STATUSES = {"normative", "normative_formula_expanded"}
_FORBIDDEN_RULE_PHRASES = {
    "证据不足",
    "需要补录",
    "转人工复核",
    "人工复核提示",
    "调用关联公式",
    "输出计算值、合规判断",
}


class EvidenceExclusion(BaseModel):
    entity_id: str
    library_type: str
    reason: str


@dataclass(frozen=True)
class EvidenceBuildResult:
    documents: list[EvidenceDocument]
    exclusions: list[EvidenceExclusion]


def build_evidence_documents(entities: Iterable[CodeEntity]) -> EvidenceBuildResult:
    entity_list = list(entities)
    by_id = {entity.entity_id: entity for entity in entity_list}
    documents: list[EvidenceDocument] = []
    exclusions: list[EvidenceExclusion] = []

    for entity in entity_list:
        if entity.library_type == "variable":
            exclusions.append(_exclude(entity, "supporting_variable_only"))
            continue
        if entity.library_type == "rule":
            reason = _rule_exclusion_reason(entity, by_id)
            if reason:
                formulas, tables = _valid_support(entity, by_id)
                if formulas or tables:
                    documents.append(
                        _build_rule_document(
                            entity,
                            by_id,
                            evidence_type="composite",
                            include_normative_text=False,
                        )
                    )
                else:
                    exclusions.append(_exclude(entity, reason))
                continue
            documents.append(_build_rule_document(entity, by_id))
            continue
        if entity.library_type == "formula":
            if not _is_normative_formula(entity):
                exclusions.append(_exclude(entity, "non_normative_formula"))
                continue
            documents.append(_build_formula_document(entity, by_id))
            continue
        if entity.library_type == "table":
            if not _table_rows(entity):
                exclusions.append(_exclude(entity, "table_without_rows"))
                continue
            documents.append(_build_table_document(entity, by_id))

    documents.sort(key=lambda item: item.evidence_id)
    exclusions.sort(key=lambda item: item.entity_id)
    return EvidenceBuildResult(documents=documents, exclusions=exclusions)


def _build_rule_document(
    entity: CodeEntity,
    by_id: dict[str, CodeEntity],
    *,
    evidence_type: str = "rule",
    include_normative_text: bool = True,
) -> EvidenceDocument:
    formulas, tables = _valid_support(entity, by_id)
    return _document(
        entity,
        by_id=by_id,
        evidence_type=evidence_type,
        normative_text=_clean_text(entity.original_text) if include_normative_text else "",
        formulas=formulas,
        tables=tables,
    )


def _build_formula_document(
    entity: CodeEntity,
    by_id: dict[str, CodeEntity],
) -> EvidenceDocument:
    return _document(
        entity,
        by_id=by_id,
        evidence_type="formula",
        normative_text="",
        formulas=[_formula_payload(entity.entity_id, by_id)],
        tables=[],
    )


def _build_table_document(
    entity: CodeEntity,
    by_id: dict[str, CodeEntity],
) -> EvidenceDocument:
    return _document(
        entity,
        by_id=by_id,
        evidence_type="table",
        normative_text="",
        formulas=[],
        tables=[_table_payload(entity.entity_id, by_id)],
    )


def _document(
    entity: CodeEntity,
    *,
    by_id: dict[str, CodeEntity],
    evidence_type: str,
    normative_text: str,
    formulas: list[EvidenceFormula],
    tables: list[EvidenceTable],
) -> EvidenceDocument:
    source_ids = [entity.entity_id]
    source_ids.extend(formula.formula_id for formula in formulas)
    source_ids.extend(table.table_id for table in tables)
    source_ids = list(dict.fromkeys(source_ids))
    source_files = list(
        dict.fromkeys(
            by_id[source_id].source_file
            for source_id in source_ids
            if source_id in by_id and by_id[source_id].source_file
        )
    )
    prompt_text = _render_prompt_text(
        entity=entity,
        normative_text=normative_text,
        formulas=formulas,
        tables=tables,
    )
    applicability = _safe_applicability(entity)
    return EvidenceDocument(
        evidence_id=f"EVID_{entity.entity_id}",
        primary_entity_id=entity.entity_id,
        evidence_type=evidence_type,
        standard_code=entity.standard_code,
        standard_name=entity.standard_name,
        version=entity.version,
        chapter=entity.chapter,
        clause=entity.clause,
        title=entity.name,
        source=entity.source,
        normative_text=normative_text,
        applicability=applicability,
        formulas=formulas,
        tables=tables,
        source_entity_ids=source_ids,
        source_files=source_files or [entity.source_file],
        prompt_text=prompt_text,
        search_text=prompt_text,
    )


def _formula_payload(formula_id: str, by_id: dict[str, CodeEntity]) -> EvidenceFormula:
    entity = by_id[formula_id]
    raw = entity.raw_payload
    math = raw.get("math") if isinstance(raw.get("math"), dict) else {}
    execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
    variable_ids = math.get("variables") if isinstance(math.get("variables"), list) else []
    variables = [
        _variable_payload(by_id[variable_id])
        for variable_id in variable_ids
        if variable_id in by_id and by_id[variable_id].library_type == "variable"
    ]
    output_id = str(math.get("output") or "")
    output = (
        _variable_payload(by_id[output_id])
        if output_id in by_id and by_id[output_id].library_type == "variable"
        else None
    )
    return EvidenceFormula(
        formula_id=entity.entity_id,
        source=entity.source,
        expression=str(math.get("expression") or execution.get("expression") or ""),
        variables=variables,
        output=output,
    )


def _variable_payload(entity: CodeEntity) -> EvidenceVariable:
    raw = entity.raw_payload
    name = str(raw.get("name") or entity.name or "")
    description = str(raw.get("definition") or raw.get("description") or "")
    return EvidenceVariable(
        symbol=str(raw.get("symbol") or entity.execution.get("symbol") or ""),
        name=name,
        definition=description if description != name else "",
        unit=str(raw.get("unit") or ""),
    )


def _table_payload(table_id: str, by_id: dict[str, CodeEntity]) -> EvidenceTable:
    entity = by_id[table_id]
    raw = entity.raw_payload
    structure = raw.get("structure") if isinstance(raw.get("structure"), dict) else {}
    schema = raw.get("schema") if isinstance(raw.get("schema"), dict) else {}
    units = structure.get("unit") or schema.get("unit") or schema.get("units") or {}
    return EvidenceTable(
        table_id=entity.entity_id,
        source=entity.source,
        title=entity.name,
        units={str(key): str(value) for key, value in units.items()} if isinstance(units, dict) else {},
        rows=[_sanitize_table_value(row, by_id) for row in _table_rows(entity)],
    )


def _render_prompt_text(
    *,
    entity: CodeEntity,
    normative_text: str,
    formulas: list[EvidenceFormula],
    tables: list[EvidenceTable],
) -> str:
    lines = [f"规范：{entity.standard_code} {entity.standard_name}".strip()]
    if entity.clause:
        lines.append(f"条文：{entity.clause}")
    if entity.name:
        lines.append(f"标题：{entity.name}")
    if normative_text:
        lines.append(f"规范内容：{normative_text}")
    for formula in formulas:
        lines.append(f"公式：{formula.expression}")
        definitions = []
        for variable in formula.variables:
            label = variable.name or variable.symbol
            detail = variable.definition or label
            unit = f"，单位 {variable.unit}" if variable.unit and variable.unit != "-" else ""
            definitions.append(f"{variable.symbol or label}：{detail}{unit}")
        if definitions:
            lines.append("变量：" + "；".join(definitions))
        if formula.output:
            output = formula.output
            lines.append(f"输出：{output.symbol or output.name}：{output.definition or output.name}")
    for table in tables:
        if table.title:
            lines.append(f"表格：{table.title}")
        if table.units:
            lines.append("单位：" + json.dumps(table.units, ensure_ascii=False, sort_keys=True))
        lines.append("数据：" + json.dumps(table.rows, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def _rule_exclusion_reason(
    entity: CodeEntity,
    by_id: dict[str, CodeEntity],
) -> str | None:
    text = _clean_text(entity.original_text)
    if not text:
        return "rule_without_normative_text"
    if text.startswith(("摘要：", "摘要:")):
        return "summary_rule"
    if any(phrase in text for phrase in _FORBIDDEN_RULE_PHRASES):
        return "subjective_or_workflow_text"
    if _normalized_rule_text(text, entity.clause) == _normalized_rule_text(entity.name, ""):
        return "title_only_rule"
    if not _has_valid_support(entity, by_id) and _is_generic_placeholder(entity, text):
        return "generic_placeholder_rule"
    return None


def _has_valid_support(entity: CodeEntity, by_id: dict[str, CodeEntity]) -> bool:
    return any(
        target in by_id and _is_normative_formula(by_id[target])
        for target in entity.related_entities.get("formulas", [])
    ) or any(
        target in by_id and bool(_table_rows(by_id[target]))
        for target in entity.related_entities.get("tables", [])
    )


def _valid_support(
    entity: CodeEntity,
    by_id: dict[str, CodeEntity],
) -> tuple[list[EvidenceFormula], list[EvidenceTable]]:
    formulas = [
        _formula_payload(target, by_id)
        for target in entity.related_entities.get("formulas", [])
        if target in by_id and _is_normative_formula(by_id[target])
    ]
    tables = [
        _table_payload(target, by_id)
        for target in entity.related_entities.get("tables", [])
        if target in by_id and _table_rows(by_id[target])
    ]
    return formulas, tables


def _is_generic_placeholder(entity: CodeEntity, text: str) -> bool:
    if re.search(r"\d+(?:\.\d+)?\s*(?:mm|cm|m|kN|MPa|GPa|%|年)\b", text, re.IGNORECASE):
        return False
    if any(
        phrase in text
        for phrase in ("相关规定", "符合本条规定", "规范限值要求", "相应公式计算")
    ):
        return True
    normalized_text = _normalized_rule_text(text, entity.clause)
    normalized_title = _normalized_rule_text(entity.name, "")
    similarity = SequenceMatcher(None, normalized_text, normalized_title).ratio()
    has_heading_prefix = bool(
        re.match(r"^(?:第?\d|附录[A-Z]|[A-Z]\.\d|\d+\.\d)", text, re.IGNORECASE)
    )
    return has_heading_prefix and len(text) <= 65 and similarity >= 0.55


def _is_normative_formula(entity: CodeEntity) -> bool:
    if entity.library_type != "formula":
        return False
    status = str(entity.raw_payload.get("formula_status") or "")
    math = entity.raw_payload.get("math") if isinstance(entity.raw_payload.get("math"), dict) else {}
    return status in _NORMATIVE_FORMULA_STATUSES and bool(math.get("expression"))


def _table_rows(entity: CodeEntity) -> list[dict[str, Any]]:
    rows = entity.raw_payload.get("data")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _sanitize_table_value(value: Any, by_id: dict[str, CodeEntity]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_table_value(item, by_id)
            for key, item in value.items()
            if not str(key).startswith("related_")
            and str(key) not in {"audit", "action", "execution", "review_note", "generation_note"}
        }
    if isinstance(value, list):
        return [_sanitize_table_value(item, by_id) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"(?<![A-Za-z0-9])(?:R|F|T|VAR)_[A-Za-z0-9_]+",
            lambda match: _entity_reference_label(match.group(0), by_id),
            value,
        )
    return value


def _entity_reference_label(entity_id: str, by_id: dict[str, CodeEntity]) -> str:
    entity = by_id.get(entity_id)
    if entity is None:
        return ""
    if entity.name:
        return entity.name
    symbol = str(entity.raw_payload.get("symbol") or "")
    return symbol or entity.source


def _safe_applicability(entity: CodeEntity) -> dict[str, Any]:
    allowed = {
        "stage",
        "object",
        "member_type",
        "member_types",
        "materials",
        "limit_state",
        "limit_states",
        "matching_method",
        "inputs",
        "outputs",
    }
    return {
        key: value
        for key, value in entity.applicability.items()
        if key in allowed and value not in [None, "", [], {}]
    }


def _normalized_rule_text(text: str, clause: str) -> str:
    value = _clean_text(text)
    if clause:
        value = re.sub(rf"^第?{re.escape(clause)}条?[：:]?", "", value)
    return re.sub(r"[\s，。；、,:：！？!?（）()《》]", "", value)


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _exclude(entity: CodeEntity, reason: str) -> EvidenceExclusion:
    return EvidenceExclusion(
        entity_id=entity.entity_id,
        library_type=entity.library_type,
        reason=reason,
    )
