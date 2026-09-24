from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


CodeLibraryType = Literal["rule", "formula", "table", "variable"]


class CodeLibraryMetadata(BaseModel):
    standard_code: str = ""
    standard_name: str = ""
    version: str = ""
    library_type: str = ""
    item_count: int | None = None
    source_path: str = ""
    raw_payload: Dict[str, Any] = Field(default_factory=dict)


class CodeEntity(BaseModel):
    entity_id: str
    library_type: CodeLibraryType
    standard_code: str
    standard_name: str = ""
    version: str = ""
    chapter: str = ""
    clause: str = ""
    name: str = ""
    source: str = ""
    source_file: str = ""
    original_text: str = ""
    description: str = ""
    applicability: Dict[str, Any] = Field(default_factory=dict)
    execution: Dict[str, Any] = Field(default_factory=dict)
    related_entities: Dict[str, List[str]] = Field(default_factory=dict)
    raw_payload: Dict[str, Any] = Field(default_factory=dict)

    def search_text(self) -> str:
        parts = [
            self.standard_code,
            self.standard_name,
            self.library_type,
            self.entity_id,
            self.chapter,
            self.clause,
            self.name,
            self.source,
            self.original_text,
            self.description,
        ]
        return "\n".join(str(part) for part in parts if part not in [None, "", [], {}])

    def semantic_text(self) -> str:
        applicability = _compact_dict(self.applicability)
        related = _compact_dict(self.related_entities)
        execution = _compact_dict(self.execution)
        parts = [
            f"规范: {self.standard_code} {self.standard_name}".strip(),
            f"实体类型: {self.library_type}",
            f"实体ID: {self.entity_id}",
            f"章节条文: {self.chapter} {self.clause}".strip(),
            f"名称: {self.name}",
            f"来源: {self.source}",
            f"正文: {self.original_text or self.description}",
            f"适用条件: {applicability}" if applicability else "",
            f"执行信息: {execution}" if execution else "",
            f"关联实体: {related}" if related else "",
        ]
        return "\n".join(part for part in parts if part)


class CodeLibrary(BaseModel):
    metadata: CodeLibraryMetadata
    entities: List[CodeEntity]


class AuditIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    entity_id: str | None = None


class AuditReport(BaseModel):
    source_path: str
    entity_count: int
    errors: List[AuditIssue] = Field(default_factory=list)
    warnings: List[AuditIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class CodeQuery(BaseModel):
    query_text: str
    standard_codes: List[str] = Field(default_factory=list)
    stage: str | None = None
    member_types: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    limit_states: List[str] = Field(default_factory=list)
    library_types: List[CodeLibraryType] = Field(default_factory=list)
    retrieval_mode: Literal["auto", "keyword_only", "vector_only", "hybrid"] = "auto"
    top_k: int = 10
    expanded_limit: int = 24


class EvidenceVariable(BaseModel):
    symbol: str = ""
    name: str = ""
    definition: str = ""
    unit: str = ""


class EvidenceFormula(BaseModel):
    formula_id: str
    source: str = ""
    expression: str
    variables: List[EvidenceVariable] = Field(default_factory=list)
    output: EvidenceVariable | None = None


class EvidenceTable(BaseModel):
    table_id: str
    source: str = ""
    title: str = ""
    units: Dict[str, str] = Field(default_factory=dict)
    rows: List[Dict[str, Any]] = Field(default_factory=list)


class EvidenceDocument(BaseModel):
    evidence_id: str
    primary_entity_id: str
    evidence_type: Literal["rule", "formula", "table", "composite"]
    standard_code: str
    standard_name: str = ""
    version: str = ""
    chapter: str = ""
    clause: str = ""
    title: str = ""
    source: str = ""
    normative_text: str = ""
    applicability: Dict[str, Any] = Field(default_factory=dict)
    formulas: List[EvidenceFormula] = Field(default_factory=list)
    tables: List[EvidenceTable] = Field(default_factory=list)
    source_entity_ids: List[str] = Field(default_factory=list)
    source_files: List[str] = Field(default_factory=list)
    prompt_text: str
    search_text: str
    content_status: Literal["prompt_ready"] = "prompt_ready"
    source_verification: Literal["not_verified_against_official_text"] = (
        "not_verified_against_official_text"
    )
    keyword_rank: int | None = None
    vector_rank: int | None = None
    fused_rank: int | None = None


class RetrievalTrace(BaseModel):
    retrieval_mode: Literal["keyword_only", "vector_only", "hybrid"] = "keyword_only"
    keyword_result_ids: List[str] = Field(default_factory=list)
    expanded_relation_count: int = 0
    vector_result_ids: List[str] = Field(default_factory=list)


class CodeEvidenceBundle(BaseModel):
    request: CodeQuery
    retrieval_status: Literal["ok", "no_valid_evidence", "invalid_index"]
    index_version: str = ""
    evidence_documents: List[EvidenceDocument] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace = Field(default_factory=RetrievalTrace)


def _compact_dict(value: Dict[str, Any]) -> str:
    items: list[str] = []
    for key, item in value.items():
        if item in [None, "", [], {}]:
            continue
        items.append(f"{key}={item}")
    return "; ".join(items)
