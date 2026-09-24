from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import yaml
from pydantic import BaseModel, Field

from .code_rag.schemas import CodeQuery, EvidenceDocument
from .code_rag.service import CodeRAGService


StructuralEvidenceStage = Literal["dimension_design", "reinforcement_design"]


class StructuralEvidenceUnavailableError(RuntimeError):
    """Raised when RAG was explicitly enabled but its index cannot be used."""


class StructuralRAGSettings(BaseModel):
    enabled: bool = False
    retrieval_mode: Literal["auto", "keyword_only", "vector_only", "hybrid"] = "auto"
    top_k: int = Field(default=4, ge=1)
    max_evidence_per_task: int = Field(default=8, ge=1)


class StructuralEvidenceContext(BaseModel):
    task_id: str
    stage: StructuralEvidenceStage
    retrieval_status: Literal["disabled", "ok", "no_valid_evidence"]
    queries: List[CodeQuery] = Field(default_factory=list)
    evidence_documents: List[EvidenceDocument] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    excluded_evidence_ids: List[str] = Field(default_factory=list)
    prompt_text: str = ""
    index_version: str = ""
    query_hash: str = ""
    retrieval_trace: List[Dict[str, Any]] = Field(default_factory=list)


def evidence_context_to_prompt_payload(
    context: StructuralEvidenceContext | Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    if context is None:
        return None
    normalized = (
        context
        if isinstance(context, StructuralEvidenceContext)
        else StructuralEvidenceContext.model_validate(context)
    )
    if normalized.retrieval_status == "disabled":
        return None
    return {
        "retrieval_status": normalized.retrieval_status,
        "evidence_ids": list(normalized.evidence_ids),
        "index_version": normalized.index_version,
        "query_hash": normalized.query_hash,
        "content": normalized.prompt_text,
    }


def load_structural_rag_settings(config_path: str) -> StructuralRAGSettings:
    path = Path(config_path)
    if not path.exists():
        return StructuralRAGSettings()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("code_rag") if isinstance(payload, dict) else {}
    return StructuralRAGSettings.model_validate(section or {})


def build_structural_queries(
    task: Dict[str, Any],
    *,
    stage: StructuralEvidenceStage,
    settings: StructuralRAGSettings,
) -> List[CodeQuery]:
    context_terms = _task_context_terms(task)
    if stage == "dimension_design":
        intents = [
            "混凝土强度设计值表",
            "普通钢筋最小混凝土保护层厚度",
        ]
        library_types = ["rule", "table"]
    elif stage == "reinforcement_design":
        intents = [
            "普通钢筋混凝土盖梁最小保护层厚度",
            "受拉钢筋最小锚固长度",
            "盖梁斜截面抗剪承载力验算",
            "钢筋混凝土轴心受压构件正截面承载力",
            "轴心受压构件稳定系数",
            "螺旋箍筋约束混凝土轴心受压承载力",
        ]
        library_types = ["rule", "formula", "table"]
    else:  # pragma: no cover - Literal plus public validation guard
        raise ValueError(f"不支持的结构设计规范检索阶段: {stage}")

    return [
        CodeQuery(
            query_text=" ".join(part for part in [intent, context_terms] if part),
            standard_codes=["JTG 3362-2018"],
            library_types=library_types,
            retrieval_mode=settings.retrieval_mode,
            top_k=settings.top_k,
        )
        for intent in intents
    ]


def retrieve_structural_evidence(
    task: Dict[str, Any],
    *,
    stage: StructuralEvidenceStage,
    config_path: str,
    service: Any | None = None,
) -> StructuralEvidenceContext:
    settings = load_structural_rag_settings(config_path)
    task_id = _task_id(task, stage)
    if not settings.enabled:
        return StructuralEvidenceContext(
            task_id=task_id,
            stage=stage,
            retrieval_status="disabled",
        )

    queries = build_structural_queries(task, stage=stage, settings=settings)
    rag_service = service or CodeRAGService()
    documents: List[EvidenceDocument] = []
    seen_ids: set[str] = set()
    excluded_ids: List[str] = []
    traces: List[Dict[str, Any]] = []
    index_versions: List[str] = []

    for request in queries:
        bundle = rag_service.retrieve(request)
        traces.append({
            "query_text": request.query_text,
            "retrieval_status": bundle.retrieval_status,
            "retrieval_trace": bundle.retrieval_trace.model_dump(),
        })
        if bundle.retrieval_status == "invalid_index":
            raise StructuralEvidenceUnavailableError(
                f"规范RAG索引不可用，阶段={stage}，任务={task_id}。"
            )
        if bundle.index_version and bundle.index_version not in index_versions:
            index_versions.append(bundle.index_version)
        for document in bundle.evidence_documents:
            if not _is_applicable(document):
                if document.evidence_id not in excluded_ids:
                    excluded_ids.append(document.evidence_id)
                continue
            if document.evidence_id in seen_ids:
                continue
            seen_ids.add(document.evidence_id)
            documents.append(document)
            if len(documents) >= settings.max_evidence_per_task:
                break
        if len(documents) >= settings.max_evidence_per_task:
            break

    status = "ok" if documents else "no_valid_evidence"
    request_payload = [request.model_dump(mode="json") for request in queries]
    query_hash = hashlib.sha256(
        json.dumps(request_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return StructuralEvidenceContext(
        task_id=task_id,
        stage=stage,
        retrieval_status=status,
        queries=queries,
        evidence_documents=documents,
        evidence_ids=[document.evidence_id for document in documents],
        excluded_evidence_ids=excluded_ids,
        prompt_text=_render_prompt_text(documents, status=status),
        index_version="|".join(index_versions),
        query_hash=query_hash,
        retrieval_trace=traces,
    )


def _task_id(task: Dict[str, Any], stage: str) -> str:
    for key in ("task_id", "单元编号", "分组编号", "桥梁编号"):
        value = task.get(key)
        if value not in [None, ""]:
            return str(value)
    return f"{stage}-unknown"


def _task_context_terms(task: Dict[str, Any]) -> str:
    values: List[str] = []
    nested = task.get("桥墩尺寸信息")
    if isinstance(nested, dict):
        basic = nested.get("基本信息")
        if isinstance(basic, dict):
            for key in ("system_type", "pier_role", "concrete_grade", "rebar_grade"):
                value = basic.get(key)
                if value not in [None, ""]:
                    values.append(str(value))
    return " ".join(values)


def _is_applicable(document: EvidenceDocument) -> bool:
    applicability = document.applicability or {}
    evidence_text = "\n".join(
        value
        for value in (document.title, document.normative_text, document.prompt_text)
        if value
    )
    if "预应力钢筋" in evidence_text and "普通钢筋" not in evidence_text:
        return False
    materials = _applicability_values(applicability, ("material", "materials", "applicable_material"))
    if materials and "reinforced_concrete" not in materials:
        return False
    objects = _applicability_values(applicability, ("object", "objects"))
    if objects:
        object_text = "\n".join(objects)
        prestress_only = (
            ("预应力" in object_text or "prestressed_concrete" in object_text)
            and "钢筋混凝土" not in object_text
            and "reinforced_concrete" not in object_text
        )
        if prestress_only:
            return False
    members = _applicability_values(
        applicability,
        ("member_type", "member_types", "applicable_member"),
    )
    allowed_member_markers = (
        "盖梁",
        "cap_beam",
        "一般规定",
        "受弯构件",
        "混凝土构件",
        "墩柱",
        "受压构件",
        "轴心受压",
        "柱式墩",
        "pier_column",
        "column",
    )
    if members and not any(
        marker in member
        for member in members
        for marker in allowed_member_markers
    ):
        return False
    return True


def _applicability_values(payload: Dict[str, Any], keys: Sequence[str]) -> set[str]:
    values: set[str] = set()
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            values.update(str(item) for item in value)
        elif value not in [None, "", {}, []]:
            values.add(str(value))
    return values


def _render_prompt_text(
    documents: Sequence[EvidenceDocument],
    *,
    status: str,
) -> str:
    if status != "ok":
        return "本任务未检索到适用的规范证据，不得据此声称满足规范。"
    lines = [
        "## 规范证据",
        "以下内容来自项目规范四库。只可在其适用条件范围内使用，并在输出中保留证据ID。",
    ]
    for index, document in enumerate(documents, start=1):
        lines.extend([
            f"### 证据{index}：{document.evidence_id}",
            document.prompt_text,
            f"来源：{document.source}",
        ])
    return "\n".join(lines)
