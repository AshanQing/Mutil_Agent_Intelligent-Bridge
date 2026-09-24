from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .embedding import SentenceTransformerBackend
from .hybrid import reciprocal_rank_fusion
from .paths import DEFAULT_DB_PATH, DEFAULT_EVIDENCE_DOCUMENTS_PATH
from .query import query_keyword_index
from .schemas import CodeEvidenceBundle, CodeQuery, EvidenceDocument, RetrievalTrace
from .vector_index import EmbeddingBackend, VectorIndexError, query_vector_index


class CodeRAGService:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        *,
        evidence_path: Path | None = None,
        index_dir: Path | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.index_dir = Path(index_dir) if index_dir is not None else self.db_path.parent
        self.evidence_path = (
            Path(evidence_path)
            if evidence_path is not None
            else self._evidence_path_from_manifest()
        )
        self.embedding_backend = embedding_backend

    def _evidence_path_from_manifest(self) -> Path:
        manifest_path = self.index_dir / "index_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return DEFAULT_EVIDENCE_DOCUMENTS_PATH
        value = str(manifest.get("evidence_file") or "")
        return Path(value) if value else DEFAULT_EVIDENCE_DOCUMENTS_PATH

    def validate_index(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"ok": False, "reason": "index_not_found", "db_path": str(self.db_path)}

        con = sqlite3.connect(str(self.db_path))
        try:
            evidence_count = int(
                con.execute("SELECT COUNT(*) FROM evidence_documents").fetchone()[0]
            )
            fts_count = int(con.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0])
            source_entity_count = int(
                con.execute("SELECT COUNT(*) FROM source_entities").fetchone()[0]
            )
            metadata = dict(con.execute("SELECT key, value FROM index_metadata").fetchall())
            source_rows = con.execute(
                "SELECT source_file, sha256 FROM source_files ORDER BY source_file"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            return {
                "ok": False,
                "reason": "invalid_schema",
                "db_path": str(self.db_path),
                "detail": str(exc),
            }
        finally:
            con.close()

        stale_sources = []
        for source_file, expected_hash in source_rows:
            path = Path(source_file)
            actual_hash = _sha256_file(path) if path.exists() else None
            if actual_hash != expected_hash:
                stale_sources.append(
                    {
                        "source_file": source_file,
                        "expected_sha256": expected_hash,
                        "actual_sha256": actual_hash,
                    }
                )
        vector_errors = self._validate_vector_files(evidence_count)
        counts_ok = evidence_count > 0 and evidence_count == fts_count
        reason = None
        if stale_sources:
            reason = "stale_source_files"
        elif vector_errors:
            reason = "invalid_vector_index"
        return {
            "ok": counts_ok and not stale_sources and not vector_errors,
            "reason": reason,
            "db_path": str(self.db_path),
            "evidence_document_count": evidence_count,
            "fts_count": fts_count,
            "source_entity_count": source_entity_count,
            "source_file_count": len(source_rows),
            "stale_sources": stale_sources,
            "vector_errors": vector_errors,
            "schema_version": metadata.get("schema_version", ""),
            "built_at": metadata.get("built_at", ""),
        }

    def _validate_vector_files(self, evidence_count: int) -> list[str]:
        manifest_path = self.index_dir / "index_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            return [f"invalid_manifest:{exc}"]
        if not manifest.get("embedding_model"):
            return []

        embedding_path = self.index_dir / "embeddings.npy"
        ids_path = self.index_dir / "embedding_ids.json"
        errors: list[str] = []
        for path, hash_key in (
            (embedding_path, "embedding_sha256"),
            (ids_path, "embedding_ids_sha256"),
        ):
            if not path.exists():
                errors.append(f"missing_file:{path.name}")
            elif _sha256_file(path) != manifest.get(hash_key):
                errors.append(f"hash_mismatch:{path.name}")
        if errors:
            return errors

        try:
            vectors = np.load(embedding_path, allow_pickle=False)
            ids_payload = json.loads(ids_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return [f"unreadable_vector_index:{exc}"]
        evidence_ids = list(ids_payload.get("evidence_ids") or [])
        expected_dimension = int(manifest.get("embedding_dimension") or 0)
        if vectors.shape != (evidence_count, expected_dimension):
            errors.append(
                f"shape_mismatch:{vectors.shape}!={(evidence_count, expected_dimension)}"
            )
        if len(evidence_ids) != evidence_count or len(evidence_ids) != len(set(evidence_ids)):
            errors.append("evidence_id_count_or_uniqueness_mismatch")
        if ids_payload.get("embedding_model") != manifest.get("embedding_model"):
            errors.append("embedding_model_mismatch")
        if ids_payload.get("evidence_file_sha256") != _sha256_file(self.evidence_path):
            errors.append("evidence_hash_mismatch")
        return errors

    def retrieve(
        self,
        request: CodeQuery,
        *,
        expand_relations: bool = True,
    ) -> CodeEvidenceBundle:
        del expand_relations  # Relations are assembled offline into each evidence document.
        validation = self.validate_index()
        if not validation["ok"]:
            return CodeEvidenceBundle(
                request=request,
                retrieval_status="invalid_index",
                index_version=str(validation.get("schema_version") or ""),
            )

        mode = self._resolve_retrieval_mode(request.retrieval_mode)
        if mode in {"vector_only", "hybrid"} and self.embedding_backend is None:
            self.embedding_backend = self._load_default_embedding_backend()
        if mode in {"vector_only", "hybrid"} and self.embedding_backend is None:
            return CodeEvidenceBundle(
                request=request,
                retrieval_status="invalid_index",
                index_version=f"{validation['schema_version']}@{validation['built_at']}",
            )

        candidate_limit = max(50, request.top_k * 5)
        keyword_rows = []
        if mode in {"keyword_only", "hybrid"}:
            keyword_rows = query_keyword_index(
                db_path=self.db_path,
                text=request.query_text,
                standards=request.standard_codes,
                library_types=list(request.library_types),
                top_k=candidate_limit,
            )
        keyword_documents = {
            document.evidence_id: document
            for document in _apply_structured_filters(
                [_document_from_keyword_row(row) for row in keyword_rows], request
            )
        }

        vector_rows: list[dict[str, Any]] = []
        vector_documents: dict[str, EvidenceDocument] = {}
        if mode in {"vector_only", "hybrid"}:
            try:
                all_vector_rows = query_vector_index(
                    query_text=request.query_text,
                    evidence_path=self.evidence_path,
                    index_dir=self.index_dir,
                    backend=self.embedding_backend,
                    top_k=100000,
                )
            except (
                FileNotFoundError,
                VectorIndexError,
                ValueError,
                RuntimeError,
                json.JSONDecodeError,
            ):
                return CodeEvidenceBundle(
                    request=request,
                    retrieval_status="invalid_index",
                    index_version=f"{validation['schema_version']}@{validation['built_at']}",
                )
            loaded_documents = self._load_evidence_documents(
                [str(row["evidence_id"]) for row in all_vector_rows]
            )
            filtered_documents = _apply_structured_filters(
                [
                    document
                    for document in loaded_documents
                    if _matches_basic_filters(document, request)
                ],
                request,
            )
            allowed_ids = {document.evidence_id for document in filtered_documents}
            vector_rows = [
                row for row in all_vector_rows if row["evidence_id"] in allowed_ids
            ][:candidate_limit]
            ranks = {str(row["evidence_id"]): int(row["rank"]) for row in vector_rows}
            vector_documents = {
                document.evidence_id: document.model_copy(
                    update={"vector_rank": ranks[document.evidence_id]}
                )
                for document in filtered_documents
                if document.evidence_id in ranks
            }

        documents = _rank_documents(
            mode=mode,
            keyword_documents=keyword_documents,
            vector_documents=vector_documents,
            top_k=candidate_limit,
        )
        documents = _apply_query_concept_gate(documents, request.query_text)[: request.top_k]
        return CodeEvidenceBundle(
            request=request,
            retrieval_status="ok" if documents else "no_valid_evidence",
            index_version=f"{validation['schema_version']}@{validation['built_at']}",
            evidence_documents=documents,
            retrieval_trace=RetrievalTrace(
                retrieval_mode=mode,
                keyword_result_ids=list(keyword_documents),
                vector_result_ids=[str(row["evidence_id"]) for row in vector_rows],
            ),
        )

    def _resolve_retrieval_mode(self, requested: str) -> str:
        if requested != "auto":
            return requested
        vector_files_exist = all(
            (self.index_dir / filename).exists()
            for filename in ("embeddings.npy", "embedding_ids.json")
        )
        return "hybrid" if vector_files_exist else "keyword_only"

    def _load_default_embedding_backend(self) -> EmbeddingBackend | None:
        manifest_path = self.index_dir / "index_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        model_name = str(manifest.get("embedding_model") or "")
        return SentenceTransformerBackend(model_name, offline=True) if model_name else None

    def _load_evidence_documents(self, evidence_ids: list[str]) -> list[EvidenceDocument]:
        if not evidence_ids:
            return []
        con = sqlite3.connect(str(self.db_path))
        try:
            rows = dict(
                con.execute(
                    "SELECT evidence_id, document_json FROM evidence_documents"
                ).fetchall()
            )
        finally:
            con.close()
        return [
            EvidenceDocument.model_validate(json.loads(rows[evidence_id]))
            for evidence_id in evidence_ids
            if evidence_id in rows
        ]

    def get_evidence(self, evidence_id: str) -> EvidenceDocument | None:
        con = sqlite3.connect(str(self.db_path))
        try:
            row = con.execute(
                "SELECT document_json FROM evidence_documents WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
        finally:
            con.close()
        return EvidenceDocument.model_validate(json.loads(row[0])) if row else None

    def get_source_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Return source graph data for audit tooling, never for prompt injection."""
        con = sqlite3.connect(str(self.db_path))
        try:
            row = con.execute(
                """
                SELECT normalized_entity_json, raw_payload_json
                FROM source_entities WHERE entity_id = ?
                """,
                (entity_id,),
            ).fetchone()
        finally:
            con.close()
        if not row:
            return None
        return {"entity": json.loads(row[0]), "raw_payload": json.loads(row[1])}


def render_evidence_summary(bundle: CodeEvidenceBundle) -> str:
    lines = [
        f"查询：{bundle.request.query_text}",
        f"状态：{bundle.retrieval_status}",
        f"索引：{bundle.index_version}",
        "",
        "规范证据：",
    ]
    for document in bundle.evidence_documents:
        lines.append("")
        lines.append(
            f"[{document.keyword_rank}] {document.standard_code} "
            f"{document.clause} {document.title}".strip()
        )
        lines.append(document.prompt_text)
        lines.append(f"证据ID：{document.evidence_id}；来源：{document.source}")
    return "\n".join(lines)


def _apply_structured_filters(
    documents: list[EvidenceDocument],
    request: CodeQuery,
) -> list[EvidenceDocument]:
    if not any([request.stage, request.member_types, request.materials, request.limit_states]):
        return documents

    filtered = []
    for document in documents:
        values = _flatten_applicability(document.applicability)
        if request.stage and values and request.stage not in values:
            continue
        if request.member_types and values and not any(value in values for value in request.member_types):
            continue
        if request.materials and values and not any(value in values for value in request.materials):
            continue
        if request.limit_states and values and not any(value in values for value in request.limit_states):
            continue
        filtered.append(document)
    return filtered


def _document_from_keyword_row(row: dict[str, Any]) -> EvidenceDocument:
    return EvidenceDocument.model_validate(
        {key: value for key, value in row.items() if key not in {"score", "rank"}}
        | {"keyword_rank": row.get("rank")}
    )


def _matches_basic_filters(document: EvidenceDocument, request: CodeQuery) -> bool:
    if request.standard_codes and document.standard_code not in request.standard_codes:
        return False
    if not request.library_types:
        return True
    evidence_types = set(request.library_types)
    if "rule" in evidence_types:
        evidence_types.add("composite")
    return document.evidence_type in evidence_types


def _rank_documents(
    *,
    mode: str,
    keyword_documents: dict[str, EvidenceDocument],
    vector_documents: dict[str, EvidenceDocument],
    top_k: int,
) -> list[EvidenceDocument]:
    if mode == "keyword_only":
        return list(keyword_documents.values())[:top_k]
    if mode == "vector_only":
        return list(vector_documents.values())[:top_k]

    fused = reciprocal_rank_fusion(
        keyword_ids=list(keyword_documents),
        vector_ids=list(vector_documents),
        top_k=top_k,
    )
    documents = keyword_documents | vector_documents
    return [
        documents[str(row["evidence_id"])].model_copy(
            update={
                "keyword_rank": row["keyword_rank"],
                "vector_rank": row["vector_rank"],
                "fused_rank": row["rank"],
            }
        )
        for row in fused
    ]


def _flatten_applicability(value: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in value.values():
        if isinstance(item, list):
            parts.extend(str(value) for value in item)
        elif item not in [None, "", {}, []]:
            parts.append(str(item))
    return "\n".join(parts)


def _apply_query_concept_gate(
    documents: list[EvidenceDocument],
    query_text: str,
) -> list[EvidenceDocument]:
    for trigger, required, alternatives in _QUERY_CONCEPT_GATES:
        if trigger not in query_text:
            continue
        return [
            document
            for document in documents
            if all(term in document.prompt_text for term in required)
            and any(term in document.prompt_text for term in alternatives)
        ]
    return documents


_QUERY_CONCEPT_GATES = (
    ("盖梁尺寸", ("盖梁",), ("尺寸", "宽度", "高度")),
    ("盖梁正截面受弯承载力", ("盖梁",), ("正截面", "受弯承载力")),
    ("风荷载标准值", ("风荷载",), ("标准值", "计算式", "公式")),
    ("桥涵建筑限界与桥面净宽", ("建筑限界",), ("桥面净宽", "净宽")),
    ("通航水域桥梁布置", ("通航",), ("桥梁布置", "布置要求")),
    ("通航水域桥墩布置", ("通航", "桥墩"), ("布置", "防撞")),
    ("受弯构件使用阶段挠度限值", ("受弯构件",), ("挠度限值", "最大挠度")),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
