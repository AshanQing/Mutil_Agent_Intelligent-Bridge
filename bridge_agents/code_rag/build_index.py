from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .paths import (
    DEFAULT_EVIDENCE_DOCUMENTS_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_SOURCE_ENTITIES_PATH,
)


SCHEMA_VERSION = "code_rag_sqlite_v0.4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the prompt-ready code evidence FTS5 index.")
    parser.add_argument(
        "--evidence",
        default=str(DEFAULT_EVIDENCE_DOCUMENTS_PATH),
        help="Path to evidence_documents.jsonl.",
    )
    parser.add_argument(
        "--source-entities",
        default=str(DEFAULT_SOURCE_ENTITIES_PATH),
        help="Path to source_entities.jsonl used for traceability only.",
    )
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Index output directory.")
    args = parser.parse_args()

    result = build_keyword_index(
        evidence_path=Path(args.evidence),
        source_entities_path=Path(args.source_entities),
        index_dir=Path(args.index_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_keyword_index(
    *,
    evidence_path: Path = DEFAULT_EVIDENCE_DOCUMENTS_PATH,
    source_entities_path: Path | None = None,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> dict[str, Any]:
    if not evidence_path.exists():
        raise FileNotFoundError(f"Evidence document file does not exist: {evidence_path}")
    if source_entities_path is None:
        source_entities_path = evidence_path.parent.parent / "source" / "source_entities.jsonl"
    if not source_entities_path.exists():
        raise FileNotFoundError(f"Source entity file does not exist: {source_entities_path}")

    documents = list(_read_jsonl(evidence_path))
    source_entities = list(_read_jsonl(source_entities_path))
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "code_knowledge.db"
    manifest_path = index_dir / "index_manifest.json"

    with tempfile.TemporaryDirectory(prefix="code_rag_index_") as temp_dir:
        temp_db_path = Path(temp_dir) / "code_knowledge.db"
        stats = _build_sqlite_db(temp_db_path, documents, source_entities)
        _validate_db(temp_db_path, expected_evidence_count=len(documents))
        shutil.copy2(temp_db_path, db_path)

    for stale_vector_file in ("embeddings.npy", "embedding_ids.json"):
        stale_path = index_dir / stale_vector_file
        if stale_path.exists():
            stale_path.unlink()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at": _utc_now(),
        "database_path": str(db_path),
        "database_sha256": _sha256_file(db_path),
        "evidence_file": str(evidence_path),
        "evidence_file_sha256": _sha256_file(evidence_path),
        "source_entities_file": str(source_entities_path),
        "source_entities_file_sha256": _sha256_file(source_entities_path),
        "evidence_document_count": len(documents),
        "source_entity_count": len(source_entities),
        "relation_count": stats["relation_count"],
        "source_file_count": stats["source_file_count"],
        "source_file_hashes": stats["source_file_hashes"],
        "embedding_model": None,
        "embedding_dimension": None,
        "fts5": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "database_path": str(db_path),
        "manifest_path": str(manifest_path),
        "evidence_document_count": len(documents),
        "source_entity_count": len(source_entities),
        "relation_count": stats["relation_count"],
        "source_file_count": stats["source_file_count"],
        "database_sha256": manifest["database_sha256"],
    }


def _build_sqlite_db(
    db_path: Path,
    documents: list[dict[str, Any]],
    source_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(
            """
            CREATE TABLE source_entities (
                entity_id TEXT PRIMARY KEY,
                library_type TEXT NOT NULL,
                standard_code TEXT NOT NULL,
                source_file TEXT NOT NULL,
                normalized_entity_json TEXT NOT NULL,
                raw_payload_json TEXT NOT NULL
            );

            CREATE TABLE relations (
                source_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_entity_id TEXT NOT NULL
            );

            CREATE TABLE evidence_documents (
                evidence_id TEXT PRIMARY KEY,
                primary_entity_id TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                standard_code TEXT NOT NULL,
                standard_name TEXT,
                version TEXT,
                chapter TEXT,
                clause TEXT,
                title TEXT,
                source TEXT,
                prompt_text TEXT NOT NULL,
                search_text TEXT NOT NULL,
                document_json TEXT NOT NULL
            );

            CREATE TABLE source_files (
                source_file TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                entity_count INTEGER NOT NULL
            );

            CREATE TABLE index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE evidence_fts USING fts5(
                evidence_id UNINDEXED,
                primary_entity_id,
                standard_code,
                evidence_type,
                clause,
                title,
                prompt_text,
                search_text,
                tokenize='unicode61'
            );
            """
        )
        source_counts: dict[str, int] = {}
        for chunk in source_chunks:
            entity = chunk.get("entity") if isinstance(chunk.get("entity"), dict) else {}
            raw_payload = chunk.get("raw_payload") if isinstance(chunk.get("raw_payload"), dict) else {}
            source_file = str(entity.get("source_file") or "")
            con.execute(
                """
                INSERT INTO source_entities (
                    entity_id, library_type, standard_code, source_file,
                    normalized_entity_json, raw_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(entity.get("entity_id") or ""),
                    str(entity.get("library_type") or ""),
                    str(entity.get("standard_code") or ""),
                    source_file,
                    json.dumps(entity, ensure_ascii=False, sort_keys=True),
                    json.dumps(raw_payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            source_counts[source_file] = source_counts.get(source_file, 0) + 1
            related = entity.get("related_entities") if isinstance(entity.get("related_entities"), dict) else {}
            for relation_type, targets in related.items():
                for target in targets if isinstance(targets, list) else []:
                    con.execute(
                        "INSERT INTO relations VALUES (?, ?, ?)",
                        (str(entity.get("entity_id") or ""), str(relation_type), str(target)),
                    )

        for document in documents:
            values = (
                str(document.get("evidence_id") or ""),
                str(document.get("primary_entity_id") or ""),
                str(document.get("evidence_type") or ""),
                str(document.get("standard_code") or ""),
                str(document.get("standard_name") or ""),
                str(document.get("version") or ""),
                str(document.get("chapter") or ""),
                str(document.get("clause") or ""),
                str(document.get("title") or ""),
                str(document.get("source") or ""),
                str(document.get("prompt_text") or ""),
                str(document.get("search_text") or ""),
                json.dumps(document, ensure_ascii=False, sort_keys=True),
            )
            con.execute(
                """
                INSERT INTO evidence_documents (
                    evidence_id, primary_entity_id, evidence_type, standard_code,
                    standard_name, version, chapter, clause, title, source,
                    prompt_text, search_text, document_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            con.execute(
                "INSERT INTO evidence_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    values[0], values[1], values[3], values[2], values[7],
                    values[8], values[10], values[11],
                ),
            )

        source_hashes = {
            source_file: _sha256_file(Path(source_file))
            for source_file in sorted(source_counts)
            if source_file and Path(source_file).exists()
        }
        con.executemany(
            "INSERT INTO source_files VALUES (?, ?, ?)",
            [
                (path, source_hashes[path], source_counts[path])
                for path in sorted(source_hashes)
            ],
        )
        relation_count = int(con.execute("SELECT COUNT(*) FROM relations").fetchone()[0])
        con.executemany(
            "INSERT INTO index_metadata VALUES (?, ?)",
            [
                ("schema_version", SCHEMA_VERSION),
                ("built_at", _utc_now()),
                ("evidence_document_count", str(len(documents))),
                ("source_entity_count", str(len(source_chunks))),
                ("relation_count", str(relation_count)),
            ],
        )
        con.commit()
        return {
            "relation_count": relation_count,
            "source_file_count": len(source_hashes),
            "source_file_hashes": source_hashes,
        }
    finally:
        con.close()


def _validate_db(db_path: Path, *, expected_evidence_count: int) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        evidence_count = int(con.execute("SELECT COUNT(*) FROM evidence_documents").fetchone()[0])
        fts_count = int(con.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0])
        if evidence_count != expected_evidence_count or fts_count != expected_evidence_count:
            raise RuntimeError(
                f"Evidence index count mismatch: documents={evidence_count}, fts={fts_count}, "
                f"expected={expected_evidence_count}"
            )
        leaking = int(
            con.execute(
                "SELECT COUNT(*) FROM evidence_documents WHERE document_json LIKE '%raw_payload%'"
            ).fetchone()[0]
        )
        if leaking:
            raise RuntimeError(f"Evidence documents contain raw payload fields: {leaking}")
        con.execute("SELECT rowid FROM evidence_fts WHERE evidence_fts MATCH ? LIMIT 1", ("盖梁",)).fetchall()
    finally:
        con.close()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
