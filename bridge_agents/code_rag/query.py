from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from .paths import DEFAULT_DB_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the SQLite FTS5 code RAG index.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to code_knowledge.db.")
    parser.add_argument("--text", required=True, help="Keyword query text.")
    parser.add_argument("--standard", action="append", default=[], help="Optional standard code filter.")
    parser.add_argument("--type", action="append", default=[], help="Optional evidence type: rule/formula/table.")
    parser.add_argument("--stage", help="Optional design stage filter.")
    parser.add_argument("--member", action="append", default=[], help="Optional member type filter.")
    parser.add_argument("--material", action="append", default=[], help="Optional material filter.")
    parser.add_argument("--limit-state", action="append", default=[], help="Optional limit-state filter.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results.")
    parser.add_argument(
        "--mode",
        choices=["auto", "keyword_only", "vector_only", "hybrid"],
        default="auto",
        help="Retrieval mode. Auto uses hybrid retrieval when a vector index exists.",
    )
    parser.add_argument(
        "--format",
        choices=["summary", "json", "rows"],
        default="summary",
        help="Output a readable evidence summary, structured bundle JSON, or ranked evidence rows.",
    )
    parser.add_argument("--no-expand", action="store_true", help="Do not expand related entities.")
    args = parser.parse_args()

    from .schemas import CodeQuery
    from .service import CodeRAGService, render_evidence_summary

    request = CodeQuery(
        query_text=args.text,
        standard_codes=args.standard,
        stage=args.stage,
        member_types=args.member,
        materials=args.material,
        limit_states=args.limit_state,
        library_types=args.type,
        retrieval_mode=args.mode,
        top_k=args.top_k,
    )
    bundle = CodeRAGService(Path(args.db)).retrieve(
        request,
        expand_relations=not args.no_expand,
    )
    if args.format == "rows":
        print(
            json.dumps(
                [document.model_dump() for document in bundle.evidence_documents],
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.format == "json":
        print(bundle.model_dump_json(indent=2))
    else:
        print(render_evidence_summary(bundle))


def query_keyword_index(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    text: str,
    standards: List[str] | None = None,
    library_types: List[str] | None = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(f"Index database does not exist: {db_path}")
    terms = _query_terms(text)
    query = _fts_query_from_terms(terms)
    evidence_types = _expand_evidence_types(library_types or [])
    if not query and not terms:
        return []

    con = sqlite3.connect(str(db_path))
    try:
        fts_rows = _query_fts(
            con,
            query=query,
            standards=standards or [],
            library_types=evidence_types,
            top_k=top_k,
        ) if query else []
        like_rows = _query_like(
            con,
            terms=terms,
            standards=standards or [],
            library_types=evidence_types,
            top_k=max(top_k * 3, 20),
        )
    finally:
        con.close()

    merged: dict[str, Dict[str, Any]] = {}
    for row in fts_rows + like_rows:
        existing = merged.get(row["evidence_id"])
        if existing is None:
            merged[row["evidence_id"]] = row
        else:
            existing["rank_score"] += row["rank_score"]
            existing["score"] = min(existing["score"], row["score"])

    ordered = sorted(
        merged.values(),
        key=lambda row: (-row["rank_score"], row["sort_score"], row["evidence_id"]),
    )
    return [
        {key: value for key, value in row.items() if key not in {"rank_score", "sort_score"}}
        | {"rank": index}
        for index, row in enumerate(ordered[:top_k], start=1)
    ]


def _query_fts(
    con: sqlite3.Connection,
    *,
    query: str,
    standards: List[str],
    library_types: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    clauses = ["evidence_fts MATCH ?"]
    params: list[Any] = [query]
    standards = standards or []
    library_types = library_types or []
    if standards:
        clauses.append(f"e.standard_code IN ({','.join('?' for _ in standards)})")
        params.extend(standards)
    if library_types:
        clauses.append(f"e.evidence_type IN ({','.join('?' for _ in library_types)})")
        params.extend(library_types)
    params.append(int(top_k))

    sql = f"""
        SELECT
            e.evidence_id,
            e.primary_entity_id,
            e.evidence_type,
            e.standard_code,
            e.title,
            e.source,
            e.document_json,
            bm25(evidence_fts) AS score
        FROM evidence_fts
        JOIN evidence_documents e ON e.evidence_id = evidence_fts.evidence_id
        WHERE {' AND '.join(clauses)}
        ORDER BY score ASC, e.evidence_id ASC
        LIMIT ?
    """
    try:
        rows = con.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        {
            "score": row[7],
            "rank_score": 10.0 / rank,
            "sort_score": float(row[7] or 0),
            **json.loads(row[6]),
        }
        for rank, row in enumerate(rows, start=1)
    ]


def _query_like(
    con: sqlite3.Connection,
    *,
    terms: List[str],
    standards: List[str],
    library_types: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not terms:
        return []

    clauses = []
    params: list[Any] = []
    for term in terms:
        clauses.append("(e.search_text LIKE ? OR e.prompt_text LIKE ? OR e.title LIKE ? OR e.clause LIKE ?)")
        like = f"%{term}%"
        params.extend([like, like, like, like])
    where = [f"({' OR '.join(clauses)})"]
    if standards:
        where.append(f"e.standard_code IN ({','.join('?' for _ in standards)})")
        params.extend(standards)
    if library_types:
        where.append(f"e.evidence_type IN ({','.join('?' for _ in library_types)})")
        params.extend(library_types)

    sql = f"""
        SELECT
            e.evidence_id,
            e.title,
            e.search_text,
            e.prompt_text,
            e.document_json
        FROM evidence_documents e
        WHERE {' AND '.join(where)}
    """
    rows = con.execute(sql, params).fetchall()
    scored = []
    for row in rows:
        haystack = "\n".join([str(row[1] or ""), str(row[2] or ""), str(row[3] or "")])
        matched_terms = [term for term in terms if term in haystack]
        if not matched_terms:
            continue
        match_score = sum(_term_weight(term) for term in matched_terms)
        scored.append(
            {
                "score": -match_score,
                "rank_score": float(match_score),
                "sort_score": -float(match_score),
                **json.loads(row[4]),
            }
        )
    scored.sort(key=lambda row: (-row["rank_score"], row["evidence_id"]))
    return scored[:top_k]


def _query_terms(text: str) -> List[str]:
    normalized = re.sub(r"[\s，。；、,;:：？！?（）()]+", " ", text.replace('"', " ")).strip()
    terms: list[str] = []
    for token in normalized.split():
        _append_unique(terms, token)
        for phrase in _DOMAIN_PHRASES:
            if phrase in token:
                _append_unique(terms, phrase)
        for chinese in re.findall(r"[\u4e00-\u9fff]+", token):
            for size in (4, 3, 2):
                if len(chinese) < size:
                    continue
                for start in range(len(chinese) - size + 1):
                    ngram = chinese[start : start + size]
                    if ngram not in _CHINESE_STOP_TERMS:
                        _append_unique(terms, ngram)
    return terms[:100]


def _fts_query_from_terms(terms: List[str]) -> str:
    quoted = [f'"{term}"' for term in terms if term]
    if not quoted:
        return ""
    return " OR ".join(quoted)


def _append_unique(terms: List[str], term: str) -> None:
    if term and term not in terms:
        terms.append(term)


def _term_weight(term: str) -> int:
    if term in _TASK_PHRASE_WEIGHTS:
        return _TASK_PHRASE_WEIGHTS[term]
    base = min(len(term), 8)
    return base * 3 if term in _DOMAIN_PHRASES else base


def _expand_evidence_types(library_types: List[str]) -> List[str]:
    expanded = list(dict.fromkeys(library_types))
    if "rule" in expanded and "composite" not in expanded:
        expanded.append("composite")
    return expanded


_DOMAIN_PHRASES = {
    "作用组合",
    "荷载组合",
    "汽车荷载",
    "风荷载",
    "极限状态",
    "建筑限界",
    "桥面净空",
    "通航水域",
    "保护层",
    "锚固长度",
    "盖梁",
    "承台",
    "裂缝宽度",
    "挠度限值",
    "标准化跨径",
    "抗剪承载力",
    "盖梁尺寸",
}


_TASK_PHRASE_WEIGHTS = {
    "作用组合": 36,
    "荷载组合": 36,
    "标准化跨径": 36,
    "抗剪承载力": 30,
    "建筑限界": 30,
    "桥面净空": 30,
    "裂缝宽度": 30,
    "挠度限值": 30,
    "保护层": 30,
    "锚固长度": 30,
    "盖梁尺寸": 30,
}


_CHINESE_STOP_TERMS = {
    "什么",
    "如何",
    "怎样",
    "怎么",
    "需要",
    "考虑",
    "采用",
    "进行",
    "相关",
    "要求",
    "规定",
    "计算",
    "设计",
    "什么要",
    "么要求",
    "如何计",
    "何计算",
}


if __name__ == "__main__":
    main()
