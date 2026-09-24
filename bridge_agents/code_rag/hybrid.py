from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    *,
    keyword_ids: list[str],
    vector_ids: list[str],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    keyword_ranks = {evidence_id: rank for rank, evidence_id in enumerate(keyword_ids, start=1)}
    vector_ranks = {evidence_id: rank for rank, evidence_id in enumerate(vector_ids, start=1)}
    evidence_ids = set(keyword_ranks) | set(vector_ranks)

    def score(evidence_id: str) -> float:
        ranks = [keyword_ranks.get(evidence_id), vector_ranks.get(evidence_id)]
        return sum(1.0 / (rrf_k + rank) for rank in ranks if rank is not None)

    ordered = sorted(evidence_ids, key=lambda evidence_id: (-score(evidence_id), evidence_id))
    return [
        {
            "evidence_id": evidence_id,
            "rrf_score": score(evidence_id),
            "keyword_rank": keyword_ranks.get(evidence_id),
            "vector_rank": vector_ranks.get(evidence_id),
            "rank": rank,
        }
        for rank, evidence_id in enumerate(ordered[: max(0, int(top_k))], start=1)
    ]
