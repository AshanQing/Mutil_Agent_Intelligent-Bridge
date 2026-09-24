from bridge_agents.code_rag.hybrid import reciprocal_rank_fusion


def test_rrf_fuses_and_deduplicates_keyword_and_vector_results() -> None:
    rows = reciprocal_rank_fusion(
        keyword_ids=["EV_A", "EV_B", "EV_C"],
        vector_ids=["EV_B", "EV_D", "EV_A"],
        top_k=4,
    )

    assert [row["evidence_id"] for row in rows] == ["EV_B", "EV_A", "EV_D", "EV_C"]
    assert rows[0]["keyword_rank"] == 2
    assert rows[0]["vector_rank"] == 1
    assert rows[0]["rank"] == 1
    assert rows[2]["keyword_rank"] is None
    assert rows[3]["vector_rank"] is None
