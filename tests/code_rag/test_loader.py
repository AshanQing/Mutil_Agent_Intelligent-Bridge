from __future__ import annotations

from bridge_agents.code_rag.audit import audit_code_library
from bridge_agents.code_rag.loader import load_code_library


def test_load_rule_library_as_entity_chunks() -> None:
    library = load_code_library("data/code/JTG_3362_2018_rules.yaml")

    assert library.metadata.standard_code == "JTG 3362-2018"
    assert len(library.entities) == 307
    assert library.entities[0].library_type == "rule"
    assert library.entities[0].entity_id.startswith("R_3362_")
    assert "JTG 3362-2018" in library.entities[0].semantic_text()
    assert library.entities[0].source_file.endswith("JTG_3362_2018_rules.yaml")


def test_load_formula_relations_for_embedding_context() -> None:
    library = load_code_library("data/code/JTG_3362_2018_formulas.yaml")
    first = library.entities[0]

    assert first.library_type == "formula"
    assert first.related_entities["rules"]
    assert first.related_entities["variables"]
    assert "表达式" not in first.search_text()
    assert "expression" in first.execution


def test_audit_accepts_matching_rule_library_item_count() -> None:
    library = load_code_library("data/code/JTG_3362_2018_rules.yaml")
    report = audit_code_library(library)

    assert report.ok
    assert report.entity_count == 307
    assert not report.errors
