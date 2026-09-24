from __future__ import annotations

from pathlib import Path

from bridge_agents.code_rag.evidence import build_evidence_documents
from bridge_agents.code_rag.loader import load_code_library


def _entities():
    return [
        entity
        for path in Path("data/code").glob("*.yaml")
        for entity in load_code_library(path).entities
    ]


def test_title_only_rule_is_excluded_from_evidence_documents() -> None:
    result = build_evidence_documents(_entities())

    assert "R_3362_CH08_8_4_2" not in {
        document.primary_entity_id for document in result.documents
    }
    exclusion = next(
        item for item in result.exclusions if item.entity_id == "R_3362_CH08_8_4_2"
    )
    assert exclusion.reason == "title_only_rule"


def test_substantive_rule_contains_only_prompt_ready_normative_content() -> None:
    result = build_evidence_documents(_entities())
    document = next(
        item for item in result.documents if item.primary_entity_id == "R_3362_CH04_004"
    )

    assert "装配式板桥不大于10m" in document.normative_text
    assert "人工复核" not in document.prompt_text
    assert "manual_review" not in document.prompt_text
    assert "related_refs" not in document.prompt_text
    assert "R_3362_" not in document.prompt_text
    assert document.source_verification == "not_verified_against_official_text"


def test_normative_formula_embeds_variable_definitions_without_relation_ids() -> None:
    result = build_evidence_documents(_entities())
    formula_document = next(
        item
        for item in result.documents
        if item.evidence_type == "formula" and item.formulas
    )

    formula = formula_document.formulas[0]
    assert formula.expression
    assert formula.variables
    assert all(variable.symbol or variable.name for variable in formula.variables)
    assert all("VAR_" not in variable.model_dump_json() for variable in formula.variables)
    assert "related_variables" not in formula_document.prompt_text


def test_derived_cap_beam_formula_is_not_indexable_evidence() -> None:
    result = build_evidence_documents(_entities())

    assert "F_3362_CH08_8_4_4_CAP_BEAM_SHEAR_CAPACITY" not in {
        document.primary_entity_id for document in result.documents
    }
    exclusion = next(
        item
        for item in result.exclusions
        if item.entity_id == "F_3362_CH08_8_4_4_CAP_BEAM_SHEAR_CAPACITY"
    )
    assert exclusion.reason == "non_normative_formula"


def test_ai_summary_and_generic_placeholder_rules_are_excluded() -> None:
    result = build_evidence_documents(_entities())
    indexed = {document.primary_entity_id for document in result.documents}

    assert "R_D60_CH03_058" not in indexed
    assert "R_3362_CH05_5_3_3" not in indexed
    reasons = {item.entity_id: item.reason for item in result.exclusions}
    assert reasons["R_D60_CH03_058"] == "summary_rule"
    assert reasons["R_3362_CH05_5_3_3"] == "generic_placeholder_rule"


def test_summary_rule_can_only_act_as_a_structured_evidence_hub() -> None:
    result = build_evidence_documents(_entities())
    document = next(
        item
        for item in result.documents
        if item.primary_entity_id == "R_D60_CH04_005"
    )

    assert document.evidence_type == "composite"
    assert document.normative_text == ""
    assert len(document.formulas) == 2
    assert len(document.tables) == 2
    assert "摘要" not in document.prompt_text
    assert "S_ud" in document.prompt_text


def test_short_rule_with_explicit_engineering_value_is_retained() -> None:
    result = build_evidence_documents(_entities())
    document = next(
        item
        for item in result.documents
        if item.primary_entity_id == "R_B01_5_0_6_PAVEMENT_STANDARD_AXLE_LOAD"
    )

    assert "100kN" in document.normative_text
    assert "0.7MPa" in document.normative_text


def test_table_document_keeps_complete_rows_and_units() -> None:
    result = build_evidence_documents(_entities())
    document = next(
        item
        for item in result.documents
        if item.primary_entity_id == "T_B01_3_6_1_CLEARANCE_HEIGHT"
    )

    assert len(document.tables) == 1
    assert len(document.tables[0].rows) == 3
    assert document.tables[0].units["clearance_height"] == "m"
    assert "高速公路、一级公路、二级公路" in document.prompt_text


def test_table_rows_resolve_graph_ids_and_drop_relation_fields() -> None:
    result = build_evidence_documents(_entities())
    document = next(
        item
        for item in result.documents
        if item.primary_entity_id == "T_D60_4_3_7_FATIGUE_LOAD_MODEL"
    )

    assert "related_tables" not in document.prompt_text
    assert "T_D60_" not in document.prompt_text
    assert "横向车道布载系数表" in document.prompt_text
