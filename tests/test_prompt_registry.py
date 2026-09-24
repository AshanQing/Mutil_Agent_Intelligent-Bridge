from __future__ import annotations

import json

import pytest

from bridge_agents.prompt_registry import PromptRegistry


def _fixture_context(prompt_id: str) -> dict:
    common_json = json.dumps({"sample": True}, ensure_ascii=False, indent=2)
    fixtures = {
        "tasks.initial_layout_design.v1": {
            "design_input_json": common_json,
            "few_shots_json": "[]",
            "standards_text": "",
        },
        "tasks.layout_revision_design.v1": {
            "ROLE_BLOCK": "你是桥梁工程设计师。",
            "STANDARDS_BLOCK": "无。",
            "DESIGN_INPUT_JSON": common_json,
            "PREVIOUS_LAYOUT_JSON": common_json,
            "REVISION_INSTRUCTION_TEXT": "执行最小必要修正。",
            "REVISION_ADVICE_JSON": common_json,
            "OUTPUT_SCHEMA_BLOCK": "只输出 JSON。",
        },
        "tasks.design_unit_extraction.v1": {
            "layout_result_json": common_json,
            "validation_feedback_json": "",
        },
        "tasks.cap_reinforcement_design.v1": {
            "REINFORCEMENT_EXAMPLES": "[]",
            "COLUMN_REINFORCEMENT_EXAMPLES": "[]",
            "CURRENT_REINFORCEMENT_TASK_INPUT": "task: sample",
        },
        "tasks.modeling_feedback.v1": {
            "payload_json": common_json,
        },
        "tasks.design_review_assessment.v1": {
            "project_evidence": "[ARTIFACT-001] 示例成果",
            "code_evidence": "[EVID_SAMPLE] 示例规范证据",
            "index_version": "test-index",
        },
        "tasks.design_review_qa.v1": {
            "question": "示例问题",
            "history": "",
            "project_evidence": "[ARTIFACT-001] 示例成果",
            "code_evidence": "[EVID_SAMPLE] 示例规范证据",
            "index_version": "test-index",
        },
        "tasks.design_review_routing.v1": {
            "question": "示例问题",
            "available_types": "final_layout、final_deliverables",
            "artifact_menu": "- final_layout（布跨设计成果）：layout/final_layout_result.json —— 顶层字段：桥梁布跨设计",
        },
    }
    return fixtures.get(prompt_id, {})


def test_manifest_validates_registered_templates() -> None:
    registry = PromptRegistry("prompts/manifest.yaml")
    assert registry.prompts


def test_all_registered_prompts_render_with_fixtures() -> None:
    registry = PromptRegistry("prompts/manifest.yaml")
    for prompt_id in registry.prompts:
        rendered = registry.render_prompt(prompt_id, _fixture_context(prompt_id))
        content = rendered.system_content or rendered.user_content
        assert content.strip()
        assert rendered.template_sha256
        assert rendered.version


def test_missing_required_context_fails() -> None:
    registry = PromptRegistry("prompts/manifest.yaml")
    with pytest.raises(KeyError):
        registry.render_prompt("tasks.design_unit_extraction.v1", {})
