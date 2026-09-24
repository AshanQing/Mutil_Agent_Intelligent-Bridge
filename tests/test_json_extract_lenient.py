from __future__ import annotations

import pytest

from bridge_agents.utils import extract_json_object


def test_tolerates_raw_newline_inside_string():
    # LLM 常见：字符串值里直接换行，严格 json.loads 会报 Invalid control character
    text = '{"设桥总览": "第一行\n第二行", "桥位": [1, 2]}'
    data = extract_json_object(text)
    assert data["设桥总览"] == "第一行\n第二行"


def test_tolerates_raw_vertical_tab_and_other_controls():
    text = '{"note": "a\x0bb\x0cc", "value": 1}'
    data = extract_json_object(text)
    assert data["value"] == 1


def test_strips_code_fence_and_surrounding_prose():
    text = "解析如下：\n```json\n{\"a\": 1, \"b\": \"x\ny\"}\n```\n以上。"
    data = extract_json_object(text)
    assert data == {"a": 1, "b": "x\ny"}


def test_extracts_object_from_prose_without_fence():
    text = '说明文字 {"a": {"b": 2}} 结束'
    data = extract_json_object(text)
    assert data == {"a": {"b": 2}}


def test_non_object_top_level_raises():
    with pytest.raises(ValueError):
        extract_json_object("[1, 2, 3]")
