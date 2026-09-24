from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bridge_agents.skill_registry import LoadedSkill, SkillRegistry


EXPECTED_SKILL_IDS = {
    "design-coordinator",
    "initial-design",
    "layout-revision",
    "structural-design",
    "modeling-check",
    "design-review",
    "design-review-qa",
}


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry("skills")


def test_discovers_standard_skills(registry: SkillRegistry) -> None:
    discovered = registry.discover()
    assert set(discovered.keys()) == EXPECTED_SKILL_IDS


def test_list_skills_returns_loaded_skills(registry: SkillRegistry) -> None:
    skills = registry.list_skills()
    assert len(skills) == len(EXPECTED_SKILL_IDS)
    for skill in skills:
        assert isinstance(skill, LoadedSkill)
        assert skill.skill_id in EXPECTED_SKILL_IDS


def test_each_skill_has_description_and_matching_name(registry: SkillRegistry) -> None:
    for skill_id in EXPECTED_SKILL_IDS:
        skill = registry.load(skill_id)
        assert skill.skill_id == skill_id
        assert skill.description.strip()
        assert skill.content.strip()


def test_sha256_matches_content(registry: SkillRegistry) -> None:
    skill = registry.load("structural-design")
    expected = hashlib.sha256(skill.content.encode("utf-8")).hexdigest()
    assert skill.sha256 == expected
    assert len(skill.sha256) == 64


def test_skill_path_points_to_real_file(registry: SkillRegistry) -> None:
    skill = registry.load("initial-design")
    path = Path(skill.path)
    assert path.is_absolute()
    assert path.exists()
    assert path.name == "SKILL.md"


def test_validate_passes(registry: SkillRegistry) -> None:
    # 不应抛异常
    registry.validate()


def test_load_unknown_skill_raises(registry: SkillRegistry) -> None:
    with pytest.raises(KeyError):
        registry.load("does-not-exist")


def test_get_description(registry: SkillRegistry) -> None:
    desc = registry.get_description("modeling-check")
    assert "承载力验算" in desc


def test_frontmatter_parser_rejects_missing_block() -> None:
    with pytest.raises(ValueError):
        SkillRegistry.parse_frontmatter("没有 frontmatter 的内容\n")


def test_frontmatter_parser_extracts_meta_and_body() -> None:
    meta, body = SkillRegistry.parse_frontmatter(
        "---\nname: x\ndescription: d\n---\n正文内容\n"
    )
    assert meta == {"name": "x", "description": "d"}
    assert body == "正文内容"
