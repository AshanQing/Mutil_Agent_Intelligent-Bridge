from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class LoadedSkill:
    """一个已加载的标准 Agent Skill。

    content 为去掉 frontmatter 后的正文；sha256 对正文内容计算，
    用于运行日志中的可追溯哈希。
    """

    skill_id: str
    description: str
    content: str
    path: str
    sha256: str


class SkillRegistry:
    """发现、校验并加载 ``skills/<name>/SKILL.md``。

    职责边界（与 PromptRegistry 分离）：
    - SkillRegistry 只管理 Agent 的角色、职责边界、决策规则与证据使用规则；
    - PromptRegistry 只管理具体任务模板。
    本模块不授予任何执行权限；动作权限真源是
    ``bridge_agents.actions.AGENT_ACTIONS``。
    """

    def __init__(self, skills_dir: str = "skills") -> None:
        self.skills_root = Path(skills_dir)
        if not self.skills_root.is_absolute():
            self.skills_root = Path.cwd() / self.skills_root
        self.skills_root = self.skills_root.resolve()
        if not self.skills_root.exists():
            raise FileNotFoundError(f"Skills directory not found: {self.skills_root}")

    # ------------------------------------------------------------------ #
    # discovery
    # ------------------------------------------------------------------ #
    def discover(self) -> Dict[str, Path]:
        """返回 {skill_id: SKILL.md 路径}。skill_id 即子目录名。"""
        found: Dict[str, Path] = {}
        if not self.skills_root.is_dir():
            return found
        for skill_dir in sorted(self.skills_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if skill_file.is_file():
                found[skill_dir.name] = skill_file
        return found

    # ------------------------------------------------------------------ #
    # frontmatter
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
        """解析 SKILL.md 开头的 YAML frontmatter，返回 (meta, body)。

        frontmatter 必须位于文件最开头，以 ``---`` 起止。
        """
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError("SKILL.md must start with a '---' frontmatter block.")
        end = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end = idx
                break
        if end is None:
            raise ValueError("SKILL.md frontmatter is missing the closing '---'.")
        meta_raw = "\n".join(lines[1:end])
        meta = yaml.safe_load(meta_raw) or {}
        if not isinstance(meta, dict):
            raise ValueError("SKILL.md frontmatter must be a YAML mapping.")
        body = "\n".join(lines[end + 1 :]).lstrip("\n")
        return meta, body

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def load(self, skill_id: str) -> LoadedSkill:
        skill_file = self.discover().get(skill_id)
        if skill_file is None:
            raise KeyError(f"Skill not found: {skill_id}")
        content = skill_file.read_text(encoding="utf-8")
        meta, body = self.parse_frontmatter(content)
        name = meta.get("name")
        description = meta.get("description")
        if name is None:
            raise ValueError(f"SKILL.md frontmatter missing 'name': {skill_file}")
        if name != skill_id:
            raise ValueError(
                f"Skill id mismatch: directory={skill_id!r}, frontmatter name={name!r}"
            )
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"SKILL.md frontmatter missing 'description': {skill_file}")
        return LoadedSkill(
            skill_id=skill_id,
            description=description.strip(),
            content=body,
            path=str(skill_file),
            sha256=self._sha256(body),
        )

    def list_skills(self) -> List[LoadedSkill]:
        return [self.load(skill_id) for skill_id in self.discover()]

    def get_description(self, skill_id: str) -> str:
        return self.load(skill_id).description

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """校验所有 skill 可加载且 frontmatter 合法；任一失败即抛错。"""
        discovered = self.discover()
        if not discovered:
            raise ValueError(f"No SKILL.md found under {self.skills_root}")
        for skill_id in discovered:
            self.load(skill_id)
