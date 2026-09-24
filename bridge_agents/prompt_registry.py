from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def _load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick(settings: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in settings and settings[key] not in [None, ""]:
            return settings[key]
    for section in ["prompts", "paths"]:
        value = settings.get(section)
        if isinstance(value, dict):
            for key in keys:
                if key in value and value[key] not in [None, ""]:
                    return value[key]
    return default


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_id: str
    version: str
    role: str
    system_content: str
    user_content: str
    template_path: str
    template_sha256: str
    context_keys: List[str]
    rendered_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PromptRegistry:
    """Manifest-backed prompt loader with strict Jinja2 rendering."""

    def __init__(self, manifest_path: str = "prompts/manifest.yaml") -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = Path.cwd() / self.manifest_path
        self.manifest_path = self.manifest_path.resolve()
        self.prompt_root = self.manifest_path.parent
        self._manifest = self._load_manifest()
        self._env = Environment(
            loader=FileSystemLoader(str(self.prompt_root)),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )
        self.validate()

    @classmethod
    def from_config(cls, config_path: str = "config/settings.yaml") -> "PromptRegistry":
        settings = _load_settings(config_path)
        manifest_path = _pick(settings, "manifest_path", default="prompts/manifest.yaml")
        return cls(str(manifest_path))

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Prompt manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        prompts = data.get("prompts")
        if not isinstance(prompts, dict) or not prompts:
            raise ValueError(f"Prompt manifest must contain a non-empty 'prompts' mapping: {self.manifest_path}")
        return data

    @property
    def prompts(self) -> Dict[str, Dict[str, Any]]:
        return self._manifest["prompts"]

    def validate(self) -> None:
        seen = set()
        for prompt_id, spec in self.prompts.items():
            if prompt_id in seen:
                raise ValueError(f"Duplicate prompt id: {prompt_id}")
            seen.add(prompt_id)
            if not isinstance(spec, dict):
                raise ValueError(f"Prompt spec must be a mapping: {prompt_id}")
            rel_path = spec.get("path")
            if not rel_path:
                raise ValueError(f"Prompt spec missing path: {prompt_id}")
            template_path = self.prompt_root / str(rel_path)
            if not template_path.exists():
                raise FileNotFoundError(f"Prompt template not found for {prompt_id}: {template_path}")
            if not spec.get("version"):
                raise ValueError(f"Prompt spec missing version: {prompt_id}")
            if spec.get("role") not in {"system", "user"}:
                raise ValueError(f"Prompt role must be system or user for {prompt_id}")
            required_context = spec.get("required_context", [])
            if not isinstance(required_context, list):
                raise ValueError(f"required_context must be a list for {prompt_id}")

    def render_prompt(self, prompt_id: str, context: Optional[Dict[str, Any]] = None) -> RenderedPrompt:
        if prompt_id not in self.prompts:
            raise KeyError(f"Prompt id is not registered: {prompt_id}")
        spec = self.prompts[prompt_id]
        context = dict(context or {})

        missing = [key for key in spec.get("required_context", []) if key not in context]
        if missing:
            raise KeyError(f"Prompt {prompt_id} missing required context keys: {missing}")

        rel_path = str(spec["path"])
        template_path = (self.prompt_root / rel_path).resolve()
        template_text = template_path.read_text(encoding="utf-8")
        rendered = self._env.get_template(rel_path.replace("\\", "/")).render(**context)

        role = str(spec["role"])
        return RenderedPrompt(
            prompt_id=prompt_id,
            version=str(spec["version"]),
            role=role,
            system_content=rendered if role == "system" else "",
            user_content=rendered if role == "user" else "",
            template_path=str(template_path),
            template_sha256=hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
            context_keys=sorted(context.keys()),
            rendered_at=datetime.now(timezone.utc).isoformat(),
        )


_REGISTRY_CACHE: Dict[str, PromptRegistry] = {}


def get_prompt_registry(config_path: str = "config/settings.yaml") -> PromptRegistry:
    settings = _load_settings(config_path)
    manifest_path = str(_pick(settings, "manifest_path", default="prompts/manifest.yaml"))
    cache_key = str(Path(manifest_path).resolve())
    if cache_key not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[cache_key] = PromptRegistry(manifest_path)
    return _REGISTRY_CACHE[cache_key]


def render_prompt(
    prompt_id: str,
    context: Optional[Dict[str, Any]] = None,
    config_path: str = "config/settings.yaml",
) -> RenderedPrompt:
    return get_prompt_registry(config_path).render_prompt(prompt_id, context)


def render_external_prompt(
    template_path: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    role: str = "user",
) -> RenderedPrompt:
    """渲染调用方显式指定的模板，同时保留与注册 Prompt 相同的审计元数据。"""
    path = Path(template_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Explicit prompt template not found: {path}")
    if role not in {"system", "user"}:
        raise ValueError(f"Prompt role must be system or user: {role}")
    template_text = path.read_text(encoding="utf-8")
    env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
    rendered = env.from_string(template_text).render(**dict(context or {}))
    return RenderedPrompt(
        prompt_id=f"external:{path.name}",
        version="external",
        role=role,
        system_content=rendered if role == "system" else "",
        user_content=rendered if role == "user" else "",
        template_path=str(path),
        template_sha256=hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
        context_keys=sorted((context or {}).keys()),
        rendered_at=datetime.now(timezone.utc).isoformat(),
    )
