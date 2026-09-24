from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import yaml
from pydantic import BaseModel, Field, field_validator

from .code_rag.schemas import CodeQuery, EvidenceDocument
from .code_rag.service import CodeRAGService
from .human_review_store import ACCEPTANCE_ACTIONS, LEDGER_RELATIVE_PATH
from .prompt_registry import render_prompt
from .skill_registry import SkillRegistry
from .utils import build_llm, extract_json_object


class ReviewEvidenceUnavailableError(RuntimeError):
    """最终评估明确启用规范 RAG，但索引不可用。"""


class ProjectEvidenceDocument(BaseModel):
    evidence_id: str
    path: str
    artifact_type: str
    sha256: str
    content: str
    relevance_score: float = 0.0


class ProjectEvidenceBundle(BaseModel):
    output_dir: str
    question: str
    documents: List[ProjectEvidenceDocument] = Field(default_factory=list)
    prompt_text: str = ""
    catalog_text: str = ""
    human_review_text: str = ""


class EvidenceRouting(BaseModel):
    """一次路由调用的结论：要哪些成果的正文、要不要查规范、查什么。

    `source` 区分结论来自模型还是兜底：路由调用失败时退回总览类成果，
    这条记录会一并落到审计里，便于回溯"这次为什么只喂了这些证据"。
    """

    artifact_types: List[str] = Field(default_factory=list)
    need_code_rag: bool = True
    code_queries: List[str] = Field(default_factory=list)
    rationale: str = ""
    source: Literal["model", "fallback"] = "model"


class DesignAssessment(BaseModel):
    assessment_status: Literal[
        "completed",
        "completed_with_limitations",
        "incomplete",
        "manual_review_required",
    ]
    overall_conclusion: str
    stage_findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    cited_artifact_ids: List[str] = Field(default_factory=list)
    cited_code_evidence_ids: List[str] = Field(default_factory=list)
    assessment_path: str = ""
    code_retrieval_status: str = ""
    citation_sources: Dict[str, str] = Field(default_factory=dict)
    display_text: str = ""

    @field_validator("stage_findings", "limitations", "recommended_actions", mode="before")
    @classmethod
    def normalize_narrative_lists(cls, value: Any) -> List[str]:
        """兼容模型把叙述型字符串数组返回为结构化对象数组。"""
        values = value if isinstance(value, list) else ([] if value is None else [value])
        return [text for item in values if (text := _normalize_assessment_text_item(item))]


class ReviewAnswer(BaseModel):
    question: str
    answer: str
    cited_artifact_ids: List[str] = Field(default_factory=list)
    cited_code_evidence_ids: List[str] = Field(default_factory=list)
    answer_path: str = ""
    code_retrieval_status: str = ""


_PRIORITY_PATTERNS: Sequence[tuple[str, str]] = (
    ("final_layout_result.json", "final_layout"),
    ("design_manifest.json", "final_deliverables"),
    ("drawing_index.json", "drawing_package"),
    ("collision_metrics_design_result_", "collision_check"),
    ("capacity_check_batch_summary.json", "capacity_check_batch"),
    ("structural_design_result.json", "structural_design"),
    ("design_units_result.json", "design_units"),
    ("dimension_design_result.json", "dimension_design"),
    ("pier_group_result.json", "pier_group"),
    ("reinforcement_design_result.json", "reinforcement_design"),
    ("design_units.json", "design_units"),
    ("capacity_check_summary.json", "capacity_check"),
    ("dimension_design", "dimension_design"),
    ("reinforcement_result", "reinforcement_design"),
)
_CORE_ARTIFACT_TYPES: Sequence[str] = (
    "final_layout",
    "structural_design",
    "design_units",
    "dimension_design",
    "pier_group",
    "reinforcement_design",
    "capacity_check_batch",
    "final_deliverables",
    "drawing_package",
    "collision_check",
)
# 无论问什么都先给模型看的"总览类"成果：布跨方案 + 交付清单。
# 它们体量小、覆盖面广，是模型判断"这次设计做了什么"的锚点。
_BASELINE_ARTIFACT_TYPES: Sequence[str] = ("final_layout", "final_deliverables")
_ARTIFACT_TYPE_LABELS: Dict[str, str] = {
    "final_layout": "布跨设计成果",
    "structural_design": "结构设计成果",
    "design_units": "设计单元划分",
    "dimension_design": "尺寸设计成果",
    "pier_group": "桥墩分组成果",
    "reinforcement_design": "配筋设计成果",
    "capacity_check_batch": "承载力验算批次汇总",
    "capacity_check": "承载力验算汇总",
    "final_deliverables": "最终交付清单",
    "drawing_package": "图纸包索引",
    "collision_check": "障碍物碰撞校核",
    "supporting_artifact": "其他过程成果",
}
# 交付清单是"这次设计做了什么"的总账，任何问题都带上它当锚点。
# 它是唯一不做路由判断的成果：体量小、覆盖全部阶段，缺了它模型无法定位自己看到的是哪次设计。
_ALWAYS_ON_ARTIFACT_TYPES: Sequence[str] = ("final_deliverables",)
# 路由模型最多能点名几份成果正文、几条规范检索词。
# 这个上限只用来兜住跑飞的输出，不该比问答的证据预算（answer 的 max_documents）还紧——
# 否则"盘点全部成果"这类问题会被上限截掉本该覆盖的阶段。
_ROUTING_MAX_TYPES = 8
_ROUTING_MAX_QUERIES = 4
_CATALOG_MAX_NAMED = 40
# 清单/摘要里单行显示的字符上限。
_DIGEST_MAX_CHARS = 80
# 人工复核动作的中文名，用于渲染复核记录。ACCEPTANCE_ACTIONS 里的是"接受"类动作。
_REVIEW_ACTION_LABELS: Dict[str, str] = {
    "accept_and_continue": "人工接受风险并继续",
    "accept_partial_and_continue": "人工接受部分成果并继续",
    "accept_check_and_finish": "人工接受该验算结果并结束",
    "revise": "人工要求返修",
    "reject": "人工否决",
}

_ALLOWED_SUFFIXES = {".json", ".yaml", ".yml", ".md", ".txt"}
_EXCLUDED_PARTS = {
    "prompts",
    "prompt_audit",
    "logs",
    "design_review",
    "code_rag",
    "few_shots",
    "revision_instructions",
    "revision_prompts",
    "run_configs",
    "human_review",
    "checkpoints",
}
_EXCLUDED_FILE_MARKERS = ("prompt", "agent_log", "audit", "trace")
_EXCLUDED_PART_PREFIXES = ("design_run_",)

_ASSESSMENT_STAGE_KEYS = ("stage", "stage_name", "阶段", "阶段名称")
_ASSESSMENT_TEXT_KEYS = (
    "finding",
    "findings",
    "conclusion",
    "description",
    "content",
    "message",
    "text",
    "结论",
    "发现",
    "说明",
)
_ASSESSMENT_CITATION_KEYS = (
    "citations",
    "citation_ids",
    "cited_artifact_ids",
    "cited_code_evidence_ids",
    "references",
    "引用",
)


def _assessment_value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "；".join(
            text for item in value if (text := _assessment_value_text(item))
        )
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def _assessment_citations_text(value: Any) -> str:
    raw = _assessment_value_text(value)
    if not raw:
        return ""
    citations = re.findall(r"(?:ARTIFACT-|EVID_)[A-Za-z0-9_.-]+", raw)
    return " ".join(f"[{item}]" for item in dict.fromkeys(citations))


def _normalize_assessment_text_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return _assessment_value_text(item)

    stage = next(
        (_assessment_value_text(item[key]) for key in _ASSESSMENT_STAGE_KEYS if item.get(key) is not None),
        "",
    )
    finding_parts = [
        _assessment_value_text(item[key])
        for key in _ASSESSMENT_TEXT_KEYS
        if item.get(key) is not None
    ]
    finding = "；".join(part for part in finding_parts if part)
    citation_parts = [
        _assessment_citations_text(item[key])
        for key in _ASSESSMENT_CITATION_KEYS
        if item.get(key) is not None
    ]
    citations = " ".join(part for part in citation_parts if part)

    if stage or finding or citations:
        text = f"{stage}：{finding}" if stage and finding else stage or finding
        return " ".join(part for part in (text, citations) if part).strip()
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"))


def _artifact_type(path: Path) -> str:
    normalized = str(path).replace("\\", "/").lower()
    for marker, artifact_type in _PRIORITY_PATTERNS:
        if marker.lower() in normalized:
            return artifact_type
    return "supporting_artifact"


def _priority(path: Path) -> tuple[int, str]:
    normalized = str(path).replace("\\", "/").lower()
    for index, (marker, _) in enumerate(_PRIORITY_PATTERNS):
        if marker.lower() in normalized:
            return index, normalized
    return len(_PRIORITY_PATTERNS), normalized


def _is_excluded_artifact(path: Path, root: Path) -> bool:
    parts = [part.lower() for part in path.relative_to(root).parts]
    return any(part in _EXCLUDED_PARTS for part in parts) or any(
        part.startswith(prefix) for part in parts for prefix in _EXCLUDED_PART_PREFIXES
    )


def _read_artifact(path: Path, *, max_chars: int = 16000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[内容已按字符上限截断]"
    return text


def _query_terms(question: str) -> set[str]:
    text = str(question or "").lower()
    terms = set(re.findall(r"[a-z0-9_.-]{2,}", text))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    terms.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
    domain_terms = (
        "布跨", "桥梁", "尺寸", "配筋", "盖梁", "承载力", "利用率", "抗弯",
        "抗剪", "裂缝", "挠度", "疲劳", "抗震", "公式", "规范", "验算",
    )
    terms.update(term for term in domain_terms if term in text)
    return {term for term in terms if term}


def _relative_display_path(path: str | Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return Path(path).name


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# 截断后的 JSON 仍是 indent=2 的格式，顶层字段名固定在两格缩进上。
_TOP_LEVEL_KEY_RE = re.compile(r'^  "([^"]+)":', re.MULTILINE)


def _artifact_digest(document: ProjectEvidenceDocument, *, limit: int = _DIGEST_MAX_CHARS) -> str:
    """给路由模型的一行摘要：JSON 取顶层字段名，其余取首个非空行。

    只求让模型能分辨"这份文件大概是什么"，所以不做语义概括、不读取子结构。
    """
    text = (document.content or "").lstrip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload:
            keys = "、".join(str(key) for key in list(payload)[:8])
            return _truncate(f"顶层字段：{keys}", limit)
        # 超过字符上限的成果文件在读入时已被截断，重新解析必然失败；
        # 退回按缩进抽顶层字段名，比整份文件只剩一个"{"有用得多。
        keys = _TOP_LEVEL_KEY_RE.findall(text)
        if keys:
            return _truncate(f"顶层字段：{'、'.join(keys[:8])}", limit)
        return "JSON 成果（内容过长，已截断）"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return _truncate(stripped, limit)
    return "（空文件）"


def _preferred_document(
    artifact_type: str,
    group: Sequence[ProjectEvidenceDocument],
) -> ProjectEvidenceDocument:
    """同一阶段有多份产出时取哪一份：碰撞校核按轮次产出，取最新一轮；其余取优先级最高的。"""
    if artifact_type == "collision_check":
        return max(group, key=lambda item: Path(item.path).stat().st_mtime)
    return group[0]


def group_by_artifact_type(
    all_documents: Sequence[ProjectEvidenceDocument],
) -> Dict[str, List[ProjectEvidenceDocument]]:
    grouped: Dict[str, List[ProjectEvidenceDocument]] = {}
    for document in all_documents:
        grouped.setdefault(document.artifact_type, []).append(document)
    return grouped


def render_artifact_menu(
    all_documents: Sequence[ProjectEvidenceDocument],
    root: Path,
) -> str:
    """渲染给路由模型看的成果菜单：每个阶段一行——类型键、标签、代表文件、摘要。

    同一阶段的多次产出（逐墩尺寸、逐组配筋等）只列代表文件并注明还有几份，
    因为最终能进 prompt 的就是这一份；逐条罗列既挤占篇幅，也不影响模型的选择。
    """
    grouped = group_by_artifact_type(all_documents)
    entries = [
        (artifact_type, group)
        for artifact_type, group in grouped.items()
        if artifact_type != "supporting_artifact"
    ]
    # 相关度只用来决定菜单里谁排前面，选不选由模型定。
    entries.sort(
        key=lambda item: (
            -max(document.relevance_score for document in item[1]),
            _priority(Path(item[1][0].path)),
        )
    )
    lines: List[str] = []
    for artifact_type, group in entries[:_CATALOG_MAX_NAMED]:
        document = _preferred_document(artifact_type, group)
        label = _ARTIFACT_TYPE_LABELS.get(artifact_type, artifact_type)
        sibling = f"（本阶段另有 {len(group) - 1} 份产出，未列出）" if len(group) > 1 else ""
        lines.append(
            f"- {artifact_type}（{label}）："
            f"{_relative_display_path(document.path, root)} —— {_artifact_digest(document)}{sibling}"
        )
    if not lines:
        return "- （本次运行未产出可引用的成果文件）"
    if grouped.get("supporting_artifact"):
        lines.append(
            f"- 另有 {len(grouped['supporting_artifact'])} 份逐组/逐轮的过程文件未列出，"
            "它们不是阶段性成果，不作为引用依据。"
        )
    return "\n".join(lines)


def available_artifact_types(all_documents: Sequence[ProjectEvidenceDocument]) -> List[str]:
    """本次运行实际产出的成果类型键，按固定阶段顺序给出，供路由模型点名。"""
    present = {item.artifact_type for item in all_documents}
    ordered = [name for name in _CORE_ARTIFACT_TYPES if name in present]
    ordered.extend(sorted(present - set(ordered) - {"supporting_artifact"}))
    return ordered


def _select_evidence_documents(
    all_documents: Sequence[ProjectEvidenceDocument],
    wanted_types: Sequence[str],
    limit: int,
) -> List[ProjectEvidenceDocument]:
    """按路由结论选取要提供正文的成果。

    "哪些成果与问题相关"由路由模型判断，这里只做机械处理：按模型给的顺序去重
    取文件、补上始终在线的交付清单、多余截断。不做"凑满名额"的补足——没被点名
    的阶段只出现在清单里，这样问布跨就不会顺带灌进配筋与验算正文。
    模型点名的顺序就是相关度顺序，所以让它排在最前，引用编号也从它开始。
    """
    grouped = group_by_artifact_type(all_documents)
    ordered = [name for name in wanted_types if name in grouped]
    ordered.extend(name for name in _ALWAYS_ON_ARTIFACT_TYPES if name not in ordered)

    selected: List[ProjectEvidenceDocument] = []
    for artifact_type in ordered:
        group = grouped.get(artifact_type)
        if group is None:
            continue
        document = _preferred_document(artifact_type, group)
        if document not in selected:
            selected.append(document)
    return selected[:limit]


def _render_catalog(
    all_documents: Sequence[ProjectEvidenceDocument],
    selected: Sequence[ProjectEvidenceDocument],
    root: Path,
) -> str:
    """列出全部已产出成果，并标出本次提供了正文的那些。"""
    evidence_id_by_path = {item.path: item.evidence_id for item in selected}
    named = [item for item in all_documents if item.artifact_type != "supporting_artifact"]
    named.sort(key=lambda item: _priority(Path(item.path)))
    lines = [
        "## 项目成果清单",
        "本次设计已产出的成果如下；正文只提供与问题相关的一部分，",
        "带 [ARTIFACT-...] 标记的才是本次已提供正文的成果，其余只提供文件名。",
    ]
    for document in named[:_CATALOG_MAX_NAMED]:
        label = _ARTIFACT_TYPE_LABELS.get(document.artifact_type, document.artifact_type)
        evidence_id = evidence_id_by_path.get(document.path)
        marker = f" [{evidence_id}]" if evidence_id else ""
        lines.append(f"- {label}：{_relative_display_path(document.path, root)}{marker}")
    overflow = len(named) - _CATALOG_MAX_NAMED
    if overflow > 0:
        lines.append(f"- 另有 {overflow} 份同类成果未逐条列出。")
    extra = len(all_documents) - len(named)
    if extra > 0:
        lines.append(f"- 另有 {extra} 份中间/过程文件未列出。")
    return "\n".join(lines)


def scan_project_documents(
    output_dir: str | Path,
    question: str,
) -> tuple[Path, List[ProjectEvidenceDocument]]:
    """扫描输出目录下的工程成果文件，并按与问题的词面重合度打一个参考分。

    这个分数只用来给成果排个先后（例如渲染路由菜单时把更可能的排在前面），
    不决定取舍；取舍由路由模型来做。
    """
    root = Path(output_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"设计输出目录不存在: {root}")
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _ALLOWED_SUFFIXES
        and not _is_excluded_artifact(path, root)
        and not any(marker in path.name.lower() for marker in _EXCLUDED_FILE_MARKERS)
        and path.stat().st_size <= 2_000_000
    ]
    candidates.sort(key=_priority)
    terms = _query_terms(question)
    all_documents: List[ProjectEvidenceDocument] = []
    for index, path in enumerate(candidates, start=1):
        content = _read_artifact(path)
        searchable = f"{path.name}\n{content}".lower()
        term_hits = sum(1 for term in terms if term in searchable)
        type_bonus = max(0.0, 2.0 - 0.2 * _priority(path)[0])
        document = ProjectEvidenceDocument(
            evidence_id=f"ARTIFACT-{index:03d}",
            path=str(path),
            artifact_type=_artifact_type(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            content=content,
            relevance_score=float(term_hits) + type_bonus,
        )
        all_documents.append(document)
    return root, all_documents


def load_human_review_decisions(root: Path) -> List[Dict[str, Any]]:
    """读人工复核决策台账。

    台账在 `human_review/` 下，被 `_EXCLUDED_PARTS` 挡在成果扫描之外（它不是设计
    成果文件），但评估必须看得到它：人工"接受风险并继续"意味着该事项已经闭合，
    只看成果文件里的 `has_collision=true` 会把已决策的事项反复报成待复核。
    """
    ledger = root / LEDGER_RELATIVE_PATH
    if not ledger.is_file():
        return []
    records: List[Dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _render_metrics(metrics: Any) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return ""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, bool):
            value = "是" if value else "否"
        elif isinstance(value, float):
            value = f"{value:.6g}"
        parts.append(f"{key}={value}")
    return "、".join(parts)


def render_human_review_records(records: Sequence[Dict[str, Any]], root: Path) -> str:
    """把人工复核决策渲染成评估证据。

    刻意与"项目成果证据"分开成节：这是"人工已经看过并作出决定"的事实记录，
    不是待办清单，措辞上不能给模型留下"还有事没做"的印象。
    """
    if not records:
        return ""
    lines = [
        "## 人工复核记录",
        "下列事项已由人工复核并作出决策，是本次设计已经闭合的事实，不是待办事项。",
        "记录它们是为了留痕：已接受的风险不再构成「需要人工复核」的理由。",
    ]
    for index, record in enumerate(records, start=1):
        review_type = str(record.get("review_type") or "未标注类型")
        action = str(record.get("action") or "")
        label = _REVIEW_ACTION_LABELS.get(action, action or "未记录动作")
        lines.append(f"### 决策 {index}：{review_type}（{label}）")
        # 这句是整节的重点：评估最容易犯的错就是把"人工已接受"当成"还没人看"。
        # 接受类动作把事项闭合掉；返修/否决类动作不闭合，下一轮还得有人拍板。
        if action in ACCEPTANCE_ACTIONS:
            lines.append("- 决策性质：人工已接受该事项，本设计范围内已闭合，不构成待复核事项。")
        else:
            lines.append(f"- 决策性质：人工未接受该事项（{label}），此项未由本次决策闭合。")
        if record.get("decided_at"):
            lines.append(f"- 决策时间：{record['decided_at']}")
        if record.get("decision_id"):
            lines.append(f"- 决策编号：{record['decision_id']}")
        subject = record.get("subject")
        if isinstance(subject, dict) and subject.get("path"):
            lines.append(f"- 复核对象：{_relative_display_path(subject['path'], root)}")
        if record.get("extra_rounds"):
            lines.append(f"- 决策前已追加返修轮次：{record['extra_rounds']}")
        accepted_state = record.get("accepted_state")
        if isinstance(accepted_state, dict) and accepted_state.get("task_status"):
            lines.append(f"- 决策后流程状态：{accepted_state['task_status']}")
        risk = record.get("accepted_risk")
        if isinstance(risk, dict) and risk:
            scope = risk.get("scope") or "未标注范围"
            reason = risk.get("reason") or "未填写理由"
            lines.append(f"- 接受的风险：范围 {scope}；理由「{reason}」")
            if risk.get("accepted_at"):
                lines.append(f"  - 接受时间：{risk['accepted_at']}")
            metrics = _render_metrics(risk.get("metrics"))
            if metrics:
                lines.append(f"  - 风险指标：{metrics}")
        feedback = str(record.get("feedback") or "").strip()
        lines.append(f"- 人工反馈：{feedback or '（未填写）'}")
    return "\n".join(lines)


def build_project_evidence(
    root: Path,
    question: str,
    all_documents: Sequence[ProjectEvidenceDocument],
    *,
    wanted_types: Sequence[str],
    max_documents: int,
) -> ProjectEvidenceBundle:
    limit = max(1, int(max_documents))
    ranked = [
        document.model_copy(update={"evidence_id": f"ARTIFACT-{index:03d}"})
        for index, document in enumerate(
            _select_evidence_documents(all_documents, wanted_types, limit), start=1
        )
    ]
    catalog_text = _render_catalog(all_documents, ranked, root)
    human_review_text = render_human_review_records(load_human_review_decisions(root), root)
    prompt_lines = [
        catalog_text,
        "",
        "## 项目成果证据",
        "回答只能依据下列项目成果；引用格式为 [ARTIFACT-001]。",
    ]
    for document in ranked:
        prompt_lines.extend(
            [
                f"### [{document.evidence_id}] {document.artifact_type}",
                f"路径：{document.path}",
                document.content,
            ]
        )
    if human_review_text:
        prompt_lines.extend(["", human_review_text])
    return ProjectEvidenceBundle(
        output_dir=str(root),
        question=question,
        documents=ranked,
        prompt_text="\n".join(prompt_lines),
        catalog_text=catalog_text,
        human_review_text=human_review_text,
    )


def collect_project_evidence(
    output_dir: str | Path,
    question: str,
    *,
    max_documents: int = 8,
    wanted_types: Optional[Sequence[str]] = None,
) -> ProjectEvidenceBundle:
    """扫描成果并装配证据包。

    `wanted_types` 为空时按"全部核心成果"装配——整体评估这条路径用得上；
    问答路径必须先跑 `DesignReviewAgent._route_evidence` 拿到结论再传进来，
    不要在调用方用关键词猜。
    """
    root, all_documents = scan_project_documents(output_dir, question)
    return build_project_evidence(
        root,
        question,
        all_documents,
        wanted_types=list(wanted_types) if wanted_types else list(_CORE_ARTIFACT_TYPES),
        max_documents=max_documents,
    )


def _load_review_rag_settings(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {"enabled": False, "retrieval_mode": "auto", "top_k": 6}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("code_rag") if isinstance(payload, dict) else {}
    section = section if isinstance(section, dict) else {}
    return {
        "enabled": bool(section.get("enabled", False)),
        "retrieval_mode": str(section.get("retrieval_mode") or "auto"),
        "top_k": max(8, int(section.get("review_top_k") or section.get("top_k") or 8)),
    }


def _render_code_evidence(documents: Sequence[EvidenceDocument], status: str) -> str:
    if status == "disabled":
        return "## 规范证据\n本次问答未启用规范 RAG，不得声称已完成规范条文检索。"
    if status == "not_applicable":
        return (
            "## 规范证据\n"
            "本问题不涉及规范依据，本次未检索规范条文。如回答确实需要规范支撑，"
            "应说明本次未检索，不得凭记忆引用条文。"
        )
    if status == "no_valid_evidence":
        return "## 规范证据\n未检索到与问题直接相关的规范证据，应明确说明证据不足。"
    lines = ["## 规范证据", "引用格式为 [EVID_...]，不得编造证据 ID。"]
    for document in documents:
        lines.extend([f"### [{document.evidence_id}]", document.prompt_text, f"来源：{document.source}"])
    return "\n".join(lines)


def _review_evidence_applicable(document: EvidenceDocument) -> bool:
    title = document.title or ""
    if "预应力" in title and "普通钢筋" not in title:
        return False
    text = "\n".join(
        value
        for value in (document.title, document.normative_text, document.prompt_text)
        if value
    )
    return not ("预应力" in text and "钢筋混凝土" not in text and "普通钢筋" not in text)


def _extract_citations(text: str, prefix: str) -> List[str]:
    values = re.findall(rf"\[({re.escape(prefix)}[A-Za-z0-9_.-]+)\]", str(text or ""))
    return list(dict.fromkeys(values))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _replace_internal_citations(text: str, sources: Dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        evidence_id = match.group(1)
        label = sources.get(evidence_id)
        return f"〔{label}〕" if label else ""

    return re.sub(r"\[((?:ARTIFACT-|EVID_)[A-Za-z0-9_.-]+)\]", replace, str(text or ""))


def _render_assessment_display(assessment: DesignAssessment) -> str:
    status_labels = {
        "completed": "已完成",
        "completed_with_limitations": "已完成（存在方法边界）",
        "incomplete": "当前目标尚未完成",
        "manual_review_required": "需要人工复核",
    }
    sections = [
        ("整体结论", [assessment.overall_conclusion]),
        ("已完成成果与阶段结论", assessment.stage_findings),
        ("方法边界与适用范围", assessment.limitations),
        ("建议的后续工作", assessment.recommended_actions),
    ]
    lines = [f"评估状态：{status_labels[assessment.assessment_status]}"]
    for title, items in sections:
        normalized = [
            _replace_internal_citations(item, assessment.citation_sources).strip()
            for item in items
            if str(item or "").strip()
        ]
        if not normalized:
            continue
        lines.extend(["", f"【{title}】"])
        if title == "整体结论":
            lines.append(normalized[0])
        else:
            lines.extend(f"{index}. {item}" for index, item in enumerate(normalized, start=1))
    return "\n".join(lines)


CHAT_HISTORY_MAX_TURNS = 10
CHAT_MESSAGE_MAX_CHARS = 1600


def _render_conversation_history(history: Optional[Sequence[Dict[str, str]]]) -> str:
    """把多轮对话历史渲染为 QA prompt 的文本块（最近 N 轮、单条截断）。

    每轮两行"用户：… / 助手：…"，只保留最近 CHAT_HISTORY_MAX_TURNS 对；
    空历史返回空串（模板据此隐藏段落）。纯函数，便于单元测试。
    """
    if not history:
        return ""
    lines: List[str] = []
    for entry in list(history)[-CHAT_HISTORY_MAX_TURNS:]:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "").strip().lower()
        content = str(entry.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            speaker = "用户"
        elif role in {"assistant", "ai", "answer"}:
            speaker = "助手"
        else:
            speaker = role or "?"
        if len(content) > CHAT_MESSAGE_MAX_CHARS:
            content = content[:CHAT_MESSAGE_MAX_CHARS] + "…"
        lines.append(f"{speaker}：{content}")
    return "\n".join(lines)


def _index_failure_detail(service: Any) -> str:
    """把"索引无效"翻译成可操作的诊断（过期/缺失/损坏/结构不兼容）。"""
    validate = getattr(service, "validate_index", None)
    info: Dict[str, Any] = {}
    if callable(validate):
        try:
            info = validate() or {}
        except Exception:
            info = {}
    reason = str(info.get("reason") or "")
    rebuild_hint = "请运行 python -m bridge_agents.code_rag.build_index 重建索引。"
    if reason == "stale_source_files":
        stale = [str(item.get("source_file") or "") for item in (info.get("stale_sources") or [])]
        names = "、".join(name for name in stale[:6] if name) or "未知"
        return f"规范 RAG 索引已过期：源文件在索引构建后被修改（{names}）。{rebuild_hint}"
    if reason == "index_not_found":
        return f"规范 RAG 索引不存在。{rebuild_hint}"
    if reason == "invalid_vector_index":
        return f"规范 RAG 向量索引损坏或不完整。{rebuild_hint}"
    if reason == "invalid_schema":
        return f"规范 RAG 索引结构不兼容（{info.get('detail', '')}）。{rebuild_hint}"
    return f"规范 RAG 已启用，但索引不可用或已失效。{rebuild_hint}"


class DesignReviewAgent:
    """设计完成后的只读评估与专业问答 Agent。"""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        rag_service: Any | None = None,
        routing_llm: Any | None = None,
    ) -> None:
        self.llm = llm
        self.rag_service = rag_service
        # 路由是一次独立的、很轻的判断，可以单独换更便宜的模型。
        self.routing_llm = routing_llm

    def _retrieve_code_evidence(
        self,
        question: str,
        config_path: str | Path,
    ) -> tuple[str, List[EvidenceDocument], str]:
        settings = _load_review_rag_settings(config_path)
        if not settings["enabled"]:
            return "disabled", [], ""
        query = CodeQuery(
            query_text=question,
            library_types=["rule", "formula", "table"],
            retrieval_mode=settings["retrieval_mode"],
            top_k=settings["top_k"],
        )
        service = self.rag_service or CodeRAGService()
        bundle = service.retrieve(query)
        if bundle.retrieval_status == "invalid_index":
            raise ReviewEvidenceUnavailableError(_index_failure_detail(service))
        documents = [item for item in bundle.evidence_documents if _review_evidence_applicable(item)]
        status = bundle.retrieval_status if documents else "no_valid_evidence"
        return status, documents, bundle.index_version

    def _llm(self, config_path: str | Path) -> Any:
        return self.llm or build_llm(str(config_path), role="review")

    def _routing_client(self, config_path: str | Path) -> Any:
        """路由是一次极轻的判断，用便宜的那档模型。

        `settings.yaml` 里配置 `llm.routing` 即可单独指定；没配就按 build_llm
        的既有约定回退到 `llm.controller`，不必为此改配置。
        """
        if self.routing_llm is not None:
            return self.routing_llm
        if self.llm is not None:
            return self.llm
        return build_llm(str(config_path), role="routing")

    def _fallback_routing(self, question: str, reason: str = "") -> EvidenceRouting:
        """路由不可用时退回"总览类成果 + 用原话检索规范"，宁可多给也不给空。"""
        return EvidenceRouting(
            artifact_types=list(_BASELINE_ARTIFACT_TYPES),
            need_code_rag=True,
            code_queries=[question],
            rationale=reason or "路由调用未给出可用结论，回退总览类成果。",
            source="fallback",
        )

    def _route_evidence(
        self,
        question: str,
        all_documents: Sequence[ProjectEvidenceDocument],
        root: Path,
        config_path: str | Path,
    ) -> EvidenceRouting:
        """让模型判断这次回答需要哪些成果的正文、要不要查规范。

        "哪些成果和问题相关"是语义判断，交给模型；模型只负责点名阶段，
        取哪个文件、怎么编号、清单怎么渲染仍由本地确定性代码完成。
        """
        types = available_artifact_types(all_documents)
        if not types:
            return self._fallback_routing(question, "本次运行没有可引用的成果文件。")
        prompt = render_prompt(
            "tasks.design_review_routing.v1",
            {
                "question": question,
                "available_types": "、".join(types),
                "artifact_menu": render_artifact_menu(all_documents, root),
            },
            config_path=str(config_path),
        )
        try:
            response = self._routing_client(config_path).invoke(
                [("user", prompt.user_content)]
            )
            routing = EvidenceRouting.model_validate(extract_json_object(response.content))
        except Exception as error:  # 路由失败不应阻断问答，退回总览类成果
            return self._fallback_routing(question, f"路由调用失败（{type(error).__name__}），回退总览类成果。")

        # 模型可能自创类型名或重复点名，按实际产出的类型清洗后再用。
        picked: List[str] = []
        for name in routing.artifact_types:
            name = str(name).strip()
            if name in types and name not in picked:
                picked.append(name)
        queries = [str(item).strip() for item in routing.code_queries if str(item).strip()]
        routing.artifact_types = picked[:_ROUTING_MAX_TYPES]
        routing.code_queries = queries[:_ROUTING_MAX_QUERIES] or ([question] if routing.need_code_rag else [])
        if not routing.artifact_types:
            return self._fallback_routing(question, "路由未点名任何有效成果，回退总览类成果。")
        return routing

    def _retrieve_code_evidence_many(
        self,
        queries: Sequence[str],
        config_path: str | Path,
    ) -> tuple[str, List[EvidenceDocument], str]:
        documents: List[EvidenceDocument] = []
        seen_ids: set[str] = set()
        versions: List[str] = []
        statuses: List[str] = []
        for query in queries:
            status, query_documents, version = self._retrieve_code_evidence(query, config_path)
            statuses.append(status)
            if version and version not in versions:
                versions.append(version)
            for document in query_documents:
                if document.evidence_id not in seen_ids:
                    seen_ids.add(document.evidence_id)
                    documents.append(document)
        if statuses and all(status == "disabled" for status in statuses):
            return "disabled", [], ""
        return ("ok" if documents else "no_valid_evidence"), documents[:16], "|".join(versions)

    def _skill(self, skill_id: str = "design-review") -> Any:
        """整体评估与问答用各自的 skill：问答要答得灵活，评估要覆盖全面。"""
        return SkillRegistry().load(skill_id)

    @staticmethod
    def _validate_citations(
        text: str,
        project_documents: Sequence[ProjectEvidenceDocument],
        code_documents: Sequence[EvidenceDocument],
    ) -> tuple[List[str], List[str]]:
        """清洗回答中的证据引用。

        只保留能对应到实际证据文档的引用 id（成果 ARTIFACT-* / 规范 EVID_*）；
        模型可能引用未收录或不存在的 id，属生成噪音，不阻断整条回答，
        未识别的引用直接过滤，回答正文原样保留。
        """
        artifact_ids = _extract_citations(text, "ARTIFACT-")
        code_ids = _extract_citations(text, "EVID_")
        known_artifacts = {item.evidence_id for item in project_documents}
        known_code = {item.evidence_id for item in code_documents}
        valid_artifacts = [item for item in artifact_ids if item in known_artifacts]
        valid_code = [item for item in code_ids if item in known_code]
        return valid_artifacts, valid_code

    def _audit(
        self,
        root: Path,
        *,
        operation: str,
        question: str,
        project: ProjectEvidenceBundle,
        code_status: str,
        code_documents: Sequence[EvidenceDocument],
        output_path: Path,
        prompt_trace: Dict[str, Any],
        routing: Optional[EvidenceRouting] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "question": question,
            "artifact_evidence_ids": [item.evidence_id for item in project.documents],
            "artifact_paths": [item.path for item in project.documents],
            "code_retrieval_status": code_status,
            "code_evidence_ids": [item.evidence_id for item in code_documents],
            "output_path": str(output_path),
            "prompt": prompt_trace,
        }
        if routing is not None:
            record["evidence_routing"] = routing.model_dump(mode="json")
        decisions = load_human_review_decisions(root)
        if decisions:
            record["human_review_decision_ids"] = [
                item.get("decision_id") for item in decisions if item.get("decision_id")
            ]
        audit_path = root / "design_review" / "review_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def assess(self, *, output_dir: str | Path, config_path: str | Path) -> DesignAssessment:
        question = "对当前桥梁概念设计成果进行完整性、工程一致性、验算结论和方法边界评估"
        # 整体评估的问题固定且必然要覆盖全部阶段，这里不做路由，
        # 直接用核心成果全集——省掉一次调用，也避免评估被路由裁窄。
        project = collect_project_evidence(output_dir, question, max_documents=10)
        code_status, code_documents, index_version = self._retrieve_code_evidence_many(
            [
                "JTG 3362 矩形截面正截面抗弯承载力 5.2.2 相对界限受压区高度 5.2.1",
                "JTG 3362 盖梁斜截面抗剪承载力 8.4.4",
                "JTG 3362 盖梁斜截面抗剪配筋承载力 8.4.5",
            ],
            config_path,
        )
        code_text = _render_code_evidence(code_documents, code_status)
        prompt = render_prompt(
            "tasks.design_review_assessment.v1",
            {
                "project_evidence": project.prompt_text,
                "code_evidence": code_text,
                "index_version": index_version or "disabled",
            },
            config_path=str(config_path),
        )
        response = self._llm(config_path).invoke(
            [("system", self._skill().content), ("user", prompt.user_content)]
        )
        parsed = extract_json_object(response.content)
        assessment = DesignAssessment.model_validate(parsed)
        combined_text = "\n".join(
            [
                assessment.overall_conclusion,
                *assessment.stage_findings,
                *assessment.limitations,
                *assessment.recommended_actions,
                *[f"[{item}]" for item in assessment.cited_artifact_ids],
                *[f"[{item}]" for item in assessment.cited_code_evidence_ids],
            ]
        )
        artifact_ids, code_ids = self._validate_citations(combined_text, project.documents, code_documents)
        assessment.cited_artifact_ids = artifact_ids
        assessment.cited_code_evidence_ids = code_ids
        assessment.code_retrieval_status = code_status
        root = Path(output_dir).resolve()
        artifact_documents = {item.evidence_id: item for item in project.documents}
        code_evidence_documents = {item.evidence_id: item for item in code_documents}
        citation_sources: Dict[str, str] = {}
        for evidence_id in artifact_ids:
            document = artifact_documents[evidence_id]
            path = Path(document.path).resolve()
            try:
                display_path = str(path.relative_to(root))
            except ValueError:
                display_path = path.name
            citation_sources[evidence_id] = f"成果文件：{display_path}"
        for evidence_id in code_ids:
            citation_sources[evidence_id] = (
                f"规范依据：{code_evidence_documents[evidence_id].source}"
            )
        assessment.citation_sources = citation_sources
        assessment.display_text = _render_assessment_display(assessment)
        review_dir = root / "design_review"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = review_dir / f"assessment_{stamp}.json"
        assessment.assessment_path = str(output_path)
        _write_json(output_path, assessment.model_dump(mode="json"))
        _write_json(review_dir / "latest_assessment.json", assessment.model_dump(mode="json"))
        self._audit(
            root,
            operation="assessment",
            question=question,
            project=project,
            code_status=code_status,
            code_documents=code_documents,
            output_path=output_path,
            prompt_trace=prompt.to_dict(),
        )
        return assessment

    def answer(
        self,
        question: str,
        *,
        output_dir: str | Path,
        config_path: str | Path,
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> ReviewAnswer:
        """回答一个关于本次设计的问题。

        两步：先用一次很轻的路由调用判断"这次回答要哪些成果的正文、要不要查规范"，
        再据此装配证据、生成回答。路由结论连同最终喂进去的证据一起落到审计里，
        这样"模型是凭什么答的"可以事后回溯。
        """
        if not str(question or "").strip():
            raise ValueError("专业问答问题不能为空。")
        history_text = _render_conversation_history(history)
        root, all_documents = scan_project_documents(output_dir, question)
        routing = self._route_evidence(question, all_documents, root, config_path)
        project = build_project_evidence(
            root,
            question,
            all_documents,
            wanted_types=routing.artifact_types,
            max_documents=8,
        )
        if routing.need_code_rag:
            code_status, code_documents, index_version = self._retrieve_code_evidence_many(
                routing.code_queries, config_path
            )
        else:
            code_status, code_documents, index_version = "not_applicable", [], ""
        prompt = render_prompt(
            "tasks.design_review_qa.v1",
            {
                "question": question,
                "history": history_text,
                "project_evidence": project.prompt_text,
                "code_evidence": _render_code_evidence(code_documents, code_status),
                "index_version": index_version or "disabled",
            },
            config_path=str(config_path),
        )
        response = self._llm(config_path).invoke(
            [("system", self._skill("design-review-qa").content), ("user", prompt.user_content)]
        )
        answer_text = str(response.content or "").strip()
        artifact_ids, code_ids = self._validate_citations(answer_text, project.documents, code_documents)
        review_dir = root / "design_review"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        answer_path = review_dir / f"answer_{stamp}.md"
        if history_text:
            body = f"# 问题\n\n{question}\n\n# 对话上下文\n\n{history_text}\n\n# 回答\n\n{answer_text}\n"
        else:
            body = f"# 问题\n\n{question}\n\n# 回答\n\n{answer_text}\n"
        _write_text(answer_path, body)
        result = ReviewAnswer(
            question=question,
            answer=answer_text,
            cited_artifact_ids=artifact_ids,
            cited_code_evidence_ids=code_ids,
            answer_path=str(answer_path),
            code_retrieval_status=code_status,
        )
        self._audit(
            root,
            operation="answer",
            question=question,
            project=project,
            code_status=code_status,
            code_documents=code_documents,
            output_path=answer_path,
            prompt_trace=prompt.to_dict(),
            routing=routing,
        )
        return result
