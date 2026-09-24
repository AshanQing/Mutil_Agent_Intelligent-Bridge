from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge_agents.code_rag.schemas import (
    CodeEvidenceBundle,
    EvidenceDocument,
    RetrievalTrace,
)
from bridge_agents.structural_evidence import (
    StructuralEvidenceContext,
    StructuralEvidenceUnavailableError,
    load_structural_rag_settings,
    retrieve_structural_evidence,
)
from tools.dimension_design_prompt_tool import build_dimension_design_prompt_tool
from tools.reinforcement_design_prompt_tool import build_reinforcement_design_prompt_tool
from tools import dimension_design_tool as dimension_runner
from tools import reinforcement_design_tool as reinforcement_runner


def _document(evidence_id: str, title: str, *, materials: list[str] | None = None) -> EvidenceDocument:
    return EvidenceDocument(
        evidence_id=evidence_id,
        primary_entity_id=evidence_id.removeprefix("EVID_"),
        evidence_type="rule",
        standard_code="JTG 3362-2018",
        standard_name="公路钢筋混凝土及预应力混凝土桥涵设计规范",
        clause="9.1.1",
        title=title,
        source=f"JTG 3362-2018, {title}",
        normative_text=title,
        applicability={
            "stage": ["结构设计"],
            "member_type": ["墩台盖梁"],
            "material": materials or ["reinforced_concrete"],
        },
        source_entity_ids=[evidence_id.removeprefix("EVID_")],
        source_files=["data/code/JTG_3362_2018_rules.yaml"],
        prompt_text=f"规范依据：{title}",
        search_text=title,
    )


class _FakeService:
    def __init__(self, responses: list[CodeEvidenceBundle]) -> None:
        self.responses = list(responses)
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def _bundle(request, *documents: EvidenceDocument, status: str = "ok") -> CodeEvidenceBundle:
    return CodeEvidenceBundle(
        request=request,
        retrieval_status=status,
        index_version="test-index-v1",
        evidence_documents=list(documents),
        retrieval_trace=RetrievalTrace(retrieval_mode="keyword_only"),
    )


def _write_config(path: Path, *, enabled: bool) -> None:
    path.write_text(
        "code_rag:\n"
        f"  enabled: {'true' if enabled else 'false'}\n"
        "  retrieval_mode: keyword_only\n"
        "  top_k: 4\n"
        "  max_evidence_per_task: 6\n",
        encoding="utf-8",
    )


def test_disabled_structural_rag_does_not_call_service(tmp_path: Path) -> None:
    """Catches accidental retrieval when the user explicitly disabled code_rag."""
    config_path = tmp_path / "settings.yaml"
    _write_config(config_path, enabled=False)
    service = _FakeService([])

    context = retrieve_structural_evidence(
        {"单元编号": "1-R-1", "桥型": "预应力混凝土T梁"},
        stage="dimension_design",
        config_path=str(config_path),
        service=service,
    )

    assert context.retrieval_status == "disabled"
    assert context.evidence_ids == []
    assert service.requests == []


def test_dimension_retrieval_merges_and_deduplicates_evidence(tmp_path: Path) -> None:
    """Catches duplicate evidence injection across the dimension query profile."""
    config_path = tmp_path / "settings.yaml"
    _write_config(config_path, enabled=True)
    settings = load_structural_rag_settings(str(config_path))
    cover = _document("EVID_COVER", "最小混凝土保护层")
    strength = _document("EVID_STRENGTH", "混凝土强度设计值")

    from bridge_agents.structural_evidence import build_structural_queries

    queries = build_structural_queries(
        {"单元编号": "1-R-1", "桥型": "预应力混凝土T梁"},
        stage="dimension_design",
        settings=settings,
    )
    service = _FakeService([
        _bundle(queries[0], strength, cover),
        _bundle(queries[1], cover),
    ])

    context = retrieve_structural_evidence(
        {"单元编号": "1-R-1", "桥型": "预应力混凝土T梁"},
        stage="dimension_design",
        config_path=str(config_path),
        service=service,
    )

    assert context.retrieval_status == "ok"
    assert context.evidence_ids == ["EVID_STRENGTH", "EVID_COVER"]
    assert "EVID_STRENGTH" in context.prompt_text
    assert "EVID_COVER" in context.prompt_text
    assert len(service.requests) == 2
    assert all(request.materials == [] for request in service.requests)
    assert all(request.member_types == [] for request in service.requests)
    assert service.requests[0].query_text.startswith("混凝土强度设计值表")
    assert service.requests[1].query_text.startswith("普通钢筋最小混凝土保护层厚度")


def test_invalid_index_is_distinct_from_user_disabled(tmp_path: Path) -> None:
    """Catches silent fallback when code_rag is enabled but its index is invalid."""
    config_path = tmp_path / "settings.yaml"
    _write_config(config_path, enabled=True)
    settings = load_structural_rag_settings(str(config_path))

    from bridge_agents.structural_evidence import build_structural_queries

    queries = build_structural_queries(
        {"task_id": "R-1"},
        stage="reinforcement_design",
        settings=settings,
    )
    service = _FakeService([
        _bundle(queries[0], status="invalid_index"),
    ])

    with pytest.raises(StructuralEvidenceUnavailableError, match="规范RAG索引不可用"):
        retrieve_structural_evidence(
            {"task_id": "R-1"},
            stage="reinforcement_design",
            config_path=str(config_path),
            service=service,
        )


def test_non_applicable_material_evidence_is_excluded(tmp_path: Path) -> None:
    """Catches injecting prestressing-only clauses into reinforced-concrete cap design."""
    config_path = tmp_path / "settings.yaml"
    _write_config(config_path, enabled=True)
    settings = load_structural_rag_settings(str(config_path))

    from bridge_agents.structural_evidence import build_structural_queries

    queries = build_structural_queries(
        {"task_id": "R-2"},
        stage="reinforcement_design",
        settings=settings,
    )
    prestress_only = _document(
        "EVID_PRESTRESS",
        "预应力钢筋锚固长度",
        materials=["prestressed_concrete"],
    )
    cover = _document("EVID_COVER", "普通钢筋保护层")
    responses = [
        _bundle(query, *(prestress_only, cover) if index == 0 else ())
        for index, query in enumerate(queries)
    ]

    context = retrieve_structural_evidence(
        {"task_id": "R-2"},
        stage="reinforcement_design",
        config_path=str(config_path),
        service=_FakeService(responses),
    )

    assert context.evidence_ids == ["EVID_COVER"]
    assert context.excluded_evidence_ids == ["EVID_PRESTRESS"]


def test_general_reinforced_concrete_rule_is_applicable_to_cap_task(tmp_path: Path) -> None:
    """Catches general detailing clauses being discarded for lacking a cap-specific member tag."""
    config_path = tmp_path / "settings.yaml"
    _write_config(config_path, enabled=True)
    settings = load_structural_rag_settings(str(config_path))

    from bridge_agents.structural_evidence import build_structural_queries

    queries = build_structural_queries(
        {"task_id": "R-3"},
        stage="reinforcement_design",
        settings=settings,
    )
    general_cover = _document("EVID_GENERAL_COVER", "普通钢筋保护层")
    general_cover.applicability = {
        "stage": ["构造设计", "施工图设计"],
        "member_type": ["一般规定"],
        "object": ["钢筋混凝土受弯构件"],
    }
    responses = [
        _bundle(query, *(general_cover,) if index == 0 else ())
        for index, query in enumerate(queries)
    ]

    context = retrieve_structural_evidence(
        {"task_id": "R-3"},
        stage="reinforcement_design",
        config_path=str(config_path),
        service=_FakeService(responses),
    )

    assert context.evidence_ids == ["EVID_GENERAL_COVER"]


def test_prestressing_only_text_is_excluded_when_metadata_has_no_material(tmp_path: Path) -> None:
    """Catches prestressing-only tables with sparse metadata entering ordinary-rebar design."""
    config_path = tmp_path / "settings.yaml"
    _write_config(config_path, enabled=True)
    settings = load_structural_rag_settings(str(config_path))

    from bridge_agents.structural_evidence import build_structural_queries

    queries = build_structural_queries(
        {"task_id": "R-4"},
        stage="reinforcement_design",
        settings=settings,
    )
    prestress_table = _document("EVID_PRETENSION_TABLE", "预应力钢筋锚固长度")
    prestress_table.applicability = {"inputs": ["concrete_grade"]}
    responses = [
        _bundle(query, *(prestress_table,) if index == 0 else ())
        for index, query in enumerate(queries)
    ]

    context = retrieve_structural_evidence(
        {"task_id": "R-4"},
        stage="reinforcement_design",
        config_path=str(config_path),
        service=_FakeService(responses),
    )

    assert context.evidence_ids == []
    assert context.excluded_evidence_ids == ["EVID_PRETENSION_TABLE"]


def test_reinforcement_queries_include_column_intents(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    _write_config(config_path, enabled=True)
    settings = load_structural_rag_settings(str(config_path))

    from bridge_agents.structural_evidence import build_structural_queries

    queries = build_structural_queries(
        {"task_id": "R-column"},
        stage="reinforcement_design",
        settings=settings,
    )
    texts = " ".join(query.query_text for query in queries)
    assert "轴心受压" in texts
    assert "螺旋箍筋" in texts


def test_column_member_evidence_is_applicable() -> None:
    from bridge_agents.structural_evidence import _is_applicable

    column_doc = _document("EVID_COLUMN_AXIAL", "钢筋混凝土轴心受压构件承载力")
    column_doc.applicability = {
        "stage": ["结构设计"],
        "member_type": ["墩柱", "受压构件"],
        "material": ["reinforced_concrete"],
    }
    assert _is_applicable(column_doc) is True


def _evidence_context(stage: str) -> StructuralEvidenceContext:
    document = _document("EVID_COVER", "普通钢筋最小保护层")
    return StructuralEvidenceContext(
        task_id="1-1-G1",
        stage=stage,
        retrieval_status="ok",
        evidence_documents=[document],
        evidence_ids=[document.evidence_id],
        prompt_text="## 规范证据\n证据ID：EVID_COVER\n普通钢筋最小保护层",
    )


def test_dimension_prompt_ignores_obsolete_structural_evidence_context(tmp_path: Path) -> None:
    """Catches obsolete design-time evidence being injected by a legacy caller."""
    samples = tmp_path / "dimension_samples.yaml"
    samples.write_text(
        "- drawing_id: S1\n"
        "  pier role: 中间墩\n"
        "  system_type: 先简支后连续\n"
        "  deck_width: 12.5\n"
        "  next_deck_width: 12.5\n"
        "  span_length: 30\n",
        encoding="utf-8",
    )
    template = tmp_path / "dimension_template.json"
    template.write_text(
        json.dumps({"当前任务": {"任务需输出内容": {}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    unit = {
        "桥梁编号": "1-R",
        "单元编号": "1-R-1",
        "联号": "第一联",
        "桥型": "预应力混凝土先简支后连续T梁",
        "本联信息": {"跨径组合": "4×30", "跨数": 4},
        "桥面宽度信息": {"起点宽度": 12.5, "终点宽度": 12.5},
        "本联设计分组": [
            {"分组编号": "G1", "墩位角色": "中间墩", "包含桥墩号列表": ["1"]}
        ],
    }

    payload = build_dimension_design_prompt_tool(
        single_unit_input=unit,
        samples_yaml_path=str(samples),
        template_json_path=str(template),
        evidence_context=_evidence_context("dimension_design"),
    )

    assert "规范证据" not in payload["prompt_json"]
    assert "EVID_COVER" not in payload["llm_prompt_text"]


def test_reinforcement_prompt_ignores_obsolete_structural_evidence_context(
    tmp_path: Path,
) -> None:
    """Catches obsolete design-time evidence being injected by a legacy caller."""
    template = tmp_path / "reinforcement.txt"
    template.write_text(
        "CUSTOM\n{{REINFORCEMENT_EXAMPLES}}\n{{CURRENT_REINFORCEMENT_TASK_INPUT}}",
        encoding="utf-8",
    )
    task = {
        "task_id": "1-1-G1",
        "桥墩尺寸信息": {"基本信息": {"pier_role": "中间墩"}},
    }

    payload = build_reinforcement_design_prompt_tool(
        reinforcement_task=task,
        template_yaml_path=str(template),
        evidence_context=_evidence_context("reinforcement_design"),
    )

    assert "规范证据" not in payload["prompt_yaml"]
    assert "EVID_COVER" not in payload["llm_prompt_text"]


def _single_dimension_unit() -> dict:
    return {
        "桥梁编号": "1-R",
        "单元编号": "1-R-1",
        "联号": "第一联",
        "桥型": "预应力混凝土先简支后连续T梁",
        "本联信息": {"跨径组合": "4×30", "跨数": 4},
        "桥面宽度信息": {"起点宽度": 12.5, "终点宽度": 12.5},
        "本联设计分组": [
            {"分组编号": "G1", "墩位角色": "中间墩", "包含桥墩号列表": ["1"]}
        ],
    }


def _write_dimension_fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    samples = tmp_path / "dimension_samples.yaml"
    samples.write_text(
        "- drawing_id: S1\n"
        "  pier role: 中间墩\n"
        "  system_type: 先简支后连续\n"
        "  deck_width: 12.5\n"
        "  next_deck_width: 12.5\n"
        "  span_length: 30\n",
        encoding="utf-8",
    )
    template = tmp_path / "dimension_template.json"
    template.write_text(
        json.dumps({"当前任务": {"任务需输出内容": {}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    return samples, template


def test_dimension_runner_does_not_retrieve_or_inject_code_evidence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Catches design-time RAG being reintroduced into dimension generation."""
    config_path = tmp_path / "settings.yaml"
    _write_config(config_path, enabled=True)
    samples, template = _write_dimension_fixture_files(tmp_path)

    class FakeLlm:
        def invoke(self, messages):
            assert "EVID_COVER" not in messages[-1][1]
            return SimpleNamespace(content='{"分组尺寸设计结果": [], "桥墩尺寸映射关系": []}')

    monkeypatch.setattr(
        dimension_runner,
        "retrieve_structural_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("尺寸设计不得检索规范 RAG")
        ),
        raising=False,
    )
    monkeypatch.setattr(dimension_runner, "_build_llm", lambda _path: FakeLlm())

    result = dimension_runner.dimension_design_tool(
        design_units={"single_unit_inputs": [_single_dimension_unit()]},
        config_path=str(config_path),
        output_dir=str(tmp_path / "output"),
        samples_yaml_path=str(samples),
        template_json_path=str(template),
    )

    assert result["success"] is True
    assert result["evidence_bundles"] == {}


def test_reinforcement_runner_does_not_retrieve_code_evidence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Catches design-time RAG being reintroduced into reinforcement generation."""
    config_path = tmp_path / "settings.yaml"
    _write_config(config_path, enabled=True)
    task = {
        "task_id": "1-1-G1",
        "桥梁编号": "B1",
        "单元编号": "U1",
        "分组编号": "G1",
        "桥墩尺寸信息": {"基本信息": {"pier_role": "中间墩"}},
    }
    monkeypatch.setattr(
        reinforcement_runner,
        "extract_reinforcement_tasks_from_dimension_result",
        lambda **_kwargs: [task],
    )
    monkeypatch.setattr(
        reinforcement_runner,
        "retrieve_structural_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("配筋设计不得检索规范 RAG")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        reinforcement_runner,
        "cap_load_design_tool",
        lambda **kwargs: {
            "success": True,
            "analysis_load_input": {"task_id": kwargs["task_id"]},
            "load_design_output": {},
            "output_files": {},
        },
    )
    monkeypatch.setattr(
        reinforcement_runner,
        "cap_internal_force_analysis_tool",
        lambda **_kwargs: {
            "success": True,
            "internal_force_output": {
                "combined_envelopes_full_beam": {"ULS": {}},
                "note_moment_sign": "test",
            },
            "output_files": {},
        },
    )
    monkeypatch.setattr(
        reinforcement_runner,
        "cap_force_control_info_tool",
        lambda **_kwargs: {
            "success": True,
            "force_control_info": {},
            "output_files": {},
        },
    )
    monkeypatch.setattr(
        reinforcement_runner,
        "build_reinforcement_design_prompt_tool",
        lambda **_kwargs: {
            "prompt_yaml": {},
            "llm_prompt_text": "joint reinforcement task",
            "prompt_id": "test.reinforcement",
            "prompt_version": "1",
            "template_sha256": "sha",
        },
    )

    class FakeLlm:
        def invoke(self, _messages):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "reinforcement": {
                            "pier_cap": {"z_patterns": {}},
                            "pier_column": {"longitudinal_bars": []},
                        }
                    },
                    ensure_ascii=False,
                )
            )

    monkeypatch.setattr(reinforcement_runner, "_build_llm", lambda _path: FakeLlm())

    result = reinforcement_runner.reinforcement_design_tool(
        dimension_design_result={"sample": True},
        config_path=str(config_path),
        output_dir=str(tmp_path / "output"),
    )

    assert result["success"] is True
    assert result["evidence_bundles"] == {}
