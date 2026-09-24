from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge_agents.code_rag.schemas import (
    CodeEvidenceBundle,
    CodeQuery,
    EvidenceDocument,
    RetrievalTrace,
)
from bridge_agents.design_review import (
    CHAT_HISTORY_MAX_TURNS,
    DesignReviewAgent,
    ReviewEvidenceUnavailableError,
    _render_conversation_history,
    _review_evidence_applicable,
    available_artifact_types,
    build_project_evidence,
    collect_project_evidence,
    load_human_review_decisions,
    render_artifact_menu,
    render_human_review_records,
    scan_project_documents,
)


def _write_project_outputs(root: Path) -> None:
    layout = root / "layout_revision" / "final_layout_result.json"
    layout.parent.mkdir(parents=True)
    layout.write_text(
        json.dumps({"桥梁布跨设计": [{"桥梁编号": "1-L", "跨径组合": "3×30"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    capacity = root / "capacity_check" / "capacity_check_batch_summary.json"
    capacity.parent.mkdir(parents=True)
    capacity.write_text(
        json.dumps(
            {
                "overall_check": {"all_ok": True, "compression_zone_ok": True},
                "utilization_summary": {"max_utilization": 0.82, "task_id": "1-L-1-G1"},
                "formula_coverage": {"not_covered": ["裂缝", "挠度"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    prompt = root / "prompts" / "ignored_prompt.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("这不是工程成果", encoding="utf-8")
    loose_prompt = root / "design_run_001" / "prompt.txt"
    loose_prompt.parent.mkdir(parents=True)
    loose_prompt.write_text("这同样不是工程成果", encoding="utf-8")


def _enabled_config(root: Path) -> Path:
    path = root / "settings.yaml"
    path.write_text("code_rag:\n  enabled: true\n  retrieval_mode: keyword_only\n", encoding="utf-8")
    return path


def _evidence_bundle(query: CodeQuery, *, status: str = "ok") -> CodeEvidenceBundle:
    documents = []
    if status == "ok":
        documents = [
            EvidenceDocument(
                evidence_id="EVID_R_3362_CH08_8_4_5",
                primary_entity_id="R_3362_CH08_8_4_5",
                evidence_type="rule",
                standard_code="JTG 3362-2018",
                clause="8.4.5",
                title="盖梁斜截面抗剪承载力",
                source="JTG 3362-2018, 8.4.5条",
                normative_text="盖梁斜截面抗剪承载力规定。",
                prompt_text="[EVID_R_3362_CH08_8_4_5] 盖梁斜截面抗剪承载力规定。",
                search_text="盖梁 抗剪",
            )
        ]
    return CodeEvidenceBundle(
        request=query,
        retrieval_status=status,
        index_version="test-index",
        evidence_documents=documents,
        retrieval_trace=RetrievalTrace(retrieval_mode="keyword_only"),
    )


class FakeRAGService:
    def __init__(self, status: str = "ok") -> None:
        self.status = status
        self.queries: list[CodeQuery] = []

    def retrieve(self, query: CodeQuery) -> CodeEvidenceBundle:
        self.queries.append(query)
        return _evidence_bundle(query, status=self.status)


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return SimpleNamespace(content=self.responses.pop(0))


def _routing_payload(**overrides) -> dict:
    payload = {
        "artifact_types": ["final_layout", "capacity_check_batch"],
        "need_code_rag": True,
        "code_queries": ["盖梁 斜截面抗剪承载力"],
        "rationale": "问题同时涉及布跨与验算结论。",
    }
    payload.update(overrides)
    return payload


class FakeRoutingLLM:
    """路由调用专用 fake：`llm=` 留给生成回答，两个调用互不抢响应。"""

    def __init__(self, payload: dict | None = None, *, raw: str | None = None) -> None:
        self.payload = _routing_payload() if payload is None else payload
        self.raw = raw
        self.messages: list = []

    def invoke(self, messages):
        self.messages.append(messages)
        if self.raw is not None:
            return SimpleNamespace(content=self.raw)
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


def _user_prompt(llm: FakeLLM) -> str:
    """取本次调用真正喂给模型的用户消息（messages 是 [(role, content), ...]，首条是 system）。"""
    messages = llm.messages[0]
    return messages[-1][1]


def _agent(
    llm: FakeLLM,
    *,
    rag=None,
    routing=None,
) -> DesignReviewAgent:
    return DesignReviewAgent(
        llm=llm,
        rag_service=rag if rag is not None else FakeRAGService(),
        routing_llm=routing if routing is not None else FakeRoutingLLM(),
    )


def test_project_evidence_prioritizes_engineering_results_and_excludes_prompts(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)

    bundle = collect_project_evidence(tmp_path, "盖梁验算最大利用率是多少", max_documents=4)

    assert bundle.documents
    assert any("capacity_check_batch_summary.json" in item.path for item in bundle.documents)
    assert all("ignored_prompt.txt" not in item.path for item in bundle.documents)
    assert all("prompt.txt" not in item.path for item in bundle.documents)
    assert "0.82" in bundle.prompt_text
    assert all(item.evidence_id.startswith("ARTIFACT-") for item in bundle.documents)


_FULL_RESULT_FILES = {
    "structural_design/structural_design_result.json": {"status": "completed"},
    "structural_design/design_units/design_units_result.json": {"units": 5},
    "structural_design/dimension_design/dimension_design_result.json": {"groups": 5},
    "structural_design/pier_group/pier_group_result.json": {"groups": 5},
    "structural_design/reinforcement_design/reinforcement_design_result.json": {"groups": 10},
    "deliverables/design_manifest.json": {"status": "completed"},
    "deliverables/drawings/drawing_index.json": {"drawing_groups": 5},
    "collision_detection/collision_metrics_design_result_001.json": {"collision_count": 0},
}


def _write_full_outputs(root: Path) -> None:
    """写出十个阶段的完整成果，用于验证证据选取。"""
    _write_project_outputs(root)
    for relative_path, payload in _FULL_RESULT_FILES.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_project_evidence_prefers_complete_results_and_excludes_process_files(tmp_path: Path) -> None:
    _write_full_outputs(tmp_path)
    process_file = tmp_path / "revision_instructions" / "revision_instruction_round_1.txt"
    process_file.parent.mkdir(parents=True)
    process_file.write_text("盖梁 承载力 布跨 配筋", encoding="utf-8")

    bundle = collect_project_evidence(tmp_path, "总结布跨、结构、配筋、验算和图纸", max_documents=10)

    artifact_types = {item.artifact_type for item in bundle.documents}
    assert {
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
    } <= artifact_types
    assert all("revision_instruction" not in item.path for item in bundle.documents)
    assert [item.evidence_id for item in bundle.documents] == [
        f"ARTIFACT-{index:03d}" for index in range(1, len(bundle.documents) + 1)
    ]


def catalog_marked_count(bundle) -> int:
    """统计清单里被标记为"本次已提供正文"的成果条数。"""
    return sum(
        1
        for line in bundle.catalog_text.splitlines()
        if line.startswith("- ") and "[ARTIFACT-" in line
    )


def test_project_evidence_keeps_named_stages_on_topic(tmp_path: Path) -> None:
    """路由只点名布跨时只给布跨正文，其余阶段只出现在清单里。"""
    _write_full_outputs(tmp_path)

    bundle = collect_project_evidence(
        tmp_path,
        "目前初步设桥布跨阶段的设计结果是什么",
        max_documents=8,
        wanted_types=["final_layout"],
    )

    types = [item.artifact_type for item in bundle.documents]
    # 模型点名的阶段排在最前（引用编号从它起），交付清单补在后面
    assert types == ["final_layout", "final_deliverables"]
    # 未被选中的阶段仍要在清单里可见，供用户指名追问。
    assert "design_manifest.json" in bundle.catalog_text
    assert "reinforcement_design_result.json" in bundle.catalog_text
    assert catalog_marked_count(bundle) == len(bundle.documents)


def test_project_evidence_takes_only_the_named_stage(tmp_path: Path) -> None:
    _write_full_outputs(tmp_path)

    bundle = collect_project_evidence(
        tmp_path,
        "盖梁配筋的箍筋怎么布置",
        max_documents=8,
        wanted_types=["reinforcement_design"],
    )

    types = [item.artifact_type for item in bundle.documents]
    assert "reinforcement_design" in types
    assert "capacity_check_batch" not in types
    assert "final_deliverables" in types  # 交付清单始终在线


def test_project_evidence_defaults_to_every_core_stage(tmp_path: Path) -> None:
    """不传 wanted_types 时按核心成果全集装配——整体评估走这条默认。"""
    _write_full_outputs(tmp_path)

    bundle = collect_project_evidence(tmp_path, "总结所有阶段的成果", max_documents=10)

    assert {item.artifact_type for item in bundle.documents} == {
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
    }


def test_catalog_marks_exactly_the_evidence_with_body(tmp_path: Path) -> None:
    _write_full_outputs(tmp_path)

    bundle = collect_project_evidence(
        tmp_path,
        "桥位 2 的跨径组合是什么",
        max_documents=8,
        wanted_types=["final_layout"],
    )

    for item in bundle.documents:
        relative = Path(item.path).resolve().relative_to(tmp_path.resolve()).as_posix()
        assert relative in bundle.catalog_text
        assert f"[{item.evidence_id}]" in bundle.catalog_text
    assert catalog_marked_count(bundle) == len(bundle.documents)


def test_artifact_menu_lists_type_keys_with_digests(tmp_path: Path) -> None:
    """路由菜单要能自解释：类型键、中文标签、路径、一行摘要。"""
    _write_full_outputs(tmp_path)
    root, documents = scan_project_documents(tmp_path, "布跨")

    menu = render_artifact_menu(documents, root)

    assert "- final_layout（布跨设计成果）：" in menu
    assert "layout_revision/final_layout_result.json" in menu
    assert "顶层字段：桥梁布跨设计" in menu
    assert "- final_deliverables（最终交付清单）：" in menu
    # 过程文件不进菜单
    assert "ignored_prompt" not in menu


def test_artifact_menu_dedupes_repeated_stage_outputs_and_summarizes_big_json(tmp_path: Path) -> None:
    """同一阶段的多次产出只列代表文件；被字符上限截断的 JSON 仍要能给出顶层字段。"""
    _write_project_outputs(tmp_path)
    big = tmp_path / "structural_design" / "reinforcement_design" / "reinforcement_design_result.json"
    big.parent.mkdir(parents=True)
    big.write_text(
        json.dumps({"reinforcement": {"pier_cap": "x" * 20000}}, ensure_ascii=False),
        encoding="utf-8",
    )
    for index in (1, 2):
        sibling = big.parent / f"reinforcement_result_{index}.json"
        sibling.write_text(json.dumps({"group": index}), encoding="utf-8")

    root, documents = scan_project_documents(tmp_path, "配筋")
    menu = render_artifact_menu(documents, root)

    assert menu.count("reinforcement_design（配筋设计成果）") == 1
    assert "reinforcement_design_result.json" in menu
    assert "本阶段另有 2 份产出，未列出" in menu
    # 20000 字符的成果会被截断成非法 JSON，摘要不能退化成单独一个"{"
    assert "顶层字段：reinforcement" in menu
    assert "—— {（" not in menu


def test_available_types_follows_stage_order(tmp_path: Path) -> None:
    _write_full_outputs(tmp_path)
    _, documents = scan_project_documents(tmp_path, "布跨")

    types = available_artifact_types(documents)

    assert types[:3] == ["final_layout", "structural_design", "design_units"]
    assert set(types) == {
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
    }


def test_review_agent_assesses_project_with_project_and_code_evidence(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "assessment_status": "completed_with_limitations",
                    "overall_conclusion": "当前概念设计流程已完成，承载力验算通过。[ARTIFACT-002]",
                    "stage_findings": ["最大利用率为0.82。[ARTIFACT-002]"],
                    "limitations": ["裂缝和挠度尚未覆盖。[ARTIFACT-002]"],
                    "recommended_actions": ["施工图阶段补充正常使用极限状态验算。"],
                    "cited_artifact_ids": ["ARTIFACT-002"],
                    "cited_code_evidence_ids": ["EVID_R_3362_CH08_8_4_5"],
                },
                ensure_ascii=False,
            )
        ]
    )
    rag = FakeRAGService()
    agent = _agent(llm, rag=rag)

    result = agent.assess(output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    assert result.assessment_status == "completed_with_limitations"
    assert result.cited_code_evidence_ids == ["EVID_R_3362_CH08_8_4_5"]
    assert (tmp_path / "design_review" / "latest_assessment.json").exists()
    audit = json.loads((tmp_path / "design_review" / "review_audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert audit["operation"] == "assessment"
    assert audit["code_evidence_ids"] == ["EVID_R_3362_CH08_8_4_5"]
    assert rag.queries[0].standard_codes == []
    assert len(rag.queries) == 3
    assert "已完成成果与阶段结论" in result.display_text
    assert "成果文件：capacity_check\\capacity_check_batch_summary.json" in result.display_text
    assert "ARTIFACT-" not in result.display_text
    assert "当前约定的概念设计范围" in llm.messages[0][1][1]


def test_review_agent_normalizes_structured_narrative_items(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "assessment_status": "completed_with_limitations",
                    "overall_conclusion": "当前成果可用于概念设计评估。",
                    "stage_findings": [
                        {
                            "stage": "下部结构",
                            "findings": ["承载力验算通过", "最大利用率为0.82"],
                            "citations": ["ARTIFACT-002", "EVID_R_3362_CH08_8_4_5"],
                        }
                    ],
                    "limitations": {"description": "裂缝和挠度尚未覆盖"},
                    "recommended_actions": [
                        {"content": "施工图阶段补充正常使用极限状态验算"}
                    ],
                    "cited_artifact_ids": [],
                    "cited_code_evidence_ids": [],
                },
                ensure_ascii=False,
            )
        ]
    )
    agent = _agent(llm)

    result = agent.assess(output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    assert result.stage_findings == [
        "下部结构：承载力验算通过；最大利用率为0.82 "
        "[ARTIFACT-002] [EVID_R_3362_CH08_8_4_5]"
    ]
    assert result.limitations == ["裂缝和挠度尚未覆盖"]
    assert result.recommended_actions == ["施工图阶段补充正常使用极限状态验算"]
    assert result.cited_artifact_ids == ["ARTIFACT-002"]
    assert result.cited_code_evidence_ids == ["EVID_R_3362_CH08_8_4_5"]
    assert "ARTIFACT-" not in result.display_text
    assert "EVID_" not in result.display_text
    assert "规范依据：JTG 3362-2018, 8.4.5条" in result.display_text


def test_review_agent_answers_follow_up_with_traceable_citations(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    llm = FakeLLM(["最大承载力利用率为0.82。[ARTIFACT-002] 盖梁抗剪依据见[EVID_R_3362_CH08_8_4_5]。"])
    agent = _agent(llm)

    result = agent.answer(
        "当前盖梁最大利用率是多少，抗剪依据是什么？",
        output_dir=tmp_path,
        config_path=_enabled_config(tmp_path),
    )

    assert "0.82" in result.answer
    assert result.cited_artifact_ids == ["ARTIFACT-002"]
    assert result.cited_code_evidence_ids == ["EVID_R_3362_CH08_8_4_5"]
    assert Path(result.answer_path).exists()


def test_review_agent_routes_evidence_by_model_decision(tmp_path: Path) -> None:
    """路由点名哪个阶段，就只喂哪个阶段的正文。"""
    _write_full_outputs(tmp_path)
    llm = FakeLLM(["布跨成果见 [ARTIFACT-002]。"])
    routing = FakeRoutingLLM(
        _routing_payload(
            artifact_types=["final_layout"],
            need_code_rag=False,
            code_queries=[],
            rationale="问的是布跨阶段结果，不涉及规范条文。",
        )
    )
    rag = FakeRAGService()
    agent = _agent(llm, rag=rag, routing=routing)

    result = agent.answer(
        "目前初步设桥布跨阶段的设计结果是什么",
        output_dir=tmp_path,
        config_path=_enabled_config(tmp_path),
    )

    user_prompt = llm.messages[-1][1][1]
    # 清单里仍会列出配筋成果的文件名，但正文段只应有被点名的阶段
    evidence_body = user_prompt.split("## 项目成果证据", 1)[1]
    assert "final_layout_result.json" in evidence_body
    assert "reinforcement_design_result.json" not in evidence_body
    assert "reinforcement_design_result.json" in user_prompt
    # 不需要规范时直接跳过检索，并在提示词里说明本次未检索
    assert rag.queries == []
    assert "本次未检索规范条文" in user_prompt
    assert result.code_retrieval_status == "not_applicable"
    assert result.cited_code_evidence_ids == []


def test_review_agent_routing_sees_the_artifact_menu(tmp_path: Path) -> None:
    _write_full_outputs(tmp_path)
    routing = FakeRoutingLLM(_routing_payload(artifact_types=["final_layout"]))
    agent = _agent(FakeLLM(["答。[ARTIFACT-002]"]), routing=routing)

    agent.answer("布跨结果", output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    menu_prompt = routing.messages[0][0][1]
    assert "用户问题：\n布跨结果" in menu_prompt
    assert "- final_layout（布跨设计成果）：" in menu_prompt
    assert "顶层字段：桥梁布跨设计" in menu_prompt
    assert "final_layout、structural_design" in menu_prompt  # 可选类型键清单


def test_review_agent_records_routing_decision_in_audit(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    routing = FakeRoutingLLM(
        _routing_payload(
            artifact_types=["capacity_check_batch"],
            code_queries=["盖梁 斜截面抗剪承载力", "JTG 3362 8.4.5"],
            rationale="问的是验算利用率与其规范依据。",
        )
    )
    rag = FakeRAGService()
    agent = _agent(FakeLLM(["通过。[ARTIFACT-002]"]), rag=rag, routing=routing)

    agent.answer("盖梁最大利用率是多少", output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    audit = json.loads(
        (tmp_path / "design_review" / "review_audit.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert audit["operation"] == "answer"
    assert audit["evidence_routing"]["artifact_types"] == ["capacity_check_batch"]
    assert audit["evidence_routing"]["rationale"] == "问的是验算利用率与其规范依据。"
    assert audit["evidence_routing"]["source"] == "model"
    # 路由给的检索词逐条送到检索器，而不是拿用户原话去搜
    assert [query.query_text for query in rag.queries] == [
        "盖梁 斜截面抗剪承载力",
        "JTG 3362 8.4.5",
    ]


def test_review_agent_drops_unknown_types_and_caps_routing_output(tmp_path: Path) -> None:
    """模型自创类型名或超量点名时只保留真实存在的阶段，并截断到上限。"""
    _write_full_outputs(tmp_path)
    routing = FakeRoutingLLM(
        _routing_payload(
            artifact_types=[
                "final_layout",
                "布跨设计成果",  # 中文标签，不是类型键
                "不存在的阶段",
                "final_layout",  # 重复
                "structural_design",
                "design_units",
                "dimension_design",
                "pier_group",
                "reinforcement_design",
                "capacity_check_batch",
                "final_deliverables",
                "drawing_package",
                "collision_check",
            ],
            code_queries=["q1", "q2", "q3", "q4", "q5", "q6"],
        )
    )
    agent = _agent(FakeLLM(["答。[ARTIFACT-002]"]), routing=routing)

    agent.answer("布跨与结构设计", output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    audit = json.loads(
        (tmp_path / "design_review" / "review_audit.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    # 11 个有效类型里保留前 8 个：上限与 answer 的证据预算一致
    assert audit["evidence_routing"]["artifact_types"] == [
        "final_layout",
        "structural_design",
        "design_units",
        "dimension_design",
        "pier_group",
        "reinforcement_design",
        "capacity_check_batch",
        "final_deliverables",
    ]
    assert audit["evidence_routing"]["code_queries"] == ["q1", "q2", "q3", "q4"]


def test_review_agent_falls_back_when_routing_names_nothing_usable(tmp_path: Path) -> None:
    """路由给不出有效阶段时退回总览类成果，并按用户原话检索规范。"""
    _write_project_outputs(tmp_path)
    routing = FakeRoutingLLM(_routing_payload(artifact_types=["不存在的阶段"], code_queries=[]))
    rag = FakeRAGService()
    agent = _agent(FakeLLM(["答。[ARTIFACT-002]"]), rag=rag, routing=routing)

    agent.answer("当前设计结果如何", output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    audit = json.loads(
        (tmp_path / "design_review" / "review_audit.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert audit["evidence_routing"]["source"] == "fallback"
    assert audit["evidence_routing"]["artifact_types"] == ["final_layout", "final_deliverables"]
    assert [query.query_text for query in rag.queries] == ["当前设计结果如何"]


def test_review_agent_falls_back_when_routing_is_not_json(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    routing = FakeRoutingLLM(raw="抱歉，我无法判断需要哪些成果。")
    agent = _agent(FakeLLM(["答。[ARTIFACT-002]"]), routing=routing)

    result = agent.answer("当前设计结果如何", output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    audit = json.loads(
        (tmp_path / "design_review" / "review_audit.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert audit["evidence_routing"]["source"] == "fallback"
    assert result.answer == "答。[ARTIFACT-002]"


def test_review_agent_falls_back_without_any_artifact(tmp_path: Path) -> None:
    """目录里没有成果文件时不发起路由调用，直接按总览类兜底。"""
    routing = FakeRoutingLLM()
    agent = _agent(FakeLLM(["没查到成果。[ARTIFACT-001]"]), routing=routing)

    result = agent.answer("这次设计做了什么", output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    assert routing.messages == []
    assert result.code_retrieval_status == "ok"


def test_review_agent_assessment_does_not_route(tmp_path: Path) -> None:
    """整体评估的问题固定覆盖全部阶段，不额外发一次路由调用。"""
    _write_project_outputs(tmp_path)
    routing = FakeRoutingLLM()
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "assessment_status": "completed",
                    "overall_conclusion": "成果完整。[ARTIFACT-002]",
                    "stage_findings": ["布跨已完成。[ARTIFACT-001]"],
                    "limitations": ["未覆盖裂缝。"],
                    "recommended_actions": ["补充正常使用极限状态验算。"],
                    "cited_artifact_ids": ["ARTIFACT-002"],
                    "cited_code_evidence_ids": [],
                },
                ensure_ascii=False,
            )
        ]
    )
    agent = _agent(llm, routing=routing)

    agent.assess(output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    assert routing.messages == []
    audit = json.loads(
        (tmp_path / "design_review" / "review_audit.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert "evidence_routing" not in audit


def test_review_agent_answer_uses_dedicated_qa_skill(tmp_path: Path) -> None:
    """问答复用评估 skill 会把"完整评估 + 方法边界"套在所有问题上，必须分开。"""
    _write_project_outputs(tmp_path)
    llm = FakeLLM(["按证据回答。[ARTIFACT-002]"])
    agent = _agent(llm)

    agent.answer("盖梁最大利用率是多少？", output_dir=tmp_path, config_path=_enabled_config(tmp_path))
    qa_system = llm.messages[-1][0][1]

    assert "只回答用户问到的问题" in qa_system
    assert "先给结论，再说明依据与限制" not in qa_system


def test_review_agent_assessment_keeps_full_evaluation_skill(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "assessment_status": "completed",
                    "overall_conclusion": "成果完整。[ARTIFACT-002]",
                    "stage_findings": ["布跨已完成。[ARTIFACT-001]"],
                    "limitations": ["未覆盖裂缝。"],
                    "recommended_actions": ["补充正常使用极限状态验算。"],
                    "cited_artifact_ids": ["ARTIFACT-002"],
                    "cited_code_evidence_ids": [],
                },
                ensure_ascii=False,
            )
        ]
    )
    agent = _agent(llm)

    agent.assess(output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    system_prompt = llm.messages[-1][0][1]
    assert "完整总结已完成成果及其确定性校核状态" in system_prompt


def test_review_agent_answer_injects_conversation_history(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    llm = FakeLLM(["弯矩包络结论见项目成果。[ARTIFACT-002]"])
    agent = _agent(llm)

    result = agent.answer(
        "那它的弯矩包络怎么样？",
        output_dir=tmp_path,
        config_path=_enabled_config(tmp_path),
        history=[
            {"role": "user", "content": "上一问：盖梁验算通过了吗？"},
            {"role": "assistant", "content": "上一答：验算已通过。"},
        ],
    )

    user_prompt = llm.messages[-1][1][1]  # 最近一批 invoke 消息中的 user 内容
    assert "上一问：盖梁验算通过了吗？" in user_prompt
    assert "助手：上一答：验算已通过。" in user_prompt
    assert "用户问题：那它的弯矩包络怎么样？" in user_prompt
    # 落盘留痕应包含对话上下文段
    saved = Path(result.answer_path).read_text(encoding="utf-8")
    assert "# 对话上下文" in saved
    assert "上一答：验算已通过。" in saved


def test_review_agent_answer_without_history_omits_conversation_block(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    llm = FakeLLM(["按证据回答。[ARTIFACT-002]"])
    agent = _agent(llm)

    agent.answer("盖梁最大利用率是多少？", output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    user_prompt = llm.messages[-1][1][1]
    assert "先前对话记录" not in user_prompt
    assert "助手：" not in user_prompt


def test_render_conversation_history_limits_and_truncates() -> None:
    assert _render_conversation_history(None) == ""
    assert _render_conversation_history([]) == ""

    turns = [
        {"role": "user", "content": f"问题{i}"}
        if i % 2 == 0
        else {"role": "assistant", "content": f"回答{i}"}
        for i in range(2 * (CHAT_HISTORY_MAX_TURNS + 2))
    ]
    rendered = _render_conversation_history(turns)
    # 只保留最近 CHAT_HISTORY_MAX_TURNS 对
    assert f"问题{0}" not in rendered
    assert f"问题{2 * (CHAT_HISTORY_MAX_TURNS - 1)}" in rendered

    long_text = "长" * 5000
    short = _render_conversation_history([{"role": "user", "content": long_text}])
    assert len(short) < 5000
    assert short.endswith("…")


def test_review_agent_filters_unknown_citations_without_blocking(tmp_path: Path) -> None:
    # 回答中引用不存在的证据 id 是模型生成噪音：过滤未知引用，但回答正文仍正常返回。
    _write_project_outputs(tmp_path)
    agent = _agent(FakeLLM(["结论引用了不存在的证据。[ARTIFACT-999][EVID_MISSING]"]))

    result = agent.answer("评价结果", output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    assert "结论引用了不存在的证据" in result.answer
    assert result.cited_artifact_ids == []
    assert result.cited_code_evidence_ids == []


def test_review_agent_distinguishes_disabled_rag_from_invalid_index(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    config = tmp_path / "settings.yaml"
    config.write_text("code_rag:\n  enabled: true\n  retrieval_mode: keyword_only\n", encoding="utf-8")
    agent = _agent(FakeLLM([]), rag=FakeRAGService("invalid_index"))

    with pytest.raises(ReviewEvidenceUnavailableError):
        agent.answer("盖梁抗剪依据", output_dir=tmp_path, config_path=config)

    disabled = tmp_path / "disabled.yaml"
    disabled.write_text("code_rag:\n  enabled: false\n", encoding="utf-8")
    llm = FakeLLM(["仅依据项目成果回答。[ARTIFACT-002]"])
    result = _agent(llm, rag=FakeRAGService("invalid_index")).answer(
        "最大利用率",
        output_dir=tmp_path,
        config_path=disabled,
    )
    assert result.code_retrieval_status == "disabled"


def test_review_applicability_excludes_prestress_only_evidence() -> None:
    query = CodeQuery(query_text="test")
    prestress = _evidence_bundle(query).evidence_documents[0].model_copy(
        update={"title": "体外预应力钢筋抗弯承载力"}
    )
    ordinary = _evidence_bundle(query).evidence_documents[0].model_copy(
        update={"title": "普通钢筋混凝土盖梁抗剪承载力"}
    )

    assert _review_evidence_applicable(prestress) is False
    assert _review_evidence_applicable(ordinary) is True


def _accepted_collision_decision(*, root: Path, **overrides) -> dict:
    """一条"接受风险并继续"的决策记录，字段按真实台账的样子写（subject.path 存绝对路径）。"""
    record = {
        "schema_version": 1,
        "decision_id": "94439050-cda2-43a7-af31-978f73992525",
        "review_type": "layout_collision_review",
        "action": "accept_and_continue",
        "decided_at": "2026-09-17T07:29:23.112322+00:00",
        "extra_rounds": 1,
        "subject": {
            "state_key": "layout_result",
            "path": str(root / "revision_results" / "revision_design_round_3.json"),
            "sha256": "0" * 64,
        },
        "accepted_state": {"task_status": "layout_revision_completed"},
        "accepted_risk": {
            "scope": "layout_revision",
            "reason": "人工接受剩余碰撞风险",
            "accepted_at": "2026-09-17T07:29:23.112322+00:00",
            "metrics": {
                "has_collision": True,
                "conflict_column_count": 4,
                "conflict_column_rate": 0.03225806451612903,
            },
        },
    }
    record.update(overrides)
    return record


def _write_human_review_ledger(root: Path, records: list[dict]) -> Path:
    ledger = root / "human_review" / "human_review_decisions.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return ledger


def test_load_human_review_decisions_tolerates_missing_and_malformed_ledger(tmp_path: Path) -> None:
    assert load_human_review_decisions(tmp_path) == []

    ledger = tmp_path / "human_review" / "human_review_decisions.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        '{"decision_id": "a", "action": "accept_and_continue"}\n'
        "\n"
        "{这不是合法 JSON}\n"
        '["数组不是决策记录"]\n'
        '{"decision_id": "b"}\n',
        encoding="utf-8",
    )

    records = load_human_review_decisions(tmp_path)

    # 坏行跳过而不是整份台账报废：丢一条已接受的决策会把闭合事项重新报成待复核。
    assert [record["decision_id"] for record in records] == ["a", "b"]


def test_human_review_records_read_as_closed_decision_not_todo(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    text = render_human_review_records([_accepted_collision_decision(root=tmp_path)], tmp_path)

    assert text.startswith("## 人工复核记录")
    assert "是本次设计已经闭合的事实，不是待办事项" in text
    assert "### 决策 1：layout_collision_review（人工接受风险并继续）" in text
    assert "决策性质：人工已接受该事项，本设计范围内已闭合，不构成待复核事项" in text
    assert "决策编号：94439050-cda2-43a7-af31-978f73992525" in text
    assert "复核对象：revision_results/revision_design_round_3.json" in text
    assert "决策后流程状态：layout_revision_completed" in text
    assert "接受的风险：范围 layout_revision；理由「人工接受剩余碰撞风险」" in text
    # 指标里的布尔与浮点要按中文与紧凑格式渲染，别把 True / 0.03225806451612903 原样丢给模型
    assert "has_collision=是" in text
    assert "conflict_column_count=4" in text
    assert "conflict_column_rate=0.0322581" in text
    assert "人工反馈：（未填写）" in text


def test_human_review_records_mark_non_acceptance_as_not_closed(tmp_path: Path) -> None:
    """要求返修不闭合事项——下轮还得有人拍板，不能当作已决策。"""
    record = _accepted_collision_decision(root=tmp_path, action="revise", accepted_risk=None)

    text = render_human_review_records([record], tmp_path)

    assert "（人工要求返修）" in text
    assert "决策性质：人工未接受该事项（人工要求返修），此项未由本次决策闭合。" in text
    assert "- 接受的风险：" not in text


def test_human_review_records_degrade_to_filename_for_moved_subject(tmp_path: Path) -> None:
    """复核对象被挪走或换了机器时，退回文件名而不是抛异常——历史记录不能因路径失效而丢掉。"""
    record = _accepted_collision_decision(
        root=tmp_path,
        subject={"state_key": "layout_result", "path": "Z:/不在本机/revision_design_round_3.json"},
    )

    text = render_human_review_records([record], tmp_path)

    assert "复核对象：revision_design_round_3.json" in text


def test_human_review_records_renders_nothing_without_decisions() -> None:
    assert render_human_review_records([], Path(".")) == ""


def test_project_evidence_carries_human_review_section_and_keeps_ledger_out_of_artifacts(
    tmp_path: Path,
) -> None:
    _write_project_outputs(tmp_path)
    _write_human_review_ledger(tmp_path, [_accepted_collision_decision(root=tmp_path)])

    _, all_documents = scan_project_documents(tmp_path, "布跨结果")
    bundle = build_project_evidence(
        tmp_path,
        "布跨结果",
        all_documents,
        wanted_types=["final_layout"],
        max_documents=8,
    )

    assert "## 人工复核记录" in bundle.prompt_text
    assert bundle.human_review_text in bundle.prompt_text
    # 台账是决策记录，不是设计成果：不得混进成果清单，也不该占掉一个 ARTIFACT 编号。
    assert "human_review_decisions.jsonl" not in bundle.catalog_text
    assert all("human_review" not in item.path for item in bundle.documents)


def test_project_evidence_omits_human_review_section_without_decisions(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)

    bundle = collect_project_evidence(tmp_path, "布跨结果", max_documents=4)

    assert bundle.human_review_text == ""
    assert "人工复核记录" not in bundle.prompt_text


def test_review_agent_assessment_feeds_human_review_evidence(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    _write_human_review_ledger(tmp_path, [_accepted_collision_decision(root=tmp_path)])
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "assessment_status": "completed_with_limitations",
                    "overall_conclusion": "设计已完成。[ARTIFACT-002]",
                    "stage_findings": ["布跨存在碰撞但已由人工接受。[ARTIFACT-002]"],
                    "limitations": [],
                    "recommended_actions": [],
                    "cited_artifact_ids": ["ARTIFACT-002"],
                    "cited_code_evidence_ids": [],
                },
                ensure_ascii=False,
            )
        ]
    )
    agent = _agent(llm)

    result = agent.assess(output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    user_prompt = _user_prompt(llm)
    assert "## 人工复核记录" in user_prompt
    assert "94439050-cda2-43a7-af31-978f73992525" in user_prompt
    assert result.assessment_status == "completed_with_limitations"
    audit = json.loads(
        (tmp_path / "design_review" / "review_audit.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    # 评估结论凭哪些人工决策下的，要能事后回溯。
    assert audit["human_review_decision_ids"] == ["94439050-cda2-43a7-af31-978f73992525"]


def test_review_agent_answer_also_carries_human_review_evidence(tmp_path: Path) -> None:
    _write_project_outputs(tmp_path)
    _write_human_review_ledger(tmp_path, [_accepted_collision_decision(root=tmp_path)])
    llm = FakeLLM(["布跨存在碰撞，已由人工接受。[ARTIFACT-002]"])
    agent = _agent(llm, routing=FakeRoutingLLM(_routing_payload(need_code_rag=False)))

    agent.answer("布跨阶段有个碰撞怎么办", output_dir=tmp_path, config_path=_enabled_config(tmp_path))

    # 问答路径同样要看得到决策记录：否则问到已接受的风险会答成"尚未处理"。
    assert "## 人工复核记录" in _user_prompt(llm)


def test_assessment_prompt_and_skills_state_that_accepted_risk_is_closed() -> None:
    """口径写在 Prompt 与两个 skill 里：已接受的风险不再是 manual_review_required 的理由。"""
    from bridge_agents.prompt_registry import render_prompt
    from bridge_agents.skill_registry import SkillRegistry

    rendered = render_prompt(
        "tasks.design_review_assessment.v1",
        {
            "project_evidence": "证据",
            "code_evidence": "规范",
            "index_version": "v1",
        },
        config_path="config/settings.yaml",
    )
    prompt_text = f"{rendered.system_content}\n{rendered.user_content}"
    assert "已经人工复核决策的事项一律不构成 `manual_review_required`" in prompt_text
    assert "人工复核记录" in prompt_text

    registry = SkillRegistry()
    for skill_id in ("design-review", "design-review-qa"):
        content = registry.load(skill_id).content
        assert "人工" in content and "闭合" in content, skill_id
    assert "只有最后一种能支撑 `manual_review_required`" in registry.load("design-review").content
