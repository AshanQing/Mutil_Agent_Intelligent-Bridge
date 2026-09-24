from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bridge_agents.prompt_audit import record_prompt_audit, reserve_artifact_path
from tools import dimension_design_tool as dimension_runner
from tools import reinforcement_design_tool as reinforcement_runner
from tools.dimension_design_prompt_tool import build_dimension_design_prompt_tool


def _single_unit(role: str = "中间墩") -> dict:
    return {
        "桥梁编号": "1",
        "单元编号": "1-1",
        "联号": "第一联",
        "桥型": "预应力混凝土先简支后连续T梁",
        "本联信息": {"跨径组合": "4×30", "跨数": 4},
        "桥面宽度信息": {
            "宽度类型": "标准宽度",
            "起点宽度": 12.5,
            "终点宽度": 12.5,
            "宽度变化说明": "等宽",
        },
        "本联设计分组": [
            {"分组编号": "G1", "墩位角色": role, "包含桥墩号列表": ["1", "2", "3"]}
        ],
    }


def _write_dimension_samples(path: Path, *, role: str = "中间墩", system: str = "先简支后连续") -> None:
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "drawing_id": "S1",
                    "pier role": role,
                    "system_type": system,
                    "deck_width": 12.5,
                    "next_deck_width": 12.5,
                    "span_length": 30,
                }
            ],
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _write_dimension_template(path: Path) -> None:
    path.write_text(
        json.dumps(
            {"自定义模板标记": "OVERRIDE", "当前任务": {"任务需输出内容": {}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_dimension_samples_fall_back_to_same_role_across_system(tmp_path: Path) -> None:
    samples = tmp_path / "samples.yaml"
    template = tmp_path / "template.json"
    _write_dimension_samples(samples, system="简支")
    _write_dimension_template(template)

    payload = build_dimension_design_prompt_tool(
        single_unit_input=_single_unit(),
        samples_yaml_path=str(samples),
        template_json_path=str(template),
    )

    assert payload["match_debug"][0]["匹配层级"] == "同角色跨体系"
    assert payload["match_debug"][0]["匹配样本数量"] == 1


def test_legacy_transition_role_is_normalized_with_warning(tmp_path: Path) -> None:
    samples = tmp_path / "samples.yaml"
    template = tmp_path / "template.json"
    _write_dimension_samples(samples, role="边墩")
    _write_dimension_template(template)

    payload = build_dimension_design_prompt_tool(
        single_unit_input=_single_unit("过渡墩"),
        samples_yaml_path=str(samples),
        template_json_path=str(template),
    )

    assert payload["match_debug"][0]["墩位角色"] == "边墩"
    assert any("过渡墩" in item for item in payload["compatibility_warnings"])
    assert payload["prompt_json"]["当前任务"]["任务输入"]["仅桥墩设计分组"][0]["墩位角色"] == "边墩"


def test_dimension_prompt_rejects_active_group_without_examples(tmp_path: Path) -> None:
    samples = tmp_path / "samples.yaml"
    template = tmp_path / "template.json"
    _write_dimension_samples(samples, role="边墩")
    _write_dimension_template(template)

    with pytest.raises(ValueError, match="中间墩.*未匹配到示例样本"):
        build_dimension_design_prompt_tool(
            single_unit_input=_single_unit("中间墩"),
            samples_yaml_path=str(samples),
            template_json_path=str(template),
        )


def test_dimension_runner_honors_explicit_template_override(monkeypatch, tmp_path: Path) -> None:
    samples = tmp_path / "samples.yaml"
    template = tmp_path / "template.json"
    _write_dimension_samples(samples)
    _write_dimension_template(template)

    class FakeLlm:
        def invoke(self, _messages):
            return SimpleNamespace(content='{"分组尺寸设计结果": [], "桥墩尺寸映射关系": []}')

    monkeypatch.setattr(dimension_runner, "_build_llm", lambda _config_path: FakeLlm())
    result = dimension_runner.dimension_design_tool(
        design_units={"single_unit_inputs": [_single_unit()]},
        output_dir=str(tmp_path / "output"),
        samples_yaml_path=str(samples),
        template_json_path=str(template),
    )

    assert result["success"] is True
    prompt_json = json.loads(
        (tmp_path / "output/structural_design/dimension_design/prompt_1-1.json").read_text(encoding="utf-8")
    )
    assert prompt_json["自定义模板标记"] == "OVERRIDE"
    audit_rows = [
        json.loads(line)
        for line in (tmp_path / "output/logs/prompt_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["status"] for row in audit_rows] == ["rendered", "completed"]
    assert audit_rows[0]["prompt_id"] == "external:template.json"


def _reinforcement_task() -> dict:
    return {
        "task_id": "1-1-G1",
        "桥梁编号": "1",
        "单元编号": "1-1",
        "联号": "第一联",
        "分组编号": "G1",
        "包含桥墩号列表": ["1"],
        "墩位角色": "中间墩",
        "桥墩尺寸信息": {"基本信息": {"pier_role": "中间墩"}},
    }


def test_reinforcement_saves_context_before_preprocessing_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reinforcement_runner, "extract_reinforcement_tasks_from_dimension_result", lambda **_kwargs: [_reinforcement_task()])
    monkeypatch.setattr(reinforcement_runner, "_build_llm", lambda _config_path: object())
    monkeypatch.setattr(reinforcement_runner, "cap_load_design_tool", lambda **_kwargs: {"success": False, "error": "bad load"})

    result = reinforcement_runner.reinforcement_design_tool(
        dimension_design_result={"sample": True},
        output_dir=str(tmp_path),
    )

    assert result["success"] is False
    context_path = Path(result["errors"][0]["task_context_path"])
    assert context_path.exists()
    assert json.loads(context_path.read_text(encoding="utf-8"))["reinforcement_task"]["task_id"] == "1-1-G1"
    audit_rows = [
        json.loads(line)
        for line in (tmp_path / "logs/prompt_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert audit_rows[-1]["status"] == "preprocessing_failed"


def test_reinforcement_runner_honors_explicit_template_override(monkeypatch, tmp_path: Path) -> None:
    template = tmp_path / "reinforcement.txt"
    template.write_text(
        "CUSTOM_REBAR_TEMPLATE\n{{REINFORCEMENT_EXAMPLES}}\n{{CURRENT_REINFORCEMENT_TASK_INPUT}}",
        encoding="utf-8",
    )
    monkeypatch.setattr(reinforcement_runner, "extract_reinforcement_tasks_from_dimension_result", lambda **_kwargs: [_reinforcement_task()])
    monkeypatch.setattr(reinforcement_runner, "cap_load_design_tool", lambda **_kwargs: {"success": True, "analysis_load_input": {}, "output_files": {}})
    monkeypatch.setattr(reinforcement_runner, "cap_internal_force_analysis_tool", lambda **_kwargs: {"success": True, "internal_force_output": {}, "output_files": {}})
    monkeypatch.setattr(reinforcement_runner, "cap_force_control_info_tool", lambda **_kwargs: {"success": True, "force_control_info": {}, "output_files": {}})

    class FakeLlm:
        def invoke(self, _messages):
            return SimpleNamespace(
                content=json.dumps(
                    {"reinforcement": {"pier_cap": {}, "pier_column": {}}},
                    ensure_ascii=False,
                )
            )

    monkeypatch.setattr(reinforcement_runner, "_build_llm", lambda _config_path: FakeLlm())
    result = reinforcement_runner.reinforcement_design_tool(
        dimension_design_result={"sample": True},
        output_dir=str(tmp_path),
        template_yaml_path=str(template),
    )

    assert result["success"] is True
    prompt_path = Path(result["reinforcement_design_result"]["任务3_下部结构配筋设计结果"]["分组原始结果"][0]["output_files"]["prompt_txt_path"])
    assert "CUSTOM_REBAR_TEMPLATE" in prompt_path.read_text(encoding="utf-8")
    audit_rows = [
        json.loads(line)
        for line in (tmp_path / "logs/prompt_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["status"] for row in audit_rows] == ["preprocessing", "rendered", "completed"]
    assert audit_rows[1]["prompt_id"] == "external:reinforcement.txt"


def test_prompt_audit_is_append_only_and_reserves_attempt_paths(tmp_path: Path) -> None:
    first = tmp_path / "prompt.txt"
    first.write_text("one", encoding="utf-8")
    second = reserve_artifact_path(first)
    second.write_text("two", encoding="utf-8")

    assert second.name == "prompt_attempt_2.txt"
    audit_path = record_prompt_audit(
        tmp_path,
        prompt_id="tasks.sample.v1",
        version="v1",
        template_sha256="abc",
        stage="sample_stage",
        scope_id="U1",
        attempt=2,
        rendered_path=second,
        status="rendered",
    )
    record_prompt_audit(
        tmp_path,
        prompt_id="tasks.sample.v1",
        version="v1",
        template_sha256="abc",
        stage="sample_stage",
        scope_id="U1",
        attempt=2,
        rendered_path=second,
        status="completed",
    )

    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == ["rendered", "completed"]
    assert rows[0]["rendered_path"] == str(second)
