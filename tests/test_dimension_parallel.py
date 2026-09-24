from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from tools import dimension_design_tool as dimension_runner


def _unit(unit_id: str) -> dict:
    return {
        "单元编号": unit_id,
        "桥梁编号": "B1",
        "联号": unit_id,
    }


def test_dimension_units_run_concurrently_and_keep_input_order(
    monkeypatch, tmp_path: Path
) -> None:
    barrier = threading.Barrier(2, timeout=2.0)

    class FakeLlm:
        def invoke(self, messages):
            unit_id = messages[-1][1]
            barrier.wait()
            if unit_id == "U1":
                time.sleep(0.03)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "单元编号": unit_id,
                        "分组尺寸设计结果": [],
                        "桥墩尺寸映射关系": [],
                    },
                    ensure_ascii=False,
                )
            )

    monkeypatch.setattr(dimension_runner, "_build_llm", lambda _path: FakeLlm())
    monkeypatch.setattr(
        dimension_runner,
        "build_dimension_design_prompt_tool",
        lambda *, single_unit_input, **_kwargs: {
            "prompt_json": {"单元编号": single_unit_input["单元编号"]},
            "llm_prompt_text": single_unit_input["单元编号"],
            "prompt_id": "test.dimension",
            "prompt_version": "1",
            "template_sha256": "sha",
            "match_debug": [],
            "compatibility_warnings": [],
        },
    )

    result = dimension_runner.dimension_design_tool(
        design_units={"single_unit_inputs": [_unit("U1"), _unit("U2")]},
        output_dir=str(tmp_path),
        samples_yaml_path="unused.yaml",
        max_workers=2,
    )

    assert result["success"] is True
    unit_results = result["dimension_design_result"]["任务2_下部结构尺寸设计结果"][
        "单元原始结果"
    ]
    assert [item["single_unit_input"]["单元编号"] for item in unit_results] == [
        "U1",
        "U2",
    ]
    audit_rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "prompt_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(audit_rows) == 4


def test_dimension_parallel_failure_is_isolated(monkeypatch, tmp_path: Path) -> None:
    class FakeLlm:
        def invoke(self, messages):
            unit_id = messages[-1][1]
            if unit_id == "U2":
                raise RuntimeError("model failed")
            return SimpleNamespace(
                content='{"分组尺寸设计结果": [], "桥墩尺寸映射关系": []}'
            )

    monkeypatch.setattr(dimension_runner, "_build_llm", lambda _path: FakeLlm())
    monkeypatch.setattr(
        dimension_runner,
        "build_dimension_design_prompt_tool",
        lambda *, single_unit_input, **_kwargs: {
            "prompt_json": {},
            "llm_prompt_text": single_unit_input["单元编号"],
            "prompt_id": "test.dimension",
            "prompt_version": "1",
            "template_sha256": "sha",
        },
    )

    result = dimension_runner.dimension_design_tool(
        design_units={"single_unit_inputs": [_unit("U1"), _unit("U2")]},
        output_dir=str(tmp_path),
        samples_yaml_path="unused.yaml",
        max_workers=2,
    )

    assert result["success"] is True
    assert result["unit_result_count"] == 1
    assert result["failed_unit_count"] == 1
    assert result["errors"][0]["单元编号"] == "U2"


def test_dimension_parse_failure_repairs_then_regenerates(monkeypatch, tmp_path: Path) -> None:
    class FakeLlm:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            responses = [
                "not-json",
                "still-not-json",
                '{"分组尺寸设计结果": [], "桥墩尺寸映射关系": []}',
            ]
            return SimpleNamespace(content=responses[self.calls - 1])

    llm = FakeLlm()
    monkeypatch.setattr(dimension_runner, "_build_llm", lambda _path: llm)
    monkeypatch.setattr(
        dimension_runner,
        "build_dimension_design_prompt_tool",
        lambda *, single_unit_input, **_kwargs: {
            "prompt_json": {},
            "llm_prompt_text": single_unit_input["单元编号"],
            "prompt_id": "test.dimension",
            "prompt_version": "1",
            "template_sha256": "sha",
        },
    )

    result = dimension_runner.dimension_design_tool(
        design_units={"single_unit_inputs": [_unit("1-3")]},
        output_dir=str(tmp_path),
        samples_yaml_path="unused.yaml",
        max_workers=1,
        max_format_repairs=1,
        max_regenerations=1,
    )

    assert result["stage_complete"] is True
    assert llm.calls == 3
    unit = result["dimension_design_result"]["任务2_下部结构尺寸设计结果"][
        "单元原始结果"
    ][0]
    assert [item["kind"] for item in unit["generation_attempts"]] == [
        "initial",
        "format_repair",
        "regeneration",
    ]


def test_dimension_retry_runs_only_failed_unit_and_merges_prior_success(
    monkeypatch, tmp_path: Path
) -> None:
    invoked: list[str] = []

    class FakeLlm:
        def invoke(self, messages):
            unit_id = messages[-1][1]
            invoked.append(unit_id)
            return SimpleNamespace(
                content='{"分组尺寸设计结果": [], "桥墩尺寸映射关系": []}'
            )

    monkeypatch.setattr(dimension_runner, "_build_llm", lambda _path: FakeLlm())
    monkeypatch.setattr(
        dimension_runner,
        "build_dimension_design_prompt_tool",
        lambda *, single_unit_input, **_kwargs: {
            "prompt_json": {},
            "llm_prompt_text": single_unit_input["单元编号"],
            "prompt_id": "test.dimension",
            "prompt_version": "1",
            "template_sha256": "sha",
        },
    )
    existing = {
        "任务2_下部结构尺寸设计结果": {
            "单元原始结果": [
                {
                    "unit_index": 1,
                    "single_unit_input": _unit("U1"),
                    "dimension_result": {
                        "分组尺寸设计结果": [],
                        "桥墩尺寸映射关系": [],
                    },
                }
            ]
        }
    }

    result = dimension_runner.dimension_design_tool(
        design_units={"single_unit_inputs": [_unit("U1"), _unit("U2")]},
        output_dir=str(tmp_path),
        samples_yaml_path="unused.yaml",
        max_workers=1,
        retry_unit_ids=["U2"],
        existing_dimension_design_result=existing,
    )

    assert invoked == ["U2"]
    rows = result["dimension_design_result"]["任务2_下部结构尺寸设计结果"][
        "单元原始结果"
    ]
    assert [row["single_unit_input"]["单元编号"] for row in rows] == ["U1", "U2"]
    assert result["stage_complete"] is True


def _stub_dimension_llm(monkeypatch, invoked: list[str]) -> None:
    class FakeLlm:
        def invoke(self, messages):
            unit_id = messages[-1][1]
            invoked.append(unit_id)
            return SimpleNamespace(
                content='{"分组尺寸设计结果": [], "桥墩尺寸映射关系": []}'
            )

    monkeypatch.setattr(dimension_runner, "_build_llm", lambda _path: FakeLlm())
    monkeypatch.setattr(
        dimension_runner,
        "build_dimension_design_prompt_tool",
        lambda *, single_unit_input, **_kwargs: {
            "prompt_json": {},
            "llm_prompt_text": single_unit_input["单元编号"],
            "prompt_id": "test.dimension",
            "prompt_version": "1",
            "template_sha256": "sha",
        },
    )


def test_dimension_retry_downgrades_task_id_to_unit_id(monkeypatch, tmp_path: Path) -> None:
    # 上游误传配筋任务号（2-2-1-G1）时按前缀降级到尺寸单元号（2-2），只重跑该单元。
    invoked: list[str] = []
    _stub_dimension_llm(monkeypatch, invoked)

    result = dimension_runner.dimension_design_tool(
        design_units={"single_unit_inputs": [_unit("2-1"), _unit("2-2")]},
        output_dir=str(tmp_path),
        samples_yaml_path="unused.yaml",
        max_workers=1,
        retry_unit_ids=["2-2-1-G1"],
    )

    assert invoked == ["2-2"]
    assert result["success"] is True
    assert result["retry_fallback"] is False


def test_dimension_retry_unknown_ids_fall_back_to_full_run(monkeypatch, tmp_path: Path) -> None:
    # 完全无法匹配时整体重跑，而不是让整个结构设计阶段因一个 ID 打错而判失败。
    invoked: list[str] = []
    _stub_dimension_llm(monkeypatch, invoked)

    result = dimension_runner.dimension_design_tool(
        design_units={"single_unit_inputs": [_unit("2-1"), _unit("2-2")]},
        output_dir=str(tmp_path),
        samples_yaml_path="unused.yaml",
        max_workers=1,
        retry_unit_ids=["T3", "T8"],
    )

    assert sorted(invoked) == ["2-1", "2-2"]
    assert result["success"] is True
    assert result["retry_fallback"] is True
    assert "整体重跑" in result["retry_note"]
    assert (
        "整体重跑"
        in result["dimension_design_result"]["任务2_下部结构尺寸设计结果"]["重试说明"]
    )
