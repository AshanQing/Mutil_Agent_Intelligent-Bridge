from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools import design_unit_extractor_tool as extractor


def _layout(span_combo: str = "4×30", span_count: int = 4, end: str = "K0+120") -> dict:
    """与真实产物保持一致的布跨结果：桥位列表位于"设桥总览"之下。

    早期 fixture 把"桥位列表"放在根层，与生产 schema 不符，因此校验器漏解包
    "设桥总览" 的缺陷在单测中始终是绿的。
    """
    return {
        "设桥总览": {
            "是否设桥": "是",
            "桥位数量": 1,
            "桥位列表": [
                {
                    "桥位编号": "1",
                    "线路类型": "整体式",
                    "统一布跨方案": {
                        "设桥信息": {
                            "桥跨范围": f"K0+000 - {end}",
                            "跨径组合": span_combo,
                            "总跨数": span_count,
                        },
                        "墩位与墩高": [
                            {
                                "墩号": f"{index}" + (" (桥台)" if index in {0, span_count} else ""),
                                "桩号": f"K0+{index * 30:03d}",
                                "墩位类型": "桥台" if index in {0, span_count} else "桥墩",
                            }
                            for index in range(span_count + 1)
                        ],
                    },
                    "分幅布跨方案列表": None,
                }
            ],
        }
    }


def _unit_result(role: str = "中间墩", span_combo: str = "4×30", span_count: int = 4, end: str = "K0+120") -> dict:
    return {
        "任务1_设计单元提取结果": {
            "桥梁列表": [
                {
                    "桥梁编号": "1",
                    "桥型": "预应力混凝土先简支后连续T梁",
                    "设计单元列表": [
                        {
                            "单元编号": "1-1",
                            "联号": "第一联",
                            "分联结果": span_combo,
                            "分联依据说明": "标准跨径按长度控制分联。",
                            "本联信息": {
                                "跨径组合": span_combo,
                                "跨数": span_count,
                                "起点桩号": "K0+000",
                                "终点桩号": end,
                            },
                            "桥面宽度信息": {"宽度类型": "标准宽度"},
                            "桥墩信息": [
                                {
                                    "原始墩号": "0 (桥台)",
                                    "桩号": "K0+000",
                                    "墩位角色": "桥台",
                                    "单元内顺序": 0,
                                    "是否连接墩": False,
                                },
                                {
                                    "原始墩号": "1",
                                    "桩号": "K0+030",
                                    "墩位角色": role,
                                    "单元内顺序": 1,
                                    "是否连接墩": False,
                                },
                                {
                                    "原始墩号": f"{span_count} (桥台)",
                                    "桩号": end,
                                    "墩位角色": "桥台",
                                    "单元内顺序": 2,
                                    "是否连接墩": False,
                                },
                            ],
                        }
                    ],
                    "单联尺寸设计输入": [
                        {
                            "桥梁编号": "1",
                            "单元编号": "1-1",
                            "联号": "第一联",
                            "桥型": "预应力混凝土先简支后连续T梁",
                            "本联信息": {
                                "跨径组合": span_combo,
                                "跨数": span_count,
                                "起点桩号": "K0+000",
                                "终点桩号": end,
                            },
                            "桥面宽度信息": {"宽度类型": "标准宽度"},
                            "本联设计分组": [
                                {"分组编号": "G1", "墩位角色": "桥台", "包含桥墩号列表": ["0 (桥台)", f"{span_count} (桥台)"]},
                                {"分组编号": "G2", "墩位角色": role, "包含桥墩号列表": ["1"]},
                            ],
                        }
                    ],
                }
            ]
        }
    }


def _unit_result_with_uncovered_edge_pier() -> dict:
    return {
        "任务1_设计单元提取结果": {
            "桥梁列表": [
                {
                    "桥梁编号": "1",
                    "桥型": "预应力混凝土先简支后连续T梁",
                    "设计单元列表": [
                        {
                            "单元编号": "1-1",
                            "联号": "第一联",
                            "本联信息": {
                                "跨径组合": "4×30",
                                "跨数": 4,
                                "起点桩号": "K0+000",
                                "终点桩号": "K0+120",
                            },
                            "桥面宽度信息": {"宽度类型": "标准宽度"},
                            "桥墩信息": [
                                {"原始墩号": "0 (桥台)", "桩号": "K0+000", "墩位角色": "桥台", "单元内顺序": 0, "是否连接墩": False},
                                {"原始墩号": "1", "桩号": "K0+030", "墩位角色": "中间墩", "单元内顺序": 1, "是否连接墩": False},
                                {"原始墩号": "2", "桩号": "K0+060", "墩位角色": "中间墩", "单元内顺序": 2, "是否连接墩": False},
                                {"原始墩号": "3", "桩号": "K0+090", "墩位角色": "中间墩", "单元内顺序": 3, "是否连接墩": False},
                                {"原始墩号": "4", "桩号": "K0+120", "墩位角色": "边墩", "单元内顺序": 4, "是否连接墩": True},
                            ],
                        }
                    ],
                    "单联尺寸设计输入": [
                        {
                            "桥梁编号": "1",
                            "单元编号": "1-1",
                            "联号": "第一联",
                            "桥型": "预应力混凝土先简支后连续T梁",
                            "本联信息": {
                                "跨径组合": "4×30",
                                "跨数": 4,
                                "起点桩号": "K0+000",
                                "终点桩号": "K0+120",
                            },
                            "桥面宽度信息": {"宽度类型": "标准宽度"},
                            "本联设计分组": [
                                {"分组编号": "G1", "墩位角色": "中间墩", "包含桥墩号列表": ["1", "2", "3"]},
                            ],
                        }
                    ],
                }
            ]
        }
    }


def test_bridge_catalog_unwraps_overview_wrapper() -> None:
    """回归：真实产物的"桥位列表"嵌在"设桥总览"之下，必须能映射到桥位编号。

    漏解包时 catalog 恒为空，校验器会对每条设计单元报"无法映射到设桥布跨结果"，
    且该失败与 LLM 输出无关，重试永远不会成功。
    """
    catalog = extractor._layout_bridge_catalog(_layout())

    assert set(catalog) == {"1"}
    assert catalog["1"]["设桥信息"]["总跨数"] == 4


def test_bridge_catalog_registers_side_span_schemes() -> None:
    """分幅布跨方案按"桥位编号-R/L"登记为独立条目。"""
    layout = _layout()
    layout["设桥总览"]["桥位列表"][0]["分幅布跨方案列表"] = [
        {"幅别": "右幅", "设桥信息": {"总跨数": 4}},
        {"幅别": "左幅", "设桥信息": {"总跨数": 4}},
    ]

    catalog = extractor._layout_bridge_catalog(layout)

    assert set(catalog) == {"1", "1-R", "1-L"}


def test_semantic_validation_accepts_standard_unit_at_length_limit() -> None:
    validation = extractor.validate_design_units_result(_layout(), _unit_result())

    assert validation["valid"] is True
    assert validation["errors"] == []


def test_semantic_validation_rejects_uncovered_edge_pier() -> None:
    validation = extractor.validate_design_units_result(
        _layout(span_combo="4×30", span_count=4),
        _unit_result_with_uncovered_edge_pier(),
    )

    assert validation["valid"] is False
    assert any("未覆盖" in error for error in validation["errors"])


def test_semantic_validation_rejects_transition_role_and_oversized_standard_unit() -> None:
    validation = extractor.validate_design_units_result(
        _layout("5×30", 5, "K0+150"),
        _unit_result("过渡墩", "5×30", 5, "K0+150"),
    )

    assert validation["valid"] is False
    assert any("过渡墩" in error for error in validation["errors"])
    assert any("120m" in error for error in validation["errors"])


def test_extractor_retries_once_after_semantic_failure(monkeypatch, tmp_path: Path) -> None:
    responses = [
        _unit_result("过渡墩"),
        _unit_result("中间墩"),
    ]
    calls: list[list[tuple[str, str]]] = []

    class FakeLlm:
        def invoke(self, messages):
            calls.append(messages)
            return SimpleNamespace(content=json.dumps(responses[len(calls) - 1], ensure_ascii=False))

    monkeypatch.setattr(extractor, "_build_llm", lambda _config_path: FakeLlm())

    result = extractor.extract_design_units_tool(
        layout_result=_layout(),
        output_dir=str(tmp_path),
        config_path="config/settings.yaml",
    )

    assert result["success"] is True
    assert len(calls) == 2
    assert result["semantic_validation"]["repair_attempted"] is True
    assert "过渡墩" in calls[1][1][1]
    assert Path(result["output_files"]["prompt_path"]).name == "design_unit_extraction_prompt_attempt_2.txt"


def test_extractor_stops_after_one_failed_repair(monkeypatch, tmp_path: Path) -> None:
    calls = 0

    class FakeLlm:
        def invoke(self, _messages):
            nonlocal calls
            calls += 1
            return SimpleNamespace(content=json.dumps(_unit_result("过渡墩"), ensure_ascii=False))

    monkeypatch.setattr(extractor, "_build_llm", lambda _config_path: FakeLlm())

    result = extractor.extract_design_units_tool(
        layout_result=_layout(),
        output_dir=str(tmp_path),
        config_path="config/settings.yaml",
    )

    assert result["success"] is False
    assert calls == 2
    assert result["manual_review_required"] is True
    assert result["semantic_validation"]["repair_attempted"] is True


def test_extractor_wraps_llm_configuration_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        extractor,
        "_build_llm",
        lambda _config_path: (_ for _ in ()).throw(ValueError("missing key")),
    )

    result = extractor.extract_design_units_tool(
        layout_result=_layout(),
        output_dir=str(tmp_path),
        config_path="config/settings.yaml",
    )

    assert result["success"] is False
    assert "missing key" in result["error"]


def test_extractor_honors_explicit_prompt_and_records_audit(monkeypatch, tmp_path: Path) -> None:
    prompt = tmp_path / "design_unit_override.j2"
    prompt.write_text(
        "CUSTOM_DESIGN_UNIT_TEMPLATE\n{{ layout_result_json }}\n{{ validation_feedback_json }}",
        encoding="utf-8",
    )

    class FakeLlm:
        def invoke(self, _messages):
            return SimpleNamespace(content=json.dumps(_unit_result(), ensure_ascii=False))

    monkeypatch.setattr(extractor, "_build_llm", lambda _config_path: FakeLlm())
    result = extractor.extract_design_units_tool(
        layout_result=_layout(),
        output_dir=str(tmp_path / "output"),
        prompt_path=str(prompt),
    )

    assert result["success"] is True
    assert "CUSTOM_DESIGN_UNIT_TEMPLATE" in Path(result["output_files"]["prompt_path"]).read_text(encoding="utf-8")
    rows = [
        json.loads(line)
        for line in (tmp_path / "output/logs/prompt_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[-1]["status"] == "completed"
    assert rows[-1]["prompt_id"] == "external:design_unit_override.j2"
