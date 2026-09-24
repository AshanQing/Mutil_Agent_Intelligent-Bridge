from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools import design_generation_tool
from tools.design_generation_tool import _format_few_shots_for_prompt


SAMPLE_INPUT = {
    "线路类型": "整体式",
    "K": {
        "桩号范围": "K0+000-K0+100",
        "设计线高程序列": [
            {"桩号": 0.0, "高程": 100.0},
            {"桩号": 100.0, "高程": 110.0},
        ],
        "地形线高程序列": [
            {"桩号": 0.0, "高程": 90.0},
            {"桩号": 50.0, "高程": 96.0},
            {"桩号": 100.0, "高程": 101.0},
        ],
        "平曲线结构": [],
        "横断面信息": [],
    },
    "障碍物信息": [],
}

SAMPLE_OUTPUT = {
    "线路类型": "整体式",
    "桥梁方案列表": [{"桥位编号": 1, "桥跨范围": "K0+010-K0+090"}],
}


def test_few_shot_input_uses_terrain_table_and_preserves_output() -> None:
    sample = {**SAMPLE_INPUT, "output": SAMPLE_OUTPUT, "_source_file": "sample.json"}

    text = _format_few_shots_for_prompt([sample])

    assert "### Few-shot 示例 1" in text
    assert "**K线地形特征表**" in text
    assert "K0+50" in text
    assert "105.000" in text
    assert "+9.000" in text
    assert '"桥位编号": 1' in text
    assert "设计线高程序列" not in text
    assert "地形线高程序列" not in text


def test_nested_few_shot_input_is_supported() -> None:
    sample = {"input": SAMPLE_INPUT, "output": SAMPLE_OUTPUT}

    text = _format_few_shots_for_prompt([sample])

    assert "**K线地形特征表**" in text
    assert '"桥位编号": 1' in text
    assert '"input"' not in text


def test_few_shot_terrain_table_is_smaller_than_pretty_json() -> None:
    dense_input = {
        **SAMPLE_INPUT,
        "K": {
            **SAMPLE_INPUT["K"],
            "设计线高程序列": [
                {"桩号": 0.0, "高程": 100.0},
                {"桩号": 990.0, "高程": 109.9},
            ],
            "地形线高程序列": [
                {"桩号": float(station), "高程": 90.0 + station / 200.0}
                for station in range(0, 1000, 10)
            ],
        },
    }
    sample = {**dense_input, "output": SAMPLE_OUTPUT}
    raw = json.dumps([sample], ensure_ascii=False, indent=2)

    text = _format_few_shots_for_prompt([sample])

    assert len(text) < len(raw) * 0.80


def test_generate_design_uses_compact_few_shots_and_preserves_raw_trace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "prompts:\n  initial_layout_design_prompt_id: tasks.initial_layout_design.v1\n",
        encoding="utf-8",
    )
    sample = {**SAMPLE_INPUT, "output": SAMPLE_OUTPUT, "_source_file": "sample.json"}

    class FakeLlm:
        def invoke(self, messages):
            return SimpleNamespace(
                content=json.dumps({"设桥总览": {}}, ensure_ascii=False)
            )

    monkeypatch.setattr(design_generation_tool, "build_llm", lambda *args: FakeLlm())

    result = design_generation_tool.generate_design.invoke({
        "design_input": SAMPLE_INPUT,
        "few_shots": [sample],
        "config_path": str(config_path),
        "output_dir": str(tmp_path / "output"),
    })

    assert result["success"] is True
    prompt = Path(result["saved_prompt_path"]).read_text(encoding="utf-8")
    assert "### Few-shot 示例 1（来源：sample.json）" in prompt
    assert "**K线地形特征表**" in prompt
    assert "设计线高程序列" not in prompt
    assert '"桥位编号": 1' in prompt

    saved_few_shots = json.loads(
        Path(result["saved_fewshots_path"]).read_text(encoding="utf-8")
    )
    assert "设计线高程序列" in saved_few_shots[0]["K"]


def test_generate_design_honors_explicit_template_and_records_audit(monkeypatch, tmp_path: Path) -> None:
    template = tmp_path / "layout_override.j2"
    template.write_text(
        "CUSTOM_LAYOUT_TEMPLATE\n{{ design_input_json }}\n{{ few_shots_json }}\n{{ standards_text }}",
        encoding="utf-8",
    )

    class FakeLlm:
        def invoke(self, _messages):
            return SimpleNamespace(content=json.dumps({"设桥总览": {}}, ensure_ascii=False))

    monkeypatch.setattr(design_generation_tool, "build_llm", lambda *args: FakeLlm())
    output_dir = tmp_path / "output"
    config_path = tmp_path / "empty_settings.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    result = design_generation_tool.generate_design.invoke({
        "design_input": SAMPLE_INPUT,
        "few_shots": [],
        "template_path": str(template),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
    })

    assert result["success"] is True
    assert "CUSTOM_LAYOUT_TEMPLATE" in Path(result["saved_prompt_path"]).read_text(encoding="utf-8")
    rows = [
        json.loads(line)
        for line in (output_dir / "logs/prompt_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["status"] for row in rows] == ["rendered", "completed"]
    assert rows[0]["prompt_id"] == "external:layout_override.j2"
