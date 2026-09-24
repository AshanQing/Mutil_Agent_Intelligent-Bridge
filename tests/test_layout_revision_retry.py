from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import bridge_agents.tool_actions as tool_actions


class _StubLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.outputs.pop(0))


def _run(outputs, repairs=1, monkeypatch=None):
    out_dir = tempfile.mkdtemp(prefix="retry_case_", dir=Path(__file__).resolve().parent)
    llm = _StubLLM(outputs)
    monkeypatch.setattr(tool_actions, "get_revision_llm", lambda *a, **k: llm)
    state = {
        "revision_prompt": "请修正布跨",
        "iteration_index": 0,
        "output_dir": out_dir,
        "config_path": "config/settings.yaml",
        "llm_max_format_repairs": repairs,
    }
    result = tool_actions.generate_revised_layout_action(state)
    return out_dir, llm, result


def test_retries_once_when_output_missing_required_field(monkeypatch):
    out_dir, llm, result = _run(['{"其他": 1}', '{"设桥总览": {"桥位": []}}'], monkeypatch=monkeypatch)
    try:
        assert result["error"] is None
        assert result["revision_format_attempts"] == 2
        assert len(llm.calls) == 2
        # 第二次调用应当是"修复 JSON 格式"的系统提示
        assert "修复 JSON" in llm.calls[1][0][1]
        assert Path(result["revision_result_path"]).is_file()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_fails_after_retries_and_keeps_raw_evidence(monkeypatch):
    out_dir, llm, result = _run(['{"其他": 1}', '{"其他": 2}'], monkeypatch=monkeypatch)
    try:
        assert result["error"]
        assert len(llm.calls) == 2
        raws = sorted(p.name for p in (Path(out_dir) / "revision_results").glob("*raw_failed*"))
        assert len(raws) == 2
        assert not (Path(out_dir) / "revision_results" / "revision_design_round_1.json").exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_zero_repairs_disables_retry(monkeypatch):
    out_dir, llm, result = _run(['{"其他": 1}'], repairs=0, monkeypatch=monkeypatch)
    try:
        assert result["error"]
        assert len(llm.calls) == 1
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_raw_newline_output_is_tolerated_without_retry(monkeypatch):
    out_dir, llm, result = _run(['{"设桥总览": {"note": "第一行\n第二行"}}'], monkeypatch=monkeypatch)
    try:
        assert result["error"] is None
        assert result["revision_format_attempts"] == 1
        assert len(llm.calls) == 1
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
