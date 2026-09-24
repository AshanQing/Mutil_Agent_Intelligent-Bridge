from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

from tools import reinforcement_design_tool as reinforcement_runner


def _task(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "桥梁编号": "B1",
        "单元编号": "U1",
        "分组编号": task_id,
        "包含桥墩号列表": [task_id],
        "墩位角色": "中间墩",
    }


def _joint_result() -> dict:
    # 墩柱纵向主筋必须非空：validate_joint_reinforcement 把空壳墩柱配筋判为
    # pier_column.empty，否则本文件里所有"生成成功"的用例都会走到失败分支。
    return {
        "reinforcement": {
            "pier_cap": {"z_patterns": {}},
            "pier_column": {"longitudinal_bars": [{"dia": 28, "count": 16}]},
        }
    }


def _install_preprocessing_fakes(monkeypatch, tasks: list[dict], calls: list) -> None:
    monkeypatch.setattr(
        reinforcement_runner,
        "extract_reinforcement_tasks_from_dimension_result",
        lambda **_kwargs: tasks,
    )

    def load(**kwargs):
        calls.append(("load", kwargs["task_id"], threading.get_ident()))
        return {
            "success": True,
            "analysis_load_input": {"task_id": kwargs["task_id"]},
            "load_design_output": {},
            "output_files": {},
        }

    def force(**kwargs):
        task_id = kwargs["task_id"]
        calls.append(("force", task_id, threading.get_ident()))
        return {
            "success": True,
            "internal_force_output": {
                "combined_envelopes_full_beam": {"ULS": {}},
                "note_moment_sign": "test",
            },
            "output_files": {"internal_force_output_path": f"{task_id}.json"},
        }

    def control(**kwargs):
        task_id = kwargs["task_id"]
        calls.append(("control", task_id, threading.get_ident()))
        return {
            "success": True,
            "force_control_info": {"task_id": task_id},
            "output_files": {},
        }

    monkeypatch.setattr(reinforcement_runner, "cap_load_design_tool", load)
    monkeypatch.setattr(reinforcement_runner, "cap_internal_force_analysis_tool", force)
    monkeypatch.setattr(reinforcement_runner, "cap_force_control_info_tool", control)
    monkeypatch.setattr(
        reinforcement_runner,
        "build_reinforcement_design_prompt_tool",
        lambda *, reinforcement_task, **_kwargs: {
            "prompt_yaml": {"task_id": reinforcement_task["task_id"]},
            "llm_prompt_text": reinforcement_task["task_id"],
            "prompt_id": "test.reinforcement",
            "prompt_version": "1",
            "template_sha256": "sha",
            "matched_sample_count": 1,
            "sample_load_mode": "test",
        },
    )


def test_reinforcement_preprocessing_is_serial_then_generation_is_parallel(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list = []
    tasks = [_task("T1"), _task("T2")]
    _install_preprocessing_fakes(monkeypatch, tasks, calls)
    main_thread = threading.get_ident()
    barrier = threading.Barrier(2, timeout=2.0)

    class FakeLlm:
        def invoke(self, messages):
            task_id = messages[-1][1]
            barrier.wait()
            return SimpleNamespace(content=json.dumps(_joint_result(), ensure_ascii=False))

    monkeypatch.setattr(reinforcement_runner, "_build_llm", lambda _path: FakeLlm())

    result = reinforcement_runner.reinforcement_design_tool(
        dimension_design_result={"sample": True},
        output_dir=str(tmp_path),
        max_workers=2,
    )

    assert [(stage, task_id) for stage, task_id, _ in calls] == [
        ("load", "T1"),
        ("force", "T1"),
        ("control", "T1"),
        ("load", "T2"),
        ("force", "T2"),
        ("control", "T2"),
    ]
    assert all(thread_id == main_thread for _, _, thread_id in calls)
    raw_results = result["reinforcement_design_result"][
        "任务3_下部结构配筋设计结果"
    ]["分组原始结果"]
    assert [item["reinforcement_task"]["task_id"] for item in raw_results] == [
        "T1",
        "T2",
    ]
    assert result["expected_task_count"] == 2
    assert result["completed_task_count"] == 2
    assert result["stage_complete"] is True


def test_reinforcement_repairs_invalid_format_and_saves_each_raw_response(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list = []
    _install_preprocessing_fakes(monkeypatch, [_task("T1")], calls)
    messages_seen: list = []

    class FakeLlm:
        responses = iter(["[]", json.dumps(_joint_result(), ensure_ascii=False)])

        def invoke(self, messages):
            messages_seen.append(messages)
            return SimpleNamespace(content=next(self.responses))

    monkeypatch.setattr(reinforcement_runner, "_build_llm", lambda _path: FakeLlm())

    result = reinforcement_runner.reinforcement_design_tool(
        dimension_design_result={"sample": True},
        output_dir=str(tmp_path),
        max_workers=1,
        max_format_repairs=1,
    )

    task_dir = tmp_path / "structural_design" / "reinforcement_design" / "T1"
    assert (task_dir / "raw_response_T1_attempt_1.txt").read_text(encoding="utf-8") == "[]"
    assert json.loads(
        (task_dir / "raw_response_T1_attempt_2.txt").read_text(encoding="utf-8")
    ) == _joint_result()
    assert "解析错误" in messages_seen[1][-1][1]
    assert result["stage_complete"] is True


def test_reinforcement_reports_incomplete_batch_after_repair_exhausted(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list = []
    _install_preprocessing_fakes(monkeypatch, [_task("T1")], calls)

    class FakeLlm:
        def invoke(self, _messages):
            return SimpleNamespace(content="[]")

    monkeypatch.setattr(reinforcement_runner, "_build_llm", lambda _path: FakeLlm())

    result = reinforcement_runner.reinforcement_design_tool(
        dimension_design_result={"sample": True},
        output_dir=str(tmp_path),
        max_workers=1,
        max_format_repairs=1,
    )

    assert result["success"] is False
    assert result["expected_task_count"] == 1
    assert result["completed_task_count"] == 0
    assert result["failed_task_count"] == 1
    assert result["stage_complete"] is False
    assert result["partial_result_available"] is False


def test_reinforcement_preserves_partial_results(monkeypatch, tmp_path: Path) -> None:
    calls: list = []
    _install_preprocessing_fakes(monkeypatch, [_task("T1"), _task("T2")], calls)

    class FakeLlm:
        def invoke(self, messages):
            prompt = messages[-1][1]
            if "T2" in prompt:
                return SimpleNamespace(content="[]")
            return SimpleNamespace(content=json.dumps(_joint_result(), ensure_ascii=False))

    monkeypatch.setattr(reinforcement_runner, "_build_llm", lambda _path: FakeLlm())

    result = reinforcement_runner.reinforcement_design_tool(
        dimension_design_result={"sample": True},
        output_dir=str(tmp_path),
        max_workers=2,
        max_format_repairs=1,
    )

    assert result["success"] is True
    assert result["completed_task_count"] == 1
    assert result["failed_task_count"] == 1
    assert result["stage_complete"] is False
    assert result["partial_result_available"] is True
    assert result["errors"][0]["task_id"] == "T2"


def test_reinforcement_retry_runs_only_failed_tasks_and_merges_prior_success(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list = []
    _install_preprocessing_fakes(monkeypatch, [_task("T1"), _task("T2")], calls)

    class FakeLlm:
        def invoke(self, messages):
            return SimpleNamespace(content=json.dumps(_joint_result(), ensure_ascii=False))

    monkeypatch.setattr(reinforcement_runner, "_build_llm", lambda _path: FakeLlm())
    prior_t1 = {
        "task_index": 1,
        "reinforcement_task": _task("T1"),
        "reinforcement_result": {"task_id": "T1", "prior": True},
        "output_files": {"reinforcement_result_yaml_path": "prior_T1.yaml"},
    }
    existing = {
        "reinforcement_design_result": {
            "任务3_下部结构配筋设计结果": {
                "分组原始结果": [prior_t1],
                "失败分组": [{"task_id": "T2", "error": "old parse error"}],
            }
        }
    }

    result = reinforcement_runner.reinforcement_design_tool(
        dimension_design_result={"sample": True},
        output_dir=str(tmp_path),
        retry_task_ids=["T2"],
        existing_reinforcement_design_result=existing,
        max_workers=2,
    )

    assert {(stage, task_id) for stage, task_id, _ in calls} == {
        ("load", "T2"),
        ("force", "T2"),
        ("control", "T2"),
    }
    merged = result["reinforcement_design_result"][
        "任务3_下部结构配筋设计结果"
    ]["分组原始结果"]
    assert [item["reinforcement_task"]["task_id"] for item in merged] == [
        "T1",
        "T2",
    ]
    assert merged[0]["reinforcement_result"]["prior"] is True
    assert result["completed_task_count"] == 2
    assert result["failed_task_count"] == 0
    assert result["stage_complete"] is True


def test_reinforcement_keeps_and_flags_groups_missing_from_collapsed_dimension_result(
    monkeypatch, tmp_path: Path
) -> None:
    """尺寸成果坍缩时，已通过组必须被保留并显式标记，而不是随 all_tasks 重排被静默丢弃。

    2026-09-15 示例项目K29：尺寸返修后尺寸成果只剩被返修的单元，配筋按坍缩结果重算任务清单，
    20 组 → 4 组、expected/completed 一起缩水而 stage_complete 仍为 True，最终只出 1 张图。
    """
    calls: list = []
    # 坍缩后的尺寸成果：任务清单里只剩本轮被返修的那一组 T2
    _install_preprocessing_fakes(monkeypatch, [_task("T2")], calls)

    class FakeLlm:
        def invoke(self, _messages):
            return SimpleNamespace(content=json.dumps(_joint_result(), ensure_ascii=False))

    monkeypatch.setattr(reinforcement_runner, "_build_llm", lambda _path: FakeLlm())
    prior_t1 = {
        "task_index": 1,
        "reinforcement_task": _task("T1"),
        "reinforcement_result": {"prior": True},
        "output_files": {"reinforcement_result_yaml_path": "prior_T1.yaml"},
    }
    existing = {
        "任务3_下部结构配筋设计结果": {
            "分组原始结果": [prior_t1],
            "失败分组": [],
        }
    }

    result = reinforcement_runner.reinforcement_design_tool(
        dimension_design_result={"sample": True},
        output_dir=str(tmp_path),
        retry_task_ids=["T2"],
        existing_reinforcement_design_result=existing,
        max_workers=1,
    )

    root = result["reinforcement_design_result"]["任务3_下部结构配筋设计结果"]
    assert [item["reinforcement_task"]["task_id"] for item in root["分组原始结果"]] == [
        "T2",
        "T1",
    ]
    assert root["分组原始结果"][1]["reinforcement_result"]["prior"] is True
    assert root["尺寸成果缺失分组"] == ["T1"]
    # 期望数取"本轮任务清单 ∪ 沿用组"，且沿用组计入失败清单 → 批次停在人工复核
    assert result["expected_task_count"] == 2
    assert result["completed_task_count"] == 2
    assert result["failed_task_count"] == 1
    assert result["stage_complete"] is False
    assert result["errors"][0]["task_id"] == "T1"
    assert result["errors"][0]["orphaned_from_dimension_result"] is True
