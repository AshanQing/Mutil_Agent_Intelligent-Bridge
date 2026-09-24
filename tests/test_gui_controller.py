from __future__ import annotations

import ctypes
import json
from pathlib import Path

import pytest
import yaml

from bridge_agents.gui_controller import (
    GraphV2Controller,
    GuiRunConfig,
    extract_interrupt,
    infer_stage_states,
    load_drawing_catalog,
)
from bridge_agents.gui_display import enable_windows_high_dpi
from run_graph_v2_gui import (
    blueprint_background_path,
    load_gui_form_state,
    save_gui_form_state,
)


def _base_config(path: Path) -> Path:
    payload = {
        "llm": {
            "controller": {
                "api_key": "ENV",
                "api_key_env": "DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
            }
        },
        "task": {"user_request": "旧任务"},
        "paths": {
            "data_path": "old-data",
            "input_drawing_path": "old.dxf",
            "output_dir": "old-output",
            "few_shots_dir": "data/few_shots",
        },
        "route_data": {"file_prefix": "OLD"},
        "agent": {"max_revision_rounds": 3},
        "modeling": {"max_check_revision_rounds": 2},
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def _run_config(tmp_path: Path) -> GuiRunConfig:
    return GuiRunConfig(
        user_request="请对K1+000-K2+000进行全流程设计",
        data_path=str(tmp_path / "route"),
        input_drawing_path=str(tmp_path / "input.dxf"),
        output_dir=str(tmp_path / "run-output"),
        file_prefix="K-final",
        thread_id="bridge-run-001",
        max_revision_rounds=4,
        max_check_revision_rounds=3,
        code_rag_enabled=True,
    )


def test_gui_form_state_round_trip_uses_allowlist(tmp_path: Path) -> None:
    state_path = tmp_path / "gui-state.json"
    save_gui_form_state(
        {
            "base_config": "config/settings.yaml",
            "data_path": "D:/route",
            "drawing_path": "D:/input.dxf",
            "output_dir": "D:/output/test",
            "file_prefix": "K",
            "max_revision": 4,
            "max_check_revision": 3,
            "code_rag": True,
            "user_request": "执行全流程设计",
            "thread_id": "must-not-persist",
            "external_authorized": True,
            "api_key": "must-not-persist",
        },
        state_path,
    )

    restored = load_gui_form_state(state_path)

    assert restored["user_request"] == "执行全流程设计"
    assert restored["output_dir"] == "D:/output/test"
    assert restored["code_rag"] is True
    assert "thread_id" not in restored
    assert "external_authorized" not in restored
    assert "api_key" not in restored


def test_gui_form_state_ignores_missing_corrupt_or_unknown_version(tmp_path: Path) -> None:
    state_path = tmp_path / "gui-state.json"
    assert load_gui_form_state(state_path) == {}

    state_path.write_text("not-json", encoding="utf-8")
    assert load_gui_form_state(state_path) == {}

    state_path.write_text(
        json.dumps({"version": 999, "values": {"output_dir": "D:/old"}}),
        encoding="utf-8",
    )
    assert load_gui_form_state(state_path) == {}


def test_start_writes_isolated_snapshot_and_forces_graph_v2(tmp_path: Path) -> None:
    base = _base_config(tmp_path / "settings.yaml")
    calls = []

    def fake_run_agent(user_request: str, **kwargs):
        calls.append((user_request, kwargs))
        return {"thread_id": kwargs["thread_id"], "task_status": "completed"}

    controller = GraphV2Controller(run_agent_fn=fake_run_agent)
    result = controller.start(_run_config(tmp_path), base_config_path=base)

    snapshot = Path(result["config_snapshot_path"])
    assert snapshot == tmp_path / "run-output" / "run_configs" / "bridge-run-001.yaml"
    saved = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert saved["task"]["user_request"] == "请对K1+000-K2+000进行全流程设计"
    assert saved["paths"]["output_dir"] == str(tmp_path / "run-output")
    assert saved["route_data"]["file_prefix"] == "K-final"
    assert saved["code_rag"]["enabled"] is True
    assert saved["llm"]["controller"]["api_key"] == "ENV"
    assert calls == [
        (
            "请对K1+000-K2+000进行全流程设计",
            {
                "config_path": str(snapshot),
                "use_graph_v2": True,
                "thread_id": "bridge-run-001",
            },
        )
    ]


def test_snapshot_absolutizes_relative_path_fields(tmp_path: Path) -> None:
    # 基础配置放在 <root>/config/ 下，不带 project_root，验证按
    # “config/ 的上级=项目根” 绝对化相对路径，且非路径字段不受影响。
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    base = cfg_dir / "settings.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "task": {"user_request": "旧任务"},
                "paths": {
                    "output_dir": "old-output",
                    "few_shots_dir": "data/few_shots",
                    "standards_path": "data/standards.json",
                },
                "structure": {
                    "dimension_samples_yaml_path": "samples/dimension_design_samples.yaml",
                    "force_combo_name": "ULS_basic",
                },
                "modeling": {"capacity_check_script_path": "tools/check.py"},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_run_agent(user_request: str, **kwargs):
        calls.append(kwargs)
        return {"thread_id": kwargs["thread_id"], "task_status": "completed"}

    run_config = GuiRunConfig(
        user_request="请对K1+000-K2+000进行全流程设计",
        data_path=str(tmp_path / "route"),
        input_drawing_path=str(tmp_path / "input.dxf"),
        output_dir=str(tmp_path / "run-output"),
        file_prefix="K-final",
        thread_id="bridge-run-002",
    )
    controller = GraphV2Controller(run_agent_fn=fake_run_agent)
    result = controller.start(run_config, base_config_path=base)

    saved = yaml.safe_load(Path(result["config_snapshot_path"]).read_text(encoding="utf-8"))
    assert saved["paths"]["few_shots_dir"] == str(tmp_path / "data" / "few_shots")
    assert saved["paths"]["standards_path"] == str(tmp_path / "data" / "standards.json")
    assert saved["paths"]["output_dir"] == str(tmp_path / "run-output")
    assert saved["structure"]["dimension_samples_yaml_path"] == str(
        tmp_path / "samples" / "dimension_design_samples.yaml"
    )
    assert saved["structure"]["force_combo_name"] == "ULS_basic"
    assert saved["modeling"]["capacity_check_script_path"] == str(tmp_path / "tools" / "check.py")
    assert isinstance(saved["paths"]["few_shots_dir"], str)
    assert Path(saved["paths"]["few_shots_dir"]).is_absolute()


def test_snapshot_prefers_explicit_project_root_over_config_dir(tmp_path: Path) -> None:
    root = tmp_path / "repo-root"
    base = tmp_path / "settings.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "project_root": str(root),
                    "few_shots_dir": "data/few_shots",
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    controller = GraphV2Controller(run_agent_fn=lambda *_args, **_kwargs: {"task_status": "completed"})
    result = controller.start(
        GuiRunConfig(
            user_request="请对K1+000-K2+000进行全流程设计",
            data_path=str(tmp_path / "route"),
            input_drawing_path=str(tmp_path / "input.dxf"),
            output_dir=str(tmp_path / "run-output"),
            file_prefix="K-final",
            thread_id="bridge-run-003",
        ),
        base_config_path=base,
    )

    saved = yaml.safe_load(Path(result["config_snapshot_path"]).read_text(encoding="utf-8"))
    assert saved["paths"]["few_shots_dir"] == str(root / "data" / "few_shots")


def test_run_config_rejects_missing_required_fields(tmp_path: Path) -> None:
    config = _run_config(tmp_path)
    config.output_dir = ""
    with pytest.raises(ValueError, match="输出目录"):
        config.validate()


def test_resume_reuses_snapshot_and_builds_revision_payload(tmp_path: Path) -> None:
    calls = []

    def fake_run_agent(user_request: str, **kwargs):
        calls.append((user_request, kwargs))
        return {"thread_id": kwargs["thread_id"], "task_status": "layout_revision_completed"}

    controller = GraphV2Controller(run_agent_fn=fake_run_agent)
    snapshot = _base_config(tmp_path / "snapshot.yaml")
    controller.resume(
        thread_id="bridge-run-001",
        config_snapshot_path=snapshot,
        action="continue_revision",
        feedback="避让高压线",
        extra_rounds=2,
    )

    assert calls[0][0] == ""
    assert calls[0][1]["use_graph_v2"] is True
    assert calls[0][1]["resume"] == {
        "action": "continue_revision",
        "feedback": "避让高压线",
        "extra_rounds": 2,
    }


def test_resume_rejects_action_not_offered_by_review(tmp_path: Path) -> None:
    controller = GraphV2Controller(run_agent_fn=lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="当前复核不允许"):
        controller.resume(
            thread_id="bridge-run-001",
            config_snapshot_path=_base_config(tmp_path / "snapshot.yaml"),
            action="accept_check_and_finish",
            available_actions=["continue_revision", "abort"],
        )


def test_extract_interrupt_returns_first_structured_payload() -> None:
    result = {"__interrupt__": [{"value": {"review_type": "layout_collision_review"}}]}
    assert extract_interrupt(result) == {"review_type": "layout_collision_review"}
    assert extract_interrupt({}) is None


def test_stage_states_show_completed_prefix_and_manual_review() -> None:
    states = infer_stage_states(
        {
            "active_agent": "ModelingCheckAgent",
            "completed_agents": ["InitialDesignAgent", "LayoutRevisionAgent", "StructuralDesignAgent"],
            "task_status": "manual_review_required",
            "__interrupt__": [{"value": {"review_type": "modeling_check_review"}}],
        }
    )
    assert states == {
        "initial_design": "completed",
        "layout_revision": "completed",
        "structural_design": "completed",
        "modeling_check": "review",
        "final_output": "pending",
    }


def test_stage_states_mark_final_failure() -> None:
    states = infer_stage_states({"task_status": "failed", "active_agent": "final_output"})
    assert states["final_output"] == "failed"


def test_enable_windows_high_dpi_prefers_per_monitor_v2() -> None:
    calls = []

    class User32:
        def SetProcessDpiAwarenessContext(self, value):
            calls.append(value.value)
            return 1

    class Windll:
        user32 = User32()

    result = enable_windows_high_dpi(platform_name="win32", windll=Windll())

    assert result == "per_monitor_v2"
    assert calls == [ctypes.c_void_p(-4).value]


def test_enable_windows_high_dpi_is_noop_outside_windows() -> None:
    assert enable_windows_high_dpi(platform_name="linux", windll=None) == "not_windows"


def test_blueprint_background_is_a_project_local_png() -> None:
    asset = blueprint_background_path()

    assert asset.is_file()
    assert asset.suffix.lower() == ".png"
    assert asset.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_load_drawing_catalog_maps_paged_sheets_to_openable_files(tmp_path: Path) -> None:
    drawings = tmp_path / "deliverables" / "drawings"
    group = drawings / "G1"
    group.mkdir(parents=True)
    for name in ("general.scr", "general.svg", "cap.scr", "cap.svg"):
        (group / name).write_text("data", encoding="utf-8")
    (drawings / "drawing_index.json").write_text(
        __import__("json").dumps(
            {
                "groups": [
                    {
                        "design_group_id": "G1",
                        "member_piers": ["P1", "P2"],
                        "issue_status": "verified",
                        "geometry_valid": True,
                        "sheets": [
                            {"sheet_id": "pier_general_arrangement", "title": "桥墩总体布置图", "scr_name": "general.scr", "svg_name": "general.svg"},
                            {"sheet_id": "cap_reinforcement_detail", "title": "盖梁配筋详图", "scr_name": "cap.scr", "svg_name": "cap.svg"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    catalog = load_drawing_catalog(tmp_path)

    assert catalog[0]["design_group_id"] == "G1"
    assert catalog[0]["member_piers"] == ["P1", "P2"]
    assert catalog[0]["sheets"][1] == {
        "sheet_id": "cap_reinforcement_detail",
        "title": "盖梁配筋详图",
        "scr_path": str(group / "cap.scr"),
        "svg_path": str(group / "cap.svg"),
    }


def test_snapshot_run_progress_empty_dir_returns_idle(tmp_path: Path) -> None:
    from bridge_agents.gui_controller import snapshot_run_progress

    snapshot = snapshot_run_progress(tmp_path)

    assert snapshot["running"] is False
    assert snapshot["active_stage"] is None
    assert all(value == "pending" for value in snapshot["stage_states"].values())


def test_snapshot_run_progress_detects_layout_and_dimension_done(tmp_path: Path) -> None:
    from bridge_agents.gui_controller import snapshot_run_progress

    (tmp_path / "layout_revision").mkdir()
    (tmp_path / "layout_revision" / "final_layout_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "structural_design" / "design_units").mkdir(parents=True)
    (tmp_path / "structural_design" / "design_units" / "design_units_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "structural_design" / "dimension_design").mkdir()
    (tmp_path / "structural_design" / "dimension_design" / "dimension_design_result.json").write_text("{}", encoding="utf-8")
    reinforcement = tmp_path / "structural_design" / "reinforcement_design"
    for group in ("G1", "G2"):
        group_dir = reinforcement / group
        group_dir.mkdir(parents=True)
        (group_dir / f"reinforcement_result_{group}.json").write_text("{}", encoding="utf-8")
    (reinforcement / "reinforcement_design_result.json").write_text("{}", encoding="utf-8")

    snapshot = snapshot_run_progress(tmp_path)

    assert snapshot["stage_states"]["initial_design"] == "completed"
    assert snapshot["stage_states"]["layout_revision"] == "completed"
    assert snapshot["stage_states"]["structural_design"] == "completed"
    assert snapshot["active_stage"] == "modeling_check"
    assert snapshot["total_groups"] == 2


class _FakeReviewAnswer:
    def __init__(self, **payload):
        self._payload = payload

    def model_dump(self, mode: str = "json") -> dict:
        return dict(self._payload)


class _RecordingReviewAgent:
    def __init__(self):
        self.answer_calls: list[dict] = []

    def answer(self, question, *, output_dir, config_path, history=None):
        self.answer_calls.append(
            {"question": question, "output_dir": output_dir, "config_path": config_path, "history": history}
        )
        return _FakeReviewAnswer(
            question=question, answer="依据当前成果作答。", answer_path="answer.md"
        )


def test_controller_answer_forwards_chat_history_to_review_agent() -> None:
    review = _RecordingReviewAgent()
    controller = GraphV2Controller(
        run_agent_fn=lambda *_args, **_kwargs: {},
        review_agent_factory=lambda: review,
    )
    history = [
        {"role": "user", "content": "上一问：盖梁如何？"},
        {"role": "assistant", "content": "上一答：验算通过。"},
    ]
    result = controller.answer(
        "那弯矩包络呢？",
        output_dir="out",
        config_snapshot_path="snap.yaml",
        history=history,
    )
    assert result["answer"] == "依据当前成果作答。"
    call = review.answer_calls[0]
    assert call["question"] == "那弯矩包络呢？"
    assert call["config_path"] == "snap.yaml"
    assert call["history"] == history


def test_controller_answer_history_defaults_to_none() -> None:
    review = _RecordingReviewAgent()
    controller = GraphV2Controller(
        run_agent_fn=lambda *_args, **_kwargs: {},
        review_agent_factory=lambda: review,
    )
    controller.answer("默认不带历史", output_dir="out", config_snapshot_path="snap.yaml")
    assert review.answer_calls[0]["history"] is None
