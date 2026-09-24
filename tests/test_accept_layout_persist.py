from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from bridge_agents.graph_v2 import apply_human_review_decision
from bridge_agents.gui_controller import snapshot_run_progress


def _case_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="accept_layout_", dir=Path(__file__).resolve().parent))


def _state(out_dir: Path, layout):
    round_file = out_dir / "revision_results" / "revision_design_round_3.json"
    round_file.parent.mkdir(parents=True, exist_ok=True)
    round_file.write_text(json.dumps({"原始": "round3"}, ensure_ascii=False), encoding="utf-8")
    return {
        "layout_result": layout,
        "collision_metrics": {"has_collision": True, "conflict_column_rate": 0.0957},
        "collision_items": [],
        "iteration_index": 3,
        "max_revision_rounds": 3,
        "output_dir": str(out_dir),
        "revision_result_path": str(round_file),
        "latest_handoff": {"stage": "layout_revision", "status": "revision_required", "message": "仍有碰撞"},
    }


_DECISION = {"action": "accept_and_continue", "decision_id": "t", "feedback": "", "extra_rounds": 1}


def test_accept_and_continue_persists_final_layout_result():
    out = _case_dir()
    try:
        layout = {"设桥总览": {"桥位列表": [{"桥位编号": 1}]}, "是否对称布跨": "是"}
        update = apply_human_review_decision(_state(out, layout), _DECISION)

        expected = out / "layout_revision" / "final_layout_result.json"
        assert expected.is_file()
        assert json.loads(expected.read_text(encoding="utf-8")) == layout
        assert update["final_layout_result_path"] == str(expected)
        assert update["layout_revision_result_path"] == str(expected)
        assert update["latest_handoff"]["produced_artifacts"][0]["path"] == str(expected)
        assert update["task_status"] == "layout_revision_completed"

        # 轨道判据（文件启发式）应与图状态一致
        snap = snapshot_run_progress(out)
        assert snap["stage_states"]["layout_revision"] == "completed"
        assert snap["stage_states"]["initial_design"] == "completed"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_accept_and_continue_falls_back_without_output_dir():
    out = _case_dir()
    try:
        state = _state(out, None)
        state["output_dir"] = None
        update = apply_human_review_decision(state, _DECISION)
        # 没有 output_dir 时不写文件，但仍回退到既有 revision 结果路径，不报错
        assert update["final_layout_result_path"] == state["revision_result_path"]
        assert not (out / "layout_revision" / "final_layout_result.json").exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)
