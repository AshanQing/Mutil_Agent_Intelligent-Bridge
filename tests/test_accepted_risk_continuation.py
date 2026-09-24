"""回归：人工接受风险后流程必须能正常走下去，且少数分组失败不拖累其余出图。

对应 2026-09-16 示例项目K1_400-K4_200 暴露的三个缺陷：
1. 重新运行时会话账本里的"人工接受"被静默丢弃（验算类复核记录只有 state_key、
   没有 path/sha256），于是流程又变成"无法继续"；
2. 确定性校验只要 check_passed=False 就拒绝最终放行，不认人工已接受该验算结论；
3. 出图阶段只要有一个组未出图就整包作废，已成功分组的图纸也一起丢掉了。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge_agents.agent import _discover_existing_outputs
from bridge_agents.coordinator import CoordinatorDecision, validate_decision
from bridge_agents.design_coordinator import build_coordinator_view


def _write(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _capacity_summary(*, all_ok: bool) -> dict:
    return {
        "check_type": "cap_beam_capacity_envelope_batch",
        "stage_complete": True,
        "expected_task_count": 2,
        "completed_task_count": 2,
        "failed_task_count": 0,
        "failed_task_ids": [],
        "failed_check_task_ids": [] if all_ok else ["2-2-4-G1"],
        "overall_check": {"M_pos_ok": all_ok, "M_neg_ok": all_ok, "V_ok": all_ok, "all_ok": all_ok},
        "utilization_summary": {"max_utilization": 0.9 if all_ok else 1.714, "task_id": "2-2-4-G1"},
        "task_results": [],
    }


def _complete_decision() -> CoordinatorDecision:
    return CoordinatorDecision(
        intent="full_design",
        next_stage=None,
        stage_objective="完成任务",
        reason="全部阶段已完成",
        revision_target=None,
        required_artifacts=[],
        task_complete=True,
    )


# --------------------------------------------------------------------------- #
# 1. 人工接受验算风险后，确定性校验必须允许最终放行
# --------------------------------------------------------------------------- #
def test_final_release_allowed_once_check_risk_is_accepted() -> None:
    failed_check = _capacity_summary(all_ok=False)
    state = {
        "check_result": failed_check,
        "capacity_check_result": failed_check,
        "accepted_risks": [{"scope": "modeling_check", "reason": "人工接受未通过的验算结果"}],
    }

    view = build_coordinator_view(state)

    assert view.check_passed is False
    assert view.modeling_risk_accepted is True
    # 关键：人工已接受的失败验算不再是"不能最终放行"的理由
    assert validate_decision(_complete_decision(), view) == []


def test_final_release_still_blocked_without_human_acceptance() -> None:
    failed_check = _capacity_summary(all_ok=False)
    view = build_coordinator_view(
        {
            "check_result": failed_check,
            "capacity_check_result": failed_check,
            "accepted_risks": [],
        }
    )

    assert view.check_passed is False
    assert view.modeling_risk_accepted is False
    assert validate_decision(_complete_decision(), view) == ["验算失败成果不能最终放行"]


# --------------------------------------------------------------------------- #
# 2. 跨运行继承：重新运行时人工"接受验算风险"的决定不能丢
# --------------------------------------------------------------------------- #
def _write_modeling_check_acceptance(output_dir: Path) -> None:
    ledger = output_dir / "human_review" / "human_review_decisions.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "review_type": "modeling_check_review",
        "action": "accept_check_and_finish",
        # 真实运行里验算类复核只记 state_key，path/sha256 为空
        "subject": {"path": None, "sha256": None, "state_key": "capacity_check_result"},
        "accepted_risk": {
            "scope": "modeling_check",
            "reason": "人工接受未通过的验算结果",
            "evidence_path": str(output_dir / "capacity_check" / "capacity_check_batch_summary.json"),
            "accepted_at": "2026-09-16T05:39:46+00:00",
        },
        "accepted_state": {"task_status": "completed_with_accepted_risks"},
        "decision_id": "d-1",
        "decided_at": "2026-09-16T05:39:46+00:00",
        "feedback": "",
        "extra_rounds": 1,
    }
    ledger.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def test_modeling_check_acceptance_survives_a_new_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write(output_dir / "capacity_check" / "capacity_check_batch_summary.json", _capacity_summary(all_ok=False))
    _write_modeling_check_acceptance(output_dir)

    discovered = _discover_existing_outputs(output_dir)

    assert [risk["scope"] for risk in discovered["accepted_risks"]] == ["modeling_check"]
    assert discovered["task_status"] == "completed_with_accepted_risks"
    assert discovered["human_override"] is True


def test_restored_acceptance_lets_the_run_finish(tmp_path: Path) -> None:
    """恢复出接受决定后，协调器的状态视图必须能直接放行（不再转人工复核）。"""
    output_dir = tmp_path / "output"
    _write(output_dir / "capacity_check" / "capacity_check_batch_summary.json", _capacity_summary(all_ok=False))
    _write_modeling_check_acceptance(output_dir)

    state = _discover_existing_outputs(output_dir)
    view = build_coordinator_view(state)

    assert view.modeling_risk_accepted is True
    assert validate_decision(_complete_decision(), view) == []


# --------------------------------------------------------------------------- #
# 3. 布跨类接受记录依赖 latest_revision_result_path 才能匹配
# --------------------------------------------------------------------------- #
def test_discovery_records_latest_revision_path_even_with_final_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write(output_dir / "layout_revision" / "final_layout_result.json", {"桥位列表": []})
    _write(output_dir / "revision_results" / "revision_design_round_3.json", {"桥位列表": []})

    discovered = _discover_existing_outputs(output_dir)

    # 账本里布跨类接受记录指向 revision_results/revision_design_round_3.json，
    # 若该路径不恢复，人工接受就无法匹配而被丢弃。
    assert discovered["latest_revision_result_path"].endswith("revision_design_round_3.json")


# --------------------------------------------------------------------------- #
# 4. 部分出图：少数组不出图不拖累其余组，也不让恢复逻辑永久判未完成
# --------------------------------------------------------------------------- #
def test_drawing_completeness_subtracts_recorded_excluded_groups(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write(
        output_dir / "structural_design" / "reinforcement_design" / "reinforcement_design_result.json",
        {
            "任务3_下部结构配筋设计结果": {
                "分组原始结果": [
                    {"reinforcement_task": {"task_id": "1-1-1-G1"}},
                    {"reinforcement_task": {"task_id": "1-1-1-G2"}},
                ]
            }
        },
    )
    _write(
        output_dir / "deliverables" / "drawings" / "drawing_index.json",
        {"groups": [{"design_group_id": "1-1-1-G1"}]},
    )
    _write(
        output_dir / "deliverables" / "design_manifest.json",
        {"excluded_groups": [{"design_group_id": "1-1-1-G2", "error": "验算未通过且无人工接受风险"}]},
    )

    package = _discover_existing_outputs(output_dir)["drawing_package_result"]

    assert package["generated_group_ids"] == ["1-1-1-G1"]
    assert package["excluded_group_ids"] == ["1-1-1-G2"]
    assert package["missing_group_ids"] == []
    assert package["success"] is True


def test_drawing_completeness_still_flags_unexplained_missing_groups(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write(
        output_dir / "structural_design" / "reinforcement_design" / "reinforcement_design_result.json",
        {
            "任务3_下部结构配筋设计结果": {
                "分组原始结果": [
                    {"reinforcement_task": {"task_id": "1-1-1-G1"}},
                    {"reinforcement_task": {"task_id": "1-1-1-G2"}},
                ]
            }
        },
    )
    _write(
        output_dir / "deliverables" / "drawings" / "drawing_index.json",
        {"groups": [{"design_group_id": "1-1-1-G1"}]},
    )

    package = _discover_existing_outputs(output_dir)["drawing_package_result"]

    assert package["missing_group_ids"] == ["1-1-1-G2"]
    assert package["success"] is False


def test_partial_package_is_written_instead_of_aborting(monkeypatch, tmp_path: Path, drawing_source) -> None:
    """一组验算未通过且无人工风险时，只跳过该组，其余组照常出图并落盘。"""
    from bridge_agents.drawing import package as package_module

    blocked = drawing_source.model_copy(
        update={"design_group_id": "B1-U1-2", "task_id": "B1-U1-G2", "check_status": "failed"}
    )
    monkeypatch.setattr(
        package_module, "collect_drawing_group_sources", lambda **_kwargs: [drawing_source, blocked]
    )

    update = package_module.build_final_deliverables(
        {
            "output_dir": str(tmp_path),
            "pier_group_result": {"design_groups": []},
            "dimension_design_result": {"任务2_下部结构尺寸设计结果": {}},
            "reinforcement_design_result": {"任务3_下部结构配筋设计结果": {}},
            "check_result": _capacity_summary(all_ok=False),
            "accepted_risks": [],
        }
    )

    result = update["drawing_package_result"]
    assert result["generated_group_ids"] == ["B1-U1-1"]
    assert [item["design_group_id"] for item in result["failed_groups"]] == ["B1-U1-2"]
    assert Path(tmp_path / "deliverables" / "drawings" / "drawing_index.json").is_file()
    manifest = json.loads(
        (tmp_path / "deliverables" / "design_manifest.json").read_text(encoding="utf-8")
    )
    assert [item["design_group_id"] for item in manifest["excluded_groups"]] == ["B1-U1-2"]


def test_deliverables_still_fail_when_no_group_can_be_drawn(monkeypatch, tmp_path: Path, drawing_source) -> None:
    """一组都放不出来时仍按失败处理（部分交付只有"还有组能出图"才成立）。"""
    from bridge_agents.drawing import package as package_module
    from bridge_agents.drawing.package import UnverifiedDrawingSourceError

    blocked = drawing_source.model_copy(update={"check_status": "failed"})
    monkeypatch.setattr(package_module, "collect_drawing_group_sources", lambda **_kwargs: [blocked])

    with pytest.raises(UnverifiedDrawingSourceError):
        package_module.build_final_deliverables(
            {
                "output_dir": str(tmp_path),
                "pier_group_result": {"design_groups": []},
                "dimension_design_result": {"任务2_下部结构尺寸设计结果": {}},
                "reinforcement_design_result": {"任务3_下部结构配筋设计结果": {}},
                "check_result": _capacity_summary(all_ok=False),
                "accepted_risks": [],
            }
        )
