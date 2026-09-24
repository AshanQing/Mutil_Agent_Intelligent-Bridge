# -*- coding: utf-8 -*-
"""布跨修正优化轮：满足阈值但冲突非0 → 强制一次修正并取最优作 final。"""
import sys
from pathlib import Path

sys.path.insert(0, r"<repo-root>")

import pytest

from bridge_agents.stage_agents import LayoutRevisionAgent


def make_metrics(**overrides):
    base = {
        "total_intrusion_depth_columns": 0.0,
        "avg_intrusion_depth_columns": 0.0,
        "avg_overlap_ratio_columns": 0.0,
        "conflict_column_rate": 0.0,
        "has_collision": False,
    }
    base.update(overrides)
    return base


class TestConflictDetection:
    def test_all_zero_no_conflict(self):
        assert LayoutRevisionAgent._metrics_have_any_conflict(make_metrics()) is False

    def test_conflict_rate_nonzero(self):
        assert LayoutRevisionAgent._metrics_have_any_conflict(
            make_metrics(conflict_column_rate=0.02)
        ) is True

    def test_intrusion_nonzero(self):
        assert LayoutRevisionAgent._metrics_have_any_conflict(
            make_metrics(total_intrusion_depth_columns=3.0)
        ) is True

    def test_has_collision_flag(self):
        assert LayoutRevisionAgent._metrics_have_any_conflict(
            make_metrics(has_collision=True)
        ) is True


class TestCompareMetrics:
    def test_second_better(self):
        first = make_metrics(conflict_column_rate=0.03, total_intrusion_depth_columns=8.0)
        second = make_metrics(conflict_column_rate=0.01, total_intrusion_depth_columns=2.0)
        assert LayoutRevisionAgent._compare_metrics(first, second) == 1

    def test_first_better(self):
        first = make_metrics(conflict_column_rate=0.0, total_intrusion_depth_columns=0.0)
        second = make_metrics(conflict_column_rate=0.02, total_intrusion_depth_columns=1.0)
        assert LayoutRevisionAgent._compare_metrics(first, second) == -1

    def test_equal(self):
        assert LayoutRevisionAgent._compare_metrics(make_metrics(), make_metrics()) == 0


class TestPrepareRevisionStep:
    def _agent(self):
        return LayoutRevisionAgent()

    def test_passed_no_conflict_finish(self):
        agent = self._agent()
        state = {"collision_metrics": make_metrics(), "iteration_index": 0}
        update = agent._prepare_revision_step(state)
        assert update["layout_revision_action"] == "finish_revision"

    def test_passed_with_conflict_triggers_optimization(self):
        agent = self._agent()
        state = {
            "collision_metrics": make_metrics(conflict_column_rate=0.02),
            "iteration_index": 0,
            "max_revision_rounds": 3,
        }
        update = agent._prepare_revision_step(state)
        assert update["layout_optimization_started"] is True
        assert update["layout_revision_action"] == "generate_revision_instruction"

    def test_optimization_chain_advances(self):
        agent = self._agent()
        # 已启动优化轮：无修正指令 → 生成指令
        state = {
            "collision_metrics": make_metrics(conflict_column_rate=0.02),
            "iteration_index": 0,
            "layout_optimization_started": True,
        }
        update = agent._prepare_revision_step(state)
        assert update["layout_revision_action"] == "generate_revision_instruction"

        # 已有指令无提示词 → 构建提示词
        state["revision_instruction"] = {"内容": "优化墩位"}
        update = agent._prepare_revision_step(state)
        assert update["layout_revision_action"] == "build_revision_prompt"

        # 已有提示词 → 生成新方案
        state["revision_prompt"] = "请优化布跨"
        update = agent._prepare_revision_step(state)
        assert update["layout_revision_action"] == "generate_revised_layout"

    def test_passed_after_one_round_finishes(self):
        agent = self._agent()
        state = {
            "collision_metrics": make_metrics(conflict_column_rate=0.01),
            "iteration_index": 1,
            "layout_optimization_started": True,
        }
        update = agent._prepare_revision_step(state)
        assert update["layout_revision_action"] == "finish_revision"

    def test_max_rounds_zero_no_optimization(self):
        agent = self._agent()
        state = {
            "collision_metrics": make_metrics(conflict_column_rate=0.02),
            "iteration_index": 0,
            "max_revision_rounds": 0,
        }
        update = agent._prepare_revision_step(state)
        assert update["layout_revision_action"] == "finish_revision"


class TestFinishRevisionPickBest:
    def _agent(self):
        return LayoutRevisionAgent()

    def test_current_better_keeps_current(self, tmp_path, monkeypatch):
        agent = self._agent()
        initial_layout = {"方案": "初始"}
        current_layout = {"方案": "修正后"}
        working = {
            "layout_result": current_layout,
            "collision_metrics": make_metrics(conflict_column_rate=0.01),
            "initial_layout_result": initial_layout,
            "initial_collision_metrics": make_metrics(conflict_column_rate=0.04),
            "output_dir": str(tmp_path),
        }
        update, obs, finished = agent._execute_action("finish_revision", working)
        assert finished is True
        assert update["layout_result"]["方案"] == "修正后"
        assert "当前方案" in update["message"]

    def test_initial_better_keeps_initial(self, tmp_path):
        agent = self._agent()
        initial_layout = {"方案": "初始"}
        current_layout = {"方案": "修正后"}
        working = {
            "layout_result": current_layout,
            "collision_metrics": make_metrics(conflict_column_rate=0.05),
            "initial_layout_result": initial_layout,
            "initial_collision_metrics": make_metrics(conflict_column_rate=0.01),
            "output_dir": str(tmp_path),
        }
        update, obs, finished = agent._execute_action("finish_revision", working)
        assert finished is True
        assert update["layout_result"]["方案"] == "初始"
        assert "初始方案" in update["message"]

    def test_writes_final_layout(self, tmp_path):
        agent = self._agent()
        working = {
            "layout_result": {"方案": "当前"},
            "collision_metrics": make_metrics(),
            "output_dir": str(tmp_path),
        }
        update, _, _ = agent._execute_action("finish_revision", working)
        result_path = update["final_layout_result_path"]
        assert Path(result_path).exists()
        import json

        assert json.loads(Path(result_path).read_text(encoding="utf-8"))["方案"] == "当前"


class TestRunCollisionSavesInitial:
    def _agent(self, monkeypatch):
        agent = LayoutRevisionAgent()

        def fake_run_action(spec, working):
            return {
                "collision_metrics": make_metrics(conflict_column_rate=0.02),
                "layout_result": {"方案": "检测方案"},
                "collision_items": [],
            }

        monkeypatch.setattr(agent, "_run_action", fake_run_action)
        return agent

    def test_saves_initial_on_first_detection(self, monkeypatch):
        agent = self._agent(monkeypatch)
        working = {"layout_result": {"方案": "已有"}}
        update, _, _ = agent._execute_action("run_collision_detection", working)
        assert update["initial_collision_metrics"]["conflict_column_rate"] == 0.02
        assert update["initial_layout_result"] == {"方案": "检测方案"}

    def test_does_not_overwrite_initial(self, monkeypatch):
        agent = self._agent(monkeypatch)
        working = {
            "layout_result": {"方案": "第二轮"},
            "initial_collision_metrics": make_metrics(conflict_column_rate=0.03),
            "initial_layout_result": {"方案": "首轮"},
        }
        update, _, _ = agent._execute_action("run_collision_detection", working)
        assert "initial_collision_metrics" not in update
        assert "initial_layout_result" not in update
