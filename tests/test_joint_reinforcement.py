from __future__ import annotations

import pytest

from bridge_agents.joint_reinforcement import (
    aggregate_axial_check_results,
    build_axial_check_summary,
    inject_column_heights,
    validate_joint_reinforcement,
)


def _task(task_id, pier_nos, column=None, bridge="1", unit="1-1"):
    task = {
        "task_id": task_id,
        "桥梁编号": bridge,
        "单元编号": unit,
        "包含桥墩号列表": pier_nos,
        "桥墩尺寸信息": {
            "墩柱几何信息": dict(column or {}),
        },
    }
    return task


def _pier_group_result():
    return {
        "design_groups": [
            {
                "design_group_id": "1-1-1",
                "bridge_id": "1",
                "unit_id": "1-1",
                "member_piers": ["1", "2"],
                "controlling_net_height_m": 10.2,
                "member_net_heights_m": {"1": 10.2, "2": 9.8},
            },
            {
                "design_group_id": "1-1-2",
                "bridge_id": "1",
                "unit_id": "1-1",
                "member_piers": ["3"],
                "controlling_net_height_m": 8.0,
                "member_net_heights_m": {"3": 8.0},
            },
        ]
    }


class TestInjectColumnHeights:
    def test_injects_controlling_height_and_members(self):
        tasks = [_task("t1", ["1", "2"]), _task("t2", ["3"])]
        result = inject_column_heights(tasks, _pier_group_result())
        assert result[0]["桥墩尺寸信息"]["墩柱几何信息"]["column_height"] == pytest.approx(10.2)
        assert result[0]["墩柱净高信息"]["controlling_net_height_m"] == pytest.approx(10.2)
        assert result[0]["墩柱净高信息"]["member_net_heights_m"] == {"1": 10.2, "2": 9.8}
        assert result[1]["桥墩尺寸信息"]["墩柱几何信息"]["column_height"] == pytest.approx(8.0)

    def test_unmapped_task_gets_no_height(self):
        tasks = [_task("t1", ["9"])]
        result = inject_column_heights(tasks, _pier_group_result())
        assert "column_height" not in result[0]["桥墩尺寸信息"]["墩柱几何信息"]
        assert result[0]["墩柱净高信息"]["controlling_net_height_m"] is None
        assert result[0]["墩柱净高信息"]["missing_pier_groups"] == ["9"]

    def test_same_pier_no_across_bridges_does_not_cross_match(self):
        # 回归：历史上 pier_to_group 只以墩号为键，桥2的墩1会串到桥3的墩1净高。
        # 复合键（桥号+单元号+墩号）必须保证各取各的值。
        tasks = [
            _task("b1", ["1"], bridge="1", unit="1-1"),
            _task("b2", ["1"], bridge="2", unit="2-2"),
        ]
        groups = {
            "design_groups": [
                {
                    "design_group_id": "1-1-1",
                    "bridge_id": "1",
                    "unit_id": "1-1",
                    "member_piers": ["1"],
                    "controlling_net_height_m": 10.2,
                    "member_net_heights_m": {"1": 10.2},
                },
                {
                    "design_group_id": "2-2-1",
                    "bridge_id": "2",
                    "unit_id": "2-2",
                    "member_piers": ["1"],
                    "controlling_net_height_m": 5.5,
                    "member_net_heights_m": {"1": 5.5},
                },
            ]
        }
        result = inject_column_heights(tasks, groups)
        assert result[0]["墩柱净高信息"]["controlling_net_height_m"] == pytest.approx(10.2)
        assert result[1]["墩柱净高信息"]["controlling_net_height_m"] == pytest.approx(5.5)


class TestValidateJointReinforcement:
    def test_complete_when_both_present(self):
        result = {
            "reinforcement": {
                "pier_cap": {},
                "pier_column": {"longitudinal_bars": [{"id": "N1"}]},
            }
        }
        assert validate_joint_reinforcement(result) == {"complete": True, "missing": []}

    def test_missing_column(self):
        result = {"reinforcement": {"pier_cap": {}}}
        assert validate_joint_reinforcement(result) == {"complete": False, "missing": ["pier_column"]}

    def test_empty_column_rejected(self):
        result = {
            "reinforcement": {
                "pier_cap": {},
                "pier_column": {
                    "longitudinal_bars": [],
                    "spiral_stirrups": [],
                    "ordinary_hoops": [],
                    "strengthening_bars": [],
                },
            }
        }
        assert validate_joint_reinforcement(result) == {"complete": False, "missing": ["pier_column.empty"]}

    def test_missing_reinforcement(self):
        assert validate_joint_reinforcement({}) == {"complete": False, "missing": ["reinforcement"]}


class TestAggregateAxialCheckResults:
    def _result(self, rows):
        return {"任务3_下部结构配筋设计结果": {"分组原始结果": rows}}

    def test_aggregates_failures_and_reviews(self):
        rows = [
            {"reinforcement_task": {"task_id": "t1"}, "axial_check": {"status": "computed", "check_ok": False}},
            {"reinforcement_task": {"task_id": "t2"}, "axial_check": {"status": "manual_review"}},
            {"reinforcement_task": {"task_id": "t3"}, "axial_check": {"status": "computed", "check_ok": True}},
            {"reinforcement_task": {"task_id": "t4"}, "axial_check": None},
        ]
        result = aggregate_axial_check_results(self._result(rows))
        assert result["axial_fail_task_ids"] == ["t1"]
        assert result["slenderness_review_task_ids"] == ["t2"]
        assert result["completed_checks"] == 3
        assert result["has_axial_failure"] is True
        assert result["has_slenderness_review"] is True

    def test_empty_result(self):
        result = aggregate_axial_check_results(None)
        assert result["completed_checks"] == 0
        assert result["has_axial_failure"] is False
        assert result["has_slenderness_review"] is False


class TestBuildAxialCheckSummary:
    def _result(self, rows):
        return {"任务3_下部结构配筋设计结果": {"分组原始结果": rows}}

    def _row(self, task_id, axial, height=8.0):
        return {
            "reinforcement_task": {
                "task_id": task_id,
                "包含桥墩号列表": [task_id, "extra"],
                "桥墩尺寸信息": {"墩柱几何信息": {"column_diameter": 1.4}},
                "墩柱净高信息": {"controlling_net_height_m": height},
            },
            "axial_check": axial,
        }

    def test_classifies_verdicts_and_picks_control_group(self):
        rows = [
            self._row("t1", {"status": "computed", "check_ok": True, "utilization": 0.6,
                             "slenderness_ratio": 40.0, "stability_factor_phi": 0.95}),
            self._row("t2", {"status": "computed", "check_ok": False, "utilization": 1.3,
                             "slenderness_ratio": 30.0, "stability_factor_phi": 0.98}),
            self._row("t3", {"status": "manual_review", "slenderness_ratio": 200.0}),
            self._row("t4", None),
        ]
        result = build_axial_check_summary(self._result(rows))
        assert result["expected_task_count"] == 4
        assert result["verdict_counts"] == {
            "passed": 1,
            "failed": 1,
            "manual_review": 1,
            "not_applicable": 0,
            "missing": 1,
        }
        assert result["has_axial_failure"] is True
        assert result["has_manual_review"] is True
        # 控制组取"通过组里利用率最大"；最大长细比行含手动复核组。
        assert result["control_group"]["task_id"] == "t1"
        assert result["max_slenderness_group"]["task_id"] == "t3"
        assert result["groups"][2]["column_diameter_mm"] == pytest.approx(1400.0)

    def test_empty(self):
        result = build_axial_check_summary(None)
        assert result["expected_task_count"] == 0
        assert result["verdict_counts"] == {
            "passed": 0,
            "failed": 0,
            "manual_review": 0,
            "not_applicable": 0,
            "missing": 0,
        }

    def test_non_positive_net_height_is_not_applicable(self):
        """净高非正 = 无有效柱身（桥台/埋入式墩）：历史 manual_review 也按不适用处理。"""
        rows = [
            self._row("t1", {"status": "computed", "check_ok": True, "utilization": 0.5}, height=8.0),
            # 旧版本写入的 manual_review，但净高为负 → 读取侧归一到"不适用"
            self._row("t2", {"status": "manual_review", "slenderness_ratio": None}, height=-0.91),
        ]
        result = build_axial_check_summary(self._result(rows))
        assert result["verdict_counts"]["not_applicable"] == 1
        assert result["verdict_counts"]["manual_review"] == 0
        assert result["groups"][1]["verdict"] == "not_applicable"
        assert result["has_manual_review"] is False

        aggregated = aggregate_axial_check_results(self._result(rows))
        assert aggregated["not_applicable_task_ids"] == ["t2"]
        assert aggregated["has_slenderness_review"] is False
