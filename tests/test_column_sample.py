from __future__ import annotations

import pytest

from bridge_agents.column_sample import (
    inspect_column_sample,
    inspect_column_samples,
    load_column_samples,
    score_column_sample,
    select_column_samples,
)


def _column_sample(
    sample_id,
    pier_role="中间墩",
    pier_type="柱式墩",
    diameter=1800,
    height=10000,
    spacing=7250,
    count=2,
):
    return {
        "sample_id": sample_id,
        "输入": {
            "基本信息": {"drawing_id": sample_id, "pier_role": pier_role, "pier_type": pier_type},
            "墩柱几何信息": {
                "column_count": count,
                "column_diameter": diameter,
                "column_height": height,
                "column_spacing": spacing,
            },
        },
        "输出": {"reinforcement": {"pier_column": {"longitudinal_bars": []}}},
    }


class TestInspectColumnSample:
    def test_valid_sample(self):
        result = inspect_column_sample(_column_sample("s1"))
        assert result["valid"] is True
        assert result["issues"] == []

    def test_missing_column_geometry(self):
        sample = _column_sample("s1")
        del sample["输入"]["墩柱几何信息"]
        result = inspect_column_sample(sample)
        assert result["valid"] is False
        assert any("墩柱几何信息" in issue for issue in result["issues"])

    def test_non_positive_height(self):
        result = inspect_column_sample(_column_sample("s1", height=0))
        assert result["valid"] is False

    def test_missing_pier_column_output(self):
        sample = _column_sample("s1")
        sample["输出"] = {"reinforcement": {"pier_cap": {}}}
        result = inspect_column_sample(sample)
        assert result["valid"] is False
        assert any("pier_column" in issue for issue in result["issues"])


class TestInspectColumnSamples:
    def test_splits_valid_and_rejected(self):
        samples = [_column_sample("s1"), _column_sample("s2", height=0), _column_sample("s3")]
        result = inspect_column_samples(samples)
        assert len(result["valid"]) == 2
        assert [item["sample_id"] for item in result["rejected"]] == ["s2"]


class TestScoreColumnSample:
    TARGET = {
        "基本信息": {"pier_role": "中间墩", "pier_type": "柱式墩"},
        "墩柱几何信息": {
            "column_diameter": 1800,
            "column_height": 10000,
            "column_spacing": 7250,
        },
    }

    def test_identical_sample_scores_zero(self):
        sample_input = _column_sample("s1")["输入"]
        assert score_column_sample(sample_input, self.TARGET) == pytest.approx(0.0)

    def test_role_mismatch_penalizes(self):
        sample_input = _column_sample("s1", pier_role="边墩")["输入"]
        assert score_column_sample(sample_input, self.TARGET) > score_column_sample(
            _column_sample("s1", pier_role="中间墩")["输入"], self.TARGET
        )

    def test_closer_height_scores_lower(self):
        near = _column_sample("near", height=10500)["输入"]
        far = _column_sample("far", height=20000)["输入"]
        assert score_column_sample(near, self.TARGET) < score_column_sample(far, self.TARGET)


class TestSelectColumnSamples:
    def test_selects_closest_two(self):
        samples = [
            _column_sample("s1", height=9000),
            _column_sample("s2", height=10100),
            _column_sample("s3", height=30000),
        ]
        target = {
            "基本信息": {"pier_role": "中间墩", "pier_type": "柱式墩"},
            "墩柱几何信息": {
                "column_diameter": 1800,
                "column_height": 10000,
                "column_spacing": 7250,
            },
        }
        selected = select_column_samples(samples, target, max_count=2)
        assert [s["sample_id"] for s in selected] == ["s2", "s1"]


class TestLoadColumnSamples:
    def test_loads_numbered_samples(self, tmp_path):
        import yaml

        def _without_id(sid):
            sample = _column_sample(sid)
            sample.pop("sample_id", None)
            return sample

        data = {
            "示例1": _without_id("1"),
            "示例2": _without_id("2"),
        }
        path = tmp_path / "column.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        samples = load_column_samples(str(path))
        assert len(samples) == 2
        assert samples[0]["sample_id"] in ("示例1", "示例2")
