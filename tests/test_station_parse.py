"""桩号入口解析回归：下划线/加号写法统一规范成 K{km}+{m}，供 load_data 使用。"""
from __future__ import annotations

from bridge_agents.utils import regex_extract_stations


def test_underscore_range_is_normalized_to_plus_format() -> None:
    result = regex_extract_stations("请对K1_000-K2_600进行全流程设计任务。")
    assert result == {"start_station": "K41+000", "end_station": "K42+600"}


def test_plus_range_unchanged() -> None:
    result = regex_extract_stations("请对K41+000-K42+600进行全流程设计任务。")
    assert result == {"start_station": "K41+000", "end_station": "K42+600"}


def test_legacy_example_keeps_working() -> None:
    result = regex_extract_stations(
        "请对示例高速K1+451-K3+500段进行全流程设计任务。"
    )
    assert result == {"start_station": "K1+451", "end_station": "K3+500"}


def test_single_station_yields_start_only() -> None:
    result = regex_extract_stations("请核查K1_000附近的桥位。")
    assert result == {"start_station": "K41+000", "end_station": None}


def test_no_station_returns_none() -> None:
    result = regex_extract_stations("请进行全流程设计任务。")
    assert result == {"start_station": None, "end_station": None}
