from bridge_agents.gui_modern_runtime import OperationResult, format_run_event, run_operation


def test_run_operation_preserves_name_and_payload() -> None:
    result = run_operation("start", lambda: {"task_status": "completed"})

    assert result == OperationResult(
        name="start",
        payload={"task_status": "completed"},
        error="",
        error_type="",
    )


def test_run_operation_wraps_exception_without_raising() -> None:
    def fail():
        raise ValueError("输入不完整")

    result = run_operation("start", fail)

    assert result.name == "start"
    assert result.payload == {}
    assert result.error == "输入不完整"
    assert result.error_type == "ValueError"


def test_format_run_event_translates_action_failure() -> None:
    line = format_run_event(
        {
            "event_type": "action_end",
            "agent": "ModelingCheckAgent",
            "name": "run_capacity_check",
            "status": "failed",
            "error": "结果无法解析",
        }
    )

    assert line == "[失败] 建模验算 · 承载力批量验算：结果无法解析"
