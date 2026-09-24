from __future__ import annotations

from types import SimpleNamespace

from run_multi_agent import (
    build_cli_summary,
    build_parser,
    build_resume_payload,
    prompt_for_human_decision,
)


def test_cli_parser_accepts_resume_without_user_request() -> None:
    args = build_parser().parse_args([
        "--config",
        "config/settings.yaml",
        "--resume",
        "thread-1",
        "--decision",
        "abort",
    ])

    assert args.user_request is None
    assert args.resume == "thread-1"
    assert args.decision == "abort"


def test_build_resume_payload_keeps_feedback_and_extra_rounds() -> None:
    payload = build_resume_payload(
        "continue_revision",
        feedback="调整桥位5边界",
        extra_rounds=2,
    )

    assert payload == {
        "action": "continue_revision",
        "feedback": "调整桥位5边界",
        "extra_rounds": 2,
    }


def test_cli_summary_surfaces_interrupt_without_large_state() -> None:
    review_request = {
        "review_type": "layout_collision_review",
        "available_actions": ["continue_revision", "accept_and_continue", "abort"],
    }
    result = {
        "thread_id": "thread-1",
        "task_status": "manual_review_required",
        "message": "等待人工复核",
        "layout_result": {"very": "large"},
        "few_shots": [{"very": "large"}],
        "__interrupt__": (SimpleNamespace(value=review_request),),
    }

    summary = build_cli_summary(result)

    assert summary["thread_id"] == "thread-1"
    assert summary["interrupts"] == [review_request]
    assert "layout_result" not in summary
    assert "few_shots" not in summary


def test_interactive_prompt_maps_numeric_choice_to_continue_revision() -> None:
    answers = iter(["1", "调整桥位5边界", "2"])

    decision = prompt_for_human_decision(
        {"available_actions": ["continue_revision", "accept_and_continue", "abort"]},
        input_fn=lambda prompt: next(answers),
        print_fn=lambda message: None,
    )

    assert decision == {
        "action": "continue_revision",
        "feedback": "调整桥位5边界",
        "extra_rounds": 2,
    }


def test_cli_accepts_structural_review_action_with_feedback() -> None:
    args = build_parser().parse_args(
        ["--resume", "thread-2", "--decision", "retry_failed_tasks"]
    )
    payload = build_resume_payload(
        args.decision,
        feedback="重试格式失败任务",
    )

    assert payload == {
        "action": "retry_failed_tasks",
        "feedback": "重试格式失败任务",
    }


def test_cli_preserves_layout_acceptance_feedback() -> None:
    payload = build_resume_payload(
        "accept_and_continue",
        feedback="接受当前布跨并记录碰撞风险",
    )

    assert payload == {
        "action": "accept_and_continue",
        "feedback": "接受当前布跨并记录碰撞风险",
    }
