from __future__ import annotations

import argparse
import json
import logging
import sys

import yaml


HUMAN_REVIEW_ACTIONS = (
    "continue_revision",
    "accept_and_continue",
    "retry_coordinator",
    "retry_failed_tasks",
    "accept_partial_and_continue",
    "continue_modeling_revision",
    "accept_check_and_finish",
    "abort",
)


def load_settings(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pick(settings: dict, *keys: str, default=None):
    for key in keys:
        if key in settings and settings[key] not in [None, ""]:
            return settings[key]
    for section in ["task", "paths"]:
        value = settings.get(section)
        if isinstance(value, dict):
            for key in keys:
                if key in value and value[key] not in [None, ""]:
                    return value[key]
    return default


def json_safe(obj):
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except TypeError:
        pass
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return str(obj)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行桥梁多智能体设计系统。")
    parser.add_argument(
        "user_request",
        type=str,
        nargs="?",
        default=None,
        help="用户自然语言任务；若不提供，则从 settings.yaml 的 task.user_request 读取。",
    )
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="settings.yaml 路径")
    parser.add_argument("--log", type=str, default="INFO", help="日志等级")
    parser.add_argument("--graph-v2", action="store_true", help="使用显式顶层图（DesignCoordinator 路由，替代旧队列式图）")
    parser.add_argument("--initial-state-json", type=str, default=None, help="可选开发调试入口：额外注入初始状态 JSON；普通运行通过 output_dir 自动发现已有成果。")
    parser.add_argument("--thread-id", type=str, default=None, help="为新运行指定持久化线程 ID；省略时自动生成。")
    parser.add_argument("--resume", metavar="THREAD_ID", type=str, default=None, help="恢复指定的 graph-v2 持久化线程。")
    parser.add_argument("--decision", choices=HUMAN_REVIEW_ACTIONS, default=None, help="恢复人工复核时提交的处理动作。")
    parser.add_argument("--feedback", type=str, default="", help="继续修正时传给下一轮的人工补充要求。")
    parser.add_argument("--extra-rounds", type=int, default=1, help="继续修正时增加的自动修正轮数。")
    parser.add_argument("--interactive", action="store_true", help="遇到人工复核中断时直接在终端选择动作。")
    parser.add_argument(
        "--clear-stale-resume",
        action="store_true",
        help=(
            "恢复前清除该线程残留的 __resume__/__error__ 待提交写入。"
            "上一次恢复在人工复核节点抛错时，这些残值会让后续任何新决策被旧值顶掉，"
            "表现为“提交动作没反应”或反复抛同一个错。"
        ),
    )
    parser.add_argument("--full-output", action="store_true", help="打印完整状态；默认仅打印便于人工查看的摘要。")
    return parser


def extract_interrupt_payloads(result: dict) -> list[dict]:
    payloads = []
    for item in result.get("__interrupt__") or []:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            payloads.append(value)
    return payloads


def build_resume_payload(action: str, *, feedback: str = "", extra_rounds: int = 1) -> dict:
    if action not in HUMAN_REVIEW_ACTIONS:
        raise ValueError(f"不支持的人工复核动作：{action}")
    payload = {"action": action}
    if action in {"continue_revision", "continue_modeling_revision"}:
        payload.update({
            "feedback": feedback.strip(),
            "extra_rounds": max(1, int(extra_rounds)),
        })
    elif action in {
        "accept_and_continue",
        "retry_failed_tasks",
        "accept_partial_and_continue",
        "accept_check_and_finish",
    }:
        payload["feedback"] = feedback.strip()
    return payload


def build_cli_summary(result: dict) -> dict:
    interrupts = extract_interrupt_payloads(result)
    summary = {
        key: result.get(key)
        for key in (
            "thread_id",
            "checkpoint_path",
            "task_status",
            "message",
            "error",
            "final_summary",
            "agent_log_path",
        )
        if result.get(key) is not None
    }
    if interrupts:
        summary["interrupts"] = interrupts
        summary["resume_required"] = True
    return summary


def prompt_for_human_decision(review_request: dict, *, input_fn=input, print_fn=print) -> dict:
    actions = list(review_request.get("available_actions") or [])
    labels = {
        "continue_revision": "继续修正",
        "accept_and_continue": "接受当前方案并继续后续阶段",
        "retry_coordinator": "重新交给协调器判断",
        "retry_failed_tasks": "重试结构设计失败任务",
        "accept_partial_and_continue": "接受结构设计部分成果并继续",
        "continue_modeling_revision": "继续验算返修",
        "accept_check_and_finish": "接受当前验算风险并结束",
        "abort": "中断任务",
    }
    print_fn(review_request.get("summary") or review_request.get("message") or "流程等待人工复核。")
    for index, action in enumerate(actions, start=1):
        print_fn(f"  {index}. {labels.get(action, action)} [{action}]")
    while True:
        raw_choice = input_fn("请选择操作编号：").strip()
        if raw_choice.isdigit() and 1 <= int(raw_choice) <= len(actions):
            action = actions[int(raw_choice) - 1]
            break
        if raw_choice in actions:
            action = raw_choice
            break
        print_fn("输入无效，请输入列表中的编号或动作名称。")
    if action == "abort" or action == "retry_coordinator":
        return build_resume_payload(action)
    feedback = input_fn("请输入补充修正要求（可留空）：")
    if action not in {"continue_revision", "continue_modeling_revision"}:
        return build_resume_payload(action, feedback=feedback)
    raw_rounds = input_fn("增加修正轮数（默认 1）：").strip()
    extra_rounds = int(raw_rounds) if raw_rounds else 1
    return build_resume_payload(action, feedback=feedback, extra_rounds=extra_rounds)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))

    initial_state = None
    if args.initial_state_json:
        with open(args.initial_state_json, "r", encoding="utf-8") as f:
            initial_state = json.load(f)

    user_request = args.user_request
    if not user_request and not args.resume:
        settings = load_settings(args.config)
        user_request = pick(settings, "user_request")

    if not user_request and not args.resume:
        parser.error("未提供 user_request，且 settings.yaml 中未配置 task.user_request。")
    if args.resume and not args.decision:
        parser.error("恢复人工复核线程时必须通过 --decision 指定处理动作。")
    if args.extra_rounds < 1:
        parser.error("--extra-rounds 必须大于等于 1。")

    from bridge_agents import run_agent

    resume_payload = None
    if args.resume:
        resume_payload = build_resume_payload(
            args.decision,
            feedback=args.feedback,
            extra_rounds=args.extra_rounds,
        )
    if args.resume and args.clear_stale_resume:
        from bridge_agents.checkpointing import clear_stale_resume_writes
        from pathlib import Path

        settings = load_settings(args.config)
        output_dir = pick(settings, "output_dir", default="output")
        cleared = clear_stale_resume_writes(Path(str(output_dir)), args.resume)
        if cleared:
            print(
                f"[恢复前清理] 已清除 {len(cleared)} 条残留恢复写入："
                + "; ".join(
                    f"{item['channel']}@{str(item['task_id'])[:8]}={json_safe(item['value'])}"
                    for item in cleared
                ),
                file=sys.stderr,
            )
        else:
            print("[恢复前清理] 未发现残留恢复写入。", file=sys.stderr)
    result = run_agent(
        user_request or "",
        config_path=args.config,
        initial_state=initial_state,
        use_graph_v2=args.graph_v2 or bool(args.resume),
        thread_id=args.resume or args.thread_id,
        resume=resume_payload,
    )
    while args.interactive:
        interrupts = extract_interrupt_payloads(result)
        if not interrupts:
            break
        decision = prompt_for_human_decision(interrupts[0])
        result = run_agent(
            "",
            config_path=args.config,
            use_graph_v2=True,
            thread_id=result.get("thread_id"),
            resume=decision,
        )
    output = result if args.full_output else build_cli_summary(result)
    print(json.dumps(json_safe(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
