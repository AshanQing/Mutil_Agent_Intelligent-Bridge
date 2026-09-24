from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from bridge_agents.gui_controller import STAGE_ORDER


STAGE_TITLES = {
    "initial_design": "初始设计",
    "layout_revision": "布跨修正",
    "structural_design": "结构设计",
    "modeling_check": "建模验算",
    "final_output": "成果交付",
}
STAGE_DETAILS = {
    "initial_design": "读取路线与图纸，生成初始布跨",
    "layout_revision": "碰撞检测与布跨方案修正",
    "structural_design": "尺寸设计、内力分析与配筋",
    "modeling_check": "承载力计算、反馈与复验",
    "final_output": "整理成果并绘制交付图纸",
}
STATE_LABELS = {
    "pending": "待开始",
    "running": "进行中",
    "completed": "已完成",
    "review": "待复核",
    "failed": "异常",
}
AGENT_TITLES = {
    "initial_design": "初始设计智能体",
    "layout_revision": "布跨修正智能体",
    "structural_design": "结构设计智能体",
    "modeling_check": "建模验算智能体",
    "final_output": "成果交付智能体",
}
REVIEW_ACTION_LABELS = {
    "continue_revision": "继续布跨修正",
    "accept_and_continue": "接受布跨并继续",
    "retry_coordinator": "重新交给协调器",
    "retry_failed_tasks": "重新设计失败任务",
    "accept_partial_and_continue": "接受部分成果并继续",
    "continue_modeling_revision": "重新设计并复验",
    "accept_check_and_finish": "接受风险并结束",
    "abort": "终止任务",
}


@dataclass(frozen=True)
class StageCardModel:
    key: str
    index: int
    title: str
    detail: str
    state: str
    state_label: str


@dataclass(frozen=True)
class MetricModel:
    label: str
    value: str
    caption: str
    tone: str = "neutral"


@dataclass(frozen=True)
class TimelineItem:
    title: str
    detail: str
    tone: str = "neutral"


@dataclass(frozen=True)
class ReviewModel:
    required: bool
    title: str
    message: str
    failed_ids: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class DashboardModel:
    project_name: str
    route_range: str
    run_status: str
    progress_percent: int
    active_stage: str | None
    active_agent: str
    current_scope: str
    stages: tuple[StageCardModel, ...]
    metrics: tuple[MetricModel, ...]
    timeline: tuple[TimelineItem, ...]
    review: ReviewModel
    drawing_count: int
    is_demo: bool = False
    data_badge: str = "实时成果"


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _progress_percent(stage_states: Mapping[str, str]) -> int:
    weights = {"completed": 1.0, "running": 0.5, "review": 0.5, "failed": 0.5}
    value = sum(weights.get(str(stage_states.get(stage, "pending")), 0.0) for stage in STAGE_ORDER)
    return round(value / len(STAGE_ORDER) * 100)


def _run_status(progress: Mapping[str, Any], stage_states: Mapping[str, str]) -> str:
    if "review" in stage_states.values():
        return "等待人工判断"
    if "failed" in stage_states.values():
        return "需要处理"
    if all(state == "completed" for state in stage_states.values()):
        return "已完成"
    if bool(progress.get("running")):
        return "运行中"
    if any(state in {"completed", "running"} for state in stage_states.values()):
        return "已暂停"
    return "尚未开始"


def _review_model(review: Mapping[str, Any] | None) -> ReviewModel:
    if not review:
        return ReviewModel(
            required=False,
            title="当前无需人工介入",
            message="系统会先自动重试；连续失败后，人工判断入口会显示在这里。",
        )
    failed_ids = tuple(str(value) for value in (review.get("failed_task_ids") or review.get("failed_ids") or ()))
    actions = tuple(
        REVIEW_ACTION_LABELS.get(str(action), str(action))
        for action in (review.get("allowed_actions") or review.get("actions") or ())
    )
    return ReviewModel(
        required=True,
        title="需要工程师判断",
        message=str(review.get("message") or "当前阶段需要人工选择后续处理方式。"),
        failed_ids=failed_ids,
        actions=actions,
    )


def build_dashboard_model(
    progress: Mapping[str, Any],
    *,
    drawings: Sequence[Any] = (),
    project_name: str = "未命名桥梁工程",
    route_range: str = "路线范围未设置",
    review: Mapping[str, Any] | None = None,
    recent_events: Iterable[Mapping[str, Any]] = (),
    is_demo: bool = False,
) -> DashboardModel:
    raw_states = progress.get("stage_states") or {}
    stage_states = {stage: str(raw_states.get(stage, "pending")) for stage in STAGE_ORDER}
    inferred_review: Mapping[str, Any] | None = review
    if stage_states["final_output"] == "completed" and stage_states["modeling_check"] != "completed":
        stage_states["modeling_check"] = "review"
        inferred_review = inferred_review or {
            "message": "已发现成果交付记录，但建模验算尚未完成。请补做验算，或由工程师确认是否接受该流程风险。",
            "failed_ids": ["modeling_check"],
            "actions": ["continue_modeling_revision", "accept_check_and_finish"],
        }
    active_stage = str(progress.get("active_stage") or "") or next(
        (stage for stage in STAGE_ORDER if stage_states[stage] in {"running", "review", "failed"}),
        None,
    )
    review_model = _review_model(inferred_review)
    if inferred_review is not review and review_model.required:
        review_model = ReviewModel(
            required=True,
            title="阶段依赖不完整",
            message=review_model.message,
            failed_ids=review_model.failed_ids,
            actions=review_model.actions,
        )
    if review_model.required and active_stage and stage_states.get(active_stage) == "pending":
        stage_states[active_stage] = "review"

    stages = tuple(
        StageCardModel(
            key=stage,
            index=index,
            title=STAGE_TITLES[stage],
            detail=STAGE_DETAILS[stage],
            state=stage_states[stage],
            state_label=STATE_LABELS.get(stage_states[stage], stage_states[stage]),
        )
        for index, stage in enumerate(STAGE_ORDER, start=1)
    )
    completed_groups = int(progress.get("completed_groups") or 0)
    total_groups = int(progress.get("total_groups") or 0)
    drawing_count = len(drawings)
    review_count = len(review_model.failed_ids) if review_model.required else 0
    metrics = (
        MetricModel("全流程进度", f"{_progress_percent(stage_states)}%", "按五个工程阶段统计", "primary"),
        MetricModel("配筋设计组", f"{completed_groups} / {total_groups}", "已完成 / 总数", "accent"),
        MetricModel("待人工复核", str(review_count), "需工程师作出判断", "warning" if review_count else "neutral"),
        MetricModel("图纸成果", str(drawing_count), "已进入成果目录", "success"),
    )

    timeline: list[TimelineItem] = []
    recent_log = str(progress.get("recent_log") or "").strip()
    if recent_log:
        timeline.append(TimelineItem("配筋设计进展", recent_log, "primary"))
    for event in recent_events:
        title = str(event.get("title") or event.get("action") or "运行事件")
        detail = str(event.get("detail") or event.get("message") or "")
        timeline.append(TimelineItem(title, detail, str(event.get("tone") or "neutral")))
    if not timeline:
        timeline.append(TimelineItem("等待运行事件", "启动或恢复任务后，关键阶段事件会出现在这里。"))

    return DashboardModel(
        project_name=project_name.strip() or "未命名桥梁工程",
        route_range=route_range.strip() or "路线范围未设置",
        run_status=_run_status(progress, stage_states),
        progress_percent=_progress_percent(stage_states),
        active_stage=active_stage,
        active_agent=AGENT_TITLES.get(active_stage or "", "等待协调器调度"),
        current_scope=str(progress.get("current_scope") or "等待分配设计单元"),
        stages=stages,
        metrics=metrics,
        timeline=tuple(timeline[:5]),
        review=review_model,
        drawing_count=drawing_count,
        is_demo=is_demo,
        data_badge="演示数据" if is_demo else "实时成果",
    )


def build_demo_dashboard_model() -> DashboardModel:
    return build_dashboard_model(
        {
            "running": True,
            "active_stage": "structural_design",
            "stage_states": {
                "initial_design": "completed",
                "layout_revision": "completed",
                "structural_design": "running",
                "modeling_check": "pending",
                "final_output": "pending",
            },
            "current_scope": "桥墩设计组 PG-04",
            "completed_groups": 6,
            "total_groups": 10,
            "recent_log": "PG-04 配筋结果已解析，正在生成下一设计组。",
        },
        drawings=[{"title": "总体布置图"}, {"title": "桥墩一般构造图"}, {"title": "盖梁配筋图"}],
        project_name="示例项目 9.8",
        route_range="K12+400 — K14+860",
        recent_events=(
            {"title": "尺寸设计完成", "detail": "10 个设计组尺寸结果已汇总", "tone": "success"},
            {"title": "自动恢复生效", "detail": "一次格式解析失败后自动重试成功", "tone": "warning"},
        ),
        is_demo=True,
    )
