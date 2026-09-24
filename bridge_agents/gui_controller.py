from __future__ import annotations

import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import yaml


HUMAN_REVIEW_ACTIONS = frozenset(
    {
        "continue_revision",
        "accept_and_continue",
        "retry_coordinator",
        "retry_failed_tasks",
        "accept_partial_and_continue",
        "continue_modeling_revision",
        "accept_check_and_finish",
        "abort",
    }
)

STAGE_ORDER = (
    "initial_design",
    "layout_revision",
    "structural_design",
    "modeling_check",
    "final_output",
)

AGENT_TO_STAGE = {
    "InitialDesignAgent": "initial_design",
    "LayoutRevisionAgent": "layout_revision",
    "StructuralDesignAgent": "structural_design",
    "ModelingCheckAgent": "modeling_check",
    "initial_design": "initial_design",
    "layout_revision": "layout_revision",
    "structural_design": "structural_design",
    "modeling_check": "modeling_check",
    "final_output": "final_output",
}


def load_drawing_catalog(output_dir: str | Path) -> list[Dict[str, Any]]:
    root = Path(output_dir)
    index_path = root / "deliverables" / "drawings" / "drawing_index.json"
    if not index_path.is_file():
        return []
    payload = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    groups = payload.get("groups") if isinstance(payload, dict) else []
    catalog: list[Dict[str, Any]] = []
    for item in groups if isinstance(groups, list) else []:
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("design_group_id") or "").strip()
        group_root = index_path.parent / group_id
        sheets = []
        for sheet in item.get("sheets") or []:
            if not isinstance(sheet, dict):
                continue
            scr_name = str(sheet.get("scr_name") or "").strip()
            svg_name = str(sheet.get("svg_name") or "").strip()
            sheets.append(
                {
                    "sheet_id": str(sheet.get("sheet_id") or ""),
                    "title": str(sheet.get("title") or sheet.get("sheet_id") or "未命名图纸"),
                    "scr_path": str(group_root / scr_name) if scr_name else "",
                    "svg_path": str(group_root / svg_name) if svg_name else "",
                }
            )
        if not sheets:
            scr_values = item.get("scr_paths") or ([item.get("scr_path")] if item.get("scr_path") else [])
            svg_values = item.get("svg_paths") or ([item.get("svg_path")] if item.get("svg_path") else [])
            for index, value in enumerate(scr_values):
                svg_value = svg_values[index] if index < len(svg_values) else ""
                sheets.append(
                    {
                        "sheet_id": f"sheet-{index + 1}",
                        "title": f"图纸 {index + 1}",
                        "scr_path": str(root / value) if value else "",
                        "svg_path": str(root / svg_value) if svg_value else "",
                    }
                )
        catalog.append(
            {
                "design_group_id": group_id,
                "member_piers": list(item.get("member_piers") or []),
                "issue_status": str(item.get("issue_status") or "unknown"),
                "geometry_valid": item.get("geometry_valid"),
                "sheets": sheets,
            }
        )
    return catalog


@dataclass
class GuiRunConfig:
    user_request: str
    data_path: str
    input_drawing_path: str
    output_dir: str
    file_prefix: str
    thread_id: str
    max_revision_rounds: int = 3
    max_check_revision_rounds: int = 2
    code_rag_enabled: bool = False

    def validate(self) -> None:
        required = {
            "任务描述": self.user_request,
            "路线数据目录": self.data_path,
            "输入图纸": self.input_drawing_path,
            "输出目录": self.output_dir,
            "线路文件前缀": self.file_prefix,
            "运行 ID": self.thread_id,
        }
        missing = [label for label, value in required.items() if not str(value or "").strip()]
        if missing:
            raise ValueError(f"缺少必填项：{'、'.join(missing)}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.thread_id):
            raise ValueError("运行 ID 只能包含字母、数字、点、下划线和连字符。")
        if int(self.max_revision_rounds) < 1 or int(self.max_check_revision_rounds) < 1:
            raise ValueError("修正轮数必须大于等于 1。")


def extract_interrupt(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for item in result.get("__interrupt__") or []:
        value = item.get("value") if isinstance(item, dict) else getattr(item, "value", item)
        if isinstance(value, dict):
            return value
    return None


def infer_stage_states(result: Dict[str, Any]) -> Dict[str, str]:
    states = {stage: "pending" for stage in STAGE_ORDER}
    completed_agents = set(result.get("completed_agents") or [])
    for agent_name in completed_agents:
        stage = AGENT_TO_STAGE.get(str(agent_name))
        if stage:
            states[stage] = "completed"

    active = AGENT_TO_STAGE.get(str(result.get("active_agent") or ""))
    task_status = str(result.get("task_status") or "")
    review = extract_interrupt(result)
    if review:
        review_stage = {
            "layout_collision_review": "layout_revision",
            "structural_batch_review": "structural_design",
            "modeling_check_review": "modeling_check",
        }.get(str(review.get("review_type") or ""), active)
        if review_stage:
            states[review_stage] = "review"
    elif task_status in {"completed", "completed_with_accepted_risks"}:
        for stage in STAGE_ORDER:
            states[stage] = "completed"
    elif task_status in {"failed", "error"}:
        states[active or "final_output"] = "failed"
    elif active:
        states[active] = "running"
    return states


def snapshot_run_progress(output_dir: str | Path) -> Dict[str, Any]:
    """基于输出目录文件状态推导运行进度（供 GUI 定时轮询，不依赖阻塞式 run_agent 返回）。

    返回字段：
    - running: 是否检测到活跃运行（存在 checkpoint 或日志写入）
    - active_stage: 推断的当前阶段名（STAGE_ORDER 内）或 None
    - stage_states: STAGE_ORDER 各阶段状态（pending/running/completed/review/failed）
    - current_scope: 当前正在处理的设计组/单元（从 prompt_trace 最新一条推断）
    - completed_groups / total_groups: 配筋设计组完成进度
    - recent_log: prompt_trace 最新一条记录的摘要文本
    """
    root = Path(output_dir)
    if not root.is_dir():
        return {"running": False, "active_stage": None, "stage_states": {
            stage: "pending" for stage in STAGE_ORDER
        }, "current_scope": None, "completed_groups": 0, "total_groups": 0, "recent_log": ""}

    # 1) 检测活跃运行：checkpoint 或日志目录存在且近期有写入
    trace_path = root / "logs" / "prompt_trace.jsonl"
    wal_path = root / "checkpoints" / "graph_v2.sqlite-wal"
    recent_mtime = 0.0
    for candidate in (trace_path, wal_path, root / "logs" / "llm_usage.jsonl"):
        if candidate.is_file():
            recent_mtime = max(recent_mtime, candidate.stat().st_mtime)
    running = recent_mtime > 0 and (time.time() - recent_mtime) < 300  # 5 分钟内活跃

    # 2) 各阶段完成判定（文件存在即视为已完成该阶段）
    stage_states = {stage: "pending" for stage in STAGE_ORDER}
    layout_done = (
        (root / "layout_revision" / "final_layout_result.json").is_file()
        or bool(list((root / "layout_revision").glob("final_layout_result.json")) if (root / "layout_revision").is_dir() else [])
    )
    if layout_done:
        stage_states["initial_design"] = "completed"
        stage_states["layout_revision"] = "completed"
    units_done = (root / "structural_design" / "design_units" / "design_units_result.json").is_file()
    dimension_done = (root / "structural_design" / "dimension_design" / "dimension_design_result.json").is_file()
    if units_done:
        stage_states["structural_design"] = "running"
    if dimension_done:
        stage_states["structural_design"] = "running"
    reinforcement_summary = root / "structural_design" / "reinforcement_design" / "reinforcement_design_result.json"
    structural_summary = root / "structural_design" / "structural_design_result.json"
    if reinforcement_summary.is_file() or structural_summary.is_file():
        stage_states["structural_design"] = "completed"
    capacity_summary = root / "capacity_check" / "capacity_check_batch_summary.json"
    capacity_legacy = root / "capacity_check" / "capacity_check_summary.json"
    capacity_dir = root / "capacity_check"
    capacity_done = bool(
        capacity_summary.is_file()
        or capacity_legacy.is_file()
        or (
            capacity_dir.is_dir()
            and any(
                (capacity_dir / group / "capacity_check_summary.json").is_file()
                for group in os.listdir(capacity_dir)
            )
        )
    )
    if capacity_done:
        stage_states["modeling_check"] = "completed"
    drawing_done = (root / "deliverables" / "drawings" / "drawing_index.json").is_file()
    if drawing_done:
        stage_states["final_output"] = "completed"

    # 3) 当前处理中的设计组：读 prompt_trace 最新一条
    current_scope: Optional[str] = None
    recent_log = ""
    latest_status = ""
    if trace_path.is_file():
        try:
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                scope = str(rec.get("scope_id") or "").strip()
                status = str(rec.get("status") or "").strip()
                if scope:
                    current_scope = scope
                    latest_status = status
                if status == "rendered":
                    recent_log = f"正在生成 {scope} 的配筋设计…"
                elif status == "completed":
                    recent_log = f"{scope} 配筋完成"
                elif status == "failed":
                    recent_log = f"{scope} 配筋失败：{(rec.get('metadata') or {}).get('error') or '解析失败'}"
                elif status == "preprocessing":
                    recent_log = f"准备 {scope} 的配筋设计…"
        except Exception:
            pass

    # 4) 配筋设计组进度统计
    rd = root / "structural_design" / "reinforcement_design"
    completed_groups = 0
    total_groups = 0
    if rd.is_dir():
        groups = [d for d in rd.iterdir() if d.is_dir()]
        total_groups = len(groups)
        for g in groups:
            if list(g.glob("reinforcement_result_*.json")):
                completed_groups += 1

    # 5) 推断 active_stage：仅当存在运行痕迹（checkpoint/日志）时按完成度推进
    has_trace = bool(
        (root / "checkpoints").is_dir()
        or (root / "logs").is_dir()
        or layout_done
        or (root / "structural_design").is_dir()
    )
    active_stage: Optional[str] = None
    if has_trace:
        if not layout_done:
            active_stage = "initial_design"
        elif dimension_done and not (reinforcement_summary.is_file() or structural_summary.is_file()):
            active_stage = "structural_design"
        elif (reinforcement_summary.is_file() or structural_summary.is_file()) and not capacity_done:
            active_stage = "modeling_check"
        elif capacity_done and not drawing_done:
            active_stage = "final_output"
        elif not drawing_done:
            active_stage = "structural_design"
    # 若该阶段未完成且任务活跃，标为 running；否则保持文件推导出的状态
    if active_stage and running and stage_states.get(active_stage) != "completed":
        stage_states[active_stage] = "running"

    return {
        "running": running,
        "active_stage": active_stage,
        "stage_states": stage_states,
        "current_scope": current_scope,
        "completed_groups": completed_groups,
        "total_groups": total_groups,
        "recent_log": recent_log,
        "latest_status": latest_status,
    }


# ============================================================ #
# 运行快照路径绝对化
#
# 运行快照被写到 output_dir/run_configs/<thread_id>.yaml 并作为运行时
# config_path（断点恢复也复用）。若其中仍保留相对路径（如
# data/standards.json），各工具会按“配置文件所在目录”或“进程工作目录”
# 自行锚定，导致解析到 output_dir/run_configs/data/standards.json 这类
# 不存在的位置。因此写快照前把语义为文件/目录的路径统一转为绝对路径。
# ============================================================ #
_PATH_VALUE_SUFFIXES = ("_path", "_dir", "_file", "_root")
_JSON_PATH_KEYS = frozenset({"route_scan_config_json", "layer_config_json"})
_PATH_BEARING_SECTIONS = frozenset(
    {
        "paths",
        "flashmpts",
        "structure",
        "modeling",
        "segmentation",
        "drawing_crop_and_mask",
        "obstacle_semantic_extractor",
    }
)


def _is_path_config_key(section: str, key: str) -> bool:
    if section not in _PATH_BEARING_SECTIONS:
        return False
    return key in _JSON_PATH_KEYS or key.endswith(_PATH_VALUE_SUFFIXES)


def _resolve_snapshot_root(payload: Dict[str, Any], base_config_path: str | Path) -> Path:
    """相对路径的基准：显式 project_root > config/ 的上级 > 配置文件所在目录。"""
    paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
    explicit = str(paths.get("project_root") or "").strip() if isinstance(paths, dict) else ""
    if explicit:
        return Path(os.path.expandvars(os.path.expanduser(explicit))).resolve()
    base_dir = Path(base_config_path).resolve().parent
    if base_dir.name == "config":
        return base_dir.parent
    return base_dir


def _absolutize_snapshot_paths(payload: Dict[str, Any], base_config_path: str | Path) -> None:
    """把快照中语义为文件/目录的相对路径原地改为绝对路径。"""
    root = _resolve_snapshot_root(payload, base_config_path)
    for section, section_value in payload.items():
        if section not in _PATH_BEARING_SECTIONS or not isinstance(section_value, dict):
            continue
        for key, value in list(section_value.items()):
            if not _is_path_config_key(section, key):
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            expanded = os.path.expandvars(os.path.expanduser(value.strip()))
            if not expanded or Path(expanded).is_absolute():
                continue
            section_value[key] = str((root / expanded).resolve())


class GraphV2Controller:
    def __init__(
        self,
        *,
        run_agent_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        review_agent_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if run_agent_fn is None:
            from bridge_agents import run_agent

            run_agent_fn = run_agent
        self._run_agent = run_agent_fn
        self._review_agent_factory = review_agent_factory

    @staticmethod
    def _load_base_config(path: str | Path) -> Dict[str, Any]:
        source = Path(path)
        if not source.is_file():
            raise ValueError(f"基础配置文件不存在：{source}")
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("基础配置必须是 YAML 对象。")
        return payload

    def write_config_snapshot(
        self,
        config: GuiRunConfig,
        *,
        base_config_path: str | Path,
    ) -> Path:
        config.validate()
        payload = deepcopy(self._load_base_config(base_config_path))
        payload.setdefault("task", {})["user_request"] = config.user_request.strip()
        paths = payload.setdefault("paths", {})
        paths["data_path"] = config.data_path.strip()
        paths["input_drawing_path"] = config.input_drawing_path.strip()
        paths["output_dir"] = config.output_dir.strip()
        payload.setdefault("route_data", {})["file_prefix"] = config.file_prefix.strip()
        payload.setdefault("agent", {})["max_revision_rounds"] = int(config.max_revision_rounds)
        payload.setdefault("modeling", {})["max_check_revision_rounds"] = int(
            config.max_check_revision_rounds
        )
        payload.setdefault("code_rag", {})["enabled"] = bool(config.code_rag_enabled)
        payload["code_rag"].setdefault("retrieval_mode", "auto")
        payload["code_rag"].setdefault("top_k", 10)

        # 相对路径统一锚定项目根并转绝对，避免快照被搬到 output_dir/run_configs
        # 后按“配置文件所在目录”解析到不存在的位置。
        _absolutize_snapshot_paths(payload, base_config_path)

        snapshot = Path(config.output_dir).resolve() / "run_configs" / f"{config.thread_id}.yaml"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return snapshot

    def start(
        self,
        config: GuiRunConfig,
        *,
        base_config_path: str | Path = "config/settings.yaml",
    ) -> Dict[str, Any]:
        snapshot = self.write_config_snapshot(config, base_config_path=base_config_path)
        result = dict(
            self._run_agent(
                config.user_request,
                config_path=str(snapshot),
                use_graph_v2=True,
                thread_id=config.thread_id,
            )
        )
        result["config_snapshot_path"] = str(snapshot)
        return result

    def resume(
        self,
        *,
        thread_id: str,
        config_snapshot_path: str | Path,
        action: str,
        feedback: str = "",
        extra_rounds: int = 1,
        available_actions: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        if action not in HUMAN_REVIEW_ACTIONS:
            raise ValueError(f"不支持的人工复核动作：{action}")
        offered = set(available_actions or HUMAN_REVIEW_ACTIONS)
        if action not in offered:
            raise ValueError(f"当前复核不允许执行动作：{action}")
        payload: Dict[str, Any] = {"action": action}
        if action in {"continue_revision", "continue_modeling_revision"}:
            payload.update(
                {
                    "feedback": feedback.strip(),
                    "extra_rounds": max(1, int(extra_rounds)),
                }
            )
        elif action in {
            "accept_and_continue",
            "retry_failed_tasks",
            "accept_partial_and_continue",
            "accept_check_and_finish",
        }:
            payload["feedback"] = feedback.strip()
        return dict(
            self._run_agent(
                "",
                config_path=str(config_snapshot_path),
                use_graph_v2=True,
                thread_id=thread_id,
                resume=payload,
            )
        )

    def _review_agent(self) -> Any:
        if self._review_agent_factory is not None:
            return self._review_agent_factory()
        from .design_review import DesignReviewAgent

        return DesignReviewAgent()

    def assess(self, *, output_dir: str, config_snapshot_path: str | Path) -> Dict[str, Any]:
        result = self._review_agent().assess(
            output_dir=output_dir,
            config_path=str(config_snapshot_path),
        )
        return result.model_dump(mode="json")

    def answer(
        self,
        question: str,
        *,
        output_dir: str,
        config_snapshot_path: str | Path,
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not question.strip():
            raise ValueError("专业问答问题不能为空。")
        result = self._review_agent().answer(
            question.strip(),
            output_dir=output_dir,
            config_path=str(config_snapshot_path),
            history=list(history) if history else None,
        )
        return result.model_dump(mode="json")
