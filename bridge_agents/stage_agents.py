from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from langgraph.graph import END, StateGraph

from .contracts import ArtifactRecord, RevisionRequest, StageName
from .state import AgentState
from .actions import ActionSpec, get_action_spec, get_agent_action_specs, run_action
from .agent_logger import AgentLogger, summarize_state_for_log
from .handoff import build_handoff
from .llm_generation_guard import generate_structured_output
from .prompt_registry import render_prompt
from .tool_actions import passes_collision_threshold
from .utils import (
    extract_json_object,
    get_controller_llm,
    get_structural_llm,
    json_safe,
    metric_float,
    write_json,
)

logger = logging.getLogger(__name__)


def _history_entry(**fields: Any) -> Dict[str, Any]:
    """构造结构设计阶段历史条目。

    带 recorded_at 是为了让"同一动作重复失败"也各自留痕：结构设计历史用
    append_unique 归并（用于吸收内外层图重复写入），没有时间戳时内容完全相同的
    失败条目会被去重掉，运行历史看起来像"只失败过一次"。
    """
    return {"recorded_at": datetime.now().isoformat(timespec="microseconds"), **fields}


class StageAgent:
    """阶段智能体基类。

    每个阶段智能体负责一个业务阶段。内部既可以是固定流程，也可以是 ReAct 循环。
    """

    name = "StageAgent"
    mode = "base"

    def run(self, state: AgentState) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _merge(working: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        working.update(update)
        return working

    @staticmethod
    def _stop_if_error(update: Dict[str, Any]) -> bool:
        return bool(update.get("error"))
    
    prompt_id = ""

    def _load_skill(self, state: Dict[str, Any]) -> Tuple[str, str]:
        if not self.prompt_id:
            raise ValueError(f"{self.name} 未配置 prompt_id。")
        rendered = render_prompt(
            self.prompt_id,
            {},
            config_path=state.get("config_path") or "config/settings.yaml",
        )
        prompt_trace = list(state.get("prompt_trace") or [])
        prompt_trace.append(rendered.to_dict())
        state["prompt_trace"] = prompt_trace
        return rendered.system_content or rendered.user_content, rendered.template_path

    def _activate_skill(self, working: Dict[str, Any]) -> str:
        skill_text, skill_path = self._load_skill(working)
        working["active_agent_skill_name"] = self.name
        working["active_agent_skill_path"] = skill_path
        working["active_agent_skill"] = skill_text

        agent_skills = dict(working.get("agent_skills") or {})
        agent_skills[self.name] = skill_text
        working["agent_skills"] = agent_skills

        run_logger = AgentLogger.from_state(working)
        run_logger.log_event(
            "agent_skill_loaded",
            agent=self.name,
            stage=self.name,
            name="load_agent_skill",
            status="completed" if skill_text else "empty",
            payload={
                "skill_path": skill_path,
                "prompt_id": self.prompt_id,
                "skill_length": len(skill_text),
            },
        )
        run_logger.attach(working)
        return skill_text
    def _log_action_start(self, working: Dict[str, Any], action_name: str) -> None:
        run_logger = AgentLogger.from_state(working)
        run_logger.log_action_start(
            agent=self.name,
            action=action_name,
            input_summary=summarize_state_for_log(working),
        )
        run_logger.attach(working)

    def _log_action_end(self, working: Dict[str, Any], action_name: str, update: Dict[str, Any]) -> None:
        run_logger = AgentLogger.from_state(working)
        run_logger.log_action_end(
            agent=self.name,
            action=action_name,
            output=update,
        )
        run_logger.attach(working)

    def _run_action(self, action_spec: ActionSpec, working: Dict[str, Any]) -> Dict[str, Any]:
        result = run_action(action_spec, working)  # type: ignore[arg-type]
        action_event = {
            "agent": self.name,
            "action": action_spec.name,
            "status": "completed" if result.ok else "failed",
            "expected_key": result.expected_key,
            "produced_keys": list(result.produced_keys),
            "error": result.error,
        }
        events = list(working.get("agent_events") or [])
        events.append(action_event)
        working["agent_events"] = events
        return result.update

class InitialDesignAgent(StageAgent):
    """初步设桥布跨智能体。

    当前版本采用：
    initial_design_agent_skill.txt → LLM 生成 required_steps → 脚本按 required_steps 执行。

    注意：
    - 执行起点由 LLM 判断；
    - 脚本不再根据已有字段自行决定跳过步骤；
    - 脚本只负责把 required_steps 中的步骤名映射为具体 action。
    """

    name = "InitialDesignAgent"
    mode = "llm_planned_pipeline"
    prompt_id = "agents.initial_design.v1"

    def _load_skill(self, state: Dict[str, Any]) -> str:
        rendered = render_prompt(
            self.prompt_id,
            {},
            config_path=state.get("config_path") or "config/settings.yaml",
        )
        prompt_trace = list(state.get("prompt_trace") or [])
        prompt_trace.append(rendered.to_dict())
        state["prompt_trace"] = prompt_trace
        return rendered.system_content or rendered.user_content

    def _make_stage_plan(self, working: Dict[str, Any]) -> Dict[str, Any]:
        """由 LLM 根据当前状态判断本轮初步设计阶段 required_steps。"""
        skill_text = self._load_skill(working)

        payload = {
            "task_type": "stage_planning",
            "user_intent": working.get("user_intent"),
            "user_request": working.get("user_request"),
            "state_summary": {
                "start_station_available": bool(working.get("start_station")),
                "end_station_available": bool(working.get("end_station")),

                "data_path_available": bool(working.get("data_path")),
                "file_prefix_available": bool(working.get("file_prefix")),
                "input_drawing_available": bool(working.get("input_drawing_path") or working.get("drawing_path")),

                "cropped_data_available": bool(working.get("cropped_data") or working.get("data_loader_result")),
                "plane_json_path_available": bool(working.get("plane_json_path")),

                "drawing_mask_result_available": bool(working.get("drawing_mask_result")),
                "png_path_available": bool(working.get("png_path") or working.get("jpg_path")),
                "pgw_path_available": bool(working.get("pgw_path")),
                "mask_path_available": bool(working.get("mask_path")),

                "obstacle_extractor_result_available": bool(working.get("obstacle_extractor_result")),
                "obstacle_json_path_available": bool(working.get("obstacle_json_path")),

                "design_input_available": bool(working.get("design_input")),
                "few_shots_available": bool(working.get("few_shots")),
                "few_shots_dir_available": bool(working.get("few_shots_dir")),

                "layout_result_available": bool(working.get("layout_result") or working.get("existing_layout_result")),
            },
            "allowed_steps": [
                "load_data",
                "drawing_crop_and_mask",
                "obstacle_semantic_extractor",
                "select_samples",
                "generate_layout_design",
            ],
            "required_output_schema": {
                "stage": "initial_design",
                "agent": "InitialDesignAgent",
                "agent_role_confirmed": True,
                "can_execute": True,
                "input_assessment": {},
                "execution_policy": "llm_planned_resume",
                "fixed_steps": [
                    "load_data",
                    "drawing_crop_and_mask",
                    "obstacle_semantic_extractor",
                    "select_samples",
                    "generate_layout_design",
                ],
                "required_steps": [],
                "skipped_steps": [],
                "design_focus": [],
                "constraints": [],
                "notes": "",
            },
        }
        llm = get_controller_llm(working.get("config_path") or "config/settings.yaml")
        print("\n[InitialDesignAgent stage_planning state_summary]")
        print(json.dumps(json_safe(payload["state_summary"]), ensure_ascii=False, indent=2), flush=True)

        response = llm.invoke([
            ("system", skill_text),
            ("user", json.dumps(json_safe(payload), ensure_ascii=False, indent=2)),
        ])
        print("\n[InitialDesignAgent stage_planning raw LLM response]")
        print(response.content, flush=True)

        plan = extract_json_object(response.content)
        plan["_debug_state_summary"] = payload["state_summary"]
        plan["_debug_llm_raw_response"] = response.content

        if not isinstance(plan.get("required_steps"), list):
            raise ValueError("InitialDesignAgent 阶段计划缺少 required_steps 列表。")

        return plan

    def _steps_from_stage_plan(self, stage_plan: Dict[str, Any]) -> List[ActionSpec]:
        """将 LLM 返回的 required_steps 转换为可执行 action 列表。"""
        action_registry = get_agent_action_specs(self.name)

        required_steps = stage_plan.get("required_steps")
        if not isinstance(required_steps, list):
            raise ValueError("InitialDesignAgent 阶段计划中的 required_steps 不是列表。")

        steps: List[ActionSpec] = []

        for step_name in required_steps:
            step_name = str(step_name).strip()

            if step_name not in action_registry:
                raise ValueError(f"InitialDesignAgent 阶段计划包含非法步骤: {step_name}")

            steps.append(action_registry[step_name])

        return steps

    def _plan_initial_design(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)
        try:
            stage_plan = working.get("initial_design_stage_plan") or self._make_stage_plan(working)
        except Exception as e:
            logger.error("InitialDesignAgent 阶段计划生成失败：%s", e, exc_info=True)
            return {
                "active_agent": self.name,
                "initial_design_graph_finished": True,
                "task_status": "failed",
                "message": "InitialDesignAgent 阶段计划生成失败。",
                "error": str(e),
            }

        working["initial_design_stage_plan"] = stage_plan
        history_entry = {
            "step": "stage_planning",
            "status": "completed",
            "required_steps": stage_plan.get("required_steps"),
            "skipped_steps": stage_plan.get("skipped_steps"),
            "error": None,
        }

        run_logger = AgentLogger.from_state(working)
        run_logger.log_event(
            "stage_plan",
            agent=self.name,
            stage=self.name,
            name="stage_planning",
            status="completed",
            payload=stage_plan,
        )
        run_logger.attach(working)

        update: Dict[str, Any] = {
            "active_agent": self.name,
            "initial_design_stage_plan": stage_plan,
            "initial_design_history": [history_entry],
            "initial_design_graph_steps": [],
            "initial_design_graph_index": 0,
            "initial_design_graph_action": None,
            "initial_design_graph_finished": False,
            "agent_log": working.get("agent_log"),
            "agent_events": working.get("agent_events") or [],
            "error": None,
        }
        if working.get("prompt_trace") is not None:
            update["prompt_trace"] = working["prompt_trace"]

        if stage_plan.get("can_execute") is False:
            return {
                **update,
                "initial_design_graph_finished": True,
                "task_status": "failed",
                "message": "InitialDesignAgent 判断当前输入不足，无法执行初步设桥布跨阶段。",
                "error": stage_plan.get("notes") or "can_execute=false",
            }

        try:
            steps = self._steps_from_stage_plan(stage_plan)
        except Exception as e:
            logger.error("InitialDesignAgent required_steps 解析失败：%s", e, exc_info=True)
            return {
                **update,
                "initial_design_graph_finished": True,
                "task_status": "failed",
                "message": "InitialDesignAgent required_steps 解析失败。",
                "error": str(e),
            }

        update["initial_design_graph_steps"] = [step.name for step in steps]
        return update

    @staticmethod
    def _route_after_initial_plan(state: AgentState) -> str:
        if state.get("error") or state.get("initial_design_graph_finished"):
            return "done"
        return "prepare"

    def _prepare_initial_action(self, state: AgentState) -> Dict[str, Any]:
        step_names = list(state.get("initial_design_graph_steps") or [])
        step_index = int(state.get("initial_design_graph_index") or 0)
        if step_index >= len(step_names):
            return {
                "initial_design_graph_action": None,
                "initial_design_graph_finished": True,
                "active_agent": self.name,
            }
        return {
            "initial_design_graph_action": step_names[step_index],
            "initial_design_graph_finished": False,
            "active_agent": self.name,
        }

    @staticmethod
    def _route_after_initial_prepare(state: AgentState) -> str:
        if state.get("initial_design_graph_finished"):
            return "complete"
        return "execute"

    def _execute_initial_action(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)
        step_name = str(working.get("initial_design_graph_action") or "")
        action_spec = get_agent_action_specs(self.name).get(step_name)
        if action_spec is None:
            return {
                "active_agent": self.name,
                "initial_design_graph_finished": True,
                "task_status": "failed",
                "message": "InitialDesignAgent required_steps 解析失败。",
                "error": f"InitialDesignAgent 阶段计划包含非法步骤: {step_name}",
            }

        self._log_action_start(working, step_name)
        action_update = self._run_action(action_spec, working)
        working.update(action_update)
        self._log_action_end(working, step_name, action_update)

        history_entry = {
            "step": step_name,
            "status": "failed" if action_update.get("error") else "completed",
            "expected_key": action_spec.expected_key,
            "error": action_update.get("error"),
        }
        graph_update: Dict[str, Any] = {
            **action_update,
            "active_agent": self.name,
            "initial_design_history": [history_entry],
            "initial_design_graph_index": int(working.get("initial_design_graph_index") or 0) + 1,
            "initial_design_graph_action": None,
            "initial_design_graph_finished": bool(action_update.get("error")),
            "agent_log": working.get("agent_log"),
            "agent_events": working.get("agent_events") or [],
        }
        if action_update.get("error"):
            graph_update.update({
                "task_status": "failed",
                "message": f"{self.name} 在 {step_name} 阶段失败。",
            })
        return graph_update

    @staticmethod
    def _route_after_initial_action(state: AgentState) -> str:
        if state.get("error") or state.get("initial_design_graph_finished"):
            return "done"
        return "continue"

    def _complete_initial_design(self, state: AgentState) -> Dict[str, Any]:
        return {
            "active_agent": self.name,
            "task_status": "initial_design_completed",
            "message": "初步设桥布跨阶段已按 LLM 规划的 required_steps 执行完成。",
            "error": None,
        }

    def _build_initial_design_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("plan_initial_design", self._plan_initial_design)
        builder.add_node("prepare_initial_action", self._prepare_initial_action)
        builder.add_node("execute_initial_action", self._execute_initial_action)
        builder.add_node("complete_initial_design", self._complete_initial_design)

        builder.set_entry_point("plan_initial_design")
        builder.add_conditional_edges(
            "plan_initial_design",
            self._route_after_initial_plan,
            {"prepare": "prepare_initial_action", "done": END},
        )
        builder.add_conditional_edges(
            "prepare_initial_action",
            self._route_after_initial_prepare,
            {"execute": "execute_initial_action", "complete": "complete_initial_design"},
        )
        builder.add_conditional_edges(
            "execute_initial_action",
            self._route_after_initial_action,
            {"continue": "prepare_initial_action", "done": END},
        )
        builder.add_edge("complete_initial_design", END)
        return builder.compile()

    def _attach_stage_handoff(
        self,
        update: Dict[str, Any],
        prior_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = {**prior_state, **update}
        artifacts: List[ArtifactRecord] = []
        if context.get("layout_result") is not None:
            artifacts.append(ArtifactRecord(
                artifact_type="layout_result",
                state_key="layout_result",
                producer=self.name,
            ))
        handoff = build_handoff(
            StageName.INITIAL_DESIGN.value,
            update,
            produced_artifacts=artifacts,
        )
        return {**update, "latest_handoff": handoff.to_dict()}

    def run(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)
        working["initial_design_graph_steps"] = []
        working["initial_design_graph_index"] = 0
        working["initial_design_graph_action"] = None
        working["initial_design_graph_finished"] = False

        result = self._build_initial_design_graph().invoke(working)
        internal_keys = {
            "initial_design_graph_steps",
            "initial_design_graph_index",
            "initial_design_graph_action",
            "initial_design_graph_finished",
        }
        returned = {
            key: value
            for key, value in result.items()
            if key not in internal_keys and (key not in state or state.get(key) != value)
        }
        returned["active_agent"] = self.name
        returned["error"] = result.get("error")
        if result.get("error"):
            returned["task_status"] = "failed"
            returned.setdefault("message", result.get("message") or "InitialDesignAgent 执行失败。")
        return self._attach_stage_handoff(returned, working)

class LayoutRevisionAgent(StageAgent):
    """设桥布跨修正智能体。

    该阶段保留 ReAct 循环，因为它需要根据检测指标动态决定是否修正、是否复检、是否转人工。
    """

    name = "LayoutRevisionAgent"
    mode = "react_loop"
    prompt_id = "agents.layout_revision.v1"

    def _attach_stage_handoff(
        self,
        update: Dict[str, Any],
        prior_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = {**prior_state, **update}
        artifacts: List[ArtifactRecord] = []
        if context.get("final_layout_result") is not None:
            artifacts.append(ArtifactRecord(
                artifact_type="final_layout_result",
                state_key="final_layout_result",
                path=context.get("final_layout_result_path") or context.get("layout_revision_result_path"),
                producer=self.name,
            ))

        task_status = str(update.get("task_status") or "")
        if task_status in {"layout_revision_completed", "layout_revision_skipped"}:
            recommended_next_stage = StageName.STRUCTURAL_DESIGN.value
        elif task_status in {"manual_review", "manual_review_required"}:
            recommended_next_stage = StageName.MANUAL_REVIEW.value
        else:
            recommended_next_stage = None

        handoff = build_handoff(
            StageName.LAYOUT_REVISION.value,
            update,
            produced_artifacts=artifacts,
            recommended_next_stage=recommended_next_stage,
        )
        return {**update, "latest_handoff": handoff.to_dict()}

    def _visible_state(self, state: Dict[str, Any], latest_observation: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {
            "layout_result_available": bool(state.get("layout_result") or state.get("existing_layout_result")),
            "iteration_index": int(state.get("iteration_index") or 0),
            "max_revision_rounds": int(state.get("max_revision_rounds") or 3),
            "has_collision_metrics": bool(state.get("collision_metrics")),
            "current_metrics": state.get("collision_metrics") or {},
            "collision_items_count": len(state.get("collision_items") or []),
            "revision_instruction_available": bool(state.get("revision_instruction")),
            "revision_prompt_available": bool(state.get("revision_prompt")),
            "latest_action": latest_observation.get("action") if latest_observation else None, 
            "latest_observation_success": latest_observation.get("success") if latest_observation else None,
        }

    def _execute_action(self, action: str, working: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """执行 ReAct Action。返回 update, observation, finished。"""
        if action == "run_collision_detection":
            update = self._run_action(get_action_spec(action), working)
            if (
                not working.get("initial_collision_metrics")
                and update.get("collision_metrics")
            ):
                update["initial_collision_metrics"] = update.get("collision_metrics")
                update["initial_layout_result"] = (
                    update.get("layout_result")
                    or working.get("layout_result")
                    or working.get("design_result")
                    or working.get("existing_layout_result")
                )
            observation = {
                "action": action,
                "success": not bool(update.get("error")),
                "metrics": update.get("collision_metrics") or {},
                "collision_items_count": len(update.get("collision_items") or []),
                "report_json_path": update.get("collision_report_json_path"),
                "error": update.get("error"),
            }
            return update, observation, False

        if action == "generate_revision_instruction":
            update = self._run_action(get_action_spec(action), working)
            observation = {
                "action": action,
                "success": not bool(update.get("error")),
                "revision_instruction_path": update.get("revision_instruction_path"),
                "error": update.get("error"),
            }
            return update, observation, False

        if action == "build_revision_prompt":
            update = self._run_action(get_action_spec(action), working)
            observation = {
                "action": action,
                "success": not bool(update.get("error")),
                "revision_prompt_path": update.get("revision_prompt_path"),
                "error": update.get("error"),
            }
            return update, observation, False

        if action == "generate_revised_layout":
            update = self._run_action(get_action_spec(action), working)
            # 生成新方案后进入下一轮，后续必须重新检测。
            next_iteration = int(working.get("iteration_index") or 0) + 1
            update["iteration_index"] = next_iteration
            update["collision_metrics"] = None
            update["collision_items"] = None
            update["revision_instruction"] = None
            update["revision_prompt"] = None
            observation = {
                "action": action,
                "success": not bool(update.get("error")),
                "revision_result_path": update.get("revision_result_path"),
                "next_iteration_index": next_iteration,
                "error": update.get("error"),
            }
            return update, observation, False

        if action == "finish_revision":
            current_layout = working.get("layout_result") or working.get("existing_layout_result")
            current_metrics = working.get("collision_metrics") or {}
            initial_layout = working.get("initial_layout_result")
            initial_metrics = working.get("initial_collision_metrics")
            final_layout = current_layout
            chosen_note = "当前方案"
            if (
                initial_layout is not None
                and initial_metrics
                and current_metrics
                and self._compare_metrics(initial_metrics, current_metrics) < 0
            ):
                final_layout = initial_layout
                chosen_note = "初始方案（指标更优）"
            output_dir = working.get("output_dir") or "outputs"
            result_path = os.path.join(output_dir, "layout_revision", "final_layout_result.json")

            try:
                write_json(result_path, final_layout)
            except Exception as e:
                logger.warning("布跨修正最终结果写入失败，但不阻断流程：%s", e)
                result_path = None

            update = {
                "task_status": "layout_revision_completed",
                "layout_revision_completed": True,
                "layout_revision_result": final_layout,
                "final_layout_result": final_layout,
                "layout_result": final_layout,
                "collision_metrics": current_metrics,
                "layout_revision_result_path": result_path,
                "final_layout_result_path": result_path,
                "message": f"设桥布跨方案已通过碰撞检测阈值，修正流程结束（选用{chosen_note}）。",
                "error": None,
            }
            observation = {"action": action, "success": True, "chosen": chosen_note}
            return update, observation, True

        if action == "manual_review":
            update = {
                "task_status": "manual_review_required",
                "message": "自动修正未能在轮次限制内收敛，建议转人工复核。",
                "error": None,
            }
            observation = {"action": action, "success": True}
            return update, observation, True

        update = {"error": f"非法 Action: {action}"}
        observation = {"action": action, "success": False, "error": update["error"]}
        return update, observation, True

    @staticmethod
    def _metrics_have_any_conflict(metrics: Dict[str, Any]) -> bool:
        '''指标中是否存在任何非零冲突信号。'''
        if metrics.get("has_collision"):
            return True
        for key in (
            "conflict_column_rate",
            "total_intrusion_depth_columns",
            "avg_overlap_ratio_columns",
            "avg_intrusion_depth_columns",
        ):
            if metric_float(metrics, key) > 1e-9:
                return True
        return False

    @staticmethod
    def _compare_metrics(first: Dict[str, Any], second: Dict[str, Any]) -> int:
        '''按冲突率→总侵入深度→平均重叠率→平均侵入深度的字典序比较。

        返回 -1 表示 first 更优，1 表示 second 更优，0 表示视为同等。
        '''
        for key in (
            "conflict_column_rate",
            "total_intrusion_depth_columns",
            "avg_overlap_ratio_columns",
            "avg_intrusion_depth_columns",
        ):
            first_value = metric_float(first, key)
            second_value = metric_float(second, key)
            if first_value < second_value - 1e-9:
                return -1
            if first_value > second_value + 1e-9:
                return 1
        return 0

    def _prepare_revision_step(self, state: AgentState) -> Dict[str, Any]:
        step_index = int(state.get("layout_revision_react_step") or 0) + 1
        max_react_steps = int(state.get("max_react_steps") or 16)
        update: Dict[str, Any] = {
            "layout_revision_react_step": step_index,
            "layout_revision_action": None,
            "layout_revision_action_obj": {},
            "layout_revision_finished": False,
            "active_agent": self.name,
        }

        if step_index > max_react_steps:
            return {
                **update,
                "layout_revision_finished": True,
                "task_status": "manual_review_required",
                "message": "达到最大 ReAct 步数，建议转人工复核。",
                "error": None,
            }

        metrics = state.get("collision_metrics") or {}
        iteration = int(state.get("iteration_index") or 0)
        raw_max_rounds = state.get("max_revision_rounds")
        max_rounds = int(raw_max_rounds) if raw_max_rounds is not None else 3
        if metrics and passes_collision_threshold(metrics):
            if bool(state.get("layout_optimization_started")) and iteration == 0:
                # 优化轮修正链推进：不依赖 LLM 决策，按指令→提示词→生成新方案顺序执行
                if not state.get("revision_instruction"):
                    action = "generate_revision_instruction"
                    thought = "初始方案满足阈值但存在冲突，生成优化修正指令。"
                elif not state.get("revision_prompt"):
                    action = "build_revision_prompt"
                    thought = "基于修正指令构建布跨修正提示词。"
                else:
                    action = "generate_revised_layout"
                    thought = "按修正提示词生成优化后布跨方案。"
                return {
                    **update,
                    "layout_revision_action": action,
                    "layout_revision_action_obj": {
                        "thought": thought,
                        "action": action,
                        "action_input": {},
                    },
                }
            if iteration == 0 and max_rounds >= 1 and self._metrics_have_any_conflict(metrics):
                return {
                    **update,
                    "layout_optimization_started": True,
                    "layout_revision_action": "generate_revision_instruction",
                    "layout_revision_action_obj": {
                        "thought": "初始方案满足阈值但存在冲突，执行一次优化修正后取最优结果。",
                        "action": "generate_revision_instruction",
                        "action_input": {},
                    },
                }
            return {
                **update,
                "layout_revision_action": "finish_revision",
                "layout_revision_action_obj": {
                    "thought": "碰撞指标已满足阈值，结束布跨修正。",
                    "action": "finish_revision",
                    "action_input": {},
                },
            }

        if metrics and int(state.get("iteration_index") or 0) >= int(state.get("max_revision_rounds") or 3):
            return {
                **update,
                "layout_revision_action": "manual_review",
                "layout_revision_action_obj": {
                    "thought": "修正轮次已达到上限，转人工复核。",
                    "action": "manual_review",
                    "action_input": {},
                },
            }

        return update

    @staticmethod
    def _route_after_revision_prepare(state: AgentState) -> str:
        if state.get("layout_revision_finished"):
            return "done"
        if state.get("layout_revision_action"):
            return "execute"
        return "decide"

    def _decide_revision_action(self, state: AgentState, llm: Any, skill_text: str) -> Dict[str, Any]:
        visible_state = self._visible_state(
            dict(state),
            state.get("layout_revision_latest_observation") or {},
        )
        try:
            response = llm.invoke([
                ("system", skill_text),
                ("user", json.dumps(json_safe(visible_state), ensure_ascii=False, indent=2)),
            ])
            action_obj = extract_json_object(response.content)
            action = action_obj.get("action")
        except Exception as e:
            logger.warning("LayoutRevisionAgent ReAct 输出解析失败，启用规则兜底：%s", e)
            action = self._fallback_action(dict(state))
            action_obj = {"thought": "LLM 输出解析失败，使用规则兜底。", "action": action, "action_input": {}}

        return {
            "layout_revision_action": str(action),
            "layout_revision_action_obj": action_obj,
            "active_agent": self.name,
        }

    def _execute_revision_graph_action(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)
        action = str(working.get("layout_revision_action") or "")
        action_obj = working.get("layout_revision_action_obj") if isinstance(working.get("layout_revision_action_obj"), dict) else {}
        step_index = int(working.get("layout_revision_react_step") or 0)

        self._log_action_start(working, action)
        update, observation, finished = self._execute_action(action, working)
        working.update(update)
        self._log_action_end(working, action, update)

        run_logger = AgentLogger.from_state(working)
        run_logger.log_thought(
            agent=self.name,
            thought=action_obj.get("thought", ""),
            action=action,
            observation=observation,
        )
        run_logger.attach(working)

        history_entry = {
            "react_step": step_index,
            "thought": action_obj.get("thought", ""),
            "action": action,
            "observation": observation,
        }

        graph_update: Dict[str, Any] = {
            **update,
            "active_agent": self.name,
            "layout_revision_latest_observation": observation,
            "layout_revision_finished": bool(finished),
            "revision_history": [history_entry],
            "agent_log": working.get("agent_log"),
            "agent_events": working.get("agent_events") or [],
        }

        if update.get("error"):
            graph_update.update({
                "layout_revision_finished": True,
                "task_status": "failed",
                "message": f"{self.name} 执行 {action} 失败。",
            })

        return graph_update

    @staticmethod
    def _route_after_revision_execute(state: AgentState) -> str:
        if state.get("error") or state.get("layout_revision_finished"):
            return "done"
        return "continue"

    def _build_layout_revision_graph(self, llm: Any, skill_text: str):
        builder = StateGraph(AgentState)
        builder.add_node("prepare_revision_step", self._prepare_revision_step)
        builder.add_node(
            "decide_revision_action",
            lambda state: self._decide_revision_action(state, llm, skill_text),
        )
        builder.add_node("execute_revision_action", self._execute_revision_graph_action)

        builder.set_entry_point("prepare_revision_step")
        builder.add_conditional_edges(
            "prepare_revision_step",
            self._route_after_revision_prepare,
            {
                "decide": "decide_revision_action",
                "execute": "execute_revision_action",
                "done": END,
            },
        )
        builder.add_edge("decide_revision_action", "execute_revision_action")
        builder.add_conditional_edges(
            "execute_revision_action",
            self._route_after_revision_execute,
            {
                "continue": "prepare_revision_step",
                "done": END,
            },
        )
        return builder.compile()

    def run(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)

        if self._should_skip_revision_stage(working):
            update = {
                "active_agent": self.name,
                "task_status": "layout_revision_skipped",
                "message": "检测到已有布跨修正最终结果，跳过设桥布跨修正阶段。",
                "layout_result": working.get("final_layout_result") or working.get("layout_revision_result") or working.get("layout_result"),
                "layout_revision_completed": True,
                "layout_revision_result": working.get("layout_revision_result") or working.get("final_layout_result"),
                "final_layout_result": working.get("final_layout_result") or working.get("layout_revision_result"),
                "revision_history": list(working.get("revision_history") or []),
                "error": None,
            }
            return self._attach_stage_handoff(update, working)

        working.setdefault("iteration_index", 0)
        working.setdefault("max_revision_rounds", 3)
        working["layout_revision_react_step"] = 0
        working["layout_revision_latest_observation"] = {}
        working["layout_revision_action"] = None
        working["layout_revision_action_obj"] = {}
        working["layout_revision_finished"] = False

        skill_text = self._activate_skill(working)
        llm = get_controller_llm(working.get("config_path") or "config/settings.yaml")
        max_react_steps = int(working.get("max_react_steps") or 16)
        result = self._build_layout_revision_graph(llm, skill_text).invoke(
            working,
            {"recursion_limit": max_react_steps * 3 + 8},
        )

        internal_keys = {
            "layout_revision_react_step",
            "layout_revision_latest_observation",
            "layout_revision_action",
            "layout_revision_action_obj",
            "layout_revision_finished",
        }
        returned = {
            k: v
            for k, v in result.items()
            if k not in internal_keys and (k not in state or state.get(k) != v)
        }
        returned["active_agent"] = self.name
        returned["error"] = result.get("error")
        if result.get("error"):
            returned["task_status"] = "failed"
            returned.setdefault("message", result.get("message") or "LayoutRevisionAgent 执行失败。")
        return self._attach_stage_handoff(returned, working)

    def _fallback_action(self, working: Dict[str, Any]) -> str:
        metrics = working.get("collision_metrics") or {}
        if not metrics:
            return "run_collision_detection"
        if passes_collision_threshold(metrics):
            return "finish_revision"
        if int(working.get("iteration_index") or 0) >= int(working.get("max_revision_rounds") or 3):
            return "manual_review"
        if not working.get("revision_instruction"):
            return "generate_revision_instruction"
        if not working.get("revision_prompt"):
            return "build_revision_prompt"
        return "generate_revised_layout"
    def _should_skip_revision_stage(self, working: Dict[str, Any]) -> bool:
        """判断是否跳过设桥布跨修正阶段。

        只有在本阶段已经形成最终结果时才允许跳过。
        不能仅凭已有 layout_result、design_units 或 dimension_design_result 跳过。
        """
        user_request = str(working.get("user_request") or "")
        user_intent = str(working.get("user_intent") or "")

        explicit_revision_request = any(
            key in user_request
            for key in [
                "重新检测",
                "重新校核",
                "重新复检",
                "重新修正",
                "碰撞检测",
                "布跨修正",
                "修正布跨",
                "检查布跨",
                "复检布跨",
            ]
        )

        if explicit_revision_request:
            return False

        if user_intent in ["layout_design_check_revision", "layout_check_revision"]:
            return False

        if bool(working.get("layout_revision_completed")):
            return True

        if working.get("layout_revision_result") or working.get("final_layout_result"):
            return True

        verification_result = working.get("verification_result") or {}
        if isinstance(verification_result, dict):
            if verification_result.get("status") in ["passed", "layout_revision_completed"]:
                return True

        metrics = working.get("collision_metrics") or {}
        if metrics and passes_collision_threshold(metrics):
            if (
                self._metrics_have_any_conflict(metrics)
                and int(working.get("iteration_index") or 0) == 0
                and not working.get("layout_optimization_started")
            ):
                # 满足阈值但仍有冲突：仍需执行一次优化修正后取最优
                return False
            return True

        return False


class StructuralDesignAgent(StageAgent):
    """下部结构设计阶段智能体。

    当前版本采用：
    structural_design_agent_skill.txt → LLM 生成 required_steps → 脚本按 required_steps 执行。

    注意：
    - 执行起点由 LLM 判断；
    - 脚本不再根据已有字段自行决定跳过步骤；
    - 脚本只负责把 required_steps 中的步骤名映射为具体 action。
    """

    name = "StructuralDesignAgent"
    mode = "llm_planned_pipeline"
    prompt_id = "agents.structural_design.v1"

    def _load_skill(self, state: Dict[str, Any]) -> str:
        rendered = render_prompt(
            self.prompt_id,
            {},
            config_path=state.get("config_path") or "config/settings.yaml",
        )
        prompt_trace = list(state.get("prompt_trace") or [])
        prompt_trace.append(rendered.to_dict())
        state["prompt_trace"] = prompt_trace
        return rendered.system_content or rendered.user_content

    def _make_stage_plan(self, working: Dict[str, Any]) -> Dict[str, Any]:
        """由 LLM 根据当前状态判断本轮结构设计阶段 required_steps。

        这里不再使用 STAGE_CONTROLLER_PROMPT，也不再提供确定性 fallback。
        """
        skill_text = self._load_skill(working)

        payload = {
            "task_type": "stage_planning",
            "user_intent": working.get("user_intent"),
            "user_request": working.get("user_request"),
            "state_summary": {
                "layout_result_available": bool(working.get("layout_result")),
                "existing_layout_result_available": bool(working.get("existing_layout_result")),
                "design_units_available": bool(working.get("design_units")),
                "dimension_design_available": bool(working.get("dimension_design_result")),
                "reinforcement_design_available": bool(working.get("reinforcement_design_result")),
                "structural_design_result_available": bool(working.get("structural_design_result")),
                "revision_context_available": bool(working.get("revision_context")),
                "revision_context": working.get("revision_context"),
                "feedback_decision": working.get("feedback_decision"),
            },
            "allowed_steps": [
                "extract_design_units",
                "dimension_design",
                "compute_pier_groups",
                "reinforcement_design",
                "summarize_structural_design_result",
            ],
            "required_output_schema": {
                "stage": "structural_design",
                "agent": "StructuralDesignAgent",
                "agent_role_confirmed": True,
                "can_execute": True,
                "input_assessment": {
                    "layout_result_available": True,
                    "existing_layout_result_available": False,
                    "design_units_available": False,
                    "dimension_design_available": False,
                    "reinforcement_design_available": False,
                    "structural_design_result_available": False,
                },
                "execution_policy": "llm_planned_resume",
                "fixed_steps": [
                    "extract_design_units",
                    "dimension_design",
                    "compute_pier_groups",
                    "reinforcement_design",
                    "summarize_structural_design_result",
                ],
                "required_steps": [],
                "skipped_steps": [],
                "design_focus": [],
                "constraints": [],
                "notes": "",
            },
        }

        llm = get_structural_llm(working.get("config_path") or "config/settings.yaml")
        initial_prompt = json.dumps(json_safe(payload), ensure_ascii=False, indent=2)

        def invoke_plan(user_prompt: str) -> str:
            response = llm.invoke([
                ("system", skill_text),
                ("user", user_prompt),
            ])
            return str(response.content or "")

        def parse_plan(raw: str) -> Dict[str, Any]:
            parsed = extract_json_object(raw)
            if not isinstance(parsed.get("required_steps"), list):
                raise ValueError("StructuralDesignAgent 阶段计划缺少 required_steps 列表。")
            return parsed

        generation = generate_structured_output(
            invoke=invoke_plan,
            parse=parse_plan,
            initial_prompt=initial_prompt,
            repair_prompt=lambda raw, error: (
                "请只修复以下结构设计阶段计划的 JSON 格式和缺失的 required_steps 列表，"
                "保持原计划语义，输出完整 JSON 对象。\n"
                f"错误：{error}\n原输出：\n{raw}"
            ),
            regeneration_prompt=lambda error: (
                f"{initial_prompt}\n\n上一次阶段计划无法解析：{error}\n"
                "请重新生成完整阶段计划，只输出 JSON 对象。"
            ),
            max_format_repairs=int(working.get("llm_max_format_repairs") or 1),
            max_regenerations=int(working.get("llm_max_regenerations") or 2),
        )
        plan = generation.value

        revision_context = working.get("revision_context") or {}
        target_step = revision_context.get("target_step")
        if target_step == "dimension_design":
            plan["required_steps"] = [
                "dimension_design",
                "compute_pier_groups",
                "reinforcement_design",
                "summarize_structural_design_result",
            ]
        elif target_step == "reinforcement_design":
            plan["required_steps"] = [
                "compute_pier_groups",
                "reinforcement_design",
                "summarize_structural_design_result",
            ]

        return plan

    def _steps_from_stage_plan(self, stage_plan: Dict[str, Any]) -> List[ActionSpec]:
        """将 LLM 返回的 required_steps 转换为可执行 action 列表。

        这里不判断从哪一步开始，只做字符串到函数的映射。

        确定性兜底：compute_pier_groups 生成配筋必需的墩柱净高，只要本阶段要执行
        dimension_design 或 reinforcement_design，就必须带上该步骤——
        - dimension_design 在计划中时插在其后（保持依赖顺序）；
        - 仅 reinforcement_design（断点续跑、dimension 已有被跳过）时插在其前。
        """
        action_registry = get_agent_action_specs(self.name)

        required_steps = stage_plan.get("required_steps")
        if not isinstance(required_steps, list) or not required_steps:
            raise ValueError("StructuralDesignAgent 阶段计划中的 required_steps 为空。")

        ordered_steps = list(required_steps)
        if "compute_pier_groups" in action_registry and "compute_pier_groups" not in ordered_steps:
            if "dimension_design" in ordered_steps:
                index = ordered_steps.index("dimension_design")
                ordered_steps.insert(index + 1, "compute_pier_groups")
            elif "reinforcement_design" in ordered_steps:
                index = ordered_steps.index("reinforcement_design")
                ordered_steps.insert(index, "compute_pier_groups")

        steps: List[ActionSpec] = []

        for step_name in ordered_steps:
            step_name = str(step_name).strip()

            if step_name == "summarize_structural_design_result":
                continue

            if step_name not in action_registry:
                raise ValueError(f"StructuralDesignAgent 阶段计划包含非法步骤: {step_name}")

            steps.append(action_registry[step_name])

        return steps

    def _summarize_result(self, working: Dict[str, Any]) -> Dict[str, Any]:
        dimension_batch = working.get("dimension_batch_status") or {}
        reinforcement_batch = working.get("reinforcement_batch_status") or {}
        batches_complete = not (
            dimension_batch.get("stage_complete") is False
            or reinforcement_batch.get("stage_complete") is False
        )
        return {
            "status": "completed" if batches_complete else "manual_review_required",
            "agent": self.name,
            "stage_plan": working.get("structural_stage_plan"),
            "source_layout": {
                "layout_result_available": bool(working.get("layout_result") or working.get("existing_layout_result")),
                "layout_source": "layout_result" if working.get("layout_result") else "existing_layout_result",
            },
            "design_units": working.get("design_units"),
            "dimension_design_result": working.get("dimension_design_result"),
            "pier_group_result": working.get("pier_group_result"),
            "reinforcement_design_result": working.get("reinforcement_design_result"),
            "dimension_batch_status": dimension_batch,
            "reinforcement_batch_status": reinforcement_batch,
            "handoff_to_modeling_check_agent": {
                "required_fields": [
                    "design_units",
                    "dimension_design_result",
                    "pier_group_result",
                    "reinforcement_design_result",
                ],
                "ready": batches_complete and bool(
                    working.get("design_units")
                    and working.get("dimension_design_result")
                    and working.get("reinforcement_design_result")
                ),
            },
        }

    def _plan_structural_design(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)

        try:
            # 若来自 ModelingCheckAgent 的反馈修正，则必须重新规划本轮 required_steps，
            # 避免复用上一次完整结构设计的 structural_stage_plan。
            if working.get("revision_context"):
                stage_plan = self._make_stage_plan(working)
            else:
                stage_plan = working.get("structural_stage_plan") or self._make_stage_plan(working)
        except Exception as e:
            logger.error("StructuralDesignAgent 阶段计划生成失败：%s", e, exc_info=True)
            return {
                "active_agent": self.name,
                "structural_design_graph_finished": True,
                "task_status": "failed",
                "message": "StructuralDesignAgent 阶段计划生成失败。",
                "error": str(e),
            }

        working["structural_stage_plan"] = stage_plan
        history_entry = _history_entry(
            step="stage_planning",
            status="completed",
            required_steps=stage_plan.get("required_steps"),
            skipped_steps=stage_plan.get("skipped_steps"),
            error=None,
        )

        run_logger = AgentLogger.from_state(working)
        run_logger.log_event(
            "stage_plan",
            agent=self.name,
            stage=self.name,
            name="stage_planning",
            status="completed",
            payload=stage_plan,
        )
        run_logger.attach(working)

        update: Dict[str, Any] = {
            "active_agent": self.name,
            "structural_stage_plan": stage_plan,
            "structural_design_history": [history_entry],
            "structural_design_graph_steps": [],
            "structural_design_graph_index": 0,
            "structural_design_graph_action": None,
            "structural_design_graph_finished": False,
            "agent_log": working.get("agent_log"),
            "agent_events": working.get("agent_events") or [],
            "error": None,
        }
        if working.get("prompt_trace") is not None:
            update["prompt_trace"] = working["prompt_trace"]

        if stage_plan.get("can_execute") is False:
            return {
                **update,
                "structural_design_graph_finished": True,
                "task_status": "failed",
                "message": "StructuralDesignAgent 判断当前输入不足，无法执行结构设计阶段。",
                "error": stage_plan.get("notes") or "can_execute=false",
            }

        try:
            steps = self._steps_from_stage_plan(stage_plan)
        except Exception as e:
            logger.error("StructuralDesignAgent required_steps 解析失败：%s", e, exc_info=True)
            return {
                **update,
                "structural_design_graph_finished": True,
                "task_status": "failed",
                "message": "StructuralDesignAgent required_steps 解析失败。",
                "error": str(e),
            }

        update["structural_design_graph_steps"] = [step.name for step in steps]
        return update

    @staticmethod
    def _route_after_structural_plan(state: AgentState) -> str:
        if state.get("error") or state.get("structural_design_graph_finished"):
            return "done"
        return "prepare"

    def _prepare_structural_action(self, state: AgentState) -> Dict[str, Any]:
        step_names = list(state.get("structural_design_graph_steps") or [])
        step_index = int(state.get("structural_design_graph_index") or 0)
        if step_index >= len(step_names):
            return {
                "structural_design_graph_action": None,
                "structural_design_graph_finished": True,
                "active_agent": self.name,
            }
        return {
            "structural_design_graph_action": step_names[step_index],
            "structural_design_graph_finished": False,
            "active_agent": self.name,
        }

    @staticmethod
    def _route_after_structural_prepare(state: AgentState) -> str:
        if state.get("structural_design_graph_finished"):
            return "complete"
        return "execute"

    def _execute_structural_action(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)
        step_name = str(working.get("structural_design_graph_action") or "")
        action_spec = get_agent_action_specs(self.name).get(step_name)
        if action_spec is None:
            return {
                "active_agent": self.name,
                "structural_design_graph_finished": True,
                "task_status": "failed",
                "message": "StructuralDesignAgent required_steps 解析失败。",
                "error": f"StructuralDesignAgent 阶段计划包含非法步骤: {step_name}",
            }

        self._log_action_start(working, step_name)
        action_update = self._run_action(action_spec, working)
        working.update(action_update)
        self._log_action_end(working, step_name, action_update)

        history_entry = _history_entry(
            step=step_name,
            status="failed" if action_update.get("error") else "completed",
            expected_key=action_spec.expected_key,
            error=action_update.get("error"),
        )
        graph_update: Dict[str, Any] = {
            **action_update,
            "active_agent": self.name,
            "structural_design_history": [history_entry],
            "structural_design_graph_index": int(working.get("structural_design_graph_index") or 0) + 1,
            "structural_design_graph_action": None,
            "structural_design_graph_finished": bool(action_update.get("error")),
            "agent_log": working.get("agent_log"),
            "agent_events": working.get("agent_events") or [],
        }
        if action_update.get("error"):
            graph_update.update({
                "task_status": "failed",
                "message": f"{self.name} 在 {step_name} 阶段失败。",
            })
        return graph_update

    @staticmethod
    def _route_after_structural_action(state: AgentState) -> str:
        if state.get("error") or state.get("structural_design_graph_finished"):
            return "done"
        return "continue"

    def _complete_structural_design(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)
        stage_plan = working.get("structural_stage_plan") or {}

        required_steps = stage_plan.get("required_steps") or []
        if "summarize_structural_design_result" not in required_steps:
            return {
                "active_agent": self.name,
                "structural_stage_plan": stage_plan,
                "structural_design_graph_finished": True,
                "task_status": "failed",
                "message": "StructuralDesignAgent 阶段计划未包含 summarize_structural_design_result，无法形成最终 structural_design_result。",
                "error": "required_steps 缺少 summarize_structural_design_result。",
            }

        structural_design_result = self._summarize_result(working)
        working["structural_design_result"] = structural_design_result
        dimension_batch = working.get("dimension_batch_status") or {}
        reinforcement_batch = working.get("reinforcement_batch_status") or {}
        incomplete_batch = (
            dimension_batch.get("stage_complete") is False
            or reinforcement_batch.get("stage_complete") is False
        )
        failed_task_ids = list(
            reinforcement_batch.get("failed_task_ids")
            or dimension_batch.get("failed_unit_ids")
            or []
        )

        history_entry = _history_entry(
            step="summarize_structural_design_result",
            status="completed",
            error=None,
        )

        output_dir = working.get("output_dir") or "outputs"
        result_path = os.path.join(output_dir, "structural_design", "structural_design_result.json")

        try:
            write_json(result_path, structural_design_result)
        except Exception as e:
            logger.warning("结构设计结果写入失败，但不阻断流程：%s", e)
            result_path = None

        structural_stage_summary = {
            "status": "manual_review_required" if incomplete_batch else "structural_design_completed",
            "required_steps": required_steps,
            "skipped_steps": stage_plan.get("skipped_steps") or [],
            "design_units_available": bool(working.get("design_units")),
            "dimension_design_available": bool(working.get("dimension_design_result")),
            "pier_group_available": bool(working.get("pier_group_result")),
            "reinforcement_design_available": bool(working.get("reinforcement_design_result")),
            "structural_design_result_available": bool(structural_design_result),
            "result_path": result_path,
            "dimension_batch_status": dimension_batch,
            "reinforcement_batch_status": reinforcement_batch,
        }

        return {
            "structural_stage_plan": stage_plan,
            "structural_stage_summary": structural_stage_summary,
            "structural_design_result": structural_design_result,
            "structural_design_result_path": result_path,
            "structural_design_history": [history_entry],
            "structural_design_graph_finished": True,
            "active_agent": self.name,
            "task_status": "manual_review_required" if incomplete_batch else "structural_design_completed",
            "message": (
                "结构设计批次存在失败任务，需人工复核后决定重试、接受部分成果或终止。"
                if incomplete_batch
                else "下部结构设计阶段已按 LLM 规划的 required_steps 执行完成，并已形成 structural_design_result。"
            ),
            "unresolved_manual_review": incomplete_batch,
            "failed_task_ids": failed_task_ids,
            "error": None,
        }

    def _build_structural_design_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("plan_structural_design", self._plan_structural_design)
        builder.add_node("prepare_structural_action", self._prepare_structural_action)
        builder.add_node("execute_structural_action", self._execute_structural_action)
        builder.add_node("complete_structural_design", self._complete_structural_design)

        builder.set_entry_point("plan_structural_design")
        builder.add_conditional_edges(
            "plan_structural_design",
            self._route_after_structural_plan,
            {"prepare": "prepare_structural_action", "done": END},
        )
        builder.add_conditional_edges(
            "prepare_structural_action",
            self._route_after_structural_prepare,
            {"execute": "execute_structural_action", "complete": "complete_structural_design"},
        )
        builder.add_conditional_edges(
            "execute_structural_action",
            self._route_after_structural_action,
            {"continue": "prepare_structural_action", "done": END},
        )
        builder.add_edge("complete_structural_design", END)
        return builder.compile()

    def _attach_stage_handoff(
        self,
        update: Dict[str, Any],
        prior_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = {**prior_state, **update}
        artifacts: List[ArtifactRecord] = []
        if context.get("design_units") is not None:
            artifacts.append(ArtifactRecord(
                artifact_type="design_units",
                state_key="design_units",
                producer=self.name,
            ))
        if context.get("dimension_design_result") is not None:
            artifacts.append(ArtifactRecord(
                artifact_type="dimension_design_result",
                state_key="dimension_design_result",
                producer=self.name,
            ))
        if context.get("reinforcement_design_result") is not None:
            artifacts.append(ArtifactRecord(
                artifact_type="reinforcement_design_result",
                state_key="reinforcement_design_result",
                producer=self.name,
            ))
        handoff = build_handoff(
            StageName.STRUCTURAL_DESIGN.value,
            update,
            produced_artifacts=artifacts,
        )
        return {**update, "latest_handoff": handoff.to_dict()}

    def run(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)
        working["structural_design_graph_steps"] = []
        working["structural_design_graph_index"] = 0
        working["structural_design_graph_action"] = None
        working["structural_design_graph_finished"] = False

        result = self._build_structural_design_graph().invoke(working)
        internal_keys = {
            "structural_design_graph_steps",
            "structural_design_graph_index",
            "structural_design_graph_action",
            "structural_design_graph_finished",
        }
        returned = {
            key: value
            for key, value in result.items()
            if key not in internal_keys and (key not in state or state.get(key) != value)
        }
        returned["active_agent"] = self.name
        returned["error"] = result.get("error")
        if result.get("error"):
            returned["task_status"] = "failed"
            returned.setdefault("message", result.get("message") or "StructuralDesignAgent 执行失败。")
        return self._attach_stage_handoff(returned, working)

class ModelingCheckAgent(StageAgent):
    """建模验算与反馈决策智能体。

    采用 ReAct 子循环：LLM 只决定下一步 Action，确定性工具负责验算上下文整理、
    承载力验算与结构化反馈决策。
    """

    name = "ModelingCheckAgent"
    mode = "react_loop_with_feedback"
    prompt_id = "agents.modeling_check.v1"

    def _attach_stage_handoff(
        self,
        update: Dict[str, Any],
        prior_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = {**prior_state, **update}
        artifacts: List[ArtifactRecord] = []
        if context.get("check_result") is not None:
            artifacts.append(ArtifactRecord(
                artifact_type="check_result",
                state_key="check_result",
                path=context.get("capacity_check_summary_path"),
                producer=self.name,
            ))
        elif context.get("capacity_check_result") is not None:
            artifacts.append(ArtifactRecord(
                artifact_type="capacity_check_result",
                state_key="capacity_check_result",
                path=context.get("capacity_check_summary_path"),
                producer=self.name,
            ))

        if context.get("feedback_decision") is not None:
            artifacts.append(ArtifactRecord(
                artifact_type="feedback_decision",
                state_key="feedback_decision",
                path=context.get("feedback_decision_path"),
                producer=self.name,
            ))
        if context.get("revision_context") is not None:
            artifacts.append(ArtifactRecord(
                artifact_type="revision_context",
                state_key="revision_context",
                path=context.get("revision_context_path"),
                producer=self.name,
            ))

        task_status = str(update.get("task_status") or "")
        decision = context.get("feedback_decision") if isinstance(context.get("feedback_decision"), dict) else {}
        revision_request = None
        recommended_next_stage = None

        if task_status == "modeling_check_revision_required":
            target_step = str(decision.get("target_step") or "")
            required_artifact_by_step = {
                "dimension_design": "dimension_design_result",
                "reinforcement_design": "reinforcement_design_result",
            }
            required_artifact = required_artifact_by_step.get(target_step)
            revision_request = RevisionRequest(
                target_stage=StageName.STRUCTURAL_DESIGN.value,
                reason=str(
                    decision.get("control_reason")
                    or context.get("modeling_revision_instruction")
                    or update.get("message")
                    or "建模验算要求返回结构设计阶段修正。"
                ),
                required_artifacts=[required_artifact] if required_artifact else [],
            )
            recommended_next_stage = StageName.STRUCTURAL_DESIGN.value
        elif task_status == "modeling_check_passed":
            recommended_next_stage = StageName.FINAL_OUTPUT.value
        elif task_status in {"manual_review", "manual_review_required"}:
            recommended_next_stage = StageName.MANUAL_REVIEW.value

        handoff = build_handoff(
            StageName.MODELING_CHECK.value,
            context,
            produced_artifacts=artifacts,
            recommended_next_stage=recommended_next_stage,
            revision_request=revision_request,
        )
        return {**update, "latest_handoff": handoff.to_dict()}

    def _visible_state(self, state: Dict[str, Any], latest_observation: Dict[str, Any] | None = None) -> Dict[str, Any]:
        check_result = state.get("check_result") if isinstance(state.get("check_result"), dict) else {}
        overall = check_result.get("overall_check") if isinstance(check_result.get("overall_check"), dict) else {}
        return {
            "reinforcement_yaml_path_available": bool(state.get("reinforcement_yaml_path")),
            "opensees_force_json_path_available": bool(state.get("opensees_force_json_path")),
            "analysis_result_available": bool(state.get("analysis_result")),
            "reinforcement_design_result_available": bool(state.get("reinforcement_design_result")),
            "check_result_available": bool(check_result),
            "overall_check": overall,
            "overall_all_ok": overall.get("all_ok") if overall else None,
            "control_sections_count": len(check_result.get("control_sections") or {}) if check_result else 0,
            "feedback_decision_available": bool(state.get("feedback_decision")),
            "feedback_decision": state.get("feedback_decision") or {},
            "revision_context_available": bool(state.get("revision_context")),
            "check_iteration_index": int(state.get("check_iteration_index") or 0),
            "max_check_revision_rounds": int(state.get("max_check_revision_rounds") or 2),
            "latest_action": latest_observation.get("action") if latest_observation else None,
            "latest_observation_success": latest_observation.get("success") if latest_observation else None,
        }

    def _fallback_action(self, working: Dict[str, Any]) -> str:
        check_result = working.get("check_result") if isinstance(working.get("check_result"), dict) else {}
        overall = check_result.get("overall_check") if isinstance(check_result.get("overall_check"), dict) else {}
        if not check_result:
            return "run_capacity_check"
        if overall.get("all_ok") is True:
            return "finish_check"
        if not working.get("feedback_decision"):
            return "generate_revision_instruction"
        return "finish_check"

    def _execute_action(self, action: str, working: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        if action == "run_capacity_check":
            update = self._run_action(get_action_spec(action), working)
            check_result = update.get("check_result") if isinstance(update.get("check_result"), dict) else {}
            observation = {
                "action": action,
                "success": not bool(update.get("error")),
                "overall_check": check_result.get("overall_check") if check_result else {},
                "utilization_summary": check_result.get("utilization_summary") if check_result else {},
                "summary_path": update.get("capacity_check_summary_path"),
                "error": update.get("error"),
            }
            return update, observation, False

        if action == "generate_revision_instruction":
            update = self._run_action(get_action_spec("generate_modeling_feedback"), working)
            decision = update.get("feedback_decision") if isinstance(update.get("feedback_decision"), dict) else {}
            next_iteration = int(working.get("check_iteration_index") or 0)
            if decision.get("next_action") in ["revise_reinforcement", "revise_dimension"]:
                next_iteration += 1
                update["check_iteration_index"] = next_iteration
            observation = {
                "action": action,
                "success": not bool(update.get("error")),
                "feedback_decision": decision,
                "revision_context_available": bool(update.get("revision_context")),
                "error": update.get("error"),
            }
            return update, observation, False

        if action == "finish_check":
            decision = working.get("feedback_decision") if isinstance(working.get("feedback_decision"), dict) else {}
            check_result = working.get("check_result") if isinstance(working.get("check_result"), dict) else {}
            overall = check_result.get("overall_check") if isinstance(check_result.get("overall_check"), dict) else {}

            if overall.get("all_ok") is True and not decision:
                decision = {
                    "overall_status": "pass",
                    "next_action": "pass",
                    "target_agent": "END",
                    "target_step": None,
                    "control_reason": "承载力验算通过。",
                    "requires_rerun_check": False,
                }

            if decision.get("next_action") in ["revise_reinforcement", "revise_dimension"]:
                status = "modeling_check_revision_required"
                message = "建模验算未通过，已生成返回结构设计阶段的修正上下文。"
            else:
                status = "modeling_check_passed"
                message = "建模验算通过，流程结束。"

            update = {
                "feedback_decision": decision or working.get("feedback_decision"),
                "task_status": status,
                "message": message,
                "error": None,
            }
            observation = {"action": action, "success": True, "feedback_decision": update.get("feedback_decision")}
            return update, observation, True

        update = {"error": f"非法 Action: {action}"}
        observation = {"action": action, "success": False, "error": update["error"]}
        return update, observation, True

    def _prepare_modeling_check_step(self, state: AgentState) -> Dict[str, Any]:
        step_index = int(state.get("modeling_check_react_step") or 0) + 1
        max_react_steps = int(state.get("max_modeling_check_steps") or state.get("max_react_steps") or 10)
        update: Dict[str, Any] = {
            "modeling_check_react_step": step_index,
            "modeling_check_action": None,
            "modeling_check_action_obj": {},
            "modeling_check_finished": False,
            "active_agent": self.name,
        }

        if step_index > max_react_steps:
            return {
                **update,
                "modeling_check_finished": True,
                "task_status": "manual_review_required",
                "message": "达到最大建模验算 ReAct 步数，建议转人工复核。",
                "error": None,
            }

        check_result = state.get("check_result") if isinstance(state.get("check_result"), dict) else {}
        overall = check_result.get("overall_check") if isinstance(check_result.get("overall_check"), dict) else {}
        if overall.get("all_ok") is True:
            # 承载力通过但墩柱长细比复核未决（超出表5.3.1适用上限）时，
            # 不直接 finish：交给人工复核关口决定接受风险或返回修订。
            try:
                from .joint_reinforcement import aggregate_axial_check_results  # noqa: PLC0415

                axial = aggregate_axial_check_results(state.get("reinforcement_design_result") or {})
            except Exception:
                axial = {}
            if axial.get("has_slenderness_review"):
                return {
                    **update,
                    "modeling_check_finished": True,
                    "task_status": "manual_review_required",
                    "message": (
                        "承载力验算通过，但墩柱长细比超出表5.3.1适用范围"
                        f"（任务={axial.get('slenderness_review_task_ids')}），"
                        "需人工复核决定接受风险或返回修订。"
                    ),
                    "error": None,
                }
            return {
                **update,
                "modeling_check_action": "finish_check",
                "modeling_check_action_obj": {
                    "thought": "承载力验算已通过，结束建模验算。",
                    "action": "finish_check",
                    "action_input": {},
                },
            }

        decision = state.get("feedback_decision") if isinstance(state.get("feedback_decision"), dict) else {}
        if decision.get("next_action") in ["revise_reinforcement", "revise_dimension", "pass"]:
            return {
                **update,
                "modeling_check_action": "finish_check",
                "modeling_check_action_obj": {
                    "thought": "已有验算反馈决策，结束本阶段并交给顶层调度。",
                    "action": "finish_check",
                    "action_input": {},
                },
            }

        if int(state.get("check_iteration_index") or 0) > int(state.get("max_check_revision_rounds") or 2):
            return {
                **update,
                "modeling_check_finished": True,
                "task_status": "manual_review_required",
                "message": "建模验算反馈修正达到轮次上限，建议转人工复核。",
                "error": None,
            }

        return update

    @staticmethod
    def _route_after_modeling_prepare(state: AgentState) -> str:
        if state.get("modeling_check_finished"):
            return "done"
        if state.get("modeling_check_action"):
            return "execute"
        return "decide"

    def _decide_modeling_action(self, state: AgentState, llm: Any, skill_text: str) -> Dict[str, Any]:
        visible_state = self._visible_state(
            dict(state),
            state.get("modeling_check_latest_observation") or {},
        )
        try:
            response = llm.invoke([
                ("system", skill_text),
                ("user", json.dumps(json_safe(visible_state), ensure_ascii=False, indent=2)),
            ])
            action_obj = extract_json_object(response.content)
            action = action_obj.get("action")
        except Exception as e:
            logger.warning("ModelingCheckAgent ReAct 输出解析失败，启用规则兜底：%s", e)
            action = self._fallback_action(dict(state))
            action_obj = {"thought": "LLM 输出解析失败，使用规则兜底。", "action": action, "action_input": {}}

        return {
            "modeling_check_action": str(action),
            "modeling_check_action_obj": action_obj,
            "active_agent": self.name,
        }

    def _execute_modeling_graph_action(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)
        action = str(working.get("modeling_check_action") or "")
        action_obj = working.get("modeling_check_action_obj") if isinstance(working.get("modeling_check_action_obj"), dict) else {}
        step_index = int(working.get("modeling_check_react_step") or 0)

        self._log_action_start(working, action)
        update, observation, finished = self._execute_action(action, working)
        working.update(update)
        self._log_action_end(working, action, update)

        run_logger = AgentLogger.from_state(working)
        run_logger.log_thought(
            agent=self.name,
            thought=action_obj.get("thought", ""),
            action=action,
            observation=observation,
        )
        run_logger.attach(working)

        history_entry = {
            "react_step": step_index,
            "thought": action_obj.get("thought", ""),
            "action": action,
            "observation": observation,
        }

        graph_update: Dict[str, Any] = {
            **update,
            "active_agent": self.name,
            "modeling_check_latest_observation": observation,
            "modeling_check_finished": bool(finished),
            "modeling_check_history": [history_entry],
            "agent_log": working.get("agent_log"),
            "agent_events": working.get("agent_events") or [],
        }

        if update.get("error"):
            graph_update.update({
                "modeling_check_finished": True,
                "task_status": "failed",
                "message": f"{self.name} 执行 {action} 失败。",
            })

        return graph_update

    @staticmethod
    def _route_after_modeling_execute(state: AgentState) -> str:
        if state.get("error") or state.get("modeling_check_finished"):
            return "done"
        return "continue"

    def _build_modeling_check_graph(self, llm: Any, skill_text: str):
        builder = StateGraph(AgentState)
        builder.add_node("prepare_modeling_check_step", self._prepare_modeling_check_step)
        builder.add_node(
            "decide_modeling_action",
            lambda state: self._decide_modeling_action(state, llm, skill_text),
        )
        builder.add_node("execute_modeling_action", self._execute_modeling_graph_action)

        builder.set_entry_point("prepare_modeling_check_step")
        builder.add_conditional_edges(
            "prepare_modeling_check_step",
            self._route_after_modeling_prepare,
            {
                "decide": "decide_modeling_action",
                "execute": "execute_modeling_action",
                "done": END,
            },
        )
        builder.add_edge("decide_modeling_action", "execute_modeling_action")
        builder.add_conditional_edges(
            "execute_modeling_action",
            self._route_after_modeling_execute,
            {
                "continue": "prepare_modeling_check_step",
                "done": END,
            },
        )
        return builder.compile()

    def run(self, state: AgentState) -> Dict[str, Any]:
        working: Dict[str, Any] = dict(state)
        working.setdefault("check_iteration_index", 0)
        working.setdefault("max_check_revision_rounds", 2)

        # 每次因结构修正后重新进入建模验算，清理上一轮旧结果，强制重建输入并复验。
        if working.get("revision_context") and (working.get("check_result") or working.get("feedback_decision")):
            working.pop("feedback_decision", None)
            working.pop("check_result", None)
            working.pop("capacity_check_result", None)

        working["modeling_check_react_step"] = 0
        working["modeling_check_latest_observation"] = {}
        working["modeling_check_action"] = None
        working["modeling_check_action_obj"] = {}
        working["modeling_check_finished"] = False

        skill_text = self._activate_skill(working)
        llm = get_controller_llm(working.get("config_path") or "config/settings.yaml")
        max_react_steps = int(working.get("max_modeling_check_steps") or working.get("max_react_steps") or 10)
        result = self._build_modeling_check_graph(llm, skill_text).invoke(
            working,
            {"recursion_limit": max_react_steps * 3 + 8},
        )

        internal_keys = {
            "modeling_check_react_step",
            "modeling_check_latest_observation",
            "modeling_check_action",
            "modeling_check_action_obj",
            "modeling_check_finished",
        }
        returned = {
            k: v
            for k, v in result.items()
            if k not in internal_keys and (k not in state or state.get(k) != v)
        }
        returned["active_agent"] = self.name
        returned["error"] = result.get("error")
        if result.get("error"):
            returned["task_status"] = "failed"
            returned.setdefault("message", result.get("message") or "ModelingCheckAgent 执行失败。")
        return self._attach_stage_handoff(returned, working)


AGENT_REGISTRY: Dict[str, StageAgent] = {
    "InitialDesignAgent": InitialDesignAgent(),
    "LayoutRevisionAgent": LayoutRevisionAgent(),
    "StructuralDesignAgent": StructuralDesignAgent(),
    "ModelingCheckAgent": ModelingCheckAgent(),
}
