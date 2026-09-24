from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .state import AgentState
from .tool_actions import (
    build_revision_prompt_action,
    compute_pier_groups_action,
    dimension_design_action,
    drawing_crop_and_mask_action,
    extract_design_units_action,
    generate_layout_design_action,
    generate_modeling_feedback_action,
    generate_reinforcement_drawings_action,
    generate_revised_layout_action,
    generate_revision_instruction_action,
    load_data_action,
    obstacle_semantic_extractor_action,
    reinforcement_design_action,
    run_capacity_check_action,
    run_collision_detection_action,
    select_samples_action,
)

logger = logging.getLogger(__name__)

ActionCallable = Callable[[AgentState], Dict[str, Any]]


@dataclass(frozen=True)
class ActionSpec:
    name: str
    func: ActionCallable
    expected_key: Optional[str] = None
    required_keys: Tuple[str, ...] = ()
    required_any: Tuple[Tuple[str, ...], ...] = ()
    produced_keys: Tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class ActionResult:
    action: str
    ok: bool
    update: Dict[str, Any]
    error: Optional[str] = None
    expected_key: Optional[str] = None
    produced_keys: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_update(cls, spec: ActionSpec, update: Mapping[str, Any]) -> "ActionResult":
        error = update.get("error")
        produced = tuple(k for k, v in update.items() if k != "error" and v not in [None, "", [], {}])
        return cls(
            action=spec.name,
            ok=not bool(error),
            update=dict(update),
            error=str(error) if error else None,
            expected_key=spec.expected_key,
            produced_keys=produced,
        )


def _missing_required_inputs(spec: ActionSpec, state: Mapping[str, Any]) -> Tuple[str, ...]:
    missing = [key for key in spec.required_keys if state.get(key) in [None, "", [], {}]]
    for group in spec.required_any:
        if not any(state.get(key) not in [None, "", [], {}] for key in group):
            missing.append(" or ".join(group))
    return tuple(missing)


def run_action(
    spec: ActionSpec,
    state: AgentState,
    *,
    validate_inputs: bool = True,
) -> ActionResult:
    """Run one tool action behind a stable ActionSpec/ActionResult boundary."""
    if validate_inputs:
        missing = _missing_required_inputs(spec, state)
        if missing:
            update = {"error": f"Action {spec.name} missing required state: {', '.join(missing)}"}
            return ActionResult.from_update(spec, update)

    try:
        update = spec.func(state)
    except Exception as exc:  # pragma: no cover - action functions normally catch their own errors
        logger.error("Action %s failed unexpectedly: %s", spec.name, exc, exc_info=True)
        update = {"error": str(exc)}

    if not isinstance(update, dict):
        update = {"error": f"Action {spec.name} returned non-dict update: {type(update).__name__}"}
    return ActionResult.from_update(spec, update)


ACTION_SPECS: Dict[str, ActionSpec] = {
    "load_data": ActionSpec(
        name="load_data",
        func=load_data_action,
        expected_key="cropped_data",
        required_keys=("data_path", "file_prefix", "start_station", "end_station"),
        produced_keys=("data_loader_result", "cropped_data", "plane_json_path"),
        description="Load and crop route source data.",
    ),
    "drawing_crop_and_mask": ActionSpec(
        name="drawing_crop_and_mask",
        func=drawing_crop_and_mask_action,
        expected_key="mask_path",
        required_keys=("plane_json_path",),
        required_any=(("input_drawing_path", "drawing_path"),),
        produced_keys=("drawing_mask_result", "png_path", "jpg_path", "pgw_path", "mask_path"),
        description="Crop drawing and build obstacle mask.",
    ),
    "obstacle_semantic_extractor": ActionSpec(
        name="obstacle_semantic_extractor",
        func=obstacle_semantic_extractor_action,
        expected_key="obstacle_extractor_result",
        required_keys=("mask_path", "pgw_path"),
        produced_keys=("obstacle_extractor_result", "obstacle_json_path", "merged_visualization_path"),
        description="Extract semantic obstacle records from the mask.",
    ),
    "select_samples": ActionSpec(
        name="select_samples",
        func=select_samples_action,
        expected_key="few_shots",
        required_keys=("few_shots_dir",),
        produced_keys=("few_shots", "design_input"),
        description="Select few-shot examples for layout generation.",
    ),
    "generate_layout_design": ActionSpec(
        name="generate_layout_design",
        func=generate_layout_design_action,
        expected_key="layout_result",
        required_any=(("design_input", "cropped_data", "data_loader_result"),),
        produced_keys=("design_input", "design_result", "layout_result"),
        description="Generate the initial bridge layout.",
    ),
    "run_collision_detection": ActionSpec(
        name="run_collision_detection",
        func=run_collision_detection_action,
        expected_key="collision_metrics",
        required_keys=("plane_json_path", "mask_path", "pgw_path"),
        required_any=(("layout_result", "design_result", "existing_layout_result"),),
        produced_keys=("collision_result", "collision_metrics", "collision_items"),
        description="Check layout conflicts with obstacles.",
    ),
    "generate_revision_instruction": ActionSpec(
        name="generate_revision_instruction",
        func=generate_revision_instruction_action,
        expected_key="revision_instruction",
        required_keys=("collision_metrics",),
        produced_keys=("revision_instruction", "revision_instruction_path"),
        description="Generate engineering guidance for layout revision.",
    ),
    "build_revision_prompt": ActionSpec(
        name="build_revision_prompt",
        func=build_revision_prompt_action,
        expected_key="revision_prompt",
        required_keys=("revision_instruction",),
        required_any=(("layout_result", "existing_layout_result"),),
        produced_keys=("revision_prompt", "revision_prompt_path"),
        description="Build the prompt used for revised layout generation.",
    ),
    "generate_revised_layout": ActionSpec(
        name="generate_revised_layout",
        func=generate_revised_layout_action,
        expected_key="layout_result",
        required_keys=("revision_prompt",),
        produced_keys=("revision_result_raw", "revision_result_path", "layout_result"),
        description="Generate a revised layout from the revision prompt.",
    ),
    "extract_design_units": ActionSpec(
        name="extract_design_units",
        func=extract_design_units_action,
        expected_key="design_units",
        required_any=(("layout_result", "existing_layout_result", "final_layout_result"),),
        produced_keys=("layout_result", "design_units"),
        description="Extract bridge design units for structural design.",
    ),
    "dimension_design": ActionSpec(
        name="dimension_design",
        func=dimension_design_action,
        expected_key="dimension_design_result",
        required_keys=("design_units",),
        produced_keys=("dimension_design_result", "evidence_bundles", "code_trace"),
        description="Design substructure dimensions.",
    ),
    "compute_pier_groups": ActionSpec(
        name="compute_pier_groups",
        func=compute_pier_groups_action,
        expected_key="pier_group_result",
        required_any=(
            ("layout_result", "existing_layout_result", "final_layout_result"),
        ),
        required_keys=("design_units", "dimension_design_result"),
        produced_keys=("pier_group_result",),
        description="Deterministically merge pier design groups and audit pier heights.",
    ),
    "reinforcement_design": ActionSpec(
        name="reinforcement_design",
        func=reinforcement_design_action,
        expected_key="reinforcement_design_result",
        required_keys=("dimension_design_result",),
        produced_keys=(
            "reinforcement_design_result",
            "reinforcement_yaml_path",
            "opensees_force_json_path",
            "evidence_bundles",
            "code_trace",
        ),
        description="Design reinforcement and produce analysis handoff files.",
    ),
    "run_capacity_check": ActionSpec(
        name="run_capacity_check",
        func=run_capacity_check_action,
        expected_key="check_result",
        produced_keys=("capacity_check_result", "check_result", "capacity_check_summary_path"),
        description="Run structural capacity checks.",
    ),
    "generate_modeling_feedback": ActionSpec(
        name="generate_modeling_feedback",
        func=generate_modeling_feedback_action,
        expected_key="feedback_decision",
        required_any=(("check_result", "capacity_check_result"),),
        produced_keys=("feedback_decision", "revision_context", "modeling_revision_instruction"),
        description="Generate modeling feedback and revision decision.",
    ),
    "generate_reinforcement_drawings": ActionSpec(
        name="generate_reinforcement_drawings",
        func=generate_reinforcement_drawings_action,
        expected_key="drawing_package_result",
        required_keys=(
            "pier_group_result",
            "dimension_design_result",
            "reinforcement_design_result",
            "capacity_check_result",
        ),
        produced_keys=(
            "drawing_package_result",
            "drawing_index_path",
            "design_manifest_path",
            "cad_script_paths",
            "drawing_preview_paths",
        ),
        description="Generate deterministic grouped reinforcement drawing deliverables.",
    ),
}


AGENT_ACTIONS: Dict[str, Tuple[str, ...]] = {
    "InitialDesignAgent": (
        "load_data",
        "drawing_crop_and_mask",
        "obstacle_semantic_extractor",
        "select_samples",
        "generate_layout_design",
    ),
    "LayoutRevisionAgent": (
        "run_collision_detection",
        "generate_revision_instruction",
        "build_revision_prompt",
        "generate_revised_layout",
    ),
    "StructuralDesignAgent": (
        "extract_design_units",
        "dimension_design",
        "compute_pier_groups",
        "reinforcement_design",
    ),
    "ModelingCheckAgent": (
        "run_capacity_check",
        "generate_modeling_feedback",
    ),
}


def get_action_spec(name: str) -> ActionSpec:
    try:
        return ACTION_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Action is not registered: {name}") from exc


def get_agent_action_specs(agent_name: str) -> Dict[str, ActionSpec]:
    return {name: get_action_spec(name) for name in AGENT_ACTIONS.get(agent_name, ())}
