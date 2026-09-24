from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from .state_reducers import append_unique


class AgentState(TypedDict, total=False):
    """桥梁多智能体系统共享状态。

    说明：
    - 顶层 LangGraph 只在这个 State 上做更新；
    - 各阶段 Agent 只读取自己需要的字段，并写回阶段成果；
    - 字段保持宽松，便于你逐步接入现有工具脚本。
    """

    # ---------- 用户输入与配置 ----------
    user_request: str
    config_path: Optional[str]
    user_parameters: Optional[Dict[str, Any]]
    user_artifacts: Optional[List[Dict[str, Any]]]

    # ---------- 任务分配 ----------
    task_category: Optional[str]
    user_intent: Optional[str]
    extracted_parameters: Optional[Dict[str, Any]]
    agent_sequence: Optional[List[Dict[str, Any]]]
    current_agent_index: Optional[int]
    completed_agents: Optional[List[str]]
    active_agent: Optional[str]

    # ---------- 基本任务参数 ----------
    start_station: Optional[str]
    end_station: Optional[str]

    # ---------- 路径与工具配置 ----------
    data_path: Optional[str]
    file_prefix: Optional[str]
    plane_paths: Optional[Dict[str, str]]  # multi-route: {K: plane.json, Z: plane.json}
    structures_filename: Optional[str]
    input_drawing_path: Optional[str]
    drawing_path: Optional[str]
    output_dir: Optional[str]

    config_file: Optional[str]
    checkpoint_file: Optional[str]
    device: Optional[str]
    converter_path: Optional[str]
    dxf_output_dir: Optional[str]
    layer_config_json: Optional[str]
    swap_xy: Optional[bool]
    buffer: Optional[float]
    resolution: Optional[float]
    max_image_pixels: Optional[int]
    expand_inserts: Optional[bool]
    patch_size: Optional[int]
    stride: Optional[int]
    background_id: Optional[int]
    convert_png_to_jpg: Optional[bool]
    jpg_quality: Optional[int]

    roadbed_type: Optional[str]
    section_name: Optional[str]
    route_scan_config_json: Optional[str]
    scan_width_integrated: Optional[float]
    scan_width_separated: Optional[float]
    sample_step: Optional[float]
    step_k: Optional[float]
    clip_to_route_range: Optional[bool]
    save_visualization: Optional[bool]
    show_visualization: Optional[bool]
    preview_wait_ms: Optional[int]

    prompt_manifest_path: Optional[str]
    prompt_trace: Annotated[List[Dict[str, Any]], append_unique]
    standards_path: Optional[str]
    few_shots_dir: Optional[str]
    few_shots_k: Optional[int]
    dimension_few_shots_dir: Optional[str]
    reinforcement_few_shots_dir: Optional[str]

    # ---------- 初步设桥布跨阶段成果 ----------
    data_loader_result: Optional[Dict[str, Any]]
    drawing_mask_result: Optional[Dict[str, Any]]
    obstacle_extractor_result: Optional[Dict[str, Any]]
    cropped_data: Optional[Dict[str, Any]]
    plane_json_path: Optional[str]
    png_path: Optional[str]
    jpg_path: Optional[str]
    pgw_path: Optional[str]
    mask_path: Optional[str]
    obstacle_json_path: Optional[str]
    merged_visualization_path: Optional[str]
    few_shots: Optional[List[Dict[str, Any]]]
    design_input: Optional[Dict[str, Any]]
    design_result: Optional[Dict[str, Any]]
    layout_result: Optional[Dict[str, Any]]
    existing_layout_result: Optional[Dict[str, Any]]
    initial_design_stage_plan: Optional[Dict[str, Any]]
    initial_design_history: Annotated[List[Dict[str, Any]], append_unique]
    initial_design_graph_steps: Optional[List[str]]
    initial_design_graph_index: Optional[int]
    initial_design_graph_action: Optional[str]
    initial_design_graph_finished: Optional[bool]

    # ---------- 布跨修正阶段成果 ----------
    collision_output_dir: Optional[str]
    collision_offset_left_m: Optional[float]
    collision_offset_right_m: Optional[float]
    collision_column_half_spacing_m: Optional[float]
    collision_check_radius_m: Optional[float]
    collision_mask_threshold: Optional[int]
    collision_save_visualization: Optional[bool]
    collision_result: Optional[Dict[str, Any]]
    collision_metrics: Optional[Dict[str, Any]]
    collision_items: Optional[List[Dict[str, Any]]]
    collision_report_json_path: Optional[str]
    collision_report_txt_path: Optional[str]
    collision_metrics_json_path: Optional[str]
    collision_visualization_path: Optional[str]

    max_revision_rounds: Optional[int]
    max_react_steps: Optional[int]
    # 顶层图单次 invoke 的超级步上限（0/None 表示用默认值），防止调度死循环无限空转。
    max_graph_steps: Optional[int]
    llm_max_format_repairs: Optional[int]
    llm_max_regenerations: Optional[int]
    iteration_index: Optional[int]
    layout_revision_react_step: Optional[int]
    layout_revision_latest_observation: Optional[Dict[str, Any]]
    layout_revision_action: Optional[str]
    layout_revision_action_obj: Optional[Dict[str, Any]]
    layout_revision_finished: Optional[bool]
    revision_instruction: Optional[str]
    revision_instruction_path: Optional[str]
    revision_prompt: Optional[str]
    revision_prompt_path: Optional[str]
    revision_result_raw: Optional[str]
    revision_result_path: Optional[str]
    revision_history: Annotated[List[Dict[str, Any]], append_unique]
    verification_result: Optional[Dict[str, Any]]

    layout_revision_completed: Optional[bool]
    layout_revision_result: Optional[Dict[str, Any]]
    layout_revision_result_path: Optional[str]
    final_layout_result: Optional[Dict[str, Any]]
    final_layout_result_path: Optional[str]
    latest_revision_result: Optional[Dict[str, Any]]
    latest_revision_result_path: Optional[str]

    # ---------- 人工复核与持久化恢复 ----------
    thread_id: Optional[str]
    checkpoint_path: Optional[str]
    human_review_request: Optional[Dict[str, Any]]
    human_review_decision: Optional[Dict[str, Any]]
    human_review_feedback: Optional[str]
    human_review_route: Optional[str]
    human_review_history: Annotated[List[Dict[str, Any]], append_unique]
    human_override: Optional[bool]
    unresolved_manual_review: Optional[bool]
    human_review_scope: Optional[str]
    human_review_decision_path: Optional[str]
    accepted_risks: Optional[List[Dict[str, Any]]]

    # ---------- 结构设计阶段成果 ----------
    design_units: Optional[Dict[str, Any]]
    structural_stage_plan: Optional[Dict[str, Any]]
    structural_stage_summary: Optional[Dict[str, Any]]
    structural_design_result_path: Optional[str]
    structural_design_history: Annotated[List[Dict[str, Any]], append_unique]
    dimension_design_result: Optional[Dict[str, Any]]
    dimension_batch_status: Optional[Dict[str, Any]]
    pier_group_result: Optional[Dict[str, Any]]
    superstructure_height_m: Optional[float]
    foundation_top_offset_m: Optional[float]
    reinforcement_design_result: Optional[Dict[str, Any]]
    reinforcement_batch_status: Optional[Dict[str, Any]]
    failed_task_ids: Optional[List[str]]
    structural_design_result: Optional[Dict[str, Any]]
    structural_design_graph_steps: Optional[List[str]]
    structural_design_graph_index: Optional[int]
    structural_design_graph_action: Optional[str]
    structural_design_graph_finished: Optional[bool]

    # ---------- 建模验算阶段成果 ----------
    modeling_input: Optional[Dict[str, Any]]
    model_script_path: Optional[str]
    analysis_result: Optional[Dict[str, Any]]
    check_result: Optional[Dict[str, Any]]
    feedback_decision: Optional[Dict[str, Any]]

    # OpenSees 与承载力验算路径
    reinforcement_yaml_path: Optional[str]
    reinforcement_yaml_paths: Optional[List[str]]
    opensees_force_json_path: Optional[str]
    internal_force_output_paths: Optional[List[str]]
    opensees_script_path: Optional[str]
    capacity_check_script_path: Optional[str]
    capacity_check_output_dir: Optional[str]
    capacity_check_combination_name: Optional[str]
    capacity_check_gamma_0: Optional[float]
    capacity_check_apply_gamma0: Optional[bool]
    auto_run_opensees_if_missing: Optional[bool]

    # 承载力验算结果与反馈循环
    capacity_check_result: Optional[Dict[str, Any]]
    capacity_check_results: Optional[List[Dict[str, Any]]]
    capacity_batch_status: Optional[Dict[str, Any]]
    capacity_check_summary_path: Optional[str]
    capacity_check_summary_paths: Optional[List[str]]
    capacity_envelope_sections_path: Optional[str]
    capacity_utilization_plot_path: Optional[str]
    feedback_decision_path: Optional[str]
    revision_context: Optional[Dict[str, Any]]
    revision_context_path: Optional[str]
    modeling_revision_instruction: Optional[str]
    modeling_check_history: Annotated[List[Dict[str, Any]], append_unique]
    modeling_check_react_step: Optional[int]
    modeling_check_latest_observation: Optional[Dict[str, Any]]
    modeling_check_action: Optional[str]
    modeling_check_action_obj: Optional[Dict[str, Any]]
    modeling_check_finished: Optional[bool]
    check_iteration_index: Optional[int]
    max_modeling_check_steps: Optional[int]
    max_check_revision_rounds: Optional[int]

    # ---------- 通用状态 ----------
    task_status: Optional[str]
    final_summary: Optional[Dict[str, Any]]
    message: Optional[str]
    error: Optional[str]
    logger: Optional[Any]   # 存放 AgentLogger 实例

    # ---------- 结构化运行日志 ----------
    agent_log: Optional[Dict[str, Any]]
    agent_log_path: Optional[str]
    agent_events: Annotated[List[Dict[str, Any]], append_unique]

    active_agent_skill_name: Optional[str]
    active_agent_skill_path: Optional[str]
    active_agent_skill: Optional[str]
    agent_skills: Optional[Dict[str, str]]

    # ---------- 层级框架控制层字段 ----------
    coordinator_decision: Optional[Dict[str, Any]]
    latest_handoff: Optional[Dict[str, Any]]
    artifact_registry: Optional[Dict[str, Any]]
    invalidated_artifacts: Optional[List[str]]
    revision_target: Optional[str]
    rework_required_artifacts: Optional[List[str]]
    minimal_rework_path: Optional[List[str]]
    # 各专业阶段的返修重入次数（按阶段名计数），用于轮次上限校验。
    stage_revision_rounds: Optional[Dict[str, int]]
    # 返修阶段本轮执行失败时为 True，供顶层图直接路由到人工复核而不回协调器。
    rework_failed: Optional[bool]
    # 各专业阶段连续执行失败次数（按阶段名计数）。非返修的纯失败不会进入
    # stage_revision_rounds，没有该计数时协调器会一直把阶段当成"未完成"反复调度。
    stage_failure_counts: Optional[Dict[str, int]]
    # 某阶段连续失败达到 max_stage_failures 时为 True，供顶层图路由到人工复核。
    stage_failure_limit_reached: Optional[bool]
    max_stage_failures: Optional[int]
    manual_review_reason: Optional[str]
    unresolved_code_items: Optional[List[Dict[str, Any]]]
    compliance_matrix: Optional[Dict[str, Any]]
    design_code_context: Optional[Dict[str, Any]]
    evidence_bundles: Optional[Dict[str, Any]]
    code_trace: Annotated[List[Dict[str, Any]], append_unique]

    # ---------- 设计完成后的只读评估与专业问答 ----------
    design_assessment: Optional[Dict[str, Any]]
    design_review_answer: Optional[Dict[str, Any]]
    design_review_audit_path: Optional[str]

    # ---------- 配筋绘图与最终交付 ----------
    drawing_package_result: Optional[Dict[str, Any]]
    drawing_index_path: Optional[str]
    design_manifest_path: Optional[str]
    cad_script_paths: Optional[List[str]]
    drawing_preview_paths: Optional[List[str]]

    # ---------- 设计完成后的只读评估与专业问答 ----------
    design_assessment: Optional[Dict[str, Any]]
    design_review_answer: Optional[Dict[str, Any]]
    design_review_audit_path: Optional[str]
    
