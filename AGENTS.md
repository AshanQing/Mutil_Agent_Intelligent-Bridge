# AGENTS.md

## Collaboration Rules

- 用中文与用户沟通。
- 不要迎合用户，始终保持客观回答；对架构优劣、风险和取舍要明确说明。
- 修改代码前先阅读现有实现，优先沿用项目已有结构和命名。
- 不要回滚用户或其他工具造成的改动，除非用户明确要求。
- 手工编辑文件使用 `apply_patch`。
- 当前项目路径为 `<repo-root>`。

## Current Project Context

本项目是高速公路桥梁设计多智能体系统。当前保留两条顶层运行链路：

```text
兼容图（默认）
run_multi_agent.py
  -> bridge_agents.run_agent()
  -> TaskAllocationAgent
  -> agent_executor
  -> InitialDesignAgent / LayoutRevisionAgent / StructuralDesignAgent / ModelingCheckAgent
  -> final_output

显式协调图（--graph-v2）
run_multi_agent.py
  -> bridge_agents.graph_v2.build_graph_v2()
  -> DesignCoordinatorAgent
  -> 确定性调度校验
  -> 专业 Agent
  -> StageHandoff
  -> DesignCoordinatorAgent 重新决策
```

显式协调图已经可运行，但仍通过命令行开关启用。InitialDesignAgent 和 StructuralDesignAgent 尚未子图化，专业 Agent 目前主要由顶层包装器转换为 StageHandoff，RAG 尚未接入专业 Agent。

## Important Completed Refactors

- Prompt 已统一由 `prompts/manifest.yaml` 注册，并通过 `bridge_agents/prompt_registry.py` 渲染。
- 当前有效 Prompt 位于 `prompts/agents/`、`prompts/tasks/`、`prompts/partials/`。
- 旧 Prompt、旧工具和历史资料已经移入 `misc/`：
  - `misc/archive/legacy_prompts/`
  - `misc/archive/legacy_tools/`
  - `misc/docs/`
- 普通用户配置已简化为原始输入路径和主输出目录。
- 中间成果不应从配置文件手工注入，统一从 `paths.output_dir` 自动发现。
- `README.md` 已重写，说明方法、LangGraph 结构、脚本、输入输出和技术栈。
- 规范四库 RAG 已完成证据文档、FTS5、BGE 向量、RRF 混合检索和 30 条固定评测；
- SkillRegistry、五个标准 SKILL.md、AgentContract、StageHandoff 和 ArtifactRecord 已建立；
- DesignCoordinatorAgent、确定性调度校验、成果依赖图和显式顶层图已建立。

## Runtime Input Policy

普通用户主要配置：

```yaml
task:
  user_request: ...

paths:
  data_path: ...
  input_drawing_path: ...
  output_dir: ...

route_data:
  file_prefix: ...
```

不要重新把这些中间成果路径暴露为普通配置入口：

```text
plane_json_path
mask_path
pgw_path
obstacle_json_path
existing_layout_result_path
layout_revision_result_path
existing_design_units_path
existing_dimension_design_result_path
existing_reinforcement_design_result_path
opensees_force_json_path
collision_output_dir
capacity_check_output_dir
```

这些内容应由 `bridge_agents.agent._discover_existing_outputs(output_dir)` 从输出目录扫描恢复。

## Main Architecture Improvement Direction

后续优化应从“断网后提出的运行输入逻辑问题”之后继续，重点是让系统更 LangGraph-native，同时保持渐进、可运行。

最新实施顺序已经固化到以下文档。新任务开始时必须先阅读：

```text
misc/docs/plans/CURRENT_HANDOFF.md
misc/docs/plans/规范四库RAG知识服务实施计划.md
misc/docs/plans/层级工程设计多智能体框架优化计划.md
```

独立规范四库 RAG、SkillRegistry、统一契约、DesignCoordinatorAgent 和显式顶层图已经完成首轮实现。当前优先级依次为：InitialDesignAgent 子图化、StructuralDesignAgent 子图化、专业 Agent 原生 Handoff、成果依赖接入真实最小返工、RAG 逐阶段接入和证据门控。

`misc/docs/plans/CURRENT_HANDOFF.md` 是当前进度真源，实施前必须以该文件为准。以下历史推荐顺序用于理解已完成改造和后续 checkpoint 方向，不覆盖最新交接计划。

推荐顺序：

1. State reducer 化
   - 优先处理 `prompt_trace`、`revision_history`、`modeling_check_history`、`agent_events` 等追加型字段。
   - 减少手工 `append` 和 `working.update()` 带来的覆盖风险。

2. 工具 Action 层统一
   - 为工具调用建立 `ActionSpec` / `ActionResult` / action registry / action runner。
   - 保留现有工具函数名，避免一次性破坏阶段 Agent。
   - 统一处理输入校验、异常包装、输出文件抽取和 observation 构造。

3. `LayoutRevisionAgent` 子图化
   - 将现有 ReAct `for` 循环拆为 `LayoutRevisionGraph`。
   - 节点包括检测、决策、修正指令、Prompt 构造、生成修正布跨、完成、人工复核。
   - 使用 LangGraph 条件边表达循环和退出条件。

4. `ModelingCheckAgent` 子图化
   - 将承载力验算反馈循环拆为 `ModelingCheckGraph`。
   - 明确表达验算通过、生成反馈、返回结构设计阶段修正、复验等路径。

5. Checkpoint 接入
   - 短期不要替代 `output_dir` 工程成果恢复。
   - `output_dir` 保存工程产物；checkpoint 保存图运行状态、当前节点和上下文。
   - 先在子图和 state 清晰后用 `MemorySaver` 做开发验证，再考虑 SQLite/Postgres 持久化。

6. 结构设计并行化
   - 在设计单元数据结构稳定后，再考虑用 LangGraph `Send` 并行处理多个设计单元。

## Validation Commands

常用验证命令：

```powershell
.\.venv\python.exe -m pytest tests/test_prompt_registry.py tests/test_initial_state_loading.py
.\.venv\python.exe -m compileall bridge_agents tools tests run_multi_agent.py
.\.venv\python.exe run_multi_agent.py --help
```

当前推荐解释器：

```powershell
.\.venv\python.exe
```

## Notes

- `.vscode/settings.json` 可能包含用户本地解释器设置，除非任务明确要求，否则不要主动修改。
- `scripts/` 不是杂项目录，当前工具仍依赖 `scripts.llm_client` 和 `scripts.selector`。
- `misc/` 中内容只用于历史追溯，不应被当前主体运行代码引用。
