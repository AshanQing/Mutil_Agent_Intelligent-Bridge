# 高速公路桥梁智能设计多智能体系统

本项目面向高速公路桥梁初步布跨、布跨修正、下部结构尺寸与配筋设计、OpenSees 内力分析和承载力验算反馈，构建了一个基于 LangGraph 的层级多智能体原型系统。

系统采用“LLM 规划与候选生成 + 确定性工程计算与校验 + 状态治理 + 人工复核 + 设计后规范问答”的混合方法。大语言模型承担任务理解、阶段规划和设计候选生成；路线解析、图纸处理、碰撞检测、结构分析、承载力计算、返修方向、批次完整性检查与调度合法性校验由确定性代码执行。规范 RAG 服务于设计完成后的评估和用户专业问答，不参与尺寸、配筋、荷载、分析或验算运行链。LLM 的调度结果通过工程规则校验后才能进入下一阶段。

当前推荐运行目标架构：

```powershell
$PYTHON = "python"
& $PYTHON run_multi_agent.py --config config/settings.yaml --graph-v2 --interactive
```

> `--graph-v2` 对应本文描述的目标方法架构。未传该参数时仍运行旧兼容图，供历史流程迁移和回归测试使用。

系统从任务启动到总体协调、四个专业子图、Action 执行、成果返修、人工复核和断点恢复的逐环节说明，见 [系统运行机制详解](docs/系统运行机制详解.md)。

当前主体框架保持为：

```text
run_multi_agent.py       命令行入口
bridge_agents/           LangGraph 编排、阶段 Agent、协调器、共享状态、Prompt/Skill Registry、RAG
tools/                   工程工具脚本和 LangChain Tool 封装
prompts/                 当前有效 Prompt 模板和 manifest
config/                  用户配置文件
data/                    规范四库、RAG 评测、few-shot、规则和绘图配置
samples/                 尺寸设计、配筋设计样例
scripts/                 被工具复用的轻量基础脚本
tests/                   回归测试
misc/                    非主体资料：历史归档、计划文档、面试资料、上传说明
```

运行输出目录如 `output/`、`outputs/`、`show/`、`tmp/`、缓存目录和虚拟环境默认不进入 Git。

## 1. 方法架构

### 1.1 五层混合架构

```text
总体协调与治理层
DesignCoordinatorAgent + 调度合法性校验 + Human Review
        │ 阶段目标                      ▲ StageHandoff
        ▼                               │
专业 Agent 子图层
InitialDesign / LayoutRevision / StructuralDesign / ModelingCheck
        ▲ Prompt / Skill / Few-shot     │ ActionSpec / ActionResult
        │                               ▼
知识与提示词层                    确定性工程工具层
PromptRegistry                   路线解析、图纸分割、碰撞检测
SkillRegistry + Few-shot         OpenSees、承载力验算
规范四库 RAG（最终评估与用户专业问答）
可执行公式注册表（固定 Python 实现）

状态、成果与持久化层贯穿各层
AgentState reducer + AgentContract + ArtifactRecord
成果依赖/失效传播 + output_dir + SQLite checkpoint
```

| 控制内容 | 执行主体 | 约束方式 |
| --- | --- | --- |
| 任务意图、阶段目标、专业步骤规划 | LLM Agent | 结构化输出、允许动作集合 |
| 工程计算、文件处理、批次统计 | 确定性工具 | 参数校验、异常隔离、自动测试 |
| 阶段跳转、成果放行、返修范围 | 协调器规则层 | 前置依赖、轮次、完整性与人工复核检查 |

### 1.2 系统运转机制

```text
用户任务与原始资料
  -> 扫描 output_dir 恢复工程成果
  -> DesignCoordinatorAgent 提出阶段决策
  -> 确定性调度校验
  -> 专业 Agent 子图规划或选择 Action
  -> 确定性工程工具执行
  -> 批次完整性、工程指标和成果依赖检查
  -> StageHandoff
  -> 返回协调器重新决策
  -> 下一阶段 / 定向返修 / 人工复核 / 结束
  -> final_output 确定性生成按设计组复用的配筋绘图成果包
  -> DesignReviewAgent 整体评估与后续专业问答（独立只读链）
```

调度校验检查阶段注册、前置成果、成果有效性、返修目标、轮次上限、人工复核状态和无依据阶段回跳。每个阶段结束后，协调器读取最新 `StageHandoff` 再决策，专业 Agent 不直接控制顶层路由。

### 1.3 两条顶层运行链路

```text
目标显式协调图（--graph-v2）
用户任务 -> DesignCoordinatorAgent -> 确定性调度校验 -> 专业 Agent
         -> StageHandoff -> DesignCoordinatorAgent 重新决策
         -> 下一阶段 / manual_review / final_output / error

旧兼容图（当前 CLI 默认）
用户任务 -> TaskAllocationAgent -> AgentExecutor -> 专业 Agent -> final_output
```

显式协调图代表项目最终方法框架。兼容图服务于历史成果、旧行为回归和迁移对照，后续可在 Graph V2 完成更多端到端验收后从主体系统分离。

规范四库 RAG 当前只接入最终评估与用户专业问答。结构尺寸和配筋设计使用既有任务规则与 few-shot，荷载、分析、承载力及返修方向由确定性代码处理。`code_rag.enabled=false` 只关闭设计后规范问答；显式启用后若索引缺失或过期，评估/问答会明确报错，检索不到有效依据时会记录证据不足。

## 2. LangGraph 结构

### 2.1 显式协调图（目标架构）

`bridge_agents/graph_v2.py::build_graph_v2()` 构建顶层协调图：

```text
START
  -> design_coordinator
       |-- initial_design ------> StageHandoff --+
       |-- layout_revision -----> StageHandoff --+
       |-- structural_design ---> StageHandoff --+--> design_coordinator
       |-- modeling_check -------> StageHandoff --+
       |-- manual_review <------------------------+
       |-- final_output -> END
       `-- error -------> END
```

顶层节点包括 `design_coordinator`、四个专业 Agent、`manual_review`、`final_output` 和 `error`。协调器输出的阶段选择在执行前必须通过确定性校验，检查阶段注册、前置成果、成果有效性、允许的返修目标、轮次限制和人工复核状态。连续非法调度会进入人工复核，防止错误决策形成无界循环。

每个专业 Agent 均为独立的内部 LangGraph：

```text
InitialDesignGraph
plan_initial_design
  -> prepare_initial_action
  -> execute_initial_action
  -> prepare_initial_action / complete_initial_design

LayoutRevisionGraph
prepare_revision_step
  -> decide_revision_action
  -> execute_revision_action
  -> prepare_revision_step / END

StructuralDesignGraph
plan_structural_design
  -> prepare_structural_action
  -> execute_structural_action
  -> prepare_structural_action / complete_structural_design

ModelingCheckGraph
prepare_modeling_check_step
  -> decide_modeling_action
  -> execute_modeling_action
  -> prepare_modeling_check_step / END
```

子图内部使用 `ActionSpec` 描述工具动作，由统一 Action runner 完成输入检查、异常包装、成果路径抽取和 observation 构造。追加型状态通过 reducer 合并，避免循环中覆盖 Prompt 轨迹、修正历史、验算历史和 Agent 事件。

### 2.2 兼容图

`bridge_agents/agent.py::build_graph()` 保留旧顶层链路：

```text
task_allocation
  -> agent_executor <-> agent_executor
  -> final_output / unsupported_intent / error
  -> END
```

兼容图根据 `agent_sequence` 顺序执行专业 Agent，并保留验算失败后回到结构设计的反馈路径。它用于历史行为回归和迁移对照；新方法开发、人工复核和断点恢复以 Graph V2 为准。

## 3. 阶段 Agent

### TaskAllocationAgent

位置：`bridge_agents/agent.py`

负责解析用户自然语言任务，输出：

```text
task_category
user_intent
extracted_parameters
agent_sequence
```

支持的主要意图：

```text
full_design
layout_design
layout_design_check_revision
layout_check_revision
structural_design
layout_to_reinforcement_design
reinforcement_design
reinforcement_design_verification
verification
```

如果 LLM 分发失败，会使用规则兜底，但正常情况下以 LLM 的结构化输出为准。

### InitialDesignAgent

位置：`bridge_agents/stage_agents.py`

模式：规划—动作内部子图。

允许步骤：

```text
load_data
drawing_crop_and_mask
obstacle_semantic_extractor
select_samples
generate_layout_design
```

LLM 根据已有成果规划缺失步骤；Action runner 调用确定性数据、图纸和障碍物工具，并执行 few-shot 选择与布跨生成。每一步完成后回到子图路由，已有有效成果可以被恢复并跳过重复动作。

### LayoutRevisionAgent

位置：`bridge_agents/stage_agents.py`

模式：检测—决策—执行内部子图。

可选动作：

```text
run_collision_detection
generate_revision_instruction
build_revision_prompt
generate_revised_layout
finish_revision
manual_review
```

该阶段反复执行“检测 -> 修正 -> 复检”，直到通过碰撞阈值、达到最大修正轮次，或进入人工复核。碰撞按桥梁编号和左右幅分别聚类；缺少明确障碍区间时，普通碰撞修正受标准跨径和人工复核规则约束，避免由碰撞点包络推导异常大跨径。

### StructuralDesignAgent

位置：`bridge_agents/stage_agents.py`

模式：规划—批次执行内部子图。

允许步骤：

```text
extract_design_units
dimension_design
compute_pier_groups
reinforcement_design
summarize_structural_design_result
```

该阶段将布跨结果转换成结构设计单元，完成尺寸和配筋设计，并汇总 `structural_design_result`：

- 尺寸设计以设计单元为批次并行执行，默认最大并发数为 4；
- 尺寸设计后确定性执行桥墩设计组归并：从布跨结果按墩号/桩号匹配墩高，用纵断面高程重算并核实布跨墩高，扣除柱位盖梁高度得到墩柱净高，在单联内按完整归并字段归并设计组并输出高度审计；
- 设计单元提取校验要求“本联设计分组”覆盖该联全部桥墩（含边墩/连接墩），漏分组时触发重试；边墩缺少尺寸分组时并入同单元相邻非桥台组，并在 `assumed_member_piers` 记录假设，避免边界墩静默丢失；
- 配筋设计先串行完成 OpenSees 确定性预处理，再以墩组任务并行调用模型，默认最大并发数为 3；
- 配筋已扩展为盖梁与墩柱联合配筋，一次生成 `pier_cap` 与 `pier_column`，并把设计组控制净高与墩柱 few-shot 样本注入 Prompt；
- 模型输出支持有限次格式修复，结果按稳定任务顺序合并；
- 任一批次缺失都会进入人工复核，可重试失败任务或带风险接受部分结果。

### ModelingCheckAgent

位置：`bridge_agents/stage_agents.py`

模式：验算—决策—反馈内部子图。

可选动作：

```text
run_capacity_check
generate_revision_instruction
finish_check
```

承载力验算由确定性工具执行。Action 层从配筋批次中按 `task_id` 配对全部成功分组的配筋 YAML 与 OpenSees 内力文件，逐组串行验算，并聚合通过状态、失败分组、控制截面和最大利用率。返修方向由确定性规则决定：墩柱长细比超出表5.3.1适用范围时返回尺寸设计；墩柱普通轴压承载力不足时返回联合配筋；受压区高度超过表5.2.1的 `ξb` 时返回尺寸设计；承载力未通过且受压区界限校核有效时返回配筋设计。验算失败、步骤耗尽或需要工程判断时进入人工复核，用户可继续返修、带风险结束或中止。

正截面抗弯采用 JTG 3362-2018 第5.2.2条固定 Python 执行器，并按表5.2.1校核相对界限受压区高度；盖梁斜截面抗剪采用第8.4.4条和第8.4.5条固定执行器。系统不会动态执行 YAML 表达式。每个分组输出合规矩阵，记录公式源实体、控制利用率、判定状态和未覆盖项。裂缝、挠度、疲劳、抗震及完整构造审查仍属于概念设计未覆盖范围。

墩柱轴心受压采用 JTG 3362-2018 第5.3.1条固定 Python 执行器，稳定系数按表5.3.1保守取上一级长细比表项；螺旋箍筋约束轴压按第5.3.2条单独列示，不覆盖普通轴压主结论。当前仅覆盖轴心受压，偏心受压、柱弯矩剪力与纵横向框架分析仍未覆盖，覆盖状态标记为 `partial_coverage`。

### DesignReviewAgent

位置：`bridge_agents/design_review.py`

该 Agent 位于设计主图结束后的独立只读链，负责：

- 从 `output_dir` 检索布跨、结构设计、批量验算等项目成果；
- 检索与问题相关的规范证据；
- 生成整体设计评估；
- 回答针对当前成果的后续专业问题；
- 校验回答引用的 `[ARTIFACT-...]` 与 `[EVID_...]` 是否真实存在；
- 将评估、回答和检索审计追加保存到 `output_dir/design_review/`。

评估按本系统当前约定的概念设计范围执行，不是施工图完整交付标准：先完整总结已完成成果与确定性校核结果，再列方法边界（尚未实现的计算或后续设计内容）与后续建议，不让未覆盖项淹没已完成成果；`completed_with_limitations` 表示概念设计目标已完成但存在明确方法边界，`manual_review_required` 仅用于当前范围内存在证据明确的冲突、失败或关键歧义；内部 Prompt、日志和历史中间方案不视为最终成果，不得据此否定最终方案。代码层对 `stage_findings`/`limitations`/`recommended_actions` 等数组字段做字符串规范化容错，并生成按分节排版的可读展示文本（`display_text`）供窗口与命令行输出。

它不修改设计成果、不控制主图路由，也不把 LLM 解释当作确定性计算结果。

## 4. 人机协同与断点恢复

Graph V2 使用 LangGraph interrupt 将“需要工程判断”表达为可恢复状态。交互模式下，系统暂停在 `manual_review` 节点，保存复核原因、候选动作、阶段上下文和已生成成果；用户选择后从原线程继续，无需重新输入完整任务。

主要复核动作如下：

| 场景 | 可选动作 | 含义 |
| --- | --- | --- |
| 布跨修正超限 | `continue_revision` | 增加修正机会并继续检测—修正循环 |
| 布跨人工接受 | `accept_and_continue` | 保留当前方案并进入后续阶段 |
| 结构批次不完整 | `retry_failed_tasks` | 只重试失败或缺失的设计任务 |
| 结构部分接受 | `accept_partial_and_continue` | 保留部分结果并记录接受风险 |
| 验算需返修 | `continue_modeling_revision` | 回到结构设计并按反馈定向修改 |
| 验算人工结束 | `accept_check_and_finish` | 保留未关闭问题并带风险结束 |
| 协调异常 | `retry_coordinator` | 重新执行协调决策 |
| 任意复核 | `abort` | 中止当前任务 |

布跨、结构部分成果或验算结果的人工接受都会写入 `accepted_risks`，最终状态标记为 `completed_with_accepted_risks`，与自动验算通过保持可区分。接受动作只表示流程授权，不代表工程指标自动合格。

Graph V2 默认在以下位置保存 SQLite checkpoint：

```text
output_dir/checkpoints/graph_v2.sqlite
```

两类恢复信息分工如下：

- `output_dir` 中的 JSON、YAML、图像和分析文件保存工程成果，可由 `_discover_existing_outputs()` 扫描恢复；
- `output_dir/human_review/human_review_decisions.jsonl` 追加保存人工决定、受审成果路径及 SHA256，用于新线程恢复仍有效的长期授权；
- checkpoint 保存图节点、线程状态、interrupt 和运行上下文，用于从暂停位置继续。

使用相同 `thread_id` 可恢复人工复核任务；若 checkpoint 不存在，系统仍可从输出目录发现已有工程成果和已持久化的接受决定，但不能还原精确的图中断位置。接受决定只有在受审成果路径与 SHA256 均保持一致时才会恢复；成果变化后旧授权自动失效并重新进入相应校核。

## 5. 状态、契约与成果治理

跨 Agent 数据通过统一状态和显式契约传递：

```text
AgentState
  |- 当前任务、阶段和运行控制字段
  |- 工程成果字段
  |- prompt_trace / agent_events 等 reducer 追加字段
  `- handoff / manual_review / accepted_risks

AgentContract
  |- required_inputs
  |- produced_outputs
  `- allowed_handoffs

StageHandoff
  |- source_stage / target_stage
  |- status / summary
  |- artifacts
  `- feedback / retry metadata
```

工程成果按依赖关系组织：

```text
route_and_obstacles
  -> layout_result
  -> design_units
  -> dimension_design
  -> reinforcement_design
  -> structural_analysis
  -> capacity_check
  -> drawing_package
  -> final_deliverables
```

协调器根据依赖图判断阶段是否具备执行条件，并在返修时失效受影响的下游成果。该机制支持定向返工，避免尺寸或配筋变化后继续复用失效验算结果。

### 5.1 配筋绘图成果

Graph V2 和兼容图在 `final_output` 共用确定性函数 `build_final_deliverables()`。当桥墩设计组、尺寸设计、联合配筋和验算成果齐全时，系统按唯一 `design_group_id` 生成一套图纸，同组桥墩通过 `member_piers` 复用该成果。绘图不增加 Agent，也不调用 LLM。

绘图前对配筋表达式与几何做确定性工程诊断，不再静默产出错误图纸：盖梁命名折线路径（如弯起钢筋）必须连续非空；墩柱环形加强箍（N4）必须位于纵筋笼内侧，其环径必须小于纵筋（N1）中心圆直径；纵筋/箍筋分布分区按标准间距排布时，末段采用不大于标准间距的收口间距以保证保护层端点准确，超出标准间距一个以上则不自动修补。诊断以 error 级标出问题项，原因以明确中文说明返回，供定向返修。

配筋 YAML/JSON 是设计真源。绘图器将其转换为与 CAD 格式无关的 Drawing IR，再由同一份 IR 输出 AutoCAD Script 和 SVG 预览：

```text
deliverables/
  design_manifest.json
  drawings/
    drawing_index.json
    <design_group_id>/
      drawing_ir.json
      reinforcement_<design_group_id>.scr
      reinforcement_<design_group_id>.svg
      bar_schedule_<design_group_id>.csv
      geometry_diagnostics.json
```

自动验算通过的成果标记为 `verified`。显式允许未验算预览时标记为 `draft_unverified`，图面显示“自动验算未完成，仅供复核”；人工接受验算风险时标记为 `accepted_risk`，图面显示“含人工接受风险”，风险原文同时写入成果清单。

主流程自动生成绘图；已有设计成果也可离线导出，不调用在线模型：

```powershell
python run_export_drawings.py --config config/settings.yaml
```

未验算草稿只能通过显式参数生成：

```powershell
python run_export_drawings.py --config config/settings.yaml --allow-unverified
```

配筋、内力或绘图逻辑升级后，可对某次完整运行的既有成果离线复算反力、轴压与长细比，并按最新逻辑重建配筋绘图成果（不重新调用在线设计模型，人工接受风险随成果保留）：

```powershell
python tools\recompute_existing_reinforcement_outputs.py <output_dir>
```

`.scr` 应在 AutoCAD 空白公制模型空间中通过 `SCRIPT` 命令执行。脚本不包含保存、另存、打印命令。SVG 只作预览，配筋图属于概念/初步设计表达，不能替代施工图深化和人工审签。详细格式与人工验收步骤见 [CAD 配筋成果说明](docs/cad_drawing/README.md)。

## 6. Prompt、few-shot 与知识服务

当前有效 Prompt 统一放在 `prompts/`，由 `prompts/manifest.yaml` 注册，并通过 `bridge_agents/prompt_registry.py` 渲染。

```text
prompts/manifest.yaml       Prompt 注册表
prompts/agents/             阶段 Agent 行为 Prompt
prompts/tasks/              具体任务 Prompt 模板
prompts/partials/           可复用片段
```

业务代码不直接拼路径，而是引用稳定的 `prompt_id`：

```python
render_prompt("agents.task_allocation.v1", {}, config_path="config/settings.yaml")
render_prompt("tasks.initial_layout_design.v1", context, config_path="config/settings.yaml")
```

`PromptRegistry` 使用 Jinja2 `StrictUndefined` 渲染模板，缺少必需上下文会直接失败。每次渲染会记录：

```text
prompt_id
version
role
template_path
template_sha256
context_keys
rendered_at
```

这些记录写入 `prompt_trace`，并追加保存到 `output_dir/logs/prompt_trace.jsonl`，用于追踪一次运行使用的 Prompt、模板版本、哈希、任务编号、尝试次数、渲染路径和执行状态。重跑采用追加式文件名，避免覆盖历史 Prompt。

初始布跨当前输入和 few-shot 样例统一压缩为 K/Z 线地形特征表，选择器兼容原始数组、嵌套对象和表格数据，并使用有效性掩码排除缺失点。尺寸样本采用“同角色同体系 -> 同角色跨体系”的匹配顺序；有效样本为空时会阻止模型调用并给出诊断。配筋任务在 OpenSees 预处理前保存任务上下文，成功取得内力后再保存最终 Prompt。

墩柱配筋模板显式约定：环形加强箍（N4）布置在纵向主筋（N1）纵筋笼内侧，其 `hoop_diameter` 必须小于 N1 的 `bar_centerline_diameter`，禁止用“柱径减两倍保护层”这类外圈公式推算 N4；任务输入的柱径/柱高/柱距进入绘图表达式环境后统一按 mm 语义使用，禁止再次乘 1000。规则与代码侧工程自检（见 §5.1）配套，防止加强箍与纵筋同环或外翻、单位重复换算。

规范四库 RAG 位于 `bridge_agents/code_rag/`，支持证据文档、SQLite FTS5、BGE 向量和 RRF 混合检索，使用说明见 `docs/code_rag/README.md`。最终评估和问答同时检索项目成果与规范证据，并记录证据 ID、索引版本和引用审计。可执行公式注册表由固定 Python 实现，公式及限值从规范知识中离线整理后固化，不在计算时调用 RAG。

## 7. 项目代码架构

### 7.1 主体目录

```text
BridgeDesign_Agent/
  run_multi_agent.py       命令行入口、交互复核和恢复入口
  bridge_agents/           多智能体状态、图、契约、协调和运行时
  tools/                   桥梁设计确定性工具与模型任务封装
  prompts/                 Prompt manifest、Agent/任务模板和复用片段
  config/                  普通运行配置与模型配置
  data/                    few-shot、规则、示例数据和绘图配置
  tests/                   单元、图路由、回归和 RAG 测试
  docs/code_rag/           规范知识服务使用说明
  misc/                    历史代码、计划和追溯材料
```

### 7.2 `bridge_agents` 关键模块

| 模块 | 职责 |
| --- | --- |
| `agent.py` | 兼容图、阶段包装器、输出目录成果发现 |
| `graph_v2.py` | 显式协调图、人工复核路由和最终状态 |
| `stage_agents.py` | 四个专业 Agent 及其内部 LangGraph |
| `design_coordinator.py` | 协调器模型调用和结构化决策 |
| `coordinator.py` | 确定性调度校验、非法调度兜底 |
| `state.py` | `AgentState` 及 reducer 定义 |
| `actions.py`、`tool_actions.py` | Action 规范、注册和统一执行 |
| `contracts.py`、`handoff.py` | Agent 契约与 `StageHandoff` |
| `dependencies.py` | 成果依赖、有效性与返工传播 |
| `checkpointing.py` | Graph V2 SQLite checkpoint |
| `batch_execution.py` | 有界并发、失败收集和稳定结果合并 |
| `prompt_registry.py`、`prompt_audit.py` | Prompt 注册、渲染和审计 |
| `skill_registry.py` | 专业能力说明和 SKILL 注册 |
| `code_rag/` | 规范四库索引、检索和评测 |
| `structural_evidence.py` | 既有结构设计检索适配层；尺寸和配筋运行器已停止调用，保留用于历史追溯与后续清理 |
| `code_formula_registry.py` | 已映射规范公式的固定 Python 执行器注册表 |
| `capacity_batch.py` | 多分组承载力聚合与规范合规矩阵 |
| `design_review.py` | 设计完成后的成果评估、规范问答和引用审计 |
| `drawing/` | 设计组输入、Drawing IR、工程视图、几何校验、SCR/SVG 和成果包 |
| `result_catalog.py` | 全阶段成果目录服务（`ResultArtifact`/`ResultCatalog`），供成果中心与评估链复用 |
| `preview_images.py` | 窗口内图片预览管线（Pillow 解码、缩放、缓存，JPEG/高 DPI） |
| `initial_visualizer.py` | 确定性初设可视化：平面叠加图与纵断面示意（输出 `catalog_views/`） |

### 7.3 工具层

主要工程工具位于 `tools/`。

```text
tools/data_loader_tool.py                         读取纬地路线资料，生成设计输入与平曲线 JSON
tools/drawing_mask_preprocess_tool.py             图纸转换、裁剪、栅格化、语义分割掩码生成
tools/obstacle_semantic_extractor_tool.py         从掩码和路线资料提取障碍物语义信息
tools/sample_selector_tool.py                     从 few-shot 库中检索相似样例
tools/design_generation_tool.py                   生成初始布跨方案
tools/collision_detection_tool.py                 设桥布跨碰撞检测
tools/revision_instruction_tool.py                生成布跨修正指令
tools/layout_revision_prompt_tool.py              构造布跨修正 Prompt
tools/design_unit_extractor_tool.py               从布跨方案提取结构设计单元
tools/dimension_design_tool.py                    下部结构尺寸设计
tools/reinforcement_design_tool.py                配筋设计
tools/capacity_check_tool.py                      承载力验算入口
tools/cap_beam_capacity_envelope_overlay_refined.py 盖梁承载力包络验算脚本
tools/recompute_existing_reinforcement_outputs.py 对既有配筋成果离线复算反力/轴力/长细比并重建绘图（不调用在线模型）
tools/plot_structural_results.py                  结构结果可视化辅助脚本
```

`scripts/` 中的文件不是杂项，它们被主体工具复用：

```text
scripts/llm_client.py       LLM 调用封装
scripts/selector.py         few-shot 相似样例选择
scripts/prompt_builder.py   历史兼容的 Prompt 构造辅助
```

## 8. 输入配置

普通用户主要修改 `config/settings.yaml`。

最小输入包括：

```yaml
task:
  user_request: 请对K1_000-K2_600进行全流程设计任务。

paths:
  data_path: path/to/route_data/K线
  input_drawing_path: path/to/input_drawing.dxf
  output_dir: output/layout_agent_run_example_K1_000-K2_600

route_data:
  file_prefix: K-终版

structure:
  dimension_max_workers: 4
  reinforcement_max_workers: 3
  reinforcement_max_format_repairs: 1

modeling:
  max_modeling_check_steps: 6
  max_check_revision_rounds: 3
  auto_run_opensees_if_missing: false

code_rag:
  enabled: true
  retrieval_mode: keyword_only
  top_k: 3
  max_evidence_per_task: 8
```

字段含义：

```text
task.user_request       用户自然语言设计任务
paths.data_path         纬地原始资料目录
paths.input_drawing_path 输入图纸路径，通常是 DXF/DWG 转换后的图纸
paths.output_dir        本次运行主输出目录，也是断点续跑扫描目录
route_data.file_prefix  纬地文件前缀，用于定位 .pm/.WID/.zdm/.dmx 文件
structure.*             尺寸和配筋批处理并发数、模型格式修复次数
modeling.*              验算子图最大步骤数和结构返修轮次
modeling.auto_run_opensees_if_missing 分组缺少 OpenSees 内力文件时是否自动补跑分析（默认 false，需人工确认后显式开启）
code_rag.enabled        是否在结构设计与最终问答中启用规范检索
code_rag.retrieval_mode auto / keyword_only / vector_only / hybrid
```

如果 `data_path` 下只有一个 `.pm` 或 `.PM` 文件，系统会自动推断 `file_prefix`。如果目录内有多条路线资料，建议显式填写。

不建议在普通配置中填写中间成果路径。以下内容由系统从 `output_dir` 自动发现：

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
capacity_check_output_dir
collision_output_dir
```

保留 `--initial-state-json` 只是为了开发调试，普通用户不需要使用。

## 9. 输出结构

系统以 `paths.output_dir` 为主输出目录。典型结果结构如下：

```text
output_dir/
  plane_from_loader/
    K_plane.json
  mask/
    obstacle_mask_*.png
  obstacle_semantic/
    complete_obstacles_grouped_*.json
    merged_vis_*.jpg
  design_run_YYYYMMDD_HHMMSS/
    input_data.json
    few_shots.json
    prompt.txt
    response_full.md
    design_result.json
  collision_detection/
    collision_metrics_*.json
    collision_report_*.json
    collision_summary_*.txt
    collision_vis_*.jpg
  revision_instructions/
    revision_instruction_round_*.txt
  revision_prompts/
    revision_prompt_round_*.txt
  revision_results/
    revision_design_round_*.json
  layout_revision/
    final_layout_result.json
  structural_design/
    design_units/
      design_units_result.json
    dimension_design/
      dimension_design_result.json
    pier_group/
      pier_group_result.json
    reinforcement_design/
      reinforcement_design_result.json
      */reinforcement_result_*.yaml
      */internal_force_output_full_beam_*.json
    structural_design_result.json
  capacity_check/
    capacity_check_batch_summary.json
    <task_id>/
      capacity_check_summary.json
  modeling_check/
    feedback_decision.json
    revision_context.json
  human_review/
    human_review_decisions.jsonl
  checkpoints/
    graph_v2.sqlite
  logs/
    agent_run_log.json
    prompt_trace.jsonl
  design_review/
    latest_assessment.json
    assessment_*.json
    answer_*.md
    review_audit.jsonl
```

断点续跑扫描优先级：

```text
layout_revision/final_layout_result.json
    >
最新 design_run_*/design_result.json
    >
递归发现的其他 design_result.json
```

结构设计、配筋、验算等阶段也会按固定目录模式自动发现已有成果。各模型任务同时保留渲染 Prompt、原始响应、解析结果和执行状态，便于定位单个批次失败原因。

## 10. 技术栈与依赖

核心依赖见 `requirements.txt`：

```text
python-dotenv
langgraph
langgraph-checkpoint-sqlite
langchain-core
langchain-openai
openai
PyYAML
numpy
pandas
pydantic
jinja2
sentence-transformers==2.7.0
transformers==4.35.2
pytest
httpx
tqdm
opencv-python
matplotlib
ezdxf
openseespy==3.5.1.11
openseespywin==3.5.1.11（Windows）
```

视觉分割相关依赖见 `requirements-vision.txt`：

```text
torch==2.1.0+cpu
torchvision==0.16.0+cpu
mmengine
mmcv==2.1.0
mmsegmentation==1.2.2
ftfy
```

当前推荐使用项目已配置的 `bridge_design` Conda 环境：

```powershell
python
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-vision.txt
```

## 11. 运行方式

### Graph V2 桌面控制台

Windows 用户可以通过独立桌面窗口完成任务配置、Graph V2 启动、阶段状态查看、人工复核恢复和设计后问答：

```powershell
& 'python' run_graph_v2_gui.py
```

窗口固定使用 Graph V2，不提供兼容图选择。它只暴露任务描述、路线数据、输入图纸、输出目录、线路前缀和修正轮次；中间成果继续由 `output_dir` 自动发现。最近一次填写的表单字段（任务描述、路径、前缀、轮次与 RAG 开关等）会自动保存到 `output/.graph_v2_gui_state.json`，下次启动自动恢复上次内容，避免重复填写；需要时仍可点“基础配置-选择”载入历史 `config/settings.yaml` 回填。每次启动会在以下位置保存独立配置快照，不覆盖原始 `config/settings.yaml`：

#### 现代 Graph V2 工作台（PySide6）

`run_graph_v2_gui_modern.py` 是推荐的现代桌面入口。它复用 `GraphV2Controller` 和统一成果目录服务，提供工程总览、任务输入与启动、安全终止、断点恢复、人工复核、成果预览、整体评估和多轮专业问答。旧 Tkinter 控制台继续保留，用于兼容和对照。

读取真实运行成果：

```powershell
.\.venv\python.exe run_graph_v2_gui_modern.py `
  --output-dir output\示例项目9.8 `
  --project-name "示例项目 9.8" `
  --route-range "K1+600 — K3+100" `
  --paper
```

任务输入页只暴露任务描述、路线数据、输入图纸、输出目录、线路前缀和修正轮数。运行 ID 自动生成；最近一次表单会保存到 `output/.graph_v2_gui_modern_state.json`，外发授权、运行 ID 和密钥不会持久化。Graph V2、人工恢复、整体评估和问答均在后台线程执行。

生成固定 1440×900 的论文截图并自动退出：

```powershell
.\.venv\python.exe run_graph_v2_gui_modern.py `
  --output-dir output\示例项目9.8 `
  --project-name "示例项目 9.8" `
  --route-range "K1+600 — K3+100" `
  --paper `
  --page overview `
  --screenshot output\示例项目9.8\gui_overview.png
```

`--page` 支持 `overview`、`task`、`review`、`results`、`qa` 和 `info`，可直接导出指定页面。`--demo` 会使用明确标注为“演示数据”的样例，不会与真实成果混淆。若发现最终成果已经存在、建模验算仍未完成，工作台会将该依赖冲突提升为人工判断告警。

成果问答携带最近 12 轮对话（24 条消息），回答会显示成果与规范证据编号；整体评估独立展示，不进入后续对话上下文。调用任务、评估或问答前必须在当前会话勾选外发授权。

“绘图成果”页读取 `drawing_index.json`，按设计组展示桥墩总体布置图、盖梁配筋详图和墩柱配筋详图，可直接打开 SVG 预览、AutoCAD SCR 或绘图目录。旧版单文件 `scr_path/svg_path` 与新版分页 `scr_paths/svg_paths` 均可在断点恢复时识别。

“成果中心”页按 初步设计 / 布跨修正 / 结构设计 / 建模验算 / 最终交付 五个阶段汇集全部工程成果：PNG/JPG 图件在窗口内直接预览（Pillow 解码与缩放，JPEG 与高 DPI 可用）；最终交付的 SVG 图纸（受控元素集：rect/line/polyline/circle/text）由轻量渲染器 `bridge_agents/svg_preview.py` 自动转 PNG 后窗口内显示，无需外部看图程序；其余数据成果在右侧信息面板展示来源路径与诊断摘要，并支持外部打开或进入所在目录。“生成目录视图”调用确定性可视化器（`bridge_agents/initial_visualizer.py`），把布跨方案叠加到地形栅格生成平面示意图，并按墩位与墩高生成纵断面示意图；初始方案（最新 `design_run` 布跨结果）与最终修正方案（`final_layout_result.json`）分别输出到 `<output_dir>/catalog_views/initial/` 与 `<output_dir>/catalog_views/final/`，互不覆盖。地形栅格与布跨坐标基准不一致时不强行叠加，界面会提示原因。成果树由共享目录服务 `bridge_agents/result_catalog.py` 提供，供窗口与 `DesignReviewAgent` 复用。运行中“运行记录”页每 2 秒增量回放 `<output_dir>/logs/agent_run_log.json` 的动作级事件（阶段规划/动作开始/完成/失败），实时反映当前运行位置。

“成果问答”页以多轮对话形式运行：提问与回答按轮次气泡展示（Enter 发送、Shift+Enter 换行）。发送时自动携带最近对话作为上下文（上限 10 轮、单条截断），工程证据与规范条文的检索仍以本轮问题为准；整体评估结果会追加为一条回答，但不计入后续问答上下文，避免撑爆窗口。每轮问答与证据引用照常落盘到 `<output_dir>/design_review/answer_*.md`。

```text
<output_dir>/run_configs/<thread_id>.yaml
```

Graph checkpoint 仍保存在：

```text
<output_dir>/checkpoints/graph_v2.sqlite
```

设计运行和专业问答会调用配置中的外部模型。窗口要求用户先勾选外发授权，API Key 只从既有配置引用或环境变量读取，不写入运行配置快照。环境冒烟检查不会打开窗口：

```powershell
& 'python' run_graph_v2_gui.py --check
```

### 命令行运行

从配置文件读取任务：

```powershell
python run_multi_agent.py --config config/settings.yaml
```

启用显式协调图：

```powershell
python run_multi_agent.py --config config/settings.yaml --graph-v2
```

推荐的交互式完整运行方式：

```powershell
python run_multi_agent.py `
  --config config/settings.yaml `
  --graph-v2 `
  --interactive `
  --thread-id case-001
```

恢复已暂停的 Graph V2 线程：

```powershell
python run_multi_agent.py `
  --config config/settings.yaml `
  --resume case-001 `
  --decision retry_failed_tasks
```

直接传入自然语言任务：

```powershell
python run_multi_agent.py "请对K1_000-K2_600进行全流程设计任务" --config config/settings.yaml
```

开发调试时额外注入初始状态：

```powershell
python run_multi_agent.py "请基于当前已有成果继续完成验算" --config config/settings.yaml --initial-state-json path/to/state.json
```

查看命令行帮助：

```powershell
python run_multi_agent.py --help
```

设计完成后生成整体评估：

```powershell
python run_design_review.py `
  --config config/settings.yaml `
  --assess
```

针对已有成果提出后续专业问题：

```powershell
python run_design_review.py `
  --config config/settings.yaml `
  --question "当前盖梁最大利用率是多少，抗剪依据是什么？"
```

### 规范 RAG 知识服务

独立的规范四库源图谱、提示词证据层、关键词/向量/RRF 混合索引、查询和评测说明见：

```text
docs/code_rag/README.md
```

## 12. 开发验证

推荐在提交前运行：

```powershell
python -m pytest tests/code_rag
python -m pytest tests/test_skill_registry.py tests/test_contracts.py tests/test_coordinator.py tests/test_design_coordinator.py tests/test_dependencies.py tests/test_handoff.py tests/test_graph_v2.py
python -m pytest tests/test_prompt_registry.py tests/test_initial_state_loading.py
python -m compileall bridge_agents tools tests run_multi_agent.py
```

测试覆盖重点：

```text
tests/test_prompt_registry.py        Prompt manifest 校验、模板渲染、缺字段失败
tests/test_initial_state_loading.py  初始状态只从 output_dir 发现已有成果，不从配置注入中间成果
tests/code_rag/                      证据构建、索引、三路检索和内容级评测
tests/test_graph_v2.py               显式协调图路由、交接与结束条件
```

## 13. 当前能力边界

- Graph V2 已具备协调、四个专业子图、人工复核和 checkpoint，CLI 默认入口仍是兼容图；全面替换需要更多实际线路端到端验收。
- 尺寸与配筋已支持有界并发和失败任务重试；承载力验算覆盖全部成功配筋分组，当前为保证验算脚本隔离而采用逐组串行执行。
- 规范四库 RAG 当前只接入最终评估与用户专业问答；尺寸、配筋、荷载、OpenSees 分析和承载力验算运行时均不调用 RAG。
- 当前自动验算覆盖盖梁正截面抗弯、斜截面抗剪和相对界限受压区高度；裂缝、挠度、疲劳、抗震、普通钢筋混凝土盖梁完整构造审查仍需后续补充或人工复核。
- 墩柱轴心受压采用 JTG 3362-2018 第5.3.1条固定执行器，稳定系数按表5.3.1保守取上一级长细比表项，螺旋箍筋按第5.3.2条单独列示；当前仅覆盖轴心受压，偏心受压、柱弯矩剪力与纵横向框架分析未覆盖，覆盖状态标记为 `partial_coverage`。
- 当前阶段仅对桥墩（中间墩与边墩）做尺寸与配筋设计，桥台设计有意跳过；边墩缺失尺寸分组时以同单元相邻非桥台组兜底并记录 `assumed_member_piers`，正式工程需人工复核。
- 人工接受会保留风险记录；它不能替代规范审查、计算复核和最终工程签认。
- LLM 输出均需通过结构化解析和相应阶段的确定性校验。模型服务、输入图纸质量及 few-shot 覆盖度仍会影响设计结果。

## 14. 历史与杂项目录

`misc/` 用于放置不参与当前主体运行链路的材料。

```text
misc/archive/legacy_prompts/   旧版 Prompt 与模板
misc/archive/legacy_tools/     旧版脚本备份
misc/docs/                     计划文档、上传说明、面试资料、历史说明文档
```

这些内容用于查阅和追溯，不应被当前运行代码引用。当前有效的 Prompt、工具、配置和数据都保留在主体目录中。
