# 规范四库 RAG 知识服务

## 1. 最终目标与当前阶段

本模块服务于高速公路桥梁设计全流程。它接收具体设计问题，返回少量、完整、可追溯、可直接放入提示词的规范证据，包括：

- 有实际约束信息的规范条文；
- 规范公式及其必要变量定义；
- 完整表格数据及单位；
- 规范号、条文号、来源和证据 ID。

RAG 当前服务于设计完成后的整体评估和用户专业问答：根据开放问题检索规范证据，再由 LLM 结合项目成果生成带引用的回答。尺寸、配筋、荷载、结构分析和承载力验算运行链不调用 RAG。规范公式、系数和限值经过离线整理与人工核实后，由固定 Python 工具执行。

当前构建结果为 1908 个源实体、644 份提示词证据和 1264 个排除项。644 份证据已使用 `BAAI/bge-small-zh-v1.5` 生成 512 维归一化向量。内容级评测共 30 题，其中 22 题要求返回指定证据及关键内容，8 题要求明确返回无有效证据；关键词、纯向量和混合模式当前均为 30/30。这只是当前固定数据集上的可复现基线，不代表尚未测试的自然语言表达也能全部命中。

## 2. 为什么不直接检索四库 YAML

四库 YAML 是源知识图谱，包含大量关系、执行配置、审核字段和生成过程信息。这些数据适合追溯和关联装配，不适合直接进入提示词。

离线构建时新增独立的 `EvidenceDocument` 层：

```text
四库 YAML
  -> 统一源实体与关系图
  -> 内容质量与适用性筛选
  -> 按关系组装条文、公式、变量和表格
  -> 提示词证据文档
  -> SQLite FTS5 关键词索引
```

以下内容不会进入证据文档：

- `raw_payload`、`audit`、`action`、`execution`、`logic`；
- 关系 ID 列表和阶段筛选评分；
- “证据不足”“补录正文”“转人工复核”等 AI 或工作流话术；
- 只有条号和标题、没有实质内容的规则；
- `derived`、`derived_pending_review`、`explanatory` 公式；
- 独立变量条目。变量仅在公式或表格需要时内嵌。

注意：自动筛选检查字段来源、结构完整性、适用范围和无主观流程话术。项目当前将 `data/code/` 作为可信规范知识源；证据结构中的 `source_verification=not_verified_against_official_text` 是既有构建元数据，不参与本项目对知识真实性的否定性判定。

## 3. 生成目录

```text
output/code_rag/
  manifest.json                         全链路文件、数量和哈希
  source/source_entities.jsonl          1908 个源实体，含完整 YAML，仅供追溯
  evidence/evidence_documents.jsonl     可检索、可进入提示词的证据白名单
  review/excluded_entities.jsonl        被排除实体及确定性原因
  audit/audit_summary.json              数据与证据质量统计
  audit/audit_report.md                 可读审计报告
  index/code_knowledge.db               SQLite 源图谱与证据索引
  index/embeddings.npy                  644 x 512 归一化证据向量
  index/embedding_ids.json              向量行与 evidence_id 的稳定映射
  index/index_manifest.json             索引版本与源文件哈希
  models/                               项目内 BGE 模型缓存
  evaluation/                           三种检索模式的独立评测结果
```

`output/code_rag/` 可随时重建，不应人工修改。`data/code/` 才是知识源。

## 4. 完整构建

```cmd
.\.venv\python.exe -m bridge_agents.code_rag.dataset
.\.venv\python.exe -m bridge_agents.code_rag.build_index
.\.venv\python.exe -m bridge_agents.code_rag.build_embeddings
.\.venv\python.exe -m bridge_agents.code_rag.evaluate --mode keyword_only
.\.venv\python.exe -m bridge_agents.code_rag.evaluate --mode vector_only
.\.venv\python.exe -m bridge_agents.code_rag.evaluate --mode hybrid
```

Embedding 依赖固定为 `sentence-transformers==2.7.0` 和 `transformers==4.35.2`，与项目现有 `torch 2.1.0+cpu` 兼容。首次构建会下载模型；查询从 `output/code_rag/models/` 的具体 snapshot 离线加载，不访问网络。

## 5. 查询

自然语言查询：

```cmd
.\.venv\python.exe -m bridge_agents.code_rag.query --text "汽车荷载作用组合"
```

存在完整向量索引时，默认 `--mode auto` 使用混合检索；否则使用关键词检索。也可显式选择：

```cmd
.\.venv\python.exe -m bridge_agents.code_rag.query --text "汽车荷载作用组合" --mode keyword_only
.\.venv\python.exe -m bridge_agents.code_rag.query --text "汽车荷载作用组合" --mode vector_only
.\.venv\python.exe -m bridge_agents.code_rag.query --text "汽车荷载作用组合" --mode hybrid
```

限定规范、证据类型和数量：

```cmd
.\.venv\python.exe -m bridge_agents.code_rag.query --text "汽车荷载作用组合" --standard "JTG D60-2015" --type rule --top-k 5
```

结构化设计查询：

```cmd
.\.venv\python.exe -m bridge_agents.code_rag.query --text "混凝土盖梁尺寸与构件验算" --standard "JTG 3362-2018" --stage "构件计算" --member "墩台盖梁" --material "reinforced_concrete" --top-k 5
```

PowerShell 换行必须使用反引号并保证反引号是该行最后一个字符；Windows CMD 不支持反引号，建议直接使用单行命令。

参数：

```text
--text          必填，自然语言问题、工程术语或精确条文号
--standard      可选，规范号，可重复
--type          可选，rule/formula/table，可重复
--stage         可选，设计阶段
--member        可选，构件类型，可重复
--material      可选，材料，可重复
--limit-state   可选，极限状态，可重复
--top-k         可选，证据文档数量，默认 10
--mode          auto/keyword_only/vector_only/hybrid，默认 auto
--format        summary/json/rows，默认 summary
```

`summary` 直接展示 `prompt_text`；`json` 返回结构化 `CodeEvidenceBundle`；`rows` 用于查看关键词排名。三种格式都不会返回原始 YAML。

## 6. 证据文档结构

```text
evidence_id             最终成果引用的证据编号
primary_entity_id       四库源图谱中的主实体编号，仅用于追溯
standard_code           规范号
chapter / clause        章节与条文号
title                   条文、公式或表格标题
normative_text          有实质内容的规范内容字段
formulas                公式表达式和内嵌变量定义
tables                  表格单位和完整数据行
applicability           结构化过滤标签，不作为规范正文
prompt_text             允许进入 LLM 提示词的确定性文本
source_entity_ids       证据装配使用的源实体编号
source_files            源 YAML 文件
source_verification     是否已对照官方规范核验
```

关系只在离线构建阶段用于装配。例如一条有效规则关联规范公式和取值表时，最终返回一份包含三者的证据文档，而不是再返回一串关系实体。

## 7. SQLite 职责

数据库仍使用 SQLite，因为当前数据规模不需要独立向量数据库。SQLite 负责结构化持久化和关键词索引，后续向量存储是并列的检索能力，不需要为关键词和向量复制两套知识内容。

```text
source_entities       完整源实体，仅供审计和追溯
relations             四库关系，仅供离线装配和审计
evidence_documents    提示词证据白名单
evidence_fts          只索引 evidence_documents
source_files          源文件 SHA256
index_metadata        Schema 和构建信息
```

源 YAML 变化后，`CodeRAGService.validate_index()` 会返回索引失效，必须显式重建。

## 8. 当前运行边界

只有以下数据流属于当前系统的运行时 RAG：

```text
用户问题或评审问题
  -> CodeRAGService 检索规范证据
  -> 同时读取 output_dir 中的项目成果
  -> 将问题、成果证据和规范证据加入 LLM 上下文
  -> 生成回答并校验证据引用
```

当前实际接入状态：

| 环节 | 是否调用 RAG | 运行语义 |
| --- | --- | --- |
| 初始布跨与布跨修正 | 否 | 使用既有工程规则、few-shot 与确定性碰撞校核 |
| 结构尺寸设计 | 否 | 使用设计任务规则和尺寸 few-shot 生成候选 |
| 盖梁—墩柱联合配筋 | 否 | 使用配筋任务规则、盖梁/墩柱 few-shot 和确定性内力摘要生成候选 |
| 荷载计算与 OpenSees 分析 | 否 | 固定工具脚本直接计算 |
| 盖梁承载力与墩柱轴压验算 | 否 | 固定 Python 公式注册表执行并记录公式 ID |
| 确定性返修路由 | 否 | 根据验算字段和失败类型选择尺寸或配筋返修 |
| 最终评估与用户问答 | 是 | 检索三套规范和项目成果，由 LLM 生成带引用回答 |

最终评估与问答位于 `bridge_agents/design_review.py`。尺寸与配筋运行器已经停止调用 `bridge_agents/structural_evidence.py`；该模块暂留用于历史追溯和后续清理。

从规范 YAML 中整理公式、系数、表格和适用条件，再写成固定 Python 执行器，属于离线规范知识结构化。运行时计算不进行语义检索。可执行公式只来自 `bridge_agents/code_formula_registry.py` 中经过固定输入映射和单元测试的 Python 函数，YAML 公式字符串不会被动态求值。

`code_rag.enabled=false` 只关闭评估和问答中的规范检索。`enabled=true` 时若索引不可用，评估/问答显式报错；若 `retrieval_status=no_valid_evidence`，回答必须说明当前规范库没有找到有效依据。

## 9. 评测边界

`data/code_rag/evaluation_queries.yaml` 当前有 30 条内容级检索问题，覆盖设桥布跨、布跨修正、结构尺寸、配筋设计、建模和结构验算六个阶段。

- 正样本使用 `required_evidence` 指定必需证据，并用 `must_contain` 验证返回的 `prompt_text` 确实包含可用条文、公式、数值或表格字段。
- 负样本使用 `expected_status: no_valid_evidence`，验证标题条目、空规则或主观生成内容不会被当成规范依据返回。
- 当前固定基线为 30/30 通过，查询通过率、必需证据召回率和内容断言通过率均为 100%。

当前三路真实评测结果如下：

| 模式 | 通过问题 | 必需证据召回率 | 内容断言通过率 |
| --- | ---: | ---: | ---: |
| `keyword_only` | 30/30 | 100% | 100% |
| `vector_only` | 30/30 | 100% | 100% |
| `hybrid` | 30/30 | 100% | 100% |

三种模式使用同一评测集并生成独立报告，不能用混合结果掩盖单路召回退化。当前问题与知识库措辞仍较接近，后续接入 Agent 前需要增加真实设计提问、同义改写、歧义问题和跨规范耦合问题，单独验证开放问法的泛化能力。

## 10. 验证

```cmd
.\.venv\python.exe -m pytest tests/code_rag
.\.venv\python.exe -m compileall bridge_agents/code_rag tests/code_rag
```
