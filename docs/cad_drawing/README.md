# CAD 配筋成果说明

## 数据职责

联合配筋 YAML/JSON 保存设计真源；`drawing_ir.json` 保存确定性绘图表达；`.scr` 面向 AutoCAD 展示；SVG 面向快速预览；CSV 保存规则化钢筋表。SCR 和 SVG 均不参与结构计算。

系统按 `design_group_id` 生成成果。同一设计组中的多个桥墩共用一套尺寸、配筋、验算和图纸，适用桥墩保存在 `member_piers`，不会逐墩复制相同图纸。

## 自动输出

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

`drawing_index.json` 记录成员桥墩、发图状态、文件相对路径、输入哈希、输出哈希和几何校验结果。目录替换以单个设计组为边界，不删除整个 `deliverables`。

## AutoCAD 执行

1. 新建空白公制图形并进入模型空间；
2. 输入 `SCRIPT`；
3. 选择对应 `.scr`；
4. 等待脚本执行到 `ZOOM E`；
5. 检查命令行、图层、轮廓、钢筋、尺寸和文字；
6. 人工确认后再另行保存图形。

脚本使用国际化命令前缀，显式建立和切换图层，不包含 `QSAVE`、`SAVEAS` 或 `PLOT`。首版未使用原生 DIM 对象，尺寸语义在导出时确定性展开为线和文字，以减少语言、版本和 DIMSTYLE 差异。

SCR 采用无 BOM UTF-8 文本。图层建立、线宽和非连续线型分别结束各自的 `-LAYER` 会话；实体颜色通过 `CECOLOR` 的裸 RGB 值设置，避免 TrueColor 子提示和系统变量 `_BYLAYER` 值造成命令错位。`.scr` 不能通过 AutoCAD 的“打开图形”加载，必须在空白图形中执行 `SCRIPT`。

SVG 使用固定 `1600 × 1200` 预览画布、深色背景、按包围盒等比例缩放和非缩放线宽。尺寸线展开为界线、尺寸线和数值文字，引线同时显示标注文字；这些表达只服务于快速审阅。

## 状态标记

| issue_status | 图面标记 | 含义 |
| --- | --- | --- |
| `verified` | 无风险附注 | 当前自动验算通过 |
| `draft_unverified` | 自动验算未完成，仅供复核 | 仅由显式 `--allow-unverified` 生成 |
| `accepted_risk` | 含人工接受风险 | 人工授权流程结束，原验算风险仍保留 |

## 当前边界

- 图纸用于概念/初步设计展示；
- 尚未覆盖完整锚固、接头、钢筋下料长度和全部施工构造；
- SVG 只作视觉预览；
- 真实 AutoCAD 版本兼容性需要按 [验收记录模板](autocad_acceptance.md) 人工确认；
- 绘图过程不调用规范 RAG 或在线模型。
