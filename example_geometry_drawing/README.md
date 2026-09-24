# 桥梁主梁几何绘图（geometry_drawing）

## 概述

本目录是一个**独立、完整的桥梁主梁二维几何生成与绘图模块**。输入一份符合参数符号约定的设计 Excel（`.xlsx`）或设计结果 JSON（`.json`），自动生成：

| 输出 | 内容 | 格式 |
|------|------|------|
| 横断面图元 | 端横隔板、等截面段、中横隔板（+ 可选墩顶截面）的 line / arc / region | JSON |
| 半联立面图元 | 半跨立面箱梁轮廓、箱室空腔、人洞空腔的 line / arc / region | JSON |
| 半联平面图元 | 半跨平面箱梁轮廓、箱室空腔的 line / arc / region | JSON |
| 合图 | 以上三视图平移到统一画布的完整布局 | JSON |
| SVG 预览图 | 每个视图（横断面/立面/平面）的可视化预览 | SVG |
| AutoCAD SCR | 分视图和整套的 AutoCAD 命令脚本 | SCR |

实现使用 **Python 标准库**，不依赖任何第三方包。

## 目录结构

```
geometry_drawing/
├── convert_geometry.py              ← 【唯一入口】修改参数后点击运行
├── requirements.txt                 # 空，无第三方依赖
├── README.md                        # 本文件
│
├── data/
│   ├── geometry_section_specs.json  # 横断面拓扑定义（4种断面类型）
│   └── geometry_config.json         # 全局配置（图层/线型/配色/SVG/CAD/几何参数）
│
├── main_girder_geometry/            # 核心包
│   ├── __init__.py
│   ├── geometry_models.py           # 数据模型：Point, Line, Arc, Region, SectionGeometry, …
│   ├── geometry_reader.py           # 输入：读取 .xlsx（Excel参数表）或 .json（设计结果）
│   ├── geometry_generator.py        # 横断面：生成关键断面几何（箱梁/实心横隔板+人洞）
│   ├── longitudinal_layout.py      # 纵向布局：解析0号块类型（加厚/过渡/有箱室）
│   ├── elevation_generator.py       # 立面：生成半联立面轮廓与空腔
│   ├── plan_generator.py           # 平面：生成半联平面轮廓与空腔
│   ├── geometry_preview.py          # SVG预览：渲染横断面/立面/平面的预览图
│   ├── cad_export.py               # CAD输出：生成AutoCAD .scr脚本 + 三视图合图平移
│   ├── geometry_config.py           # 配置加载：从JSON读取并类型化返回所有可配置项
│   └── __pycache__/                # 缓存目录
│
└── runs/                            # 【输出目录】所有运行结果
    ├── geometry_results/            # 示例结果（默认参数）
    ├── geometry_results_40+64+40/   # 示例结果（40+64+40三跨连续梁）
    └── geometry_results_2×88/       # 示例结果（2×88两跨连续梁）
```

## 数据流：从输入到输出的完整管线

```
  ┌──────────────┐
  │ 输入          │
  │ .xlsx 或 .json│
  └──────┬───────┘
         │
         ▼
  ┌──────────────────┐
  │ geometry_reader   │  解析Excel参数表或设计JSON
  │ load_design_      │  → DesignParameters
  │ parameters()      │    (longitudinal, cross_section, global_parameters)
  └──────┬───────────┘
         │
    ┌────┴────────────┬──────────────┐
    ▼                 ▼              ▼
┌──────────┐   ┌────────────┐  ┌──────────┐
│geometry_ │   │elevation_  │  │plan_     │
│generator │   │generator   │  │generator │
│          │   │            │  │          │
│横断面几何│   │立面几何    │  │平面几何  │
│生成      │   │生成        │  │生成      │
└────┬─────┘   └─────┬──────┘  └────┬─────┘
     │               │              │
     ▼               ▼              ▼
┌──────────────────────────┐
│    三份分视图 JSON        │
│  cross_sections.json     │
│  elevation.json          │
│  plan.json               │
└────────┬─────────────────┘
         │
         │ build_combined_layout()（cad_export.py）
         │ 将三视图平移到统一画布
         ▼
┌─────────────────────────────────────┐
│  complete_girder_layout.json (合图)  │
└────────┬────────────────────────────┘
         │
    ┌────┴────────────┐
    ▼                 ▼
┌──────────┐   ┌──────────────┐
│geometry_ │   │cad_export    │
│preview   │   │              │
│          │   │              │
│SVG预览图 │   │AutoCAD .scr  │
│.svg      │   │命令脚本       │
└──────────┘   └──────────────┘
```

## 输入格式

### Excel 参数表（`.xlsx`）

程序直接读取 `.xlsx` 工作簿的特定工作表，解析参数符号-数值对：

| 工作表 | 符号列 | 数值列 | 对应参数分区 |
|--------|--------|--------|-------------|
| `纵向参数` | E 列 | G 列 | `longitudinal` |
| `横断面参数` | E 列 | G 列 | `cross_section` |
| `增补全局参数-主要的QA(Q PART)` | C 列 | D 列 | `global_parameters`（含 `span_layout` 等） |

参数符号遵循约定命名（如 `H_pier` = 墩顶梁高, `B_top` = 顶板宽度, `t_t_std` = 标准段顶板厚度 等）。支持 Excel 内嵌字符串、数值、布尔值和共享字符串。自动检测 Excel 错误标记（`#DIV/0!`, `#N/A` 等）并报告。

### 设计结果 JSON（`.json`）

也可直接传入上游设计任务输出的 JSON，程序读取 `output.longitudinal`、`output.cross_section` 和 `input` 三个分区。

## 断面拓扑类型

`data/geometry_section_specs.json` 定义了4种关键横断面的拓扑结构和所需参数：

| 断面 ID | 拓扑 | 说明 |
|---------|------|------|
| `END-DIAPHRAGM` | `solid_diaphragm_with_manhole` | 端横隔板：实心截面 + 人洞 |
| `CONSTANT-SECTION` | `box_girder` | 等截面段：标准箱梁截面（单箱单室） |
| `MID-DIAPHRAGM` | `solid_diaphragm_with_manhole` | 中横隔板：实心截面 + 人洞 |
| `PIER-SECTION` | `box_girder` | 墩顶截面：有箱室0号块箱梁（仅当 `n_box_0_side > 0` 时生成） |

每种断面通过配置映射到对应的纵向/横断面参数名，实现参数驱动的几何生成。

## 0号块（支座段）的三种布局模式

立面生成根据纵向参数自动判定0号块类型：

| 类型 | 标识 | 特征 |
|------|------|------|
| **加厚型** | `thickened` | 横墙 + 顶/底/腹板渐变加厚段 + 过渡段 |
| **倒角过渡型** | `transition` | 横墙 + 顶板倒角 + 过渡段（无加厚渐变） |
| **有箱室型** | `boxed` | 箱室 + 外侧横墙 + 顶/底/腹板加厚段（仅当 `n_box_0_side > 0`） |

## 输出文件详解

### JSON 几何文件

所有 JSON 文件共享统一的 schema（`schema_version: "2.0.0"`，坐标单位 `mm`，角度单位 `degree`）。

#### 单视图 JSON（`cross_sections.json` / `elevation.json` / `plan.json`）

```json
{
  "schema_version": "2.0.0",
  "units": "mm",
  "angle_unit": "degree",
  "source": { "type": "xlsx", "path": "..." },
  "views": {
    "cross_section": {           // 或 "elevation" / "plan"
      "sections": [
        {
          "id": "END-DIAPHRAGM",  // 断面标识
          "entities": {
            "lines":  [{ "id":"...", "layer":"XS-OUTLINE", "start":{"x":…,"y":…}, "end":{…} }, …],
            "arcs":   [{ "id":"...", "layer":"XS-OUTLINE", "center":{…}, "radius":…, "start_angle_deg":…, "end_angle_deg":…, "clockwise":… }, …],
            "regions":[{ "id":"...", "layer":"XS-CONCRETE", "outer":[{ "entity_id":"...", "direction":"forward" }, …], "holes":[[{…}, …]] }]
          }
        }, …
      ]
    }
  }
}
```

**实体类型说明：**

- **Line（直线）**：含 `id`、`layer`、`start`/`end` 坐标、`linetype`
- **Arc（圆弧）**：含 `id`、`layer`、`center`、`radius`、`start_angle_deg`/`end_angle_deg`、`clockwise`、`linetype`
- **Region（封闭面域）**：含 `id`、`layer`、`outer`（外轮廓引用序列）、`holes`（孔洞引用序列）。每个引用含 `entity_id`（指向同 section 内的 line/arc）和 `direction`（`"forward"` 顺定义方向 / `"reverse"` 反向遍历）

**图层命名规则：**

| 前缀 | 视图 | 图层示例 |
|------|------|----------|
| `XS-` | 横断面 Cross Section | `XS-OUTLINE` 外轮廓, `XS-VOID` 箱室孔洞, `XS-MANHOLE` 人洞, `XS-CONCRETE` 混凝土面域 |
| `EL-` | 立面 Elevation | `EL-OUTLINE`, `EL-VOID`, `EL-CUT` 截断线, `EL-CONCRETE` |
| `PL-` | 平面 Plan | `PL-OUTLINE`, `PL-VOID`, `PL-CUT`, `PL-CONCRETE` |

#### 图层映射逻辑：从表格参数到点线图层

本节详细说明每个视图（横断面/立面/平面）中，设计参数如何驱动几何坐标的生成，以及每个几何边如何被赋予具体的图层标签。

##### 1. 参数 → 坐标点

以立面为例，`elevation_generator.py` 的 `generate_elevation()` 函数流程：

```
DesignParameters
  ├── span_layout → 计算 X 轴桩号序列
  ├── H_end, H_pier, profile_top_L/R → 逐桩插值梁高
  ├── t_t_std, t_b_std, t_t_var_seq, t_b_var_seq → 逐桩插值顶/底板厚
  └── 0号块类型判定（加厚/倒角过渡/有箱室）→ 调整局部厚度
        ↓
  generate_elevation_stations() → list[LongitudinalStation]
        ↓
  boundary_function(stations, **args) → list[Point]  （upper_void / lower_void）
```

横断面和平面流程类似，各自有独立的 station 生成和 boundary 函数。

##### 2. 坐标点 → 图层标签

图层是**边的属性**，不是点的属性。每个 builder 的 `add_polygon(points, layers)` 要求 `points` 和 `layers` 等长——第 `i` 条边（从 `points[i]` 到 `points[(i+1)%n]`）使用 `layers[i]`。

```python
# _ElevationBuilder.add_polygon() — elevation_generator.py:74
def add_polygon(self, points, layers):
    for i, start in enumerate(points):
        self.add_line(start, points[(i+1) % len(points)], layers[i])
```

`add_line()` 调用 `get_layer_style(layer).linetype` 解析线型，连同图层名一起存入 `Line` 对象：

```python
Line(id="el_half_girder_elevation_line_001",
     start=Point(0, 0), end=Point(1000, 500),
     layer="EL-CUT", linetype="BYLAYER")
```

##### 3. 各视图的图层分布

**横断面**（`geometry_generator.py`）—— 单图层循环：

横断面不涉及半桥截断，所有边使用统一的 `add_line_loop(points, layer)` 方法，整个环路共用一个图层名。图层分布简单：

| 区域 | 图层 | 边数 |
|------|------|------|
| 箱梁外轮廓 | `XS-OUTLINE` | ~11 条（含顶板倒角 profile 段） |
| 箱室空洞 | `XS-VOID` | 8 条（闭合环路） |
| 人洞轮廓 | `XS-MANHOLE` | 8 条（闭合环路） |
| 人洞辅助线 | `XS-MANHOLE-REF` | 4 条（对角线） |
| 混凝土面域 | `XS-CONCRETE` | Region（outer=外轮廓, hole=箱室+人洞） |

##### 图层决策依据：点按几何角色分组 → 组内边共享图层

程序判断"哪条边属于哪个图层"的根源在于：**用不同类别的 Excel 参数计算不同几何角色的 Y 坐标**。

每个桩号 `x` 处的 `LongitudinalStation` 携带 4 个 Y 值，它们由不同类别的参数算出：

```python
# elevation_generator.py — LongitudinalStation 的 4 个 Y 属性
# 坐标系：y=0 = 梁顶面

outer_bottom_y = -height                           # 梁底外轮廓
                  └── 用梁高参数：H_end, H_pier, profile_top_L/R, profile_bottom_L/R

void_top_y     = -top_thickness                    # 上部空洞下边界（顶板底面）
                  └── 用顶板厚参数：t_t_std, t_t_var_seq, t_t_var_delta

void_bottom_y  = -height + bottom_thickness        # 下部空洞上边界（底板顶面）
                  └── 用底板厚参数：t_b_std, t_b_var_seq, t_b_var_delta

# outer_top_y 未显式定义 —— y=0 就是梁顶，不需要计算
```

也就是说，同一个 `x` 坐标处：

| Y 坐标 | 几何含义 | 依赖的 Excel 参数类别 | → 图层 |
|--------|---------|----------------------|--------|
| `0` | 梁顶面 | （无需参数） | `EL-OUTLINE` |
| `-top_thickness` | 顶板底面 = 空洞上边界 | **顶板厚度参数** | `EL-VOID` |
| `-height + bottom_thickness` | 底板顶面 = 空洞下边界 | **底板厚度参数** | `EL-VOID` |
| `-height` | 梁底面 | **梁高参数** | `EL-OUTLINE` |

**第一步：算点，按几何角色分组**

`boundary_function` 内部调用 `boundary_y(station)`，它返回 `void_top_y` 或 `void_bottom_y`——也就是**专门取空洞边界 Y 值**。因此它返回的点数组天然就是 void 边界：

```python
# elevation_generator.py — 立面的点数组
upper_void  = boundary_function(stations, use_top=True,  ...)   # 用 void_top_y → 上部空洞边界
lower_void  = boundary_function(stations, use_top=False, ...)   # 用 void_bottom_y → 下部空洞边界
outer_bottom = [Point(item.x, item.outer_bottom_y) for ...]      # 用 outer_bottom_y → 底部外轮廓
upper_void  = boundary_function(stations, use_top=True,  ...)   # 上部 void 边界点
lower_void  = boundary_function(stations, use_top=False, ...)   # 下部 void 边界点
outer_bottom = [Point(item.x, item.outer_bottom_y) for ...]      # 底部外轮廓点
```

| 点数组 | 几何含义 | 属于这个数组的边 → 对应图层 |
|--------|---------|---------------------------|
| `upper_void` | 上部箱室/人洞的上边界 | `EL-VOID` |
| `lower_void` | 下部箱室/人洞的下边界 | `EL-VOID` |
| `outer_bottom` | 梁底外轮廓线 | `EL-OUTLINE` |
| （单点）`left_end` / `right_end` | 梁端 / 跨中端点 | 看它连接谁（见第二步） |

**第二步：拼接成多边形，组间过渡边手动指定图层**

各组点按固定顺序拼接为一个平列表，图层数组同步拼接：

```
以立面上部为例，假设 upper_void = [A, B, C, D, E]（5个点）：

upper_points = [left_end,  A,   B,   C,   D,   E,   right_end]    ← 7个点
                └─梁端──┘  └──── upper_void ────┘  └─跨中──┘

upper_layers = ["EL-OUTLINE", "EL-VOID","EL-VOID","EL-VOID","EL-VOID", "EL-CUT", "EL-OUTLINE"]
                 │             └─── upper_void组内4条边 ────┘         │         │
                 左过渡边                                              右过渡边   闭合边
                 left_end→A                                           E→right   right→left
```

| 边的类型 | 位置 | 图层 | 决定方式 |
|----------|------|------|----------|
| **组内边** | `upper_void` 内部相邻点之间 | `EL-VOID` | 由点所在数组决定：`*["EL-VOID"] * (len(upper_void)-1)` |
| **组内边** | `outer_bottom` 内部相邻点之间 | `EL-OUTLINE` | 由点所在数组决定：`*["EL-OUTLINE"] * (len(outer_bottom)-1)` |
| **过渡边** | 梁端单点 → void 数组首点 | `EL-OUTLINE` | 手动硬编码（梁的实际终点） |
| **过渡边** | void 数组末点 → 跨中单点 | `EL-CUT` | 手动硬编码（半桥截断处） |
| **闭合边** | 末点 → 首点（封闭多边形） | `EL-OUTLINE` | 手动硬编码（外侧轮廓） |

**第三步：`add_polygon` 逐边绑定**

```python
# 每个边从 layers[i] 取图层名，存入 Line 对象
def add_polygon(self, points, layers):
    for i, start in enumerate(points):
        self.add_line(start, points[(i+1) % len(points)], layers[i])
```

生成的 `Line` 对象：
```
Line(start=left_end, end=A,     layer="EL-OUTLINE")
Line(start=A,        end=B,     layer="EL-VOID")
Line(start=B,        end=C,     layer="EL-VOID")
...
Line(start=E,        end=right, layer="EL-CUT")
Line(start=right,    end=left,  layer="EL-OUTLINE")
```

**总结**：图层的"知识"来源于——点属于哪个几何数组（`upper_void` / `outer_bottom` / …），以及该点在数组中的位置（首/末/内部）。这些规则在代码中硬编码，与输入 Excel 的具体数值无关。设计师调整板厚、跨度后，只要点序列结构不变，图层就会自动正确分配。

**立面**（`elevation_generator.py`）—— 每条边独立指定图层：

**立面**（`elevation_generator.py`）—— 每条边独立指定图层：

半联立面取桥梁的一半（左端=梁端，右端=跨中截断处）。上部 void 多边形示例如下：

```
点序列:  [left_end(梁端),  upper_void[0],  upper_void[1],  ...,  upper_void[-1],  right_end(跨中)]
          │                │                                               │                │
边图层:   EL-OUTLINE       EL-VOID          EL-VOID          EL-VOID       EL-CUT           EL-OUTLINE
          (梁端竖边)        (void上边界)      (void上边界)      (void上边界)   (跨中截断边)      (底部闭合边)
```

下部 void 多边形同理——梁端侧 `EL-OUTLINE`，跨中侧 `EL-CUT`，底部外轮廓边 `EL-OUTLINE`，void 内部边 `EL-VOID`。

| 图层 | 出现位置 | 含义 |
|------|----------|------|
| `EL-OUTLINE` | 梁端竖边、底部闭合边 | 梁的实际外轮廓 |
| `EL-VOID` | 箱室/人洞上边界 | 空洞轮廓 |
| `EL-CUT` | 跨中侧竖边 | 取半桥的人工截断位置 |
| `EL-CONCRETE` | Region 填充 | 混凝土实体区域 |

> **注意**：`EL-CUT` 只出现在跨中侧（`right_end`），因为只有那里是人工截断。梁端侧（`left_end`）是梁的实际终点，使用 `EL-OUTLINE`。

**平面**（`plan_generator.py`）—— 与立面对称：

平面取桥梁的一半（左端=梁端，右端=跨中）。上部多边形：

```
点序列:  [upper_inner[0](梁端),  ...,  upper_inner[-1](跨中),  (right_end, half_width),  (left_end, half_width)]
          │                       │                          │                           │
边图层:   PL-VOID                  PL-VOID                    PL-CUT                      PL-OUTLINE
          (void内边界)              (void内边界)                (跨中截断边)                  (梁端竖边→闭合)
```

| 图层 | 出现位置 | 含义 |
|------|----------|------|
| `PL-OUTLINE` | 梁端竖边、外侧长边 | 梁的实际外轮廓 |
| `PL-VOID` | 箱室内边界 | 空洞轮廓 |
| `PL-CUT` | 跨中侧竖边 | 取半桥的人工截断位置 |
| `PL-CONCRETE` | Region 填充 | 混凝土实体区域 |

> **注意**：与立面相同，`PL-CUT` 只出现在跨中侧。

##### 4. 图层名 → 视觉样式

图层名是唯一的"钥匙"。所有视觉属性（线型、颜色、线宽、虚线样式）在输出阶段通过 `get_layer_style()` 统一查询：

```python
# geometry_config.py
def get_layer_style(layer: str) -> LayerStyle:
    # 1. 先查 data/geometry_config.json 的 "layers" 分区
    # 2. 未命中则回退 _DEFAULT_LAYERS（13个硬编码图层）
    # 3. 仍未命中则返回兜底值
```

查询优先级：`用户JSON配置 > 硬编码默认值 > 兜底值`

**SVG 输出**（`geometry_preview.py`）：
```python
style = get_layer_style(entity["layer"])
stroke = style.color        # → SVG stroke 属性
width  = style.lineweight   # → SVG stroke-width 属性
dash   = _DASH[layer]       # → SVG stroke-dasharray（独立查表）
```

**CAD SCR 输出**（`cad_export.py`）：
```
_.-LAYER _LT BYLAYER EL-CUT      ← 图层线型
_.-LAYER _LW 0.30 EL-CUT         ← 图层线宽
_.CECOLOR 96,125,139              ← 逐实体颜色（#607d8b → RGB）
```

颜色通过 `CECOLOR` 系统变量逐实体赋予而非 `-LAYER Color`，以确保跨 AutoCAD 版本的兼容性。Region 面域不在 SCR 中输出（它是逻辑结构，不是绘图命令）。

##### 5. 完整数据流

```
Excel (.xlsx)
  │  geometry_reader.py 解析 "纵向参数"/"横断面参数"/"增补全局参数" 三张Sheet
  ▼
DesignParameters {longitudinal, cross_section, global_parameters}
  │
  ├── generate_key_sections()    → 逐桩插值 → builder.add_line_loop() → XS-* 图层
  ├── generate_elevation()       → 逐桩插值 → builder.add_polygon()  → EL-* 图层
  └── generate_plan()            → 逐桩插值 → builder.add_polygon()  → PL-* 图层
  │
  ▼
SectionGeometry (含 Line/Arc/Region，每个实体携带 layer 字段)
  │  .to_dict() → JSON
  ├── geometry_preview.py   → get_layer_style(layer) → SVG
  └── cad_export.py         → get_layer_style(layer) → SCR
```

修改 `data/geometry_config.json` 中任意图层的颜色/线宽/线型即可全局改变该图层的视觉呈现，无需改动生成代码。

#### 合图 JSON（`complete_girder_layout.json`）

将横断面、立面、平面的全部实体平移到统一画布坐标系，按行排列（上中下三行：立面 / 平面 / 横断面），水平自动居中。坐标系原点为 `(0, 0)`，实体 `id` 加前缀以区分来源。

### SVG 预览图

默认生成的预览图（按 `PREVIEW_FILENAMES` 配置命名）：

| 文件 | 内容 |
|------|------|
| `01_end_diaphragm.svg` | 端横隔板横断面 |
| `02_constant_section.svg` | 等截面段横断面 |
| `03_mid_diaphragm.svg` | 中横隔板横断面 |
| `04_pier_section.svg` | 墩顶截面横断面（如有箱室0号块） |
| `04_half_girder_elevation.svg` | 半联立面 |
| `05_half_girder_plan.svg` | 半联平面 |

预览图特点：坐标原点标有绿色虚线十字；混凝土面域填充菱形花纹；外轮廓、孔洞、截断线颜色和线型不同；底部有图例。

### AutoCAD SCR 脚本

生成的是 AutoCAD 命令行脚本（`.scr`），可在 AutoCAD 中通过 `SCRIPT` 命令执行，自动绘制所有图层。文件组织：

```
cad/
├── complete-girder-layout.scr        ← 整套合图
├── cross_sections/
│   ├── end-diaphragm.scr
│   ├── constant-section.scr
│   ├── mid-diaphragm.scr
│   └── pier-section.scr（如有）
├── elevation/
│   └── half-girder-elevation.scr
└── plan/
    └── half-girder-plan.scr
```

SCR 脚本按图层分组输出 `LINE` 和 `ARC` 命令（region 不在 SCR 中输出，因为它是逻辑结构而非绘图命令），最后执行 `ZOOM E` 适配视图。

## 使用流程

### 1. 修改任务参数

打开 `convert_geometry.py`，修改顶部的任务参数：

```python
# 输入文件：Excel 参数表或设计结果 JSON
INPUT_FILE = TASK_ROOT.parent.parent / "单箱_无砟_连续_40+64+40.xlsx"
# 或使用绝对路径
# INPUT_FILE = Path(r"D:\项目资料\design_result.json")

# 输出目录
OUTPUT_DIR = TASK_ROOT / "runs" / "geometry_results_40+64+40"

# 是否生成 SVG 预览
CREATE_PREVIEW = True

# 是否生成 AutoCAD SCR 脚本
CREATE_CAD = True
```

### 2. 运行

在 VSCode 中打开 `convert_geometry.py`，点击 **运行 Python 文件**，或命令行：

```bash
python convert_geometry.py
```

### 3. 查看结果

程序终端输出会列出所有生成文件的路径和每个视图的图元统计（直线数、圆弧数、区域数）。

## 全局配置（`data/geometry_config.json`）

`data/geometry_config.json` 是**本模块唯一的全局配置文件**，集中管理图层样式、线型、配色、SVG 预览参数、CAD 导出设置和几何引擎常量。修改此文件即可自定义所有绘图参数，无需改动任何 Python 代码。

**加载机制：** `main_girder_geometry/geometry_config.py` 在首次访问时读取该 JSON 文件，解析为类型化的 dataclass 对象供各模块使用。若文件不存在或格式错误，程序会打印警告并回退到内置默认值（与当前行为完全一致），管线可零配置运行。

### 配置分区

| 分区 | 键 | 说明 |
|------|-----|------|
| **layers** | `"XS-OUTLINE"`, `"EL-VOID"`, … | 每个图层的线型、颜色、线宽、虚线样式 |
| **cad** | `"layout_gap_mm"`, `"default_linetype"`, … | CAD 合图间距、精度格式、线型命令输出 |
| **reader** | `"sheets"`, `"global_sheet"` | Excel 工作表名及符号/数值列映射 |

### 图层样式（`layers`）

每个图层可配置以下属性：

```json
"XS-OUTLINE": {
  "linetype":    "Continuous",   // AutoCAD 线型名：Continuous / Dashed / DASHDOT 等
  "color":       "#172033",      // 十六进制颜色（SVG 及 CAD 层颜色）
  "lineweight":  "0.50",         // CAD 线宽（mm）
  "description": "横断面外轮廓"   // 图层用途说明
}
```

**当前定义的 13 个图层：**

| 图层名 | 线型 | 颜色 | 线宽 | 用途 |
|--------|------|------|------|------|
| `XS-OUTLINE` | Continuous | #172033 | 0.50 | 横断面外轮廓 |
| `XS-VOID` | Dashed | #172033 | 0.35 | 横断面箱室孔洞 |
| `XS-MANHOLE` | Dashed | #172033 | 0.35 | 横断面人洞 |
| `XS-MANHOLE-REF` | Dashed | #d32f2f | 0.35 | 横断面人洞参考线 |
| `XS-CONCRETE` | Continuous | #8aa0b8 | 0.25 | 横断面混凝土面域 |
| `EL-OUTLINE` | Continuous | #172033 | 0.50 | 立面外轮廓 |
| `EL-VOID` | Dashed | #172033 | 0.35 | 立面箱室及人洞空腔 |
| `EL-CUT` | Dashed | #607d8b | 0.30 | 立面截断线 |
| `EL-CONCRETE` | Continuous | #8aa0b8 | 0.25 | 立面混凝土面域 |
| `PL-OUTLINE` | Continuous | #172033 | 0.50 | 平面外轮廓 |
| `PL-VOID` | Dashed | #172033 | 0.35 | 平面箱室空腔 |
| `PL-CUT` | Dashed | #607d8b | 0.30 | 平面截断线 |
| `PL-CONCRETE` | Continuous | #8aa0b8 | 0.25 | 平面混凝土面域 |

### CAD 输出配置（`cad`）

- `layout_gap_mm`：合图中三视图之间的间距（mm）
- `layout_origin_x` / `layout_origin_y`：合图整体在 CAD 画布中的原点坐标（mm），默认为 `(0, 0)`
- `format_precision`：坐标数值精度格式（`".12g"`）
- `output_linetype_commands`：设为 `true` 时 SCR 脚本会输出 `LINETYPE` 命令加载线型
- `default_linetype`：图层未明确指定线型时的默认值

### Reader 配置（`reader`）

定义 Excel 输入文件的各工作表名、参数符号所在列和数值所在列。若上游 Excel 模板有调整，只需修改此处分区即可适配。

## 运行环境

- Python 3.8+
- 无第三方依赖（仅使用 `json`, `math`, `pathlib`, `xml.etree`, `zipfile`, `re`, `copy`, `dataclasses` 等标准库）
