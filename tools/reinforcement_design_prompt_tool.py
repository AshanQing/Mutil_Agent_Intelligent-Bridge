from __future__ import annotations

import json
import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from bridge_agents.column_sample import (
    inspect_column_samples,
    load_column_samples,
    select_column_samples,
)
from bridge_agents.prompt_registry import render_prompt


# ============================================================
# 数据读取与通用工具
# ============================================================
def load_yaml_file(path: Union[str, Path]) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML文件不存在: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def read_text_file(path: Union[str, Path]) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文本文件不存在: {path}")
    return path.read_text(encoding="utf-8-sig")


def load_json_file(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {path}")
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"JSON文件为空: {path}")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"JSON文件顶层必须是对象: {path}")
    return data


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if text.lower() in {"none", "null", "nan", "", "-"}:
            return default
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not nums:
            return default
        return float(nums[0])
    except Exception:
        return default


def normalize_role(role: Any) -> str:
    text = str(role or "").strip()
    mapping = {
        "桥台": "桥台",
        "边墩": "边墩",
        "中间墩": "中间墩",
        "中墩": "中间墩",
        "联接边墩": "边墩",
        "连接边墩": "边墩",
    }
    return mapping.get(text, text)


def infer_system_type_from_bridge_type(bridge_type: Any) -> str:
    text = str(bridge_type or "").strip()
    normalized = (
        text.replace("预应力混凝土", "预应力砼")
        .replace("混凝土", "砼")
        .replace("连续T构", "连续T梁")
        .replace("先简支后连续体系", "先简支后连续")
    )
    if "先简支后连续" in normalized:
        return "先简支后连续"
    if "简支" in normalized and "连续" not in normalized:
        return "简支"
    if "连续T梁" in normalized or "连续" in normalized:
        return "先简支后连续"
    if "T梁" in normalized:
        return "简支"
    return normalized or "未知结构类型"


def infer_representative_span_length(span_combo: Any) -> Optional[float]:
    if not span_combo:
        return None
    text = str(span_combo).replace("×", "x").replace("X", "x")
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return None
    values = [float(n) for n in nums]
    span_candidates = [v for v in values if v >= 10]
    return max(span_candidates) if span_candidates else None


def safe_filename(text: Any) -> str:
    value = str(text or "unknown")
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fa5]+", "_", value).strip("_") or "unknown"


def to_pretty_yaml(data: Any) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, indent=2)


# ============================================================
# 从尺寸设计结果中提取配筋任务
# ============================================================
def _extract_dimension_unit_items(dimension_design_result: Any) -> List[Dict[str, Any]]:
    """兼容 dimension_design_tool 的输出结构。"""
    if not dimension_design_result:
        return []

    if isinstance(dimension_design_result, dict):
        if isinstance(dimension_design_result.get("dimension_design_result"), dict):
            return _extract_dimension_unit_items(dimension_design_result["dimension_design_result"])

        root = dimension_design_result.get("任务2_下部结构尺寸设计结果", dimension_design_result)
        if isinstance(root, dict) and isinstance(root.get("单元原始结果"), list):
            return [item for item in root["单元原始结果"] if isinstance(item, dict)]

    return []


def extract_reinforcement_tasks_from_dimension_result(
    dimension_design_result: Dict[str, Any],
    default_cover_mm: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """将尺寸设计结果转化为“按尺寸分组”的盖梁配筋任务列表。"""
    default_cover_mm = default_cover_mm or {}
    cover_x = default_cover_mm.get("cover_x", 50.0)
    cover_y = default_cover_mm.get("cover_y", 50.0)
    cover_z = default_cover_mm.get("cover_z", 60.0)

    unit_items = _extract_dimension_unit_items(dimension_design_result)
    tasks: List[Dict[str, Any]] = []

    for unit_index, item in enumerate(unit_items, start=1):
        single_unit = item.get("single_unit_input", {}) or {}
        dim_result = item.get("dimension_result", {}) or {}

        bridge_id = single_unit.get("桥梁编号")
        unit_id = single_unit.get("单元编号") or f"unit_{unit_index}"
        joint_no = single_unit.get("联号")
        bridge_type = single_unit.get("桥型")
        system_type = infer_system_type_from_bridge_type(bridge_type)

        unit_info = single_unit.get("本联信息", {}) or {}
        deck_info = single_unit.get("桥面宽度信息", {}) or {}
        span_length = infer_representative_span_length(unit_info.get("跨径组合"))
        deck_width = safe_float(deck_info.get("起点宽度"))
        next_deck_width = safe_float(deck_info.get("终点宽度"), deck_width)

        group_results = dim_result.get("分组尺寸设计结果", []) or []
        if not isinstance(group_results, list):
            continue

        for group_index, group in enumerate(group_results, start=1):
            if not isinstance(group, dict):
                continue

            role = normalize_role(group.get("墩位角色") or group.get("墩位角色类型"))
            if not role:
                # 尝试从单联设计分组中反查角色。
                gid = group.get("分组编号")
                for src_group in single_unit.get("本联设计分组", []) or []:
                    if src_group.get("分组编号") == gid:
                        role = normalize_role(src_group.get("墩位角色"))
                        break

            if role == "桥台":
                continue

            t_beam = group.get("T梁布置", {}) or {}
            front = t_beam.get("前跨", {}) or {}
            rear = t_beam.get("后跨", {}) or {}
            cap = group.get("盖梁尺寸", {}) or {}
            column = group.get("墩柱尺寸", {}) or {}
            foundation = group.get("基础", {}) or {}

            def _edge(span_obj: Dict[str, Any]) -> Dict[str, Optional[float]]:
                edge = span_obj.get("左右两侧距盖梁边缘距离", {}) or {}
                return {
                    "left": safe_float(edge.get("左")),
                    "right": safe_float(edge.get("右")),
                }

            task_id = f"{bridge_id}-{unit_id}-{group.get('分组编号') or group_index}"
            task_input = {
                "基本信息": {
                    "drawing_id": safe_filename(task_id),
                    "remarks": f"由尺寸设计结果自动生成的盖梁配筋任务，桥梁={bridge_id}，单元={unit_id}，分组={group.get('分组编号')}",
                    "system_type": system_type,
                    "pier_role": role,
                    "pier_type": group.get("墩柱类型"),
                    "foundation_type": group.get("基础类型") or foundation.get("类型"),
                },
                "上部布置与支承条件": {
                    "span_length": span_length,
                    "deck_width": deck_width,
                    "next_deck_width": next_deck_width,
                    "deck_slope": None,
                    "is_double_support": bool(group.get("是否双排支座", False)),
                    "support_symmetry": None,
                    "front_span": {
                        "t_beam_count": front.get("T梁数量"),
                        "t_beam_spacing": safe_float(front.get("T梁间距")),
                        "support_center_distance": safe_float(front.get("支座中心距"), safe_float(front.get("T梁间距"))),
                        "support_edge_distance": _edge(front),
                    },
                    "rear_span": {
                        "t_beam_count": rear.get("T梁数量"),
                        "t_beam_spacing": safe_float(rear.get("T梁间距")),
                        "support_center_distance": safe_float(rear.get("支座中心距"), safe_float(rear.get("T梁间距"))),
                        "support_edge_distance": _edge(rear),
                    },
                },
                "盖梁几何信息": {
                    "cap_length": safe_float(cap.get("长度")),
                    "cap_width": safe_float(cap.get("宽度")),
                    "cap_height_mid": safe_float(cap.get("中高")),
                    "cap_height_end": safe_float(cap.get("端高")),
                    "cantilever_length": safe_float(cap.get("悬臂")),
                    "cap_slope": None,
                },
                "墩柱几何信息": {
                    "column_count": column.get("数量"),
                    "column_diameter": safe_float(column.get("直径")),
                    "column_spacing": safe_float(column.get("中心间距")),
                    "column_tie_beam": {
                        "exists": False,
                        "width": None,
                        "height": None,
                    },
                },
                "基础信息": {
                    "pile_diameter": safe_float(foundation.get("桩基直径")),
                    "lower_tie_beam": {
                        "exists": False,
                        "width": None,
                        "height": None,
                    },
                },
                "保护层与控制参数": {
                    "cover_x": cover_x,
                    "cover_y": cover_y,
                    "cover_z": cover_z,
                },
            }

            tasks.append({
                "task_id": task_id,
                "桥梁编号": bridge_id,
                "单元编号": unit_id,
                "联号": joint_no,
                "分组编号": group.get("分组编号"),
                "包含桥墩号列表": group.get("包含桥墩号列表", []),
                "墩位角色": role,
                "桥墩尺寸信息": task_input,
                "source_dimension_group": group,
                "source_single_unit_input": single_unit,
            })

    return tasks


# ============================================================
# 示例样本选择与转换
# ============================================================
def _extract_samples(samples_data: Any) -> List[Dict[str, Any]]:
    """从配筋样本库中提取样本列表。

    兼容三种格式：
    1. 顶层直接为 list；
    2. 顶层 dict 中包含 samples / 示例样本 / 配筋样本 / reinforcement_samples；
    3. 顶层为 示例1、示例2、sample_1、case_1 等编号键，每个值为一个样本对象。
    """
    if isinstance(samples_data, list):
        return [s for s in samples_data if isinstance(s, dict)]

    if isinstance(samples_data, dict):
        for key in ["samples", "示例样本", "配筋样本", "reinforcement_samples"]:
            value = samples_data.get(key)
            if isinstance(value, list):
                return [s for s in value if isinstance(s, dict)]
            if isinstance(value, dict):
                return [v for v in value.values() if isinstance(v, dict)]

        numbered_samples: List[Dict[str, Any]] = []
        for key, value in samples_data.items():
            if not isinstance(value, dict):
                continue
            key_text = str(key).strip().lower()
            looks_like_sample_key = (
                key_text.startswith("示例")
                or key_text.startswith("样本")
                or key_text.startswith("sample")
                or key_text.startswith("case")
            )
            has_sample_fields = (
                "输入" in value
                or "input" in value
                or "桥墩尺寸信息" in value
                or "输出" in value
                or "output" in value
                or "配筋结果" in value
                or "reinforcement" in value
            )
            if looks_like_sample_key and has_sample_fields:
                sample = dict(value)
                sample.setdefault("sample_id", key)
                numbered_samples.append(sample)

        if numbered_samples:
            return numbered_samples

        value_samples = [
            dict(v)
            for v in samples_data.values()
            if isinstance(v, dict) and (
                "输入" in v
                or "input" in v
                or "输出" in v
                or "output" in v
                or "reinforcement" in v
            )
        ]
        return value_samples

    return []


def _sample_input(sample: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(sample.get("输入"), dict):
        return sample["输入"]
    if isinstance(sample.get("input"), dict):
        return sample["input"]
    if isinstance(sample.get("桥墩尺寸信息"), dict):
        return sample["桥墩尺寸信息"]
    return sample


def _sample_output(sample: Dict[str, Any]) -> Any:
    for key in ["输出", "output", "配筋结果", "reinforcement", "reinforcement_result"]:
        if key in sample:
            return sample[key]
    # 若样本本身就是完整配筋对象，则保留。
    if "pier_cap" in sample or "z_patterns" in sample or "skeleton_definitions" in sample:
        return sample
    return sample.get("result", {})


def _get_nested(data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    obj: Any = data
    for key in path:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key)
    return obj if obj is not None else default


def _score_sample(sample: Dict[str, Any], task_input: Dict[str, Any]) -> float:
    sin = _sample_input(sample)
    t_basic = task_input.get("基本信息", {}) or {}
    t_upper = task_input.get("上部布置与支承条件", {}) or {}
    t_cap = task_input.get("盖梁几何信息", {}) or {}
    t_col = task_input.get("墩柱几何信息", {}) or {}

    s_basic = sin.get("基本信息", sin) if isinstance(sin, dict) else {}
    s_upper = sin.get("上部布置与支承条件", sin) if isinstance(sin, dict) else {}
    s_cap = sin.get("盖梁几何信息", sin) if isinstance(sin, dict) else {}
    s_col = sin.get("墩柱几何信息", sin) if isinstance(sin, dict) else {}

    score = 0.0

    # 角色和体系优先匹配。
    t_role = normalize_role(t_basic.get("pier_role") or t_basic.get("墩位角色"))
    s_role = normalize_role(s_basic.get("pier_role") or s_basic.get("墩位角色") or sin.get("pier role"))
    if t_role and s_role and t_role != s_role:
        score += 100.0

    t_sys = str(t_basic.get("system_type") or "").strip()
    s_sys = str(s_basic.get("system_type") or sin.get("system_type") or "").strip()
    if t_sys and s_sys and t_sys != s_sys:
        score += 20.0

    numeric_pairs = [
        (safe_float(t_upper.get("deck_width")), safe_float(s_upper.get("deck_width") or sin.get("deck_width")), 1.0),
        (safe_float(t_upper.get("next_deck_width")), safe_float(s_upper.get("next_deck_width") or sin.get("next_deck_width")), 0.5),
        (safe_float(t_upper.get("span_length")), safe_float(s_upper.get("span_length") or sin.get("span_length")), 0.3),
        (safe_float(t_cap.get("cap_length")), safe_float(s_cap.get("cap_length") or sin.get("cap_length")), 1.0),
        (safe_float(t_cap.get("cap_width")), safe_float(s_cap.get("cap_width") or sin.get("cap_width")), 2.0),
        (safe_float(t_cap.get("cap_height_mid")), safe_float(s_cap.get("cap_height_mid") or sin.get("cap_height_mid")), 2.0),
        (safe_float(t_col.get("column_diameter")), safe_float(s_col.get("column_diameter") or sin.get("column_diameter")), 2.0),
        (safe_float(t_col.get("column_spacing")), safe_float(s_col.get("column_spacing") or sin.get("column_spacing")), 1.0),
    ]
    for tv, sv, weight in numeric_pairs:
        if tv is not None and sv is not None:
            score += weight * abs(tv - sv)

    return score


def match_reinforcement_samples(
    samples: List[Dict[str, Any]],
    task_input: Dict[str, Any],
    max_count: int = 2,
) -> List[Dict[str, Any]]:
    if not samples:
        return []
    ranked = sorted(samples, key=lambda s: _score_sample(s, task_input))
    return ranked[:max_count]


def make_reinforcement_sample_object(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id") or sample.get("draw_id") or sample.get("drawing_id") or sample.get("id"),
        "输入": _sample_input(sample),
        "输出": _sample_output(sample),
    }


# ============================================================
# Prompt 组装
# ============================================================
def load_fixed_reinforcement_examples(
    samples_yaml_path: Optional[Union[str, Path]],
    fixed_count: int = 2,
) -> Dict[str, Any]:
    """读取固定配筋示例。

    当前配筋样本数量较少，不进行相似度筛选，直接取样本文件中的前 fixed_count 个样本。
    若样本文件暂时不是严格 YAML，也不阻断流程，而是把原文作为固定案例文本拼入 Prompt。
    """
    if not samples_yaml_path:
        return {"sample_objects": [], "raw_examples_text": "", "load_mode": "none"}

    path = Path(samples_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"配筋样本文件不存在: {path}")

    raw_text = path.read_text(encoding="utf-8-sig")

    try:
        samples = _extract_samples(yaml.safe_load(raw_text))
    except Exception:
        # 样本中如果包含大量未加引号的冒号说明文字，YAML 解析可能失败。
        # 配筋阶段只需要固定案例参考时，直接拼接原文比强行解析更稳。
        return {
            "sample_objects": [],
            "raw_examples_text": raw_text.strip(),
            "load_mode": "raw_text_fallback",
        }

    fixed_samples = samples[:max(0, int(fixed_count or 0))]
    return {
        "sample_objects": [make_reinforcement_sample_object(s) for s in fixed_samples],
        "raw_examples_text": "",
        "load_mode": "structured_fixed",
    }


def _samples_to_prompt_text(sample_objects: List[Dict[str, Any]], raw_examples_text: str = "") -> str:
    if sample_objects:
        blocks: List[str] = []
        for idx, sample in enumerate(sample_objects, start=1):
            blocks.append(f"示例样本_{idx}:")
            blocks.append(to_pretty_yaml(sample).rstrip())
        return "\n".join(blocks)
    if raw_examples_text:
        return raw_examples_text.strip()
    return "示例样本_1:\n  输入: {}\n  输出: {}\n示例样本_2:\n  输入: {}\n  输出: {}"


def _build_column_examples_text(
    task_input: Dict[str, Any],
    column_samples_yaml_path: Optional[Union[str, Path]],
    max_examples: int,
) -> str:
    """加载、质检并近邻选择墩柱样本，渲染为 Prompt 文本。"""
    if not column_samples_yaml_path or not Path(column_samples_yaml_path).exists():
        return ""
    try:
        raw_samples = load_column_samples(str(column_samples_yaml_path))
        valid_samples = inspect_column_samples(raw_samples)["valid"]
        target = {
            "基本信息": task_input.get("基本信息", {}),
            "墩柱几何信息": task_input.get("墩柱几何信息", {}),
        }
        selected = select_column_samples(valid_samples, target, max_count=max_examples)
        return _samples_to_prompt_text(
            [make_reinforcement_sample_object(s) for s in selected]
        )
    except Exception:
        return ""


def _build_current_input_yaml(
    reinforcement_task: Dict[str, Any],
    force_control_info: Optional[Dict[str, Any]] = None,
) -> str:
    task_input = reinforcement_task.get("桥墩尺寸信息") or reinforcement_task
    payload = {
        "桥墩尺寸信息": task_input,
        "墩柱净高信息": reinforcement_task.get("墩柱净高信息"),
        "内力控制信息": (force_control_info or {}).get("内力控制信息", force_control_info or {}),
        "当前任务元信息": {
            "task_id": reinforcement_task.get("task_id"),
            "桥梁编号": reinforcement_task.get("桥梁编号"),
            "单元编号": reinforcement_task.get("单元编号"),
            "联号": reinforcement_task.get("联号"),
            "分组编号": reinforcement_task.get("分组编号"),
            "包含桥墩号列表": reinforcement_task.get("包含桥墩号列表", []),
            "墩位角色": reinforcement_task.get("墩位角色"),
        },
    }
    return to_pretty_yaml(payload).rstrip()


def build_reinforcement_design_prompt_tool(
    reinforcement_task: Dict[str, Any],
    template_yaml_path: Optional[Union[str, Path]] = None,
    samples_yaml_path: Optional[Union[str, Path]] = None,
    column_samples_yaml_path: Optional[Union[str, Path]] = None,
    max_examples: int = 2,
    force_control_info: Optional[Dict[str, Any]] = None,
    config_path: str = "config/settings.yaml",
    prompt_id: str = "tasks.cap_reinforcement_design.v1",
    evidence_context: Any = None,
) -> Dict[str, Any]:
    """组装单个尺寸分组的盖梁配筋设计 Prompt。

    当前版本使用文本 Prompt 模板，并将“尺寸信息 + 内力控制信息 + 固定配筋示例”共同写入 Prompt。
    参数名 template_yaml_path 保持兼容旧调用，实际可传入 .txt 模板路径。
    """
    # 兼容旧调用签名；配筋设计 Prompt 不再接受运行时规范检索内容。
    _ = evidence_context
    task_input = reinforcement_task.get("桥墩尺寸信息") or reinforcement_task
    fixed_payload = load_fixed_reinforcement_examples(
        samples_yaml_path=samples_yaml_path,
        fixed_count=max_examples,
    )
    fixed_objects = fixed_payload.get("sample_objects", [])
    raw_examples_text = fixed_payload.get("raw_examples_text", "")
    load_mode = fixed_payload.get("load_mode", "none")

    examples_text = _samples_to_prompt_text(fixed_objects, raw_examples_text)
    column_examples_text = _build_column_examples_text(
        task_input=task_input,
        column_samples_yaml_path=column_samples_yaml_path,
        max_examples=max_examples,
    )
    current_input_yaml = _build_current_input_yaml(
        reinforcement_task=reinforcement_task,
        force_control_info=force_control_info,
    )
    if template_yaml_path:
        template_text = read_text_file(template_yaml_path)
        template_source = str(template_yaml_path)
        rendered = None
        template_sha256 = hashlib.sha256(Path(template_yaml_path).read_bytes()).hexdigest()
        llm_prompt_text = (
            template_text
            .replace("{{REINFORCEMENT_EXAMPLES}}", examples_text)
            .replace("{{COLUMN_REINFORCEMENT_EXAMPLES}}", column_examples_text)
            .replace("{{CURRENT_REINFORCEMENT_TASK_INPUT}}", current_input_yaml)
            .replace("{{CODE_EVIDENCE}}", "")
        )
    else:
        rendered = render_prompt(
            prompt_id,
            {
                "REINFORCEMENT_EXAMPLES": examples_text,
                "COLUMN_REINFORCEMENT_EXAMPLES": column_examples_text,
                "CURRENT_REINFORCEMENT_TASK_INPUT": current_input_yaml,
            },
            config_path=config_path,
        )
        template_source = rendered.template_path
        template_sha256 = rendered.template_sha256
        llm_prompt_text = rendered.user_content or rendered.system_content

    prompt_yaml = {
        "template_path": template_source,
        "prompt_id": rendered.prompt_id if rendered else f"external:{Path(template_source).name}",
        "prompt_version": rendered.version if rendered else "external",
        "template_sha256": template_sha256,
        "样本使用方式": "固定案例拼接，不进行相似度检索或排序。",
        "样本读取诊断": {
            "sample_load_mode": load_mode,
            "matched_sample_count": len(fixed_objects),
            "samples_yaml_path": str(samples_yaml_path) if samples_yaml_path else None,
        },
        "示例样本集": fixed_objects,
        "输入": {
            "桥墩尺寸信息": task_input,
            "内力控制信息": (force_control_info or {}).get("内力控制信息", force_control_info or {}),
        },
        "当前任务元信息": {
            "task_id": reinforcement_task.get("task_id"),
            "桥梁编号": reinforcement_task.get("桥梁编号"),
            "单元编号": reinforcement_task.get("单元编号"),
            "联号": reinforcement_task.get("联号"),
            "分组编号": reinforcement_task.get("分组编号"),
            "包含桥墩号列表": reinforcement_task.get("包含桥墩号列表", []),
            "墩位角色": reinforcement_task.get("墩位角色"),
        },
    }
    return {
        "success": True,
        "prompt_yaml": prompt_yaml,
        "llm_prompt_text": llm_prompt_text,
        "matched_sample_count": len(fixed_objects),
        "matched_samples": fixed_objects,
        "sample_selection_mode": "fixed_concat",
        "sample_load_mode": load_mode,
        "force_control_info_available": bool(force_control_info),
        "prompt_id": rendered.prompt_id if rendered else f"external:{Path(template_source).name}",
        "prompt_version": rendered.version if rendered else "external",
        "template_sha256": template_sha256,
    }

    llm_prompt_text = (
        to_pretty_yaml(prompt)
        + "\n请严格按照模板中“输出格式”返回最终 YAML 结构。"
        + "只输出 YAML，不要输出解释、分析过程或 Markdown 代码块。"
    )

    return {
        "success": True,
        "prompt_yaml": prompt,
        "llm_prompt_text": llm_prompt_text,
        "matched_sample_count": len(fixed_objects),
        "matched_samples": fixed_objects,
        "sample_selection_mode": "fixed_concat",
        "sample_load_mode": load_mode,
    }
