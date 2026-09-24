from __future__ import annotations

import json
import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from bridge_agents.prompt_registry import render_prompt


# ============================================================
# 数据读取
# ============================================================
def load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")

    raw = json_path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"JSON文件为空: {json_path}")

    return json.loads(raw)


def load_yaml(yaml_path: Union[str, Path]) -> List[Dict[str, Any]]:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML文件不存在: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    samples = data if isinstance(data, list) else []

    # 兜底修复：有些样本把 pier role 缩进到了 rear_support_edge_distance 下。
    for s in samples:
        if isinstance(s, dict) and not s.get("pier role"):
            rear = s.get("rear_support_edge_distance")
            if isinstance(rear, dict) and "pier role" in rear:
                s["pier role"] = rear.get("pier role")

    return samples


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if text.lower() in {"none", "null", "nan", ""}:
            return default
        return float(text)
    except Exception:
        return default


def normalize_role(role: Any) -> str:
    if role is None:
        return ""
    text = str(role).strip()
    mapping = {
        "桥台": "桥台",
        "边墩": "边墩",
        "中间墩": "中间墩",
        "中墩": "中间墩",
        "联接边墩": "边墩",
        "连接边墩": "边墩",
        "过渡墩": "边墩",
    }
    return mapping.get(text, text)


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


# ============================================================
# 单联输入标准化
# ============================================================
def infer_representative_span_length(span_combo: str) -> Optional[float]:
    if not span_combo:
        return None

    text = str(span_combo).replace("×", "x").replace("X", "x")
    nums = re.findall(r"(\d+(?:\.\d+)?)", text)
    if not nums:
        return None

    values = [float(n) for n in nums]
    span_candidates = [v for v in values if v >= 10]
    return max(span_candidates) if span_candidates else None


def infer_system_type_from_bridge_type(bridge_type: Any) -> str:
    text = str(bridge_type or "").strip()
    normalized = (
        text.replace("预应力混凝土", "预应力砼")
        .replace("混凝土", "砼")
        .replace("连续T构", "连续T梁")
        .replace("先简支后连续体系", "先简支后连续")
    )

    if not normalized:
        raise ValueError("无法从桥型推断结构类型：桥型为空")

    if "先简支后连续" in normalized:
        return "先简支后连续"
    if "简支" in normalized and "连续" not in normalized:
        return "简支"
    if "连续T梁" in normalized or "连续" in normalized:
        return "先简支后连续"
    if "T梁" in normalized:
        return "简支"

    raise ValueError(f"无法将桥型映射为样本库 system_type：{text}")


def normalize_single_unit_input(single_unit_input: Dict[str, Any]) -> Dict[str, Any]:
    data = deepcopy(single_unit_input)

    required_keys = ["桥梁编号", "单元编号", "联号", "桥型", "本联信息", "桥面宽度信息", "本联设计分组"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"single_unit_input 缺少必要字段: {key}")

    bridge_type = data.get("桥型")
    inferred_system_type = infer_system_type_from_bridge_type(bridge_type)

    deck_info = data.get("桥面宽度信息", {}) or {}
    start_width = safe_float(deck_info.get("起点宽度"))
    end_width = safe_float(deck_info.get("终点宽度"))

    # 兼容“标准宽度”但宽度只写在宽度变化说明中的情况。
    if start_width is None:
        width_text = str(deck_info.get("宽度变化说明") or "")
        nums = re.findall(r"\d+(?:\.\d+)?", width_text)
        if nums:
            start_width = float(nums[0])
    if start_width is not None and end_width is None:
        end_width = start_width
    if end_width is not None and start_width is None:
        start_width = end_width

    non_abutment_groups: List[Dict[str, Any]] = []
    skipped_groups: List[Dict[str, Any]] = []
    compatibility_warnings: List[str] = []
    for group in data.get("本联设计分组", []) or []:
        normalized_group = deepcopy(group)
        raw_role = str(group.get("墩位角色") or "").strip()
        role = normalize_role(raw_role)
        normalized_group["墩位角色"] = role
        if raw_role == "过渡墩":
            compatibility_warnings.append(
                f"分组 {group.get('分组编号')} 的历史角色“过渡墩”已兼容映射为“边墩”。"
            )
        if role == "桥台":
            skipped_groups.append(normalized_group)
        else:
            non_abutment_groups.append(normalized_group)

    data["_标准化任务输入"] = {
        "当前桥面宽度": start_width,
        "下一墩桥面宽度": end_width,
        "桥型": bridge_type,
        "结构类型": inferred_system_type,
        "跨径长度参考": infer_representative_span_length(data.get("本联信息", {}).get("跨径组合", "")),
    }
    data["_桥墩设计分组"] = non_abutment_groups
    data["_跳过分组"] = skipped_groups
    data["_兼容警告"] = compatibility_warnings
    return data


# ============================================================
# 示例样本筛选
# ============================================================
def get_foundation_type(sample: Dict[str, Any]) -> Any:
    return (
        sample.get("foundation_type")
        or sample.get("foundatauin type")
        or sample.get("foundatauin_type")
        or sample.get("基础类型")
    )


def get_pier_type(sample: Dict[str, Any]) -> Any:
    return sample.get("pier type") or sample.get("pier_type") or sample.get("墩柱类型") or sample.get("桥墩类型")


def match_group_samples(
    samples: List[Dict[str, Any]],
    deck_width: Optional[float],
    next_deck_width: Optional[float],
    structure_type: str,
    pier_role: str,
    span_length: Optional[float] = None,
    max_count: int = 3,
) -> List[Dict[str, Any]]:
    role = normalize_role(pier_role)
    sys_type = str(structure_type).strip() if structure_type else ""

    if role == "桥台":
        return []

    def score(s: Dict[str, Any]) -> float:
        s_deck = safe_float(s.get("deck_width"), 999.0)
        s_next = safe_float(s.get("next_deck_width"), 999.0)
        s_span = safe_float(s.get("span_length"), None)

        val = 0.0
        if deck_width is not None:
            val += abs(float(s_deck) - deck_width)
        if next_deck_width is not None:
            val += 0.5 * abs(float(s_next) - next_deck_width)
        if span_length is not None and s_span is not None:
            val += 0.3 * abs(float(s_span) - span_length)
        return val

    candidates: List[Dict[str, Any]] = []

    # 第一优先级：墩位角色 + 结构类型完全匹配。
    for s in samples:
        if not isinstance(s, dict):
            continue
        sample_role = normalize_role(s.get("pier role"))
        sample_sys = str(s.get("system_type", "")).strip()
        if sample_role == role and sample_sys == sys_type:
            candidates.append(s)

    # 第二优先级：角色匹配。
    if not candidates:
        for s in samples:
            if isinstance(s, dict) and normalize_role(s.get("pier role")) == role:
                candidates.append(s)

    candidates.sort(key=score)
    return candidates[:max_count]


# ============================================================
# 示例与 Prompt 组装
# ============================================================
def make_sample_object(sample: Dict[str, Any]) -> Dict[str, Any]:
    is_double_support = bool(sample.get("is_double_support", False))
    t_beam_rows = 2 if is_double_support else 1

    front_edge = sample.get("front_support_edge_distance", {}) or {}
    rear_edge = sample.get("rear_support_edge_distance", {}) or {}
    draw_id = sample.get("drawing_id", sample.get("draw_id"))

    return {
        "draw_id": draw_id,
        "输入": {
            "当前桥面宽度": safe_float(sample.get("deck_width")),
            "下一墩桥面宽度": safe_float(sample.get("next_deck_width")),
            "结构类型": sample.get("system_type"),
            "墩位角色": normalize_role(sample.get("pier role")),
        },
        "输出": {
            "墩柱类型": get_pier_type(sample),
            "基础类型": get_foundation_type(sample),
            "是否双排支座": is_double_support,
            "T梁布置": {
                "T梁排数": t_beam_rows,
                "前跨": {
                    "T梁数量": sample.get("front_span_t_beam_count"),
                    "T梁间距": safe_float(sample.get("front_span_t_beam_spacing")),
                    "左右两侧距盖梁边缘距离": {
                        "左": safe_float(front_edge.get("left")),
                        "右": safe_float(front_edge.get("right")),
                    },
                },
                "后跨": {
                    "T梁数量": sample.get("rear_span_t_beam_count"),
                    "T梁间距": safe_float(sample.get("rear_span_t_beam_spacing")),
                    "左右两侧距盖梁边缘距离": {
                        "左": safe_float(rear_edge.get("left")),
                        "右": safe_float(rear_edge.get("right")),
                    },
                },
            },
            "盖梁尺寸": {
                "长度": safe_float(sample.get("cap_length")),
                "宽度": safe_float(sample.get("cap_width")),
                "中高": safe_float(sample.get("cap_height_mid")),
                "端高": safe_float(sample.get("cap_height_end")),
                "悬臂": safe_float(sample.get("cantilever_length")),
            },
            "墩柱尺寸": {
                "数量": sample.get("column_count"),
                "直径": safe_float(sample.get("column_diameter")),
                "中心间距": safe_float(sample.get("column_spacing")),
            },
            "基础": {
                "类型": get_foundation_type(sample),
                "桩基直径": safe_float(sample.get("pile_diameter")),
            },
        },
    }


def assemble_prompt(
    samples: List[Dict[str, Any]],
    single_unit_input: Dict[str, Any],
    template: Dict[str, Any],
    max_examples_per_group: int = 3,
) -> Dict[str, Any]:
    prompt = deepcopy(template)
    normalized_unit = normalize_single_unit_input(single_unit_input)

    normalized_task = normalized_unit["_标准化任务输入"]
    deck_width = normalized_task["当前桥面宽度"]
    next_deck_width = normalized_task["下一墩桥面宽度"]
    top_system_type = normalized_task["结构类型"]
    bridge_type = normalized_task["桥型"]
    rep_span_length = normalized_task["跨径长度参考"]

    active_groups = normalized_unit.get("_桥墩设计分组", [])
    skipped_groups = normalized_unit.get("_跳过分组", [])

    group_examples = []
    match_debug = []
    for group in active_groups:
        pier_role = normalize_role(group.get("墩位角色"))

        matched = match_group_samples(
            samples=samples,
            deck_width=deck_width,
            next_deck_width=next_deck_width,
            structure_type=top_system_type,
            pier_role=pier_role,
            span_length=rep_span_length,
            max_count=max_examples_per_group,
        )
        if not matched:
            raise ValueError(
                f"分组 {group.get('分组编号')}（{pier_role}）未匹配到示例样本；"
                "已依次尝试同角色同体系和同角色跨体系。"
            )
        exact_match = any(
            normalize_role(sample.get("pier role")) == pier_role
            and str(sample.get("system_type", "")).strip() == top_system_type
            for sample in matched
        )
        match_level = "同角色同体系" if exact_match else "同角色跨体系"

        match_debug.append({
            "分组编号": group.get("分组编号"),
            "墩位角色": pier_role,
            "桥型": bridge_type,
            "映射结构类型": top_system_type,
            "桥面宽度": deck_width,
            "下一墩桥面宽度": next_deck_width,
            "跨径长度参考": rep_span_length,
            "匹配层级": match_level,
            "匹配样本数量": len(matched),
            "匹配样本draw_id": [s.get("drawing_id", s.get("draw_id")) for s in matched],
        })

        group_examples.append({
            "对应分组编号": group.get("分组编号"),
            "墩位角色": pier_role,
            "示例样本": [make_sample_object(s) for s in matched],
        })

    prompt["分组示例样本集"] = group_examples

    prompt.setdefault("当前任务", {})
    prompt["当前任务"]["任务输入"] = {
        "桥梁编号": normalized_unit.get("桥梁编号"),
        "单元编号": normalized_unit.get("单元编号"),
        "联号": normalized_unit.get("联号"),
        "桥型": normalized_unit.get("桥型"),
        "映射得到的结构类型": top_system_type,
        "本联信息": normalized_unit.get("本联信息"),
        "桥面宽度信息": normalized_unit.get("桥面宽度信息"),
        "仅桥墩设计分组": active_groups,
        "已跳过分组": [
            {
                "分组编号": g.get("分组编号"),
                "墩位角色": g.get("墩位角色"),
                "包含桥墩号列表": g.get("包含桥墩号列表", []),
                "跳过原因": "当前阶段不考虑桥台设计",
            }
            for g in skipped_groups
        ],
    }

    prompt["_匹配调试信息"] = match_debug
    prompt["_兼容警告"] = normalized_unit.get("_兼容警告", [])
    return prompt


def build_prompt_for_llm(prompt_json: Dict[str, Any]) -> str:
    # 不把 _匹配调试信息 传给 LLM，避免污染输出。
    prompt_for_llm = deepcopy(prompt_json)
    prompt_for_llm.pop("_匹配调试信息", None)
    prompt_for_llm.pop("_兼容警告", None)
    return (
        pretty_json(prompt_for_llm)
        + "\n\n请严格按照“当前任务 -> 任务需输出内容”给出的结构返回最终JSON结果。"
        + "仅输出结果JSON，不要附加解释、分析过程或多余文字。"
    )


def build_dimension_design_prompt_tool(
    single_unit_input: Dict[str, Any],
    samples_yaml_path: str,
    template_json_path: Optional[str] = None,
    max_examples_per_group: int = 3,
    config_path: str = "config/settings.yaml",
    prompt_id: str = "tasks.cap_dimension_design.v1",
    evidence_context: Any = None,
) -> Dict[str, Any]:
    # 兼容旧调用签名；尺寸设计 Prompt 不再接受运行时规范检索内容。
    _ = evidence_context
    samples = load_yaml(samples_yaml_path)
    if template_json_path:
        template = load_json(template_json_path)
        template_source = str(template_json_path)
        rendered = None
        template_sha256 = hashlib.sha256(Path(template_json_path).read_bytes()).hexdigest()
    else:
        rendered = render_prompt(prompt_id, {}, config_path=config_path)
        template = json.loads(rendered.user_content or rendered.system_content)
        template_source = rendered.template_path
        template_sha256 = rendered.template_sha256
    prompt_json = assemble_prompt(
        samples=samples,
        single_unit_input=single_unit_input,
        template=template,
        max_examples_per_group=max_examples_per_group,
    )
    llm_prompt_text = build_prompt_for_llm(prompt_json)
    return {
        "success": True,
        "prompt_json": prompt_json,
        "llm_prompt_text": llm_prompt_text,
        "match_debug": prompt_json.get("_匹配调试信息", []),
        "compatibility_warnings": prompt_json.get("_兼容警告", []),
        "template_path": template_source,
        "prompt_id": rendered.prompt_id if rendered else f"external:{Path(template_source).name}",
        "prompt_version": rendered.version if rendered else "external",
        "template_sha256": template_sha256,
    }
