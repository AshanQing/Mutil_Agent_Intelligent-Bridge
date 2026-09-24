# tools/design_generation_tool.py
# -*- coding: utf-8 -*-
"""
桥梁布跨设计 Prompt 组装与生成工具。

本版本适配：
1. settings.yaml 统一配置输入路径、规范路径和输出路径；
2. 通过 prompts/manifest.yaml 与 PromptRegistry 定位设桥布跨任务模板；
3. few-shot、当前设计输入和规范文本通过 Jinja2 上下文注入；
4. 内置竖曲线计算引擎与地形特征表生成（融合 scripts/prompt_builder.py 能力）。
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import yaml

from langchain_core.tools import tool

from bridge_agents.prompt_audit import artifact_attempt, record_prompt_audit, reserve_artifact_path
from bridge_agents.prompt_registry import render_external_prompt, render_prompt
from bridge_agents.utils import build_llm

logger = logging.getLogger(__name__)


# ======================================================
# 竖曲线计算引擎
# ======================================================
class VerticalAlignment:
    """竖曲线计算：支持抛物线型竖曲线，无 R 值时退化为直线插值。"""

    def __init__(self, pvi_list: List[Dict[str, Any]]):
        self.pvi = sorted(pvi_list, key=lambda x: x['桩号'])
        self.curves: List[Dict[str, Any]] = []
        self._calculate_curves()

    def _calculate_curves(self) -> None:
        n = len(self.pvi)
        if n < 2:
            return
        for i in range(1, n - 1):
            p_prev, p_curr, p_next = self.pvi[i - 1], self.pvi[i], self.pvi[i + 1]
            try:
                g1 = (p_curr['高程'] - p_prev['高程']) / (p_curr['桩号'] - p_prev['桩号'])
                g2 = (p_next['高程'] - p_curr['高程']) / (p_next['桩号'] - p_curr['桩号'])
                omega = g2 - g1
                R = p_curr.get('R', 0)
                if R == 0 or abs(omega) < 1e-6:
                    continue
                L = abs(omega) * R
                T = L / 2
                self.curves.append({
                    'BVC': p_curr['桩号'] - T,
                    'EVC': p_curr['桩号'] + T,
                    'H_BVC': p_curr['高程'] - T * g1,
                    'g1': g1, 'g2': g2, 'L': L,
                })
            except ZeroDivisionError:
                continue

    def get_elevation(self, station: float) -> float:
        if not self.pvi:
            return 0.0
        idx = -1
        for i in range(len(self.pvi) - 1):
            if self.pvi[i]['桩号'] <= station <= self.pvi[i + 1]['桩号']:
                idx = i
                break
        if idx == -1:
            if station < self.pvi[0]['桩号']:
                idx = 0
            else:
                idx = len(self.pvi) - 2
        if idx < 0 or idx >= len(self.pvi) - 1:
            return float(self.pvi[0]['高程'])

        p_start, p_end = self.pvi[idx], self.pvi[idx + 1]
        denom = (p_end['桩号'] - p_start['桩号'])
        if denom == 0:
            return float(p_start['高程'])
        g = (p_end['高程'] - p_start['高程']) / denom
        h_tangent = p_start['高程'] + g * (station - p_start['桩号'])

        for curve in self.curves:
            if curve['BVC'] <= station <= curve['EVC']:
                dx = station - curve['BVC']
                correction = (curve['g2'] - curve['g1']) / (2 * curve['L']) * (dx ** 2)
                return curve['H_BVC'] + curve['g1'] * dx + correction
        return float(h_tangent)


# ======================================================
# 全局缓存
# ======================================================
_CONFIG_CACHE: Dict[str, dict] = {}


# ======================================================
# 配置与路径处理
# ======================================================
def _as_resolved_path(path_value: Optional[str], *, config_path: str, project_root: Optional[str] = None) -> Optional[str]:
    if path_value in [None, ""]:
        return None
    raw = str(path_value)
    raw = os.path.expandvars(os.path.expanduser(raw))
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    if project_root:
        root = Path(os.path.expandvars(os.path.expanduser(str(project_root))))
        return str((root / p).resolve())
    cfg_dir = Path(config_path).resolve().parent
    return str((cfg_dir / p).resolve())


def _get_config(config_path: str = "config/settings.yaml") -> dict:
    cfg_path = str(Path(config_path).resolve())
    if cfg_path not in _CONFIG_CACHE:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"配置文件未找到: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        _CONFIG_CACHE[cfg_path] = cfg
        logger.info(f"[OK] 配置文件加载成功: {cfg_path}")
    return _CONFIG_CACHE[cfg_path]


def _get_project_root(config: dict, config_path: str) -> Optional[str]:
    paths = config.get("paths", {}) or {}
    project_root = paths.get("project_root") or config.get("project_root")
    if not project_root:
        return None
    return str(Path(os.path.expandvars(os.path.expanduser(str(project_root)))).resolve())


def _get_configured_path(
    config_path: str,
    explicit_value: Optional[str],
    *candidate_keys: str,
    required: bool = False,
) -> Optional[str]:
    config = _get_config(config_path)
    project_root = _get_project_root(config, config_path)
    value = explicit_value
    if value in [None, ""]:
        for key_path in candidate_keys:
            cur: Any = config
            for part in key_path.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    cur = None
                    break
                cur = cur[part]
            if cur not in [None, ""]:
                value = cur
                break
    resolved = _as_resolved_path(value, config_path=config_path, project_root=project_root)
    if required and not resolved:
        raise ValueError(f"缺少必要路径配置，请在 settings.yaml 中配置 {candidate_keys} 或显式传参")
    return resolved


def _read_reference_file(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    p = Path(path_value)
    if not p.exists():
        raise FileNotFoundError(f"参考文件不存在: {p}")
    suffix = p.suffix.lower()
    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)
    return p.read_text(encoding="utf-8")


def _extract_design_json_from_response_text(text: str) -> Dict[str, Any]:
    import re
    if not text or not text.strip():
        raise ValueError("LLM 响应为空，无法提取设桥 JSON。")
    content = text.strip()
    blocks = re.findall(
        r"```json\s*(\{.*?\})\s*```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if blocks:
        json_text = blocks[-1].strip()
    else:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM 响应中未找到 JSON 对象。")
        json_text = content[start:end + 1].strip()
    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError("提取到的 JSON 顶层不是对象。")
    if "设桥总览" not in data:
        raise ValueError('提取到的 JSON 中缺少"设桥总览"字段。')
    return data


# ======================================================
# 地形特征表生成（融合自 scripts/prompt_builder.py）
# ======================================================
def _format_station(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    sign = "-" if v < 0 else ""
    v = abs(v)
    km = int(v // 1000)
    m = v - km * 1000
    if abs(m - round(m)) < 1e-6:
        m_text = str(int(round(m)))
    else:
        m_text = f"{m:.3f}".rstrip("0").rstrip(".")
    return f"K{sign}{km}+{m_text}"


def _get_vertical_profile(line_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(line_data, dict):
        return {"桩号范围": "未知", "设计线高程序列": [], "地形线高程序列": []}
    vertical = line_data.get("纵断面结构")
    if isinstance(vertical, dict):
        return {
            "桩号范围": vertical.get("桩号范围", line_data.get("截取范围", line_data.get("桩号范围", "未知"))),
            "设计线高程序列": vertical.get("设计线高程序列", []),
            "地形线高程序列": vertical.get("地形线高程序列", []),
        }
    return {
        "桩号范围": line_data.get("桩号范围", line_data.get("截取范围", "未知")),
        "设计线高程序列": line_data.get("设计线高程序列", []),
        "地形线高程序列": line_data.get("地形线高程序列", []),
    }


def _generate_terrain_table(
    design_pvi_list: List[Dict[str, Any]],
    ground_list: List[Dict[str, Any]],
) -> str:
    """生成地形特征表：| 桩号 | 设计线高程 | 地面线高程 | 高差（设计线-地面线）|"""
    if not design_pvi_list or not ground_list:
        return "（数据缺失）"

    va = VerticalAlignment(design_pvi_list)
    df_ground = pd.DataFrame(ground_list)
    if '桩号' not in df_ground.columns or '高程' not in df_ground.columns:
        return "（地形数据格式错误）"

    df_ground = df_ground.sort_values('桩号')
    if design_pvi_list:
        min_k = min(p['桩号'] for p in design_pvi_list)
        max_k = max(p['桩号'] for p in design_pvi_list)
        df_ground = df_ground[(df_ground['桩号'] >= min_k) & (df_ground['桩号'] <= max_k)]

    rows = []
    for _, row in df_ground.iterrows():
        k, zg = float(row['桩号']), float(row['高程'])
        zd = va.get_elevation(k)
        delta = zd - zg
        rows.append(f"| {_format_station(k):>12} | {zd:8.3f} | {zg:8.3f} | {delta:+9.3f} |")

    header = "| 桩号 | 设计线高程 | 地面线高程 | 高差（设计线-地面线） |\n|---:|---:|---:|---:|"
    return header + "\n" + "\n".join(rows)


def _format_auxiliary_info(line_data: Dict[str, Any]) -> str:
    text = ""
    curves = line_data.get('平曲线结构', []) if isinstance(line_data, dict) else []
    simple_curves = []
    for c in curves:
        if not isinstance(c, dict):
            continue
        r_val = c.get("半径R") or c.get("R")
        d_val = c.get("转角Δ(°)") or c.get("转角")
        start = c.get("起点", {}).get("桩号", "") if isinstance(c.get("起点"), dict) else ""
        end = c.get("终点", {}).get("桩号", "") if isinstance(c.get("终点"), dict) else ""
        rng = c.get("桩号范围") or f"{start} - {end}"
        simple_curves.append(f"{c.get('曲线类型','曲线')}: {rng}, R={r_val}, Δ={d_val}°")
    text += "**平曲线信息**:\n" + ("\n".join(simple_curves) if simple_curves else "无数据") + "\n\n"

    sections = line_data.get('横断面信息', []) if isinstance(line_data, dict) else []
    if sections:
        text += "**横断面信息**:\n" + json.dumps(sections[:3], ensure_ascii=False, indent=2) + "\n"
    else:
        text += "**横断面信息**: 无\n"
    return text


def _flatten_obstacles(obstacles: Any) -> List[Dict[str, Any]]:
    if not obstacles:
        return []
    if isinstance(obstacles, list):
        return [x for x in obstacles if isinstance(x, dict)]
    if isinstance(obstacles, dict):
        out: List[Dict[str, Any]] = []
        def walk(obj: Any) -> None:
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        out.append(item)
            elif isinstance(obj, dict):
                if any(k in obj for k in ["range", "桩号范围", "id", "avg_width", "geometry", "advice"]):
                    out.append(obj)
                else:
                    for v in obj.values():
                        walk(v)
        walk(obstacles)
        return out
    return []


def _format_obstacles(obstacles: Any, max_items: Optional[int] = None) -> str:
    obs = _flatten_obstacles(obstacles)
    if not obs:
        return "**障碍物信息**: 无\n\n"
    show = obs if max_items is None else obs[:max_items]
    text = f"**障碍物信息**（共 {len(obs)} 条）:\n"
    for item in show:
        oid = item.get("id", item.get("障碍物编号", ""))
        rng = item.get("range", item.get("桩号范围", item.get("覆盖桩号范围", "")))
        length = item.get("length", item.get("长度", ""))
        width = item.get("avg_width", item.get("平均宽度", item.get("平均宽度(m)", "")))
        geometry = item.get("geometry", item.get("几何关系", ""))
        advice = item.get("advice", item.get("避让建议", ""))
        text += f"- id={oid}, range={rng}, length={length}, avg_width={width}, geometry={geometry}, advice={advice}\n"
    if max_items is not None and len(obs) > max_items:
        text += f"- ……其余 {len(obs) - max_items} 条障碍物略。\n"
    text += "\n"
    return text


def _format_input_data_for_prompt(target_data: Dict[str, Any]) -> str:
    """将多线路设计输入格式化为含地形特征表的 Markdown 文本。"""
    if not isinstance(target_data, dict):
        return str(target_data)

    line_type = target_data.get('线路类型', '整体式')
    data_block = f"**线路类型**: {line_type}\n\n"

    obs = target_data.get('障碍物信息', target_data.get('构造物信息', []))
    data_block += _format_obstacles(obs, max_items=None)

    for route_key in ["K", "Z", "Z1", "Z2"]:
        route_data = target_data.get(route_key)
        if not route_data:
            continue
        vertical = _get_vertical_profile(route_data)
        data_block += f"#### --- {route_key}线数据 ---\n"
        data_block += f"桩号范围: {vertical.get('桩号范围', '未知')}\n"
        data_block += _format_auxiliary_info(route_data)
        data_block += f"**{route_key}线地形特征表**:\n"
        data_block += _generate_terrain_table(
            vertical.get('设计线高程序列', []),
            vertical.get('地形线高程序列', []),
        ) + "\n\n"

    return data_block


def _split_few_shot_sample(sample: Dict[str, Any]) -> tuple[Dict[str, Any], Any]:
    """拆分兼容嵌套与扁平结构的 few-shot 输入、输出。"""
    nested_input = sample.get("input")
    if isinstance(nested_input, dict):
        return nested_input, sample.get("output", {})

    sample_input = {
        key: value
        for key, value in sample.items()
        if key not in {"output", "_source_file"}
    }
    return sample_input, sample.get("output", {})


def _format_few_shots_for_prompt(few_shots: List[Dict[str, Any]]) -> str:
    """将 few-shot 输入压缩为地形特征表，并保留完整示例输出。"""
    blocks: List[str] = []
    for index, sample in enumerate(few_shots, start=1):
        sample_input, sample_output = _split_few_shot_sample(sample)
        source_file = sample.get("_source_file")
        source_text = f"（来源：{source_file}）" if source_file else ""
        blocks.append(
            f"### Few-shot 示例 {index}{source_text}\n\n"
            f"#### 示例输入\n\n{_format_input_data_for_prompt(sample_input)}"
            f"#### 示例输出\n\n"
            f"```json\n"
            f"{json.dumps(sample_output, ensure_ascii=False, indent=2, default=str)}\n"
            f"```"
        )
    return "\n\n".join(blocks) if blocks else "无。"


# ======================================================
# 工具函数
# ======================================================
@tool
def generate_design(
    design_input: Dict[str, Any],
    few_shots: Optional[List[Dict[str, Any]]] = None,
    standards_path: Optional[str] = None,
    template_path: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    根据设计输入生成桥梁布跨设计方案，并保存 Prompt 与结果。
    """
    prompt_path = response_path = json_path = input_data_path = few_shots_path = None

    try:
        config_path = str(Path(config_path).resolve())

        standards_path = _get_configured_path(
            config_path, standards_path,
            "paths.standards_path", "prompt.standards_path", required=False,
        )
        output_dir = _get_configured_path(
            config_path, output_dir,
            "paths.output_dir", "output_dir", required=False,
        ) or "output"

        if standards_path is None:
            logger.warning("standards_path 未提供，将不加载规范条文")

        # 1. 准备输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = Path(output_dir)
        save_dir = save_root / f"design_run_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 结果将保存至: {save_dir}")

        # 2. 保存输入与 few-shot
        input_data_path = save_dir / "input_data.json"
        with open(input_data_path, "w", encoding="utf-8") as f:
            json.dump(design_input, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"💾 设计输入已保存: {input_data_path}")

        few_shots = few_shots or []
        if few_shots:
            few_shots_path = save_dir / "few_shots.json"
            with open(few_shots_path, "w", encoding="utf-8") as f:
                json.dump(few_shots, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"💾 参考样本已保存: {few_shots_path}")

        # 3. 构建 Prompt —— 使用内置地形特征表格式化
        config = _get_config(config_path)
        prompt_id = (
            ((config.get("prompts") or {}).get("initial_layout_design_prompt_id"))
            or "tasks.initial_layout_design.v1"
        )
        design_input_text = _format_input_data_for_prompt(design_input)
        prompt_context = {
            "design_input_json": design_input_text,
            "few_shots_json": _format_few_shots_for_prompt(few_shots),
            "standards_text": _read_reference_file(standards_path) if standards_path else "",
        }
        rendered_prompt = (
            render_external_prompt(template_path, prompt_context)
            if template_path
            else render_prompt(prompt_id, prompt_context, config_path=config_path)
        )
        prompt = rendered_prompt.user_content or rendered_prompt.system_content
        logger.info(f"Prompt 构建完成，长度: {len(prompt)} 字符")

        prompt_path = reserve_artifact_path(save_dir / "prompt.txt")
        prompt_path.write_text(prompt, encoding="utf-8")
        logger.info(f"💾 Prompt 已保存: {prompt_path}")
        prompt_attempt = artifact_attempt(prompt_path)
        record_prompt_audit(
            output_dir,
            prompt_id=rendered_prompt.prompt_id,
            version=rendered_prompt.version,
            template_sha256=rendered_prompt.template_sha256,
            stage="initial_design",
            scope_id=save_dir.name,
            attempt=prompt_attempt,
            rendered_path=prompt_path,
            status="rendered",
        )

        # 4. 调用 LLM（与 Agent 层统一使用 LangChain ChatOpenAI）
        logger.info("正在调用大模型生成设计方案...")
        llm = build_llm(config_path, "generation")
        messages = []
        system_prompt = (
            (config.get("llm", {}) or {})
            .get("generation", {})
            .get("system_prompt")
            or config.get("system_prompt")
        )
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("user", prompt))

        # 复杂布跨任务的模型输出可能被推理预算截断（返回空 content 或残缺 JSON），
        # 属偶发；自动重试若干次，并把每次原始响应落盘便于诊断。
        design_text = ""
        design_json: Dict[str, Any] = {}
        json_path = save_dir / "design_result.json"
        response_path = save_dir / "response_full.md"
        last_error = ""
        max_attempts = 3
        for attempt_no in range(1, max_attempts + 1):
            try:
                response = llm.invoke(messages)
                design_text = response.content or ""
                attempt_response = (
                    response_path if attempt_no == 1 else save_dir / f"response_full_attempt_{attempt_no}.md"
                )
                attempt_response.write_text(str(design_text), encoding="utf-8")
                logger.info(
                    "LLM 原始响应(第 %d 次, %d 字符)已保存: %s",
                    attempt_no,
                    len(str(design_text)),
                    attempt_response,
                )
                if not str(design_text).strip():
                    raise ValueError("LLM 返回空结果")
                design_json = _extract_design_json_from_response_text(str(design_text))
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(design_json, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 JSON 设计结果已保存: {json_path}")
                break
            except Exception as exc:
                last_error = str(exc)
                logger.warning("设桥布跨设计生成第 %d 次失败：%s", attempt_no, last_error)
                if attempt_no >= max_attempts:
                    raise ValueError(f"设桥布跨设计生成连续 {max_attempts} 次失败：{last_error}") from exc
        record_prompt_audit(
            output_dir,
            prompt_id=rendered_prompt.prompt_id,
            version=rendered_prompt.version,
            template_sha256=rendered_prompt.template_sha256,
            stage="initial_design",
            scope_id=save_dir.name,
            attempt=prompt_attempt,
            rendered_path=prompt_path,
            status="completed",
        )

        return {
            "success": True,
            "tool_name": "generate_design",
            "design_text": design_text,
            "design_json": design_json,
            "error": None,
            "input_files": {
                "template_path": rendered_prompt.template_path,
                "prompt_id": rendered_prompt.prompt_id,
                "prompt_version": rendered_prompt.version,
                "template_sha256": rendered_prompt.template_sha256,
                "standards_path": str(standards_path) if standards_path else None,
                "config_path": str(config_path),
            },
            "output_files": {
                "saved_prompt_path": str(prompt_path),
                "saved_response_path": str(response_path),
                "saved_json_path": str(json_path) if json_path.exists() else None,
                "design_json_path": str(json_path) if json_path.exists() else None,
                "output_json_path": str(json_path) if json_path.exists() else None,
                "result_json_path": str(json_path) if json_path.exists() else None,
                "saved_input_path": str(input_data_path),
                "saved_fewshots_path": str(few_shots_path) if few_shots_path else None,
                "save_dir": str(save_dir),
            },
            "saved_prompt_path": str(prompt_path),
            "saved_response_path": str(response_path),
            "saved_json_path": str(json_path) if json_path.exists() else None,
            "design_json_path": str(json_path) if json_path.exists() else None,
            "output_json_path": str(json_path) if json_path else None,
            "result_json_path": str(json_path) if json_path else None,
            "saved_input_path": str(input_data_path),
            "saved_fewshots_path": str(few_shots_path) if few_shots_path else None,
        }

    except Exception as e:
        if prompt_path and Path(prompt_path).exists() and "rendered_prompt" in locals():
            record_prompt_audit(
                output_dir or "output",
                prompt_id=rendered_prompt.prompt_id,
                version=rendered_prompt.version,
                template_sha256=rendered_prompt.template_sha256,
                stage="initial_design",
                scope_id=Path(prompt_path).parent.name,
                attempt=artifact_attempt(prompt_path),
                rendered_path=prompt_path,
                status="failed",
                metadata={"error": str(e)},
            )
        logger.error(f"生成设计失败: {e}", exc_info=True)
        return {
            "success": False,
            "tool_name": "generate_design",
            "design_text": None,
            "design_json": None,
            "error": str(e),
        }
