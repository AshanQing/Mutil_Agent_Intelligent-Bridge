from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def load_column_samples(path: str) -> List[Dict[str, Any]]:
    """加载墩柱配筋样本文件，兼容顶层列表与「示例N」编号键。"""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
    samples: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for index, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                sample = dict(item)
                sample.setdefault("sample_id", f"示例{index}")
                samples.append(sample)
    elif isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                sample = dict(value)
                sample.setdefault("sample_id", str(key))
                samples.append(sample)
    return samples


def inspect_column_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """质检单个墩柱样本，返回 valid、issues、输入与输出。"""
    sample_id = str(sample.get("sample_id") or sample.get("drawing_id") or "?")
    inp = sample.get("输入") or sample.get("input") or {}
    out = sample.get("输出") or sample.get("output") or {}
    issues: List[str] = []

    if not isinstance(inp, dict):
        return {"sample_id": sample_id, "valid": False, "issues": ["输入缺失"], "input": {}, "output": out}

    basic = inp.get("基本信息")
    if isinstance(basic, dict):
        if not basic.get("pier_role"):
            issues.append("基本信息.pier_role 缺失")
        if not basic.get("pier_type"):
            issues.append("基本信息.pier_type 缺失")
    else:
        issues.append("缺少 基本信息")

    geom = inp.get("墩柱几何信息")
    if not isinstance(geom, dict):
        issues.append("缺少 墩柱几何信息")
    else:
        for key in ("column_diameter", "column_height", "column_spacing"):
            value = _as_float(geom.get(key))
            if value is None or value <= 0:
                issues.append(f"墩柱几何信息.{key} 缺失或非正数")
        count = geom.get("column_count")
        if count is None or (isinstance(count, (int, float)) and float(count) <= 0):
            issues.append("墩柱几何信息.column_count 缺失或非正数")

    reinforcement = out.get("reinforcement") if isinstance(out, dict) else None
    if not isinstance(reinforcement, dict) or "pier_column" not in reinforcement:
        issues.append("输出缺少 reinforcement.pier_column")

    return {"sample_id": sample_id, "valid": not issues, "issues": issues, "input": inp, "output": out}


def inspect_column_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """批量质检，返回有效样本与被拒样本及原因。"""
    valid: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for sample in samples:
        result = inspect_column_sample(sample)
        if result["valid"]:
            valid.append(sample)
        else:
            rejected.append({
                "sample_id": result["sample_id"],
                "issues": result["issues"],
                "sample": sample,
            })
    return {"valid": valid, "rejected": rejected}


def _nested(data: Any, path: List[str], default: Any = None) -> Any:
    obj = data
    for key in path:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key)
    return obj if obj is not None else default


def score_column_sample(sample_input: Dict[str, Any], target: Dict[str, Any]) -> float:
    """墩柱样本近邻分数，越小越接近。按柱径、柱高、柱间距、墩柱类型与墩位角色评分。"""
    s_basic = _nested(sample_input, ["基本信息"], sample_input)
    s_geom = _nested(sample_input, ["墩柱几何信息"], sample_input)
    t_basic = _nested(target, ["基本信息"], target)
    t_geom = _nested(target, ["墩柱几何信息"], target)

    score = 0.0
    t_role = str((t_basic or {}).get("pier_role") or "").strip()
    s_role = str((s_basic or {}).get("pier_role") or "").strip()
    if t_role and s_role and t_role != s_role:
        score += 100.0

    t_type = str((t_basic or {}).get("pier_type") or "").strip()
    s_type = str((s_basic or {}).get("pier_type") or "").strip()
    if t_type and s_type and t_type != s_type:
        score += 20.0

    for key, weight in (("column_diameter", 2.0), ("column_height", 1.0), ("column_spacing", 1.0)):
        tv = _as_float((t_geom or {}).get(key))
        sv = _as_float((s_geom or {}).get(key))
        if tv is not None and sv is not None:
            score += weight * abs(tv - sv)
    return score


def select_column_samples(
    samples: List[Dict[str, Any]],
    target: Dict[str, Any],
    max_count: int = 2,
) -> List[Dict[str, Any]]:
    """选择与目标最接近的 max_count 个墩柱样本。"""
    if not samples:
        return []
    ranked = sorted(samples, key=lambda s: score_column_sample(s.get("输入", s), target))
    return ranked[:max(0, int(max_count))]

