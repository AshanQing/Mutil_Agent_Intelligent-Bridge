# -*- coding: utf-8 -*-
"""Capacity check tool for cap-beam reinforcement verification.

This module wraps the existing cap_beam_capacity_envelope_overlay_refined.py
script as a deterministic tool callable by ModelingCheckAgent.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 文件顶层必须为对象: {path}")
    return data


def _import_capacity_script(script_path: Path) -> Any:
    if not script_path.exists():
        raise FileNotFoundError(f"未找到承载力验算脚本: {script_path}")

    spec = importlib.util.spec_from_file_location("cap_beam_capacity_module", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载承载力验算脚本: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _max_utilization_from_controls(control_sections: Dict[str, Any]) -> Dict[str, Any]:
    max_item = {
        "max_utilization": None,
        "control_name": None,
        "x_m": None,
        "util_type": None,
    }
    for name, row in (control_sections or {}).items():
        if not isinstance(row, dict):
            continue
        for key in ["util_M_pos", "util_M_neg", "util_V"]:
            value = row.get(key)
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if max_item["max_utilization"] is None or v > float(max_item["max_utilization"]):
                max_item = {
                    "max_utilization": v,
                    "control_name": name,
                    "x_m": row.get("x_m"),
                    "util_type": key,
                }
    return max_item


def capacity_check_tool(
    reinforcement_yaml_path: str,
    opensees_force_json_path: str,
    output_dir: str,
    capacity_script_path: Optional[str] = None,
    opensees_script_path: Optional[str] = None,
    combination_name: str = "ULS_basic",
    gamma_0: float = 1.10,
    apply_gamma0_to_demand: bool = True,
    auto_run_opensees_if_missing: bool = False,
) -> Dict[str, Any]:
    """Run cap-beam demand/capacity envelope verification.

    Parameters are intentionally explicit so the tool can be called from
    AgentState without relying on hard-coded filenames inside the original script.
    """
    try:
        reinf_path = Path(reinforcement_yaml_path).expanduser().resolve()
        force_path = Path(opensees_force_json_path).expanduser().resolve()
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not reinf_path.exists():
            return {"success": False, "error": f"未找到配筋 YAML: {reinf_path}"}
        if not force_path.exists() and not auto_run_opensees_if_missing:
            return {"success": False, "error": f"未找到 OpenSees 内力包络 JSON: {force_path}"}

        if capacity_script_path:
            script_path = Path(capacity_script_path).expanduser().resolve()
        else:
            script_path = Path(__file__).resolve().parent / "cap_beam_capacity_envelope_overlay_refined.py"

        module = _import_capacity_script(script_path)

        # Override script globals before calling main(), avoiding subprocess-based wrappers.
        module.BASE_DIR = script_path.parent
        module.REINFORCEMENT_YAML = reinf_path
        module.OPENSEES_FORCE_JSON = force_path
        module.OUTPUT_DIR = out_dir
        module.AUTO_RUN_OPENSEES_IF_MISSING = bool(auto_run_opensees_if_missing)
        module.COMBINATION_NAME = combination_name
        module.GAMMA_0 = float(gamma_0)
        module.APPLY_GAMMA0_TO_DEMAND = bool(apply_gamma0_to_demand)
        if opensees_script_path:
            module.OPENSEES_SCRIPT = Path(opensees_script_path).expanduser().resolve()

        module.main()

        summary_path = out_dir / "capacity_check_summary.json"
        sections_path = out_dir / "capacity_envelope_sections.json"
        summary = _load_json(summary_path)
        sections = _load_json(sections_path) if sections_path.exists() else {}

        overall_check = summary.get("overall_check", {}) if isinstance(summary, dict) else {}
        control_sections = summary.get("control_sections", {}) if isinstance(summary, dict) else {}
        utilization_summary = _max_utilization_from_controls(control_sections)

        output_files = {
            "capacity_check_summary_path": str(summary_path),
            "capacity_envelope_sections_path": str(sections_path),
            "moment_overlay_path": str(out_dir / "moment_demand_vs_capacity_overlay.png"),
            "shear_overlay_path": str(out_dir / "shear_demand_vs_capacity_overlay.png"),
            "utilization_plot_path": str(out_dir / "demand_capacity_utilization.png"),
        }

        return {
            "success": True,
            "check_type": "cap_beam_capacity_envelope",
            "combination_name": combination_name,
            "overall_check": overall_check,
            "control_sections": control_sections,
            "utilization_summary": utilization_summary,
            "summary": summary,
            "sections_count": len(sections.get("capacity_sections", [])) if isinstance(sections, dict) else None,
            "output_files": output_files,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "check_type": "cap_beam_capacity_envelope",
            "error": str(e),
        }
