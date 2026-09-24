"""
collision_detection_tool.py

设桥布跨结果的视觉-几何碰撞检测工具。

功能：
1. 读取 LLM 按设桥布跨任务 Prompt 输出的设桥布跨 JSON；
2. 基于 plane_json 中的平曲线结构，将墩台桩号转换为平面坐标；
3. 基于 mask + pgw 将墩柱足迹映射到障碍物掩码图；
4. 输出逐墩柱碰撞报告、冲突摘要、可视化图和六项评估指标。

在 agent.py 中可通过：
    from tools.collision_detection_tool import collision_detection_tool
调用：
    collision_detection_tool.invoke({...})
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from langchain_core.tools import tool
except Exception:  # 允许脱离 LangChain 单独调试
    tool = None


# ============================================================
# 1. 基础工具函数
# ============================================================
def _ensure_dir(path: str | os.PathLike[str]) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _jsonable(obj: Any) -> Any:
    """将 numpy 标量、Path 等对象转换为 JSON 可序列化对象。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _station_to_float(k_str: Any) -> float:
    """支持 K12+360、12+360、12360.0 等桩号格式。"""
    if isinstance(k_str, (int, float)):
        return float(k_str)

    text = str(k_str).strip().upper().replace(" ", "")
    text = text.replace("Ｋ", "K")
    text = re.sub(r"^[A-Z]+", "", text)

    if "+" in text:
        left, right = text.split("+", 1)
        return float(left) * 1000.0 + float(right)
    return float(text)


def _station_label(k: float) -> str:
    km = int(k // 1000)
    m = k - km * 1000
    return f"K{km}+{m:06.3f}"


def _load_design_data(
    design_json_path: Optional[str] = None,
    design_result: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    兼容三种输入：
    1. 直接传 design_json_path；
    2. 传 design_result，且其中有 output_json_path / design_json_path 等路径字段；
    3. 传 design_result，且其中本身就是设桥 JSON 或包含 result/data/design 字段。
    """
    candidate_keys = [
        "design_json_path",
        "output_json_path",
        "result_json_path",
        "json_path",
        "final_json_path",
        "output_path",
    ]

    if design_json_path:
        with open(design_json_path, "r", encoding="utf-8") as f:
            return json.load(f), design_json_path

    if not design_result:
        raise ValueError("未提供 design_json_path 或 design_result，无法读取设桥布跨结果。")

    for key in candidate_keys:
        path = design_result.get(key)
        if path and os.path.exists(str(path)):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), str(path)

    # 有些 generate_design 工具会把输出文件放在 output_files 中。
    output_files = design_result.get("output_files")
    if isinstance(output_files, dict):
        for key in candidate_keys:
            path = output_files.get(key)
            if path and os.path.exists(str(path)):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f), str(path)

    # 有些工具会直接把最终 JSON 放在 result/data/design 中。
    for key in ["result", "data", "design", "design_data", "layout_result"]:
        value = design_result.get(key)
        if isinstance(value, dict) and "设桥总览" in value:
            return value, f"design_result.{key}"

    if "设桥总览" in design_result:
        return design_result, "design_result"

    raise ValueError(
        "design_result 中未找到可读取的设桥 JSON。请检查 generate_design 工具是否返回 "
        "design_json_path/output_json_path/output_files 等字段。"
    )


def _iter_layout_schemes(design_data: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    统一遍历当前设桥布跨输出格式和旧版对称布跨格式。

    产出：
        bridge: 桥位对象
        scheme_info: {
            scheme_type, side_label, scheme, pier_list, column_mode
        }
    """
    for bridge in design_data.get("桥位列表", []) or []:
        if not isinstance(bridge, dict):
            continue

        unified = bridge.get("统一布跨方案")
        if isinstance(unified, dict) and unified:
            yield bridge, {
                "scheme_type": "统一布跨方案",
                "side_label": "unified",
                "scheme": unified,
                "pier_list": unified.get("墩位与墩高", []) or [],
                "column_mode": "both_sides",
            }

        # 兼容旧模板字段：对称布跨方案。
        legacy = bridge.get("对称布跨方案")
        if isinstance(legacy, dict) and legacy:
            yield bridge, {
                "scheme_type": "对称布跨方案",
                "side_label": "unified",
                "scheme": legacy,
                "pier_list": legacy.get("墩位与墩高", []) or [],
                "column_mode": "both_sides",
            }

        split_list = bridge.get("分幅布跨方案列表")
        if isinstance(split_list, list):
            for item in split_list:
                if not isinstance(item, dict):
                    continue
                side_label = str(item.get("幅别") or item.get("方案编号") or "single_side")
                yield bridge, {
                    "scheme_type": "分幅布跨方案",
                    "side_label": side_label,
                    "scheme": item,
                    "pier_list": item.get("墩位与墩高", []) or [],
                    "column_mode": "single_side",
                }


# ============================================================
# 2. 坐标与线形系统
# ============================================================
class GeoReference:
    def __init__(self, pgw_path: str):
        if not os.path.exists(pgw_path):
            raise FileNotFoundError(f"PGW 文件不存在: {pgw_path}")
        with open(pgw_path, "r", encoding="utf-8") as f:
            values = [float(x.strip()) for x in f.readlines() if x.strip()]
        if len(values) < 6:
            raise ValueError(f"PGW 文件格式错误，应至少包含6行参数: {pgw_path}")

        self.forward = np.array(
            [
                [values[0], values[2], values[4]],
                [values[1], values[3], values[5]],
                [0.0, 0.0, 1.0],
            ]
        )
        self.inv = np.linalg.inv(self.forward)
        self.res = abs(values[0])

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        vec = np.array([x, y, 1.0])
        res = self.inv @ vec
        return int(round(res[0])), int(round(res[1]))


class AnchorRouteSystem:
    def __init__(self, json_path: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"平曲线 JSON 不存在: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.segments = data.get("平曲线结构") or data.get("segments") or []
        if not self.segments:
            raise ValueError("平曲线 JSON 中未找到 '平曲线结构'。")
        self.segments.sort(key=lambda x: self.parse_k(x.get("起点", {}).get("桩号", 0.0)))

    @staticmethod
    def parse_k(k_str: Any) -> float:
        try:
            return _station_to_float(k_str)
        except Exception:
            return 0.0

    @staticmethod
    def _get_val(node: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in node:
                return _safe_float(node[key], default)
        return default

    def get_exact_pos(self, k_target: float) -> Tuple[float, float, float]:
        target_seg = None
        for seg in self.segments:
            s_k = self.parse_k(seg.get("起点", {}).get("桩号"))
            e_k = self.parse_k(seg.get("终点", {}).get("桩号"))
            if s_k - 0.01 <= k_target <= e_k + 0.01:
                target_seg = seg
                break

        if target_seg is None:
            if k_target < self.parse_k(self.segments[0].get("起点", {}).get("桩号")):
                target_seg = self.segments[0]
            else:
                target_seg = self.segments[-1]

        s_node = target_seg.get("起点", {})
        anchor_x = self._get_val(s_node, ["X", "x"])
        anchor_y = self._get_val(s_node, ["Y", "y"])
        azi = self._get_val(s_node, ["方位角", "方位角(°)", "azimuth"])

        # 沿用你原型脚本中的坐标交换逻辑：map_x=Y, map_y=X。
        map_x, map_y = anchor_y, anchor_x
        anchor_theta = math.radians(90.0 - azi)

        s_k = self.parse_k(s_node.get("桩号"))
        dist_signed = k_target - s_k
        if abs(dist_signed) < 0.001:
            return map_x, map_y, anchor_theta

        ctype = str(target_seg.get("曲线类型", ""))
        radius = _safe_float(target_seg.get("半径R", target_seg.get("R", 0.0)))
        s_azi = self._get_val(target_seg.get("起点", {}), ["方位角", "方位角(°)", "azimuth"])
        e_azi = self._get_val(target_seg.get("终点", {}), ["方位角", "方位角(°)", "azimuth"])

        delta_deg = (e_azi - s_azi + 360.0) % 360.0
        if delta_deg > 180.0:
            delta_deg -= 360.0
        math_sign = -1.0 if delta_deg > 0 else 1.0
        if "直线" in ctype:
            math_sign = 0.0

        k_start = 0.0
        k_end = 0.0
        if "圆" in ctype:
            val = 1.0 / radius if radius else 0.0
            k_start = k_end = math_sign * val
        elif "缓" in ctype:
            val = 1.0 / radius if radius else 0.0
            max_k = math_sign * val
            k_start, k_end = (max_k, 0.0) if "后缓" in ctype else (0.0, max_k)

        step = 0.5
        curr_dist = 0.0
        curr_x, curr_y, curr_theta = map_x, map_y, anchor_theta
        direction = 1.0 if dist_signed >= 0 else -1.0
        dist = abs(dist_signed)
        seg_len = max(self.parse_k(target_seg.get("终点", {}).get("桩号")) - s_k, 1.0)

        while curr_dist < dist:
            ds = min(step, dist - curr_dist)
            ratio = (curr_dist + ds / 2.0) / seg_len
            k_now = k_start + (k_end - k_start) * ratio
            curr_x += math.cos(curr_theta) * ds * direction
            curr_y += math.sin(curr_theta) * ds * direction
            curr_theta += k_now * ds * direction
            curr_dist += ds

        return curr_x, curr_y, curr_theta


# ============================================================
# 3. 碰撞检测与指标计算
# ============================================================
class CollisionValidator:
    def __init__(self, mask_path: str, pgw_path: str, mask_threshold: int = 127):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"mask 文件不存在: {mask_path}")
        self.geo = GeoReference(pgw_path)
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise ValueError(f"Mask 读取失败: {mask_path}")
        _, self.binary_mask = cv2.threshold(self.mask, mask_threshold, 255, cv2.THRESH_BINARY)
        self.h, self.w = self.binary_mask.shape
        self.dist_map = cv2.distanceTransform(self.binary_mask, cv2.DIST_L2, 5)

    def check_pier(self, world_x: float, world_y: float, radius_m: float) -> Dict[str, Any]:
        u, v = self.geo.world_to_pixel(world_x, world_y)
        if not (0 <= u < self.w and 0 <= v < self.h):
            return {
                "status": "OUT_OF_BOUNDS",
                "msg": "墩柱中心超出掩码图范围",
                "overlap_ratio": 0.0,
                "intrusion_depth_m": 0.0,
            }

        radius_px = int(math.ceil(radius_m / self.geo.res))
        x1, x2 = max(0, u - radius_px), min(self.w, u + radius_px + 1)
        y1, y2 = max(0, v - radius_px), min(self.h, v + radius_px + 1)

        roi_mask = self.binary_mask[y1:y2, x1:x2]
        roi_dist = self.dist_map[y1:y2, x1:x2]
        if roi_mask.size == 0:
            return {"status": "ERROR", "msg": "局部检测窗口为空"}

        pier_footprint = np.zeros_like(roi_mask)
        cv2.circle(pier_footprint, (u - x1, v - y1), radius_px, 255, -1)
        intersection = cv2.bitwise_and(roi_mask, pier_footprint)
        overlap_pixels = int(np.count_nonzero(intersection))

        if overlap_pixels == 0:
            return {
                "status": "SAFE",
                "overlap_ratio": 0.0,
                "intrusion_depth_m": 0.0,
                "center_dist_m": float(self.dist_map[v, u] * self.geo.res),
            }

        pier_pixels = max(int(np.count_nonzero(pier_footprint)), 1)
        masked_dist = cv2.bitwise_and(roi_dist, roi_dist, mask=intersection)
        max_dist_m = float(np.max(masked_dist) * self.geo.res)

        return {
            "status": "COLLISION",
            "overlap_ratio": float(overlap_pixels / pier_pixels),
            "intrusion_depth_m": max_dist_m,
            "center_dist_m": float(self.dist_map[v, u] * self.geo.res),
        }

    def visualize_result(self, report: List[Dict[str, Any]], output_path: str) -> str:
        vis_img = cv2.cvtColor(self.binary_mask, cv2.COLOR_GRAY2BGR)
        for item in report:
            u, v = self.geo.world_to_pixel(item["x"], item["y"])
            radius_px = max(1, int(math.ceil(item["r_m"] / self.geo.res)))
            status = item.get("res", {}).get("status")

            if status == "COLLISION":
                cv2.circle(vis_img, (u, v), radius_px, (0, 0, 150), -1)
                cv2.circle(vis_img, (u, v), radius_px, (0, 0, 255), 2)
                depth = _safe_float(item.get("res", {}).get("intrusion_depth_m"))
                cv2.putText(
                    vis_img,
                    f"{depth:.1f}m",
                    (u, v),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
            elif status == "SAFE":
                cv2.circle(vis_img, (u, v), radius_px, (0, 255, 0), 2)
            else:
                cv2.circle(vis_img, (u, v), radius_px, (255, 0, 0), 1)
        cv2.imwrite(output_path, vis_img)
        return output_path


class DesignCollisionEvaluator:
    def __init__(self, collision_status: str = "COLLISION"):
        self.collision_status = collision_status.upper()

    def calculate_metrics(self, report_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_items = [item for item in report_data if isinstance(item, dict)]
        total_columns = len(valid_items)
        conflict_columns = [
            item
            for item in valid_items
            if str(item.get("res", {}).get("status", "")).upper() == self.collision_status
        ]
        conflict_column_count = len(conflict_columns)
        intrusion_depths = [
            _safe_float(item.get("res", {}).get("intrusion_depth_m", 0.0))
            for item in conflict_columns
        ]
        overlap_ratios = [
            _safe_float(item.get("res", {}).get("overlap_ratio", 0.0))
            for item in conflict_columns
        ]
        return {
            "total_columns": total_columns,
            "conflict_column_count": conflict_column_count,
            "conflict_column_rate": conflict_column_count / total_columns if total_columns else 0.0,
            "total_intrusion_depth_columns": float(sum(intrusion_depths)),
            "avg_intrusion_depth_columns": float(np.mean(intrusion_depths)) if intrusion_depths else 0.0,
            "avg_overlap_ratio_columns": float(np.mean(overlap_ratios)) if overlap_ratios else 0.0,
        }


# ============================================================
# 4. 主工具函数
# ============================================================
def _build_column_points(
    cx: float,
    cy: float,
    theta: float,
    column_mode: str,
    side_label: str,
    offset_left_m: float,
    offset_right_m: float,
    column_half_spacing_m: float,
) -> List[Tuple[str, float, float]]:
    nx, ny = math.cos(theta + math.pi / 2.0), math.sin(theta + math.pi / 2.0)

    def left_cols() -> List[Tuple[str, float, float]]:
        return [
            ("Left-1", cx + nx * offset_left_m + nx * column_half_spacing_m, cy + ny * offset_left_m + ny * column_half_spacing_m),
            ("Left-2", cx + nx * offset_left_m - nx * column_half_spacing_m, cy + ny * offset_left_m - ny * column_half_spacing_m),
        ]

    def right_cols() -> List[Tuple[str, float, float]]:
        return [
            ("Right-1", cx - nx * offset_right_m + nx * column_half_spacing_m, cy - ny * offset_right_m + ny * column_half_spacing_m),
            ("Right-2", cx - nx * offset_right_m - nx * column_half_spacing_m, cy - ny * offset_right_m - ny * column_half_spacing_m),
        ]

    if column_mode == "both_sides":
        return left_cols() + right_cols()

    side = side_label.upper()
    if any(key in side for key in ["左", "LEFT", "Z线".upper(), "Z"]):
        return left_cols()
    if any(key in side for key in ["右", "RIGHT", "K线".upper(), "K"]):
        return right_cols()

    # 无法判断幅别时，保守按路线中心线两根柱检测。
    return [
        ("Center-1", cx + nx * column_half_spacing_m, cy + ny * column_half_spacing_m),
        ("Center-2", cx - nx * column_half_spacing_m, cy - ny * column_half_spacing_m),
    ]


def _resolve_route_key(side_label: str) -> str:
    """Determine which route (K/Z) a separated-roadbed scheme belongs to."""
    upper = str(side_label).upper()
    if any(k in upper for k in ["左", "LEFT", "Z"]):
        return "Z"
    return "K"  # default: right / K / unified


def _run_collision_detection(
    design_json_path: Optional[str],
    design_result: Optional[Dict[str, Any]],
    plane_paths: Dict[str, str],
    mask_path: str,
    pgw_path: str,
    output_dir: str,
    offset_left_m: float,
    offset_right_m: float,
    column_half_spacing_m: float,
    check_radius_m: float,
    mask_threshold: int,
    save_visualization: bool,
) -> Dict[str, Any]:
    design_data, design_source = _load_design_data(design_json_path, design_result)

    output_dir = _ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(design_source).stem if os.path.exists(str(design_source)) else "design_result"

    # Build per-route AnchorRouteSystem instances.
    route_systems: Dict[str, AnchorRouteSystem] = {}
    for rid, path in plane_paths.items():
        if os.path.exists(path):
            route_systems[rid] = AnchorRouteSystem(path)
    if not route_systems:
        raise ValueError("plane_paths 中没有有效的平曲线 JSON 文件。")
    default_route = route_systems.get("K") or next(iter(route_systems.values()))
    validator = CollisionValidator(mask_path, pgw_path, mask_threshold=mask_threshold)

    full_report: List[Dict[str, Any]] = []
    collision_logs: List[str] = [
        f"[*] 开始校验设计: {base_name}",
        f"[*] 使用平曲线: {', '.join(sorted(route_systems.keys()))}",
    ]
    route_keys_used: set = set()

    for bridge, scheme_info in _iter_layout_schemes(design_data):
        bridge_id = bridge.get("桥位编号") or "unknown_bridge"
        # For separated roadbeds, select the correct plane curve per side label.
        route_key = _resolve_route_key(scheme_info["side_label"])
        route = route_systems.get(route_key, default_route)
        route_keys_used.add(route_key)
        for pier in scheme_info["pier_list"]:
            if not isinstance(pier, dict):
                continue
            pier_id = pier.get("墩号") or pier.get("墩台号") or "unknown_pier"
            station_raw = pier.get("桩号")
            if not station_raw:
                continue

            k_val = route.parse_k(station_raw)
            cx, cy, theta = route.get_exact_pos(k_val)
            cols_world = _build_column_points(
                cx=cx,
                cy=cy,
                theta=theta,
                column_mode=scheme_info["column_mode"],
                side_label=scheme_info["side_label"],
                offset_left_m=offset_left_m,
                offset_right_m=offset_right_m,
                column_half_spacing_m=column_half_spacing_m,
            )

            for col_name, wx, wy in cols_world:
                res = validator.check_pier(wx, wy, check_radius_m)
                item = {
                    "bridge_id": bridge_id,
                    "scheme_type": scheme_info["scheme_type"],
                    "side_label": scheme_info["side_label"],
                    "pier_id": pier_id,
                    "col_name": col_name,
                    "station": str(station_raw),
                    "k_val": k_val,
                    "station_label": _station_label(k_val),
                    "x": wx,
                    "y": wy,
                    "r_m": check_radius_m,
                    "res": res,
                }
                full_report.append(item)

                if res.get("status") == "COLLISION":
                    collision_logs.append(
                        "[WARN] "
                        f"桥位:{bridge_id} | 方案:{scheme_info['scheme_type']} | 幅别:{scheme_info['side_label']} | "
                        f"墩号:{pier_id} | 位置:{col_name} | 桩号:{_station_label(k_val)} | "
                        f"状态:COLLISION | 侵入:{_safe_float(res.get('intrusion_depth_m')):.2f}m | "
                        f"重叠:{_safe_float(res.get('overlap_ratio')):.1%}"
                    )

    evaluator = DesignCollisionEvaluator()
    metrics = evaluator.calculate_metrics(full_report)
    metrics["has_collision"] = metrics["conflict_column_count"] > 0
    metrics["design_source"] = design_source

    report_json_path = os.path.join(output_dir, f"collision_report_{base_name}_{timestamp}.json")
    report_txt_path = os.path.join(output_dir, f"collision_summary_{base_name}_{timestamp}.txt")
    metrics_json_path = os.path.join(output_dir, f"collision_metrics_{base_name}_{timestamp}.json")
    vis_path = os.path.join(output_dir, f"collision_vis_{base_name}_{timestamp}.jpg")

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(full_report), f, indent=2, ensure_ascii=False)

    if len(collision_logs) == 1:
        collision_logs.append("[OK] 未检测到墩柱与障碍物掩码冲突。")
    collision_logs.extend(
        [
            "-" * 60,
            f"总墩柱数: {metrics['total_columns']}",
            f"冲突墩柱数: {metrics['conflict_column_count']}",
            f"冲突率: {metrics['conflict_column_rate']:.1%}",
            f"累计侵入深度: {metrics['total_intrusion_depth_columns']:.3f} m",
            f"平均侵入深度: {metrics['avg_intrusion_depth_columns']:.3f} m",
            f"平均重叠比例: {metrics['avg_overlap_ratio_columns']:.4f}",
        ]
    )
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(collision_logs))

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2, ensure_ascii=False)

    visualization_path = None
    if save_visualization:
        visualization_path = validator.visualize_result(full_report, vis_path)

    return {
        "success": True,
        "has_collision": metrics["has_collision"],
        "metrics": _jsonable(metrics),
        "collision_items": [item for item in _jsonable(full_report) if item.get("res", {}).get("status") == "COLLISION"],
        "report": _jsonable(full_report),
        "output_files": {
            "report_json_path": report_json_path,
            "report_txt_path": report_txt_path,
            "metrics_json_path": metrics_json_path,
            "visualization_path": visualization_path,
        },
        "message": "存在墩柱障碍物冲突。" if metrics["has_collision"] else "未检测到墩柱障碍物冲突。",
    }


def collision_detection(
    design_json_path: Optional[str] = None,
    design_result: Optional[Dict[str, Any]] = None,
    plane_json_path: Optional[str] = None,
    plane_paths: Optional[Dict[str, str]] = None,
    mask_path: Optional[str] = None,
    pgw_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    offset_left_m: float = 7.0,
    offset_right_m: float = 7.0,
    column_half_spacing_m: float = 3.45,
    check_radius_m: float = 1.25,
    mask_threshold: int = 127,
    save_visualization: bool = True,
) -> Dict[str, Any]:
    """碰撞检测入口。

    plane_paths: 分离式路基多线路平曲线 {route_id: plane_json_path}。
                 plane_json_path 作为单线路兜底。
    """
    try:
        if not plane_paths and plane_json_path:
            plane_paths = {"K": str(plane_json_path)}

        required = {
            "plane_paths": plane_paths,
            "mask_path": mask_path,
            "pgw_path": pgw_path,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"碰撞检测缺少必要输入: {missing}")

        if not output_dir:
            output_dir = os.path.join(os.getcwd(), "collision_detection_output")

        return _run_collision_detection(
            design_json_path=design_json_path,
            design_result=design_result,
            plane_paths=plane_paths,
            mask_path=str(mask_path),
            pgw_path=str(pgw_path),
            output_dir=str(output_dir),
            offset_left_m=offset_left_m,
            offset_right_m=offset_right_m,
            column_half_spacing_m=column_half_spacing_m,
            check_radius_m=check_radius_m,
            mask_threshold=mask_threshold,
            save_visualization=save_visualization,
        )
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "has_collision": None,
            "metrics": None,
            "collision_items": [],
            "report": [],
            "output_files": {},
        }


if tool is not None:
    collision_detection_tool = tool("collision_detection_tool")(collision_detection)
else:
    collision_detection_tool = collision_detection


if __name__ == "__main__":
    # 本地调试时在这里填写路径，Agent 调用时不走该入口。
    result = collision_detection(
        design_json_path=None,
        design_result=None,
        plane_json_path=None,
        mask_path=None,
        pgw_path=None,
        output_dir="./collision_detection_output",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
