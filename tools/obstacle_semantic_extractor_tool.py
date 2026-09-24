# tools/obstacle_semantic_extractor_tool.py
# -*- coding: utf-8 -*-

import os
import json
import cv2
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from langchain_core.tools import tool


# ============================================================
# 1. 基础组件：PGW 坐标映射与平曲线定位
#    基本沿用 Obstacle_data_reader.py 的逻辑
# ============================================================

class GeoReference:
    """PGW 世界文件坐标转换器：实现大地/工程平面坐标与图像像素坐标的互转。"""

    def __init__(self, pgw_path: str):
        if not os.path.exists(pgw_path):
            raise FileNotFoundError(f"PGW missing: {pgw_path}")

        with open(pgw_path, "r", encoding="utf-8") as f:
            vals = [float(x.strip()) for x in f.readlines() if x.strip()]

        if len(vals) < 6:
            raise ValueError(f"Invalid PGW file, expected 6 numbers: {pgw_path}")

        self.forward = np.array([
            [vals[0], vals[2], vals[4]],
            [vals[1], vals[3], vals[5]],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self.inv = np.linalg.inv(self.forward)
        self.res = abs(vals[0]) if abs(vals[0]) > 1e-12 else 1.0

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        vec = np.array([x, y, 1.0], dtype=float)
        res = self.inv @ vec
        return int(round(res[0])), int(round(res[1]))

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        vec = np.array([u, v, 1.0], dtype=float)
        res = self.forward @ vec
        return float(res[0]), float(res[1])


class AnchorRouteSystem:
    """基于平曲线 JSON 的“桩号 -> 平面坐标 + 切线角”定位系统。"""

    def __init__(self, json_path: str, route_id: str = "K"):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Plane JSON missing: {json_path}")

        self.json_path = json_path
        self.route_id = route_id.upper()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.segments = data.get("平曲线结构", [])
        if not self.segments:
            raise ValueError(f"No 平曲线结构 found in: {json_path}")

        self.segments.sort(key=lambda x: self.parse_k(x["起点"]["桩号"]))

    def get_route_range(self) -> Tuple[float, float]:
        vals = []
        for seg in self.segments:
            vals.append(self.parse_k(seg["起点"]["桩号"]))
            vals.append(self.parse_k(seg["终点"]["桩号"]))
        return min(vals), max(vals)

    @staticmethod
    def parse_k(k_str: Any) -> float:
        if isinstance(k_str, (float, int)):
            return float(k_str)

        text = str(k_str).strip().upper()

        for prefix in ["K", "Z", "Y", "L", "R", "CK", "AK", "BK"]:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break

        cleaned = [ch for ch in text if ch.isdigit() or ch in ["+", ".", "-"]]
        text = "".join(cleaned)

        try:
            if "+" in text:
                a, b = text.split("+", 1)
                is_negative = text.strip().startswith("-")
                base_k = abs(float(a)) * 1000.0 + float(b)
                return -base_k if is_negative else base_k

            return float(text)

        except Exception:
            return 0.0

    @staticmethod
    def format_k(k_val: float, prefix: str = "K") -> str:
        k_val = round(float(k_val), 3)
        sign_str = "-" if k_val < 0 else ""
        abs_val = abs(k_val)

        k_part = int(abs_val // 1000)
        m_part = abs_val - k_part * 1000

        if abs(m_part - round(m_part)) < 1e-9:
            return f"{prefix}{sign_str}{k_part}+{int(round(m_part)):03d}"

        return f"{prefix}{sign_str}{k_part}+{m_part:07.3f}"

    @staticmethod
    def _get_val(node: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in node and node[key] not in [None, ""]:
                try:
                    return float(node[key])
                except Exception:
                    pass
        return default

    def get_exact_pos(self, k_target: float) -> Tuple[float, float, float]:
        """返回指定桩号处的 (地图X, 地图Y, 切线方向角弧度)。"""

        target_seg = None

        for seg in self.segments:
            if (
                self.parse_k(seg["起点"]["桩号"]) - 0.01
                <= k_target
                <= self.parse_k(seg["终点"]["桩号"]) + 0.01
            ):
                target_seg = seg
                break

        if target_seg is None:
            if k_target < self.parse_k(self.segments[0]["起点"]["桩号"]):
                target_seg = self.segments[0]
            else:
                target_seg = self.segments[-1]

        s_node = target_seg["起点"]

        # HintCAD 坐标互换逻辑，沿用原脚本
        map_x = self._get_val(s_node, ["Y"])
        map_y = self._get_val(s_node, ["X"])

        azi = self._get_val(s_node, ["方位角", "方位角(°)"])
        anchor_theta = math.radians(90.0 - azi)

        s_k = self.parse_k(s_node["桩号"])
        dist_raw = k_target - s_k

        if abs(dist_raw) < 0.001:
            return map_x, map_y, anchor_theta

        ctype = str(target_seg.get("曲线类型", ""))
        R = float(target_seg.get("半径R", 0) or 0)

        s_azi = self._get_val(target_seg["起点"], ["方位角", "方位角(°)"])
        e_azi = self._get_val(target_seg["终点"], ["方位角", "方位角(°)"])

        delta_deg = (e_azi - s_azi + 360.0) % 360.0
        if delta_deg > 180.0:
            delta_deg -= 360.0

        math_sign = 0.0 if "直线" in ctype else (-1.0 if delta_deg > 0 else 1.0)

        k_start = 0.0
        k_end = 0.0

        if "圆" in ctype:
            k_start = k_end = math_sign * (1.0 / R if abs(R) > 1e-12 else 0.0)

        elif "缓" in ctype:
            max_k = math_sign * (1.0 / R if abs(R) > 1e-12 else 0.0)
            k_start, k_end = (max_k, 0.0) if "后缓" in ctype else (0.0, max_k)

        step = 0.5
        curr_dist = 0.0
        curr_x = map_x
        curr_y = map_y
        curr_theta = anchor_theta

        direction = 1.0 if dist_raw >= 0 else -1.0
        dist = abs(dist_raw)
        seg_len = max(self.parse_k(target_seg["终点"]["桩号"]) - s_k, 1e-9)

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
# 2. 核心组件：障碍物语义分析器
#    基本沿用 Obstacle_data_reader.py 的逻辑
# ============================================================

class ObstacleProfiler:
    """支持整体式与分离式路基的障碍物扫描器。"""

    def __init__(
        self,
        route_system: AnchorRouteSystem,
        mask_path: str,
        pgw_path: str,
        section_name: str = "Default",
        route_id: str = "K",
        carriageway: str = "both",
        roadbed_type: str = "integrated",
        scan_side: str = "both",
        station_prefix: Optional[str] = None,
        scan_width: float = 26.5,
        sample_step: float = 0.5,
    ):
        self.route = route_system
        self.geo = GeoReference(pgw_path)

        self.section_name = section_name
        self.route_id = route_id.upper()
        self.carriageway = carriageway
        self.roadbed_type = roadbed_type
        self.scan_side = scan_side
        self.station_prefix = station_prefix or self.route_id
        self.scan_width = float(scan_width)
        self.sample_step = float(sample_step)

        self.raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.raw_mask is None:
            raise ValueError(f"[{self.section_name}] Mask load failed: {mask_path}")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.mask = cv2.morphologyEx(self.raw_mask, cv2.MORPH_CLOSE, kernel)

        _, self.mask = cv2.threshold(self.mask, 100, 255, cv2.THRESH_BINARY)

        self.h, self.w = self.mask.shape

    def format_k(self, k_val: float) -> str:
        return AnchorRouteSystem.format_k(k_val, prefix=self.station_prefix)

    def _make_offsets(
        self,
        scan_width: Optional[float] = None,
        step: Optional[float] = None,
    ) -> np.ndarray:
        width = float(scan_width if scan_width is not None else self.scan_width)
        ds = float(step if step is not None else self.sample_step)

        if self.scan_side == "both":
            return np.arange(-width / 2.0, width / 2.0 + 1e-9, ds)

        if self.scan_side == "left":
            return np.arange(0.0, width + 1e-9, ds)

        return np.arange(-width, 0.0 + 1e-9, ds)

    def get_scan_slice(
        self,
        k_val: float,
        scan_width: Optional[float] = None,
        step: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        cx, cy, theta_rad = self.route.get_exact_pos(k_val)
        normal_rad = theta_rad + math.pi / 2.0

        offsets = self._make_offsets(scan_width=scan_width, step=step)

        segments = []
        current_seg = None

        for off in offsets:
            u, v = self.geo.world_to_pixel(
                cx + off * math.cos(normal_rad),
                cy + off * math.sin(normal_rad),
            )

            is_obstacle = (
                0 <= u < self.w
                and 0 <= v < self.h
                and self.mask[v, u] > 50
            )

            if is_obstacle:
                if current_seg is None:
                    current_seg = {
                        "start": float(off),
                        "end": float(off),
                    }
                else:
                    current_seg["end"] = float(off)

            else:
                if current_seg:
                    if abs(current_seg["end"] - current_seg["start"]) >= self.sample_step:
                        segments.append(current_seg)
                    current_seg = None

        if current_seg and abs(current_seg["end"] - current_seg["start"]) >= self.sample_step:
            segments.append(current_seg)

        return segments

    def scan_range(
        self,
        start_k: float,
        end_k: float,
        step_k: float = 0.5,
    ) -> List[Dict[str, Any]]:
        raw_scans = []

        step_val = abs(step_k) * (1.0 if end_k >= start_k else -1.0)

        for k in np.arange(start_k, end_k, step_val):
            segs = self.get_scan_slice(float(k))
            if segs:
                raw_scans.append({
                    "k": float(k),
                    "segs": segs,
                })

        return self._cluster_obstacles_strict(raw_scans)

    def _cluster_obstacles_strict(
        self,
        raw_scans: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        obstacles = []
        active_obs = []

        MAX_GAP = 6.0
        MAX_SHIFT = 5.0

        for scan in raw_scans:
            k = scan["k"]
            segs = scan["segs"]

            for obs in active_obs:
                obs["updated_this_round"] = False

            for seg in segs:
                matched = False
                seg_center = (seg["start"] + seg["end"]) / 2.0

                for obs in active_obs:
                    obs_center = (obs["last_l"] + obs["last_r"]) / 2.0

                    if (
                        abs(k - obs["last_k"]) <= MAX_GAP
                        and not obs["updated_this_round"]
                        and abs(seg_center - obs_center) < MAX_SHIFT
                    ):
                        obs.update({
                            "end_k": k,
                            "last_k": k,
                            "min_offset": min(obs["min_offset"], seg["start"]),
                            "max_offset": max(obs["max_offset"], seg["end"]),
                            "last_l": seg["start"],
                            "last_r": seg["end"],
                            "updated_this_round": True,
                        })

                        obs["profile"].append((k, seg["start"], seg["end"]))

                        matched = True
                        break

                if not matched:
                    active_obs.append({
                        "id": 0,
                        "start_k": k,
                        "end_k": k,
                        "last_k": k,
                        "min_offset": seg["start"],
                        "max_offset": seg["end"],
                        "last_l": seg["start"],
                        "last_r": seg["end"],
                        "profile": [(k, seg["start"], seg["end"])],
                        "updated_this_round": True,
                        "finished": False,
                    })

            clean_active = []

            for obs in active_obs:
                if not obs["updated_this_round"] and abs(k - obs["last_k"]) > MAX_GAP:
                    obs["finished"] = True
                    obstacles.append(obs)
                else:
                    clean_active.append(obs)

            active_obs = clean_active

        obstacles.extend(active_obs)

        result = []

        for obs in obstacles:
            if abs(obs["end_k"] - obs["start_k"]) > 2.0:
                copied = dict(obs)
                copied.update({
                    "roadbed_type": self.roadbed_type,
                    "route_id": self.route_id,
                    "carriageway": self.carriageway,
                    "scan_side": self.scan_side,
                    "station_prefix": self.station_prefix,
                })
                result.append(copied)

        return result

    def generate_prompt_description(
        self,
        obstacles: List[Dict[str, Any]],
        start_uid: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        descriptions = []
        uid = int(start_uid)

        for obs in obstacles:
            length = abs(obs["end_k"] - obs["start_k"])

            avg_width = (
                float(np.mean([abs(r - l) for _, l, r in obs["profile"]]))
                if obs["profile"]
                else 0.0
            )

            ratio = (
                sum(1 for _, l, r in obs["profile"] if l < -0.5 and r > 0.5)
                / max(len(obs["profile"]), 1)
            )

            side_sum = sum((pl + pr) / 2.0 for _, pl, pr in obs["profile"])

            obs["id"] = uid

            desc = {
                "id": uid,
                "range": f"{self.format_k(obs['start_k'])} ~ {self.format_k(obs['end_k'])}",
                "length": f"{length:.1f}m",
                "avg_width": f"{avg_width:.1f}m",
                "roadbed_type": self.roadbed_type,
            }

            if ratio > 0.3:
                desc.update({
                    "geometry": f"横向截断线路，平均物理宽度 {avg_width:.1f}m。",
                    "advice": f"建议采用 {max(30, length + 5):.0f}m 以上跨径跨越。",
                })

            else:
                side_zh = "右侧" if side_sum < 0 else "左侧"
                avoid_dir = "右" if side_zh == "左侧" else "左"

                desc.update({
                    "geometry": f"位于线路{side_zh}的侧向侵入。",
                    "advice": f"建议向{avoid_dir}侧偏置墩位。",
                })

            descriptions.append(desc)
            uid += 1

        return descriptions, uid

    def visualize_scan(
        self,
        obstacles: List[Dict[str, Any]],
        draw_k_range: Tuple[float, float],
        window_name: Optional[str] = None,
        show: bool = True,
        wait_ms: int = 0,
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """生成单条路线扫描示意图。"""

        if not show and not save_path:
            return None

        vis_img = cv2.cvtColor(self.raw_mask, cv2.COLOR_GRAY2BGR)

        center_pts = []
        bound_a_pts = []
        bound_b_pts = []

        start_k, end_k = draw_k_range
        direction = 1.0 if end_k >= start_k else -1.0

        offsets = self._make_offsets(
            scan_width=self.scan_width,
            step=max(self.scan_width, 1.0),
        )

        off_min = float(np.min(offsets))
        off_max = float(np.max(offsets))

        for k in np.arange(start_k, end_k, 2.0 * direction):
            cx, cy, theta = self.route.get_exact_pos(float(k))
            normal = theta + math.pi / 2.0

            center_pts.append(self.geo.world_to_pixel(cx, cy))
            bound_a_pts.append(
                self.geo.world_to_pixel(
                    cx + off_min * math.cos(normal),
                    cy + off_min * math.sin(normal),
                )
            )
            bound_b_pts.append(
                self.geo.world_to_pixel(
                    cx + off_max * math.cos(normal),
                    cy + off_max * math.sin(normal),
                )
            )

        def draw_polyline(pts, color, thickness=1):
            if pts:
                cv2.polylines(
                    vis_img,
                    [np.array(pts, dtype=np.int32).reshape((-1, 1, 2))],
                    False,
                    color,
                    thickness,
                )

        draw_polyline(bound_a_pts, (150, 150, 150), 1)
        draw_polyline(bound_b_pts, (150, 150, 150), 1)
        draw_polyline(center_pts, (255, 0, 0), 2)

        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]

        for obs in obstacles:
            color = colors[int(obs.get("id", 0)) % len(colors)]

            for k, l, r in obs["profile"]:
                cx, cy, theta = self.route.get_exact_pos(float(k))
                normal = theta + math.pi / 2.0

                u1, v1 = self.geo.world_to_pixel(
                    cx + l * math.cos(normal),
                    cy + l * math.sin(normal),
                )
                u2, v2 = self.geo.world_to_pixel(
                    cx + r * math.cos(normal),
                    cy + r * math.sin(normal),
                )

                cv2.line(vis_img, (u1, v1), (u2, v2), color, 2)

            if obs["profile"]:
                mid = obs["profile"][len(obs["profile"]) // 2]
                mu, mv = self.geo.world_to_pixel(
                    *self.route.get_exact_pos(float(mid[0]))[:2]
                )

                label = f"{obs.get('id', 0)}"

                cv2.putText(
                    vis_img,
                    label,
                    (mu, mv),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    vis_img,
                    label,
                    (mu, mv),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, vis_img)

        if show:
            win = window_name or f"Obstacle scan - {self.section_name}"
            try:
                cv2.imshow(win, vis_img)
                cv2.waitKey(int(wait_ms))
                cv2.destroyWindow(win)
            except Exception as e:
                print(f"[警告] 无法弹窗展示。原因: {e}")

        return vis_img

    def draw_scan_on_canvas(
        self,
        canvas,
        obstacles: List[Dict[str, Any]],
        draw_k_range: Tuple[float, float],
        route_color=(0, 0, 255),
        bound_color=(150, 150, 150),
        obstacle_color=(0, 255, 255),
        label_prefix="",
    ) -> None:
        """将当前路线数据绘制到外部画布，用于生成合并图。"""

        center_pts = []
        bound_a_pts = []
        bound_b_pts = []

        start_k, end_k = draw_k_range
        direction = 1.0 if end_k >= start_k else -1.0

        offsets = self._make_offsets(
            scan_width=self.scan_width,
            step=max(self.scan_width, 1.0),
        )

        off_min = float(np.min(offsets))
        off_max = float(np.max(offsets))

        for k in np.arange(start_k, end_k, 2.0 * direction):
            cx, cy, theta = self.route.get_exact_pos(float(k))
            normal = theta + math.pi / 2.0

            center_pts.append(self.geo.world_to_pixel(cx, cy))
            bound_a_pts.append(
                self.geo.world_to_pixel(
                    cx + off_min * math.cos(normal),
                    cy + off_min * math.sin(normal),
                )
            )
            bound_b_pts.append(
                self.geo.world_to_pixel(
                    cx + off_max * math.cos(normal),
                    cy + off_max * math.sin(normal),
                )
            )

        def draw_polyline(pts, color, thickness=1):
            if pts:
                cv2.polylines(
                    canvas,
                    [np.array(pts, dtype=np.int32).reshape((-1, 1, 2))],
                    False,
                    color,
                    thickness,
                )

        draw_polyline(bound_a_pts, bound_color, 1)
        draw_polyline(bound_b_pts, bound_color, 1)
        draw_polyline(center_pts, route_color, 2)

        for obs in obstacles:
            for k, l, r in obs["profile"]:
                cx, cy, theta = self.route.get_exact_pos(float(k))
                normal = theta + math.pi / 2.0

                u1, v1 = self.geo.world_to_pixel(
                    cx + l * math.cos(normal),
                    cy + l * math.sin(normal),
                )
                u2, v2 = self.geo.world_to_pixel(
                    cx + r * math.cos(normal),
                    cy + r * math.sin(normal),
                )

                cv2.line(canvas, (u1, v1), (u2, v2), obstacle_color, 2)

            if obs["profile"]:
                mid = obs["profile"][len(obs["profile"]) // 2]
                mu, mv = self.geo.world_to_pixel(
                    *self.route.get_exact_pos(float(mid[0]))[:2]
                )

                label = f"{label_prefix}{obs.get('id', '')}"

                cv2.putText(
                    canvas,
                    label,
                    (mu, mv),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    canvas,
                    label,
                    (mu, mv),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )


# ============================================================
# 3. 从第一步纬地读取结果中生成 plane.json
# ============================================================

def ensure_dict(data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """兼容 agent 传入 dict 或 JSON 字符串。"""

    if isinstance(data, dict):
        return data

    if isinstance(data, str):
        return json.loads(data)

    raise TypeError("输入数据必须是 dict 或 JSON 字符串")


def sanitize_filename(name: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in ["_", "-", "."] else "_"
        for ch in str(name)
    )

    while "__" in safe:
        safe = safe.replace("__", "_")

    return (safe.strip("_") or "section")[:120]


def station_to_float_from_text(k_str: Any) -> float:
    return AnchorRouteSystem.parse_k(k_str)


def write_plane_json_from_route_data(
    route_data: Dict[str, Any],
    output_dir: str,
    route_id: str,
) -> str:
    """
    从第一步 data_loader_tool 的输出结果中提取“平曲线结构”，写出内部 plane.json。

    route_data 可能是：
        1. 单线路数据本身；
        2. 多线路数据中的 route_data["线路数据"]["K"] / ["Z1"]。
    """

    if not isinstance(route_data, dict):
        raise TypeError("route_data 必须是 dict")

    plane_segments = route_data.get("平曲线结构", [])

    if not plane_segments:
        raise ValueError(f"route_id={route_id} 的数据中未找到 平曲线结构，无法生成 plane.json")

    plane_data = {
        "平曲线结构": plane_segments
    }

    plane_dir = os.path.join(output_dir, "plane_from_loader")
    os.makedirs(plane_dir, exist_ok=True)

    plane_path = os.path.join(
        plane_dir,
        f"{sanitize_filename(route_id)}_plane.json"
    )

    with open(plane_path, "w", encoding="utf-8") as f:
        json.dump(plane_data, f, ensure_ascii=False, indent=2)

    return plane_path


def get_route_data_map(preprocessed_route_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    将第一步 data_loader_tool 输出统一转成：
        {
          "K": {...},
          "Z1": {...},
          ...
        }

    兼容两种形式：
    1. 单线路：
        {
          "线路编号": "...",
          "平曲线结构": [...]
        }

    2. 多线路：
        {
          "线路数据": {
            "K": {...},
            "Z1": {...}
          }
        }
    """

    if "线路数据" in preprocessed_route_data:
        route_map = preprocessed_route_data.get("线路数据", {})
        if not isinstance(route_map, dict):
            raise ValueError("线路数据 字段必须是 dict")
        return route_map

    route_id = (
        preprocessed_route_data.get("线路编号")
        or preprocessed_route_data.get("route_id")
        or "K"
    )

    return {
        str(route_id).upper(): preprocessed_route_data
    }


def parse_scan_range_from_loader(
    preprocessed_route_data: Dict[str, Any],
    start_k: Optional[Union[str, float]],
    end_k: Optional[Union[str, float]],
) -> Tuple[float, float]:
    """
    获取扫描起终点。

    优先使用工具参数 start_k/end_k；
    若未传，则尝试从第一步结果中的“截取范围”解析。
    """

    if start_k is not None and end_k is not None:
        return (
            station_to_float_from_text(start_k),
            station_to_float_from_text(end_k),
        )

    range_text = preprocessed_route_data.get("截取范围", "")
    if not range_text:
        raise ValueError("未提供 start_k/end_k，且第一步结果中无 截取范围")

    text = str(range_text)

    for sep in ["~", "～", "-", "—", "–", "至", "到"]:
        text = text.replace(sep, "|")

    parts = [p.strip() for p in text.split("|") if p.strip()]

    if len(parts) < 2:
        raise ValueError(f"无法从 截取范围 解析起终点: {range_text}")

    return (
        station_to_float_from_text(parts[0]),
        station_to_float_from_text(parts[1]),
    )


def infer_default_route_settings(
    route_id: str,
    roadbed_type: str,
) -> Dict[str, str]:
    """
    根据路线编号和路基类型推断默认扫描参数。

    与 Obstacle_data_reader.py 的默认习惯保持一致：
    - 整体式：K / both / both
    - 分离式：K 默认 right；Z/Z1/Z2 默认 left
    """

    rid = str(route_id).upper()

    if roadbed_type == "separated":
        if rid == "K":
            return {
                "carriageway": "right",
                "scan_side": "right",
                "station_prefix": "K",
            }

        return {
            "carriageway": "left",
            "scan_side": "left",
            "station_prefix": "K",
        }

    return {
        "carriageway": "both",
        "scan_side": "both",
        "station_prefix": "K",
    }


# ============================================================
# 4. 执行函数：从第一步结果 + mask/pgw 提取障碍物语义
# ============================================================

def run_obstacle_semantic_extraction_from_loader(
    preprocessed_route_data: Union[str, Dict[str, Any]],
    mask_path: str,
    pgw_path: str,
    output_dir: str,
    roadbed_type: str = "integrated",
    start_k: Optional[Union[str, float]] = None,
    end_k: Optional[Union[str, float]] = None,
    section_name: str = "section",
    route_scan_config: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    scan_width_integrated: float = 26.5,
    scan_width_separated: float = 13.25,
    sample_step: float = 0.5,
    step_k: float = 0.5,
    clip_to_route_range: bool = True,
    save_visualization: bool = True,
    show_visualization: bool = False,
    preview_wait_ms: int = 0,
) -> Dict[str, Any]:
    """
    从第一步纬地读取结果中提取平曲线结构，生成内部 plane.json，
    再结合 mask / pgw 扫描障碍物语义信息。
    """

    route_data_obj = ensure_dict(preprocessed_route_data)

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"mask 文件不存在: {mask_path}")

    if not os.path.exists(pgw_path):
        raise FileNotFoundError(f"pgw 文件不存在: {pgw_path}")

    os.makedirs(output_dir, exist_ok=True)

    start_val, end_val = parse_scan_range_from_loader(
        route_data_obj,
        start_k=start_k,
        end_k=end_k,
    )

    route_map = get_route_data_map(route_data_obj)

    # 可选扫描配置
    custom_scan_config = None
    if route_scan_config:
        if isinstance(route_scan_config, str):
            custom_scan_config = json.loads(route_scan_config)
        else:
            custom_scan_config = route_scan_config

    if isinstance(custom_scan_config, list):
        custom_config_map = {
            str(item.get("route_id", "")).upper(): item
            for item in custom_scan_config
            if isinstance(item, dict)
        }
    elif isinstance(custom_scan_config, dict):
        custom_config_map = {
            str(k).upper(): v
            for k, v in custom_scan_config.items()
            if isinstance(v, dict)
        }
    else:
        custom_config_map = {}

    grouped_by_route: Dict[str, List[Dict[str, Any]]] = {}
    raw_obstacles_by_route: Dict[str, List[Dict[str, Any]]] = {}
    plane_paths: Dict[str, str] = {}
    visual_items = []

    uid = 1

    for route_id, one_route_data in route_map.items():
        rid = str(route_id).upper()

        plane_json_path = write_plane_json_from_route_data(
            route_data=one_route_data,
            output_dir=output_dir,
            route_id=rid,
        )

        plane_paths[rid] = plane_json_path

        default_settings = infer_default_route_settings(
            route_id=rid,
            roadbed_type=roadbed_type,
        )

        route_cfg = custom_config_map.get(rid, {})

        carriageway = route_cfg.get(
            "carriageway",
            default_settings["carriageway"],
        )

        scan_side = route_cfg.get(
            "scan_side",
            default_settings["scan_side"],
        )

        station_prefix = route_cfg.get(
            "station_prefix",
            default_settings["station_prefix"],
        )

        scan_width = float(
            route_cfg.get(
                "scan_width",
                scan_width_separated if roadbed_type == "separated" else scan_width_integrated,
            )
        )

        route_step_k = float(route_cfg.get("step_k", step_k))
        route_sample_step = float(route_cfg.get("sample_step", sample_step))

        route_sys = AnchorRouteSystem(
            plane_json_path,
            route_id=rid,
        )

        route_min_k, route_max_k = route_sys.get_route_range()

        scan_start_k = start_val
        scan_end_k = end_val

        if clip_to_route_range:
            low = min(start_val, end_val)
            high = max(start_val, end_val)

            clipped_low = max(low, route_min_k)
            clipped_high = min(high, route_max_k)

            if clipped_low >= clipped_high:
                print(f"[警告] 路线 {rid} 超出扫描范围，跳过。")
                continue

            if start_val <= end_val:
                scan_start_k, scan_end_k = clipped_low, clipped_high
            else:
                scan_start_k, scan_end_k = clipped_high, clipped_low

        profiler = ObstacleProfiler(
            route_system=route_sys,
            mask_path=mask_path,
            pgw_path=pgw_path,
            section_name=f"{section_name}_{rid}_{carriageway}",
            route_id=rid,
            carriageway=carriageway,
            roadbed_type=roadbed_type,
            scan_side=scan_side,
            station_prefix=station_prefix,
            scan_width=scan_width,
            sample_step=route_sample_step,
        )

        obstacles = profiler.scan_range(
            scan_start_k,
            scan_end_k,
            step_k=route_step_k,
        )

        prompt_desc, uid = profiler.generate_prompt_description(
            obstacles,
            start_uid=uid,
        )

        route_key = f"{rid}_{carriageway}"
        grouped_by_route[route_key] = prompt_desc
        raw_obstacles_by_route[route_key] = obstacles

        single_vis_path = None

        if save_visualization:
            single_vis_path = os.path.join(
                output_dir,
                "obstacle_semantic",
                f"vis_{sanitize_filename(section_name)}_{sanitize_filename(route_key)}.jpg",
            )

        profiler.visualize_scan(
            obstacles=obstacles,
            draw_k_range=(scan_start_k, scan_end_k),
            window_name=f"{section_name} - {route_key}",
            show=show_visualization,
            wait_ms=preview_wait_ms,
            save_path=single_vis_path,
        )

        visual_items.append({
            "profiler": profiler,
            "obstacles": obstacles,
            "range": (scan_start_k, scan_end_k),
            "route_id": rid,
            "carriageway": carriageway,
            "route_key": route_key,
        })

    merged_vis_path = None

    if save_visualization and visual_items:
        base_profiler = visual_items[0]["profiler"]
        merged_canvas = cv2.cvtColor(base_profiler.raw_mask, cv2.COLOR_GRAY2BGR)

        for item in visual_items:
            rid = item["route_id"]
            carriageway = item["carriageway"]

            if rid == "K":
                route_color = (0, 0, 255)
                obstacle_color = (0, 255, 255)
                label_prefix = "K-"
            elif rid == "Z":
                route_color = (255, 0, 0)
                obstacle_color = (0, 255, 0)
                label_prefix = "Z-"
            else:
                route_color = (255, 0, 255)
                obstacle_color = (0, 255, 255)
                label_prefix = f"{rid}-"

            item["profiler"].draw_scan_on_canvas(
                canvas=merged_canvas,
                obstacles=item["obstacles"],
                draw_k_range=item["range"],
                route_color=route_color,
                bound_color=(160, 160, 160),
                obstacle_color=obstacle_color,
                label_prefix=label_prefix,
            )

            if item["obstacles"]:
                k0 = item["obstacles"][0]["profile"][0][0]
            else:
                k0 = item["range"][0]

            x0, y0, _ = item["profiler"].route.get_exact_pos(float(k0))
            u0, v0 = item["profiler"].geo.world_to_pixel(x0, y0)

            cv2.putText(
                merged_canvas,
                f"{rid}_{carriageway}",
                (u0 + 10, v0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                route_color,
                2,
                cv2.LINE_AA,
            )

        merged_vis_path = os.path.join(
            output_dir,
            "obstacle_semantic",
            f"merged_vis_{sanitize_filename(section_name)}.jpg",
        )

        os.makedirs(os.path.dirname(merged_vis_path), exist_ok=True)
        cv2.imwrite(merged_vis_path, merged_canvas)

    total_count = sum(len(v) for v in grouped_by_route.values())

    output_json_path = os.path.join(
        output_dir,
        "obstacle_semantic",
        f"complete_obstacles_grouped_{sanitize_filename(section_name)}.json",
    )

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    output_data = {
        section_name: grouped_by_route
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "section_name": section_name,
        "roadbed_type": roadbed_type,
        "start_k": start_val,
        "end_k": end_val,
        "plane_paths": plane_paths,
        "obstacle_count": total_count,
        "obstacle_grouped": output_data,
        "obstacle_json_path": output_json_path,
        "merged_visualization_path": merged_vis_path,
        "mask_path": mask_path,
        "pgw_path": pgw_path,
    }


# ============================================================
# 5. LangChain Agent 工具封装
# ============================================================

@tool
def obstacle_semantic_extractor_tool(
    preprocessed_route_data_json: str,
    mask_path: str,
    pgw_path: str,
    output_dir: str,
    roadbed_type: str = "integrated",
    start_k: str = None,
    end_k: str = None,
    section_name: str = "section",
    route_scan_config_json: str = None,
    scan_width_integrated: float = 26.5,
    scan_width_separated: float = 13.25,
    sample_step: float = 0.5,
    step_k: float = 0.5,
    clip_to_route_range: bool = True,
    save_visualization: bool = True,
    show_visualization: bool = False,
    preview_wait_ms: int = 0,
) -> dict:
    """
    障碍物语义提取工具。

    该工具不要求用户直接提供 plane.json。
    plane.json 由第一步纬地读取结果中的“平曲线结构”自动生成。

    输入：
        preprocessed_route_data_json:
            第一阶段 data_loader_tool 的输出结果，JSON 字符串。
            可为单线路结构，也可为多线路结构：
            {
              "线路数据": {
                "K": {...},
                "Z1": {...}
              }
            }

        mask_path:
            上一环节 drawing_crop_and_mask_tool 输出的 mask_path。

        pgw_path:
            上一环节 drawing_crop_and_mask_tool 输出的 pgw_path。

        output_dir:
            输出目录。

        roadbed_type:
            integrated 或 separated。

        start_k / end_k:
            可选。若不传，则从 preprocessed_route_data_json["截取范围"] 中解析。

        section_name:
            当前区段名称。

        route_scan_config_json:
            可选。用于覆盖每条路线的扫描参数。
            示例：
            {
              "K": {
                "carriageway": "right",
                "scan_side": "right",
                "station_prefix": "K",
                "scan_width": 13.25
              },
              "Z1": {
                "carriageway": "left",
                "scan_side": "left",
                "station_prefix": "K",
                "scan_width": 13.25
              }
            }

    输出：
        {
          "success": true,
          "obstacle_grouped": {},
          "obstacle_json_path": "",
          "plane_paths": {},
          "merged_visualization_path": ""
        }
    """

    try:
        return run_obstacle_semantic_extraction_from_loader(
            preprocessed_route_data=preprocessed_route_data_json,
            mask_path=mask_path,
            pgw_path=pgw_path,
            output_dir=output_dir,
            roadbed_type=roadbed_type,
            start_k=start_k,
            end_k=end_k,
            section_name=section_name,
            route_scan_config=route_scan_config_json,
            scan_width_integrated=scan_width_integrated,
            scan_width_separated=scan_width_separated,
            sample_step=sample_step,
            step_k=step_k,
            clip_to_route_range=clip_to_route_range,
            save_visualization=save_visualization,
            show_visualization=show_visualization,
            preview_wait_ms=preview_wait_ms,
        )

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }