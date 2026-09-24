# tools/drawing_mask_preprocess_tool.py
# -*- coding: utf-8 -*-

import os
import json
import math
import subprocess
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple

import cv2
import ezdxf
import numpy as np
from tqdm import tqdm

from langchain_core.tools import tool


# ======================================================
# 1. DWG 转 DXF 模块
# ======================================================

class DwgConverter:
    def __init__(self, converter_path):
        self.converter_exe = converter_path

    def convert(self, input_dwg_path, output_dir):
        if not os.path.exists(input_dwg_path):
             raise FileNotFoundError(f"输入文件不存在: {input_dwg_path}")

        input_dir = os.path.dirname(input_dwg_path)
        cmd = [
            self.converter_exe,
            input_dir,
            output_dir,
            "ACAD2018", "DXF", "0", "0"
        ]

        print(f"正在转换 DWG: {input_dwg_path} ...")
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)

        filename = os.path.basename(input_dwg_path).replace(".dwg", ".dxf")
        output_dxf_path = os.path.join(output_dir, filename)

        if os.path.exists(output_dxf_path):
            print(f"转换成功: {output_dxf_path}")
            return output_dxf_path
        else:
            raise FileNotFoundError("DWG转换失败")


# ======================================================
# 2. 平曲线计算器
# 来源逻辑：Segmentation_extractor.py
# ======================================================

class RealAlignmentCalculator:
    def __init__(self, json_path, swap_xy=True):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到平曲线文件: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.segments = data["平曲线结构"]
        self.swap_xy = swap_xy

        for seg in self.segments:
            seg["start_k_val"] = self.parse_k(seg["起点"]["桩号"])
            seg["end_k_val"] = self.parse_k(seg["终点"]["桩号"])

    @staticmethod
    def parse_k(k_str):
        s = str(k_str).upper().replace("K", "").strip()
        sign = 1

        if s.startswith("-"):
            sign = -1
            s = s[1:]

        if "+" in s:
            km, m = s.split("+", 1)
            return sign * (float(km) * 1000.0 + float(m))

        return sign * float(s)

    def get_coords_at_k(self, k_val):
        target_seg = None

        for seg in self.segments:
            if seg["start_k_val"] - 0.01 <= k_val <= seg["end_k_val"] + 0.01:
                target_seg = seg
                break

        if target_seg is None:
            if k_val < self.segments[0]["start_k_val"]:
                seg = self.segments[0]
                p = seg["起点"]
            else:
                seg = self.segments[-1]
                p = seg["终点"]

            raw_x = p["X"]
            raw_y = p["Y"]

        else:
            s_x = target_seg["起点"]["X"]
            s_y = target_seg["起点"]["Y"]
            e_x = target_seg["终点"]["X"]
            e_y = target_seg["终点"]["Y"]

            seg_len = target_seg["end_k_val"] - target_seg["start_k_val"]
            ratio = (k_val - target_seg["start_k_val"]) / seg_len if abs(seg_len) > 1e-9 else 0.0

            raw_x = s_x + (e_x - s_x) * ratio
            raw_y = s_y + (e_y - s_y) * ratio

        if self.swap_xy:
            return raw_y, raw_x

        return raw_x, raw_y

    def get_bounding_box(self, start_k_str, end_k_str, buffer=200.0, sample_step=50.0):
        s_val = self.parse_k(start_k_str)
        e_val = self.parse_k(end_k_str)

        if e_val < s_val:
            s_val, e_val = e_val, s_val

        sample_ks = np.arange(s_val, e_val, sample_step)
        sample_ks = np.append(sample_ks, e_val)

        coords = np.array([self.get_coords_at_k(k) for k in sample_ks], dtype=float)

        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)

        return (
            float(min_x - buffer),
            float(max_x + buffer),
            float(min_y - buffer),
            float(max_y + buffer),
        )


# ======================================================
# 3. DXF 几何工具函数
# 来源逻辑：Segmentation_extractor.py
# ======================================================

def rgb_to_bgr(color_rgb):
    r, g, b = color_rgb
    return int(b), int(g), int(r)


def normalize_layer_name(layer_name):
    return str(layer_name).upper().strip()


def bbox_of_points(points):
    if not points:
        return None

    arr = np.asarray(points, dtype=float)
    return (
        float(np.min(arr[:, 0])),
        float(np.max(arr[:, 0])),
        float(np.min(arr[:, 1])),
        float(np.max(arr[:, 1])),
    )


def bbox_intersects(bbox, bounds):
    if bbox is None:
        return False

    min_x, max_x, min_y, max_y = bounds
    b_min_x, b_max_x, b_min_y, b_max_y = bbox

    if b_max_x < min_x:
        return False
    if b_min_x > max_x:
        return False
    if b_max_y < min_y:
        return False
    if b_min_y > max_y:
        return False

    return True


def sample_arc_points(cx, cy, radius, start_angle_deg, end_angle_deg, step_deg=3.0):
    start = float(start_angle_deg)
    end = float(end_angle_deg)

    while end < start:
        end += 360.0

    span = end - start
    n = max(int(math.ceil(span / step_deg)), 2)

    angles = np.linspace(start, end, n + 1)
    pts = []

    for a in angles:
        rad = math.radians(a)
        x = cx + radius * math.cos(rad)
        y = cy + radius * math.sin(rad)
        pts.append((float(x), float(y)))

    return pts


def sample_circle_points(cx, cy, radius, step_deg=3.0):
    n = max(int(math.ceil(360.0 / step_deg)), 16)
    angles = np.linspace(0.0, 360.0, n + 1)
    pts = []

    for a in angles:
        rad = math.radians(a)
        x = cx + radius * math.cos(rad)
        y = cy + radius * math.sin(rad)
        pts.append((float(x), float(y)))

    return pts


def has_lwpolyline_bulge(entity):
    try:
        for p in entity.get_points("xyseb"):
            if len(p) >= 5 and abs(float(p[4])) > 1e-12:
                return True
    except Exception:
        return False
    return False


# ======================================================
# 4. DXF 快速栅格化渲染器
# 来源逻辑：Segmentation_extractor.py
# ======================================================

class FastDxfRasterRenderer:
    def __init__(
        self,
        dxf_path,
        layer_config,
        resolution=0.5,
        expand_inserts=False,
        arc_step_deg=3.0,
        spline_flatten_tol=0.5,
        circle_step_deg=3.0,
        max_image_pixels=20000,
        background_color=(255, 255, 255),
    ):
        if not os.path.exists(dxf_path):
            raise FileNotFoundError(f"DXF 文件不存在: {dxf_path}")

        self.dxf_path = dxf_path
        self.layer_config = {
            normalize_layer_name(k): v for k, v in layer_config.items()
        }
        self.target_layers = set(self.layer_config.keys())
        self.resolution = float(resolution)
        self.expand_inserts = bool(expand_inserts)
        self.arc_step_deg = float(arc_step_deg)
        self.spline_flatten_tol = float(spline_flatten_tol)
        self.circle_step_deg = float(circle_step_deg)
        self.max_image_pixels = int(max_image_pixels)
        self.background_color = background_color

        print(f"正在读取 DXF 文件: {dxf_path}")
        self.doc = ezdxf.readfile(dxf_path)
        self.msp = self.doc.modelspace()

        self.stats = defaultdict(int)
        self.drawn_stats = defaultdict(int)
        self.unsupported_stats = defaultdict(int)

    def world_to_pixel(self, x, y, bounds):
        min_x, max_x, min_y, max_y = bounds
        col = int(round((x - min_x) / self.resolution))
        row = int(round((max_y - y) / self.resolution))
        return col, row

    def draw_polyline(self, image, points, bounds, color_bgr, thickness=1, closed=False):
        if len(points) < 2:
            return False

        bbox = bbox_of_points(points)
        if not bbox_intersects(bbox, bounds):
            return False

        pix = [self.world_to_pixel(x, y, bounds) for x, y in points]
        pix_arr = np.asarray(pix, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            image,
            [pix_arr],
            isClosed=bool(closed),
            color=color_bgr,
            thickness=int(max(1, thickness)),
            lineType=cv2.LINE_AA,
        )

        return True

    def draw_line_entity(self, image, entity, bounds, style):
        s = entity.dxf.start
        e = entity.dxf.end
        pts = [(float(s.x), float(s.y)), (float(e.x), float(e.y))]

        return self.draw_polyline(
            image=image,
            points=pts,
            bounds=bounds,
            color_bgr=style["color_bgr"],
            thickness=style["thickness"],
            closed=False,
        )

    def draw_arc_entity(self, image, entity, bounds, style):
        center = entity.dxf.center
        radius = float(entity.dxf.radius)
        start_angle = float(entity.dxf.start_angle)
        end_angle = float(entity.dxf.end_angle)

        pts = sample_arc_points(
            cx=float(center.x),
            cy=float(center.y),
            radius=radius,
            start_angle_deg=start_angle,
            end_angle_deg=end_angle,
            step_deg=self.arc_step_deg,
        )

        return self.draw_polyline(
            image=image,
            points=pts,
            bounds=bounds,
            color_bgr=style["color_bgr"],
            thickness=style["thickness"],
            closed=False,
        )

    def draw_circle_entity(self, image, entity, bounds, style):
        center = entity.dxf.center
        radius = float(entity.dxf.radius)

        pts = sample_circle_points(
            cx=float(center.x),
            cy=float(center.y),
            radius=radius,
            step_deg=self.circle_step_deg,
        )

        return self.draw_polyline(
            image=image,
            points=pts,
            bounds=bounds,
            color_bgr=style["color_bgr"],
            thickness=style["thickness"],
            closed=True,
        )

    def draw_lwpolyline_entity(self, image, entity, bounds, style):
        if has_lwpolyline_bulge(entity):
            drawn_any = False
            try:
                for ve in entity.virtual_entities():
                    if self.draw_entity(image, ve, bounds, inherited_style=style):
                        drawn_any = True
                return drawn_any
            except Exception:
                pass

        try:
            pts = [(float(p[0]), float(p[1])) for p in entity.get_points(format="xy")]
        except Exception:
            return False

        closed = bool(entity.closed)

        return self.draw_polyline(
            image=image,
            points=pts,
            bounds=bounds,
            color_bgr=style["color_bgr"],
            thickness=style["thickness"],
            closed=closed,
        )

    def draw_polyline_entity(self, image, entity, bounds, style):
        pts = []

        try:
            for v in entity.vertices:
                loc = v.dxf.location
                pts.append((float(loc.x), float(loc.y)))
        except Exception:
            return False

        if len(pts) < 2:
            return False

        closed = False
        try:
            closed = bool(entity.is_closed)
        except Exception:
            closed = False

        return self.draw_polyline(
            image=image,
            points=pts,
            bounds=bounds,
            color_bgr=style["color_bgr"],
            thickness=style["thickness"],
            closed=closed,
        )

    def draw_spline_entity(self, image, entity, bounds, style):
        try:
            pts = [
                (float(p.x), float(p.y))
                for p in entity.flattening(self.spline_flatten_tol)
            ]
        except Exception:
            return False

        return self.draw_polyline(
            image=image,
            points=pts,
            bounds=bounds,
            color_bgr=style["color_bgr"],
            thickness=style["thickness"],
            closed=False,
        )

    def draw_ellipse_entity(self, image, entity, bounds, style):
        try:
            pts = [
                (float(p.x), float(p.y))
                for p in entity.flattening(self.spline_flatten_tol)
            ]
        except Exception:
            return False

        closed = False
        try:
            closed = bool(entity.dxf.start_param == 0 and abs(entity.dxf.end_param - 2 * math.pi) < 1e-6)
        except Exception:
            closed = False

        return self.draw_polyline(
            image=image,
            points=pts,
            bounds=bounds,
            color_bgr=style["color_bgr"],
            thickness=style["thickness"],
            closed=closed,
        )

    def draw_insert_entity(self, image, entity, bounds, inherited_style=None):
        if not self.expand_inserts:
            return False

        drawn_any = False

        try:
            virtual_entities = list(entity.virtual_entities())
        except Exception:
            return False

        for ve in virtual_entities:
            if self.draw_entity(image, ve, bounds, inherited_style=inherited_style):
                drawn_any = True

        return drawn_any

    def get_style_for_entity(self, entity, inherited_style=None):
        if inherited_style is not None:
            return inherited_style

        layer = normalize_layer_name(getattr(entity.dxf, "layer", ""))
        if layer not in self.layer_config:
            return None

        cfg = self.layer_config[layer]
        color_rgb = cfg.get("color", (0, 0, 0))
        thickness = int(cfg.get("thickness", 1))

        return {
            "color_bgr": rgb_to_bgr(color_rgb),
            "thickness": thickness,
            "layer": layer,
        }

    def draw_entity(self, image, entity, bounds, inherited_style=None):
        style = self.get_style_for_entity(entity, inherited_style=inherited_style)
        if style is None:
            return False

        etype = entity.dxftype()
        self.stats[etype] += 1

        try:
            if etype == "LINE":
                ok = self.draw_line_entity(image, entity, bounds, style)

            elif etype == "LWPOLYLINE":
                ok = self.draw_lwpolyline_entity(image, entity, bounds, style)

            elif etype == "POLYLINE":
                ok = self.draw_polyline_entity(image, entity, bounds, style)

            elif etype == "ARC":
                ok = self.draw_arc_entity(image, entity, bounds, style)

            elif etype == "CIRCLE":
                ok = self.draw_circle_entity(image, entity, bounds, style)

            elif etype == "SPLINE":
                ok = self.draw_spline_entity(image, entity, bounds, style)

            elif etype == "ELLIPSE":
                ok = self.draw_ellipse_entity(image, entity, bounds, style)

            elif etype == "INSERT":
                ok = self.draw_insert_entity(image, entity, bounds, inherited_style=style)

            else:
                self.unsupported_stats[etype] += 1
                return False

        except Exception:
            self.unsupported_stats[etype] += 1
            return False

        if ok:
            self.drawn_stats[etype] += 1

        return ok

    def render(self, output_base_path, bounds):
        min_x, max_x, min_y, max_y = bounds

        width_m = max_x - min_x
        height_m = max_y - min_y

        w_px = int(math.ceil(width_m / self.resolution))
        h_px = int(math.ceil(height_m / self.resolution))

        print("\n========== 裁剪与输出参数 ==========")
        print(f"视口范围: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]")
        print(f"图像尺寸: {w_px} x {h_px} px")
        print(f"分辨率: {self.resolution} m/px")
        print("====================================\n")

        if w_px <= 0 or h_px <= 0:
            raise ValueError("图像尺寸无效，请检查 bounds。")

        if w_px > self.max_image_pixels or h_px > self.max_image_pixels:
            raise ValueError(
                f"图像尺寸过大: {w_px} x {h_px}。"
                f"请增大 RESOLUTION 或减小 BUFFER / 桩号范围。"
            )

        bg_bgr = rgb_to_bgr(self.background_color)
        image = np.full((h_px, w_px, 3), bg_bgr, dtype=np.uint8)

        total_entities = 0

        for entity in self.msp:
            total_entities += 1

            layer = normalize_layer_name(getattr(entity.dxf, "layer", ""))

            if layer in self.target_layers:
                self.draw_entity(image, entity, bounds)
            elif self.expand_inserts and entity.dxftype() == "INSERT":
                self.draw_insert_entity(image, entity, bounds, inherited_style=None)

        png_path = output_base_path + ".png"
        pgw_path = output_base_path + ".pgw"

        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        ok = cv2.imwrite(png_path, image)
        if not ok:
            raise RuntimeError(f"PNG 写入失败: {png_path}")

        with open(pgw_path, "w", encoding="utf-8") as f:
            f.write(f"{self.resolution}\n")
            f.write("0.0\n")
            f.write("0.0\n")
            f.write(f"{-self.resolution}\n")
            f.write(f"{min_x}\n")
            f.write(f"{max_y}\n")

        print("\n========== 渲染统计 ==========")
        print(f"模型空间总图元数: {total_entities}")

        print("\n目标图层内参与判断的图元类型:")
        for etype, count in sorted(self.stats.items()):
            print(f"  {etype}: {count}")

        print("\n实际绘制到视口内的图元类型:")
        for etype, count in sorted(self.drawn_stats.items()):
            print(f"  {etype}: {count}")

        if self.unsupported_stats:
            print("\n未支持或处理失败的图元类型:")
            for etype, count in sorted(self.unsupported_stats.items()):
                print(f"  {etype}: {count}")

        print("==============================\n")

        print(f"PNG 输出: {png_path}")
        print(f"PGW 输出: {pgw_path}")

        return png_path, pgw_path


# ======================================================
# 5. 大图切片分割推理器
# 来源逻辑：Mask_producer.py
# ======================================================

class MMSegLargeImageInferencer:
    def __init__(self, model, patch_size=512, stride=512, background_id=0):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.background_id = background_id

    def pad_image(self, img):
        h, w = img.shape[:2]
        pad_h = (self.patch_size - (h % self.stride)) % self.stride
        if (h + pad_h) < self.patch_size:
            pad_h = self.patch_size - h
        pad_w = (self.patch_size - (w % self.stride)) % self.stride
        if (w + pad_w) < self.patch_size:
            pad_w = self.patch_size - w

        padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        return padded_img, (h, w)

    def predict_patch(self, patch_img):
        from mmseg.apis import inference_model  # 惰性导入，避免启动时加载 mmseg 链
        result = inference_model(self.model, patch_img)
        pred = result.pred_sem_seg.data[0].cpu().numpy()
        mask = (pred != self.background_id).astype(np.uint8) * 255
        return mask

    def run(self, img_path, save_path=None):
        if not os.path.exists(img_path):
            print(f"错误: 找不到图片 {img_path}")
            return None

        print(f"[*] 正在读取大图: {os.path.basename(img_path)} ...")
        original_img = cv2.imread(img_path)
        if original_img is None:
            raise ValueError("图片读取失败，请检查路径或文件完整性。")

        padded_img, (orig_h, orig_w) = self.pad_image(original_img)
        H, W = padded_img.shape[:2]

        full_mask_accum = np.zeros((H, W), dtype=np.float32)
        weight_map = np.zeros((H, W), dtype=np.float32)

        coords = []
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                coords.append((x, y))

        print(
            f"[*] 开始重叠切片推理，共 {len(coords)} 个切片 "
            f"(Size: {self.patch_size}x{self.patch_size}, Stride: {self.stride})..."
        )

        overlap = self.patch_size - self.stride
        border = overlap // 2

        for x, y in tqdm(coords):
            patch = padded_img[y:y + self.patch_size, x:x + self.patch_size]
            mask_patch = self.predict_patch(patch).astype(np.float32)

            px0 = 0
            py0 = 0
            px1 = self.patch_size
            py1 = self.patch_size

            cx0 = x
            cy0 = y
            cx1 = x + self.patch_size
            cy1 = y + self.patch_size

            if x > 0:
                px0 += border
                cx0 += border

            if y > 0:
                py0 += border
                cy0 += border

            if x + self.patch_size < W:
                px1 -= border
                cx1 -= border

            if y + self.patch_size < H:
                py1 -= border
                cy1 -= border

            valid_mask = mask_patch[py0:py1, px0:px1]

            full_mask_accum[cy0:cy1, cx0:cx1] += valid_mask
            weight_map[cy0:cy1, cx0:cx1] += 1.0

        weight_map[weight_map == 0] = 1.0

        full_mask = (full_mask_accum / weight_map >= 127.5).astype(np.uint8) * 255
        final_mask = full_mask[:orig_h, :orig_w]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, final_mask)
            print(f"[OK] 推理完成！掩码已保存至: {save_path}")

        return final_mask


# ======================================================
# 6. 工具内部执行函数
# ======================================================

def default_layer_config() -> Dict[str, Dict[str, Any]]:
    return {
        "SXSS": {"color": (0, 0, 255), "thickness": 3},
        "DGX":  {"color": (128, 128, 128), "thickness": 1},
        "DLSS": {"color": (0, 255, 255), "thickness": 2},
        "JMD":  {"color": (128, 128, 128), "thickness": 2},
    }


def safe_station_name(k: str) -> str:
    return str(k).replace("+", "_").replace(":", "_").replace(" ", "").replace("~", "_")



def _json_loads_or_dict(value: Optional[Any]) -> Optional[Dict[str, Any]]:
    """兼容 AgentState 字典、JSON 字符串、JSON 文件路径三种输入。"""
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None

        if os.path.exists(s) and os.path.isfile(s):
            with open(s, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("design_state_json 文件内容必须是 JSON object。")
            return data

        try:
            data = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "design_state_json 既不是有效 JSON 字符串，也不是存在的 JSON 文件路径。"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError("design_state_json 必须解析为 JSON object。")
        return data

    raise TypeError("design_state_json / design_state 仅支持 dict、JSON 字符串或 JSON 文件路径。")


def _deep_get_first(obj: Any, keys: tuple[str, ...]) -> Optional[Any]:
    """在嵌套 dict/list 中递归查找第一个候选键。"""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] not in [None, ""]:
                return obj[k]
        for v in obj.values():
            found = _deep_get_first(v, keys)
            if found not in [None, ""]:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _deep_get_first(item, keys)
            if found not in [None, ""]:
                return found
    return None


def resolve_from_state(
    explicit_value: Optional[str],
    design_state: Optional[Dict[str, Any]],
    candidate_keys: tuple[str, ...],
    field_name: str,
    required: bool = True,
) -> Optional[str]:
    """从显式参数或上游 AgentState 中解析字段。"""
    if explicit_value not in [None, ""]:
        return str(explicit_value)

    if design_state:
        found = _deep_get_first(design_state, candidate_keys)
        if found not in [None, ""]:
            return str(found)

    if required:
        raise ValueError(
            f"缺少 {field_name}。该字段应由上游工具输出传入，或通过显式参数提供。候选键: {candidate_keys}"
        )
    return None


def build_drawing_crop_state_result(
    *,
    input_drawing_path: str,
    dxf_path: str,
    plane_json_path: str,
    start_k: str,
    end_k: str,
    bounds: tuple[float, float, float, float],
    png_path: str,
    pgw_path: str,
    mask_path: str,
    foreground_ratio: Optional[float],
) -> Dict[str, Any]:
    """统一工具输出结构，方便后续 obstacle_data_reader_tool 和 collision_check_tool 调用。"""
    return {
        "success": True,
        "tool_name": "drawing_crop_and_mask_tool",
        "input_files": {
            "input_drawing_path": input_drawing_path,
            "dxf_path": dxf_path,
            "plane_json_path": plane_json_path,
        },
        "station_range": {
            "start_k": start_k,
            "end_k": end_k,
        },
        "bounds": {
            "min_x": bounds[0],
            "max_x": bounds[1],
            "min_y": bounds[2],
            "max_y": bounds[3],
        },
        "output_files": {
            "png_path": png_path,
            "pgw_path": pgw_path,
            "mask_path": mask_path,
        },
        "input_drawing_path": input_drawing_path,
        "dxf_path": dxf_path,
        "plane_json_path": plane_json_path,
        "start_k": start_k,
        "end_k": end_k,
        "png_path": png_path,
        "pgw_path": pgw_path,
        "mask_path": mask_path,
        "foreground_ratio": foreground_ratio,
        "message": "图纸裁剪、PNG/PGW 输出与障碍物 mask 生成完成。",
    }


def run_drawing_crop_and_mask(
    input_drawing_path: Optional[str] = None,
    plane_json_path: Optional[str] = None,
    start_k: Optional[str] = None,
    end_k: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    device: str = "cuda:0",
    converter_path: Optional[str] = None,
    dxf_output_dir: Optional[str] = None,
    swap_xy: bool = True,
    buffer: float = 200.0,
    resolution: float = 0.5,
    max_image_pixels: int = 20000,
    expand_inserts: bool = False,
    patch_size: int = 256,
    stride: int = 256,
    background_id: int = 0,
    layer_config: Optional[Dict[str, Dict[str, Any]]] = None,
    design_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    input_drawing_path = resolve_from_state(
        input_drawing_path,
        design_state,
        ("input_drawing_path", "drawing_path", "dwg_path", "dxf_path"),
        "input_drawing_path",
    )
    plane_json_path = resolve_from_state(
        plane_json_path,
        design_state,
        ("plane_json_path", "plane_path", "horizontal_alignment_json_path", "平曲线文件路径"),
        "plane_json_path",
    )
    start_k = resolve_from_state(
        start_k,
        design_state,
        ("start_k", "start_station", "起点桩号", "range_start_k"),
        "start_k",
    )
    end_k = resolve_from_state(
        end_k,
        design_state,
        ("end_k", "end_station", "终点桩号", "range_end_k"),
        "end_k",
    )
    output_dir = resolve_from_state(
        output_dir,
        design_state,
        ("drawing_output_dir", "output_dir", "输出目录"),
        "output_dir",
    )
    config_file = resolve_from_state(
        config_file,
        design_state,
        ("config_file", "mmseg_config_file", "segmentation_config_file"),
        "config_file",
    )
    checkpoint_file = resolve_from_state(
        checkpoint_file,
        design_state,
        ("checkpoint_file", "mmseg_checkpoint_file", "segmentation_checkpoint_file"),
        "checkpoint_file",
    )

    if not os.path.exists(input_drawing_path):
        raise FileNotFoundError(f"输入图纸不存在: {input_drawing_path}")

    if not os.path.exists(plane_json_path):
        raise FileNotFoundError(f"平曲线 plane.json 不存在: {plane_json_path}")

    os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(input_drawing_path)[1].lower()

    if ext == ".dwg":
        if not converter_path:
            raise ValueError("输入为 DWG 时必须提供 converter_path")

        dxf_dir = dxf_output_dir or os.path.join(output_dir, "dxf")
        os.makedirs(dxf_dir, exist_ok=True)

        converter = DwgConverter(converter_path)
        dxf_path = converter.convert(input_drawing_path, dxf_dir)

    elif ext == ".dxf":
        dxf_path = input_drawing_path

    else:
        raise ValueError("input_drawing_path 仅支持 .dwg 或 .dxf")

    calc = RealAlignmentCalculator(
        json_path=plane_json_path,
        swap_xy=swap_xy,
    )

    bounds = calc.get_bounding_box(
        start_k_str=start_k,
        end_k_str=end_k,
        buffer=buffer,
        sample_step=50.0,
    )

    use_layer_config = layer_config or default_layer_config()

    renderer = FastDxfRasterRenderer(
        dxf_path=dxf_path,
        layer_config=use_layer_config,
        resolution=resolution,
        expand_inserts=expand_inserts,
        arc_step_deg=3.0,
        spline_flatten_tol=0.5,
        circle_step_deg=3.0,
        max_image_pixels=max_image_pixels,
    )

    safe_start = safe_station_name(start_k)
    safe_end = safe_station_name(end_k)

    output_base = os.path.join(
        output_dir,
        f"pred_{safe_start}_{safe_end}_fast"
    )

    png_path, pgw_path = renderer.render(output_base, bounds)

    print("[*] 正在加载分割模型...")
    from mmseg.apis import init_model  # 惰性导入，避免启动时加载 mmseg 链
    model = init_model(config_file, checkpoint_file, device=device)
    print("[*] 模型加载完成！")

    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    mask_path = os.path.join(
        mask_dir,
        f"obstacle_mask_{safe_start}_{safe_end}.png"
    )

    inferencer = MMSegLargeImageInferencer(
        model,
        patch_size=patch_size,
        stride=stride,
        background_id=background_id
    )

    mask_result = inferencer.run(png_path, mask_path)

    foreground_ratio = None
    if mask_result is not None:
        foreground_ratio = float((mask_result > 0).sum() / mask_result.size)
        print(f"掩码形状: {mask_result.shape}")
        print(f"前景像素占比: {foreground_ratio:.2%}")

    return build_drawing_crop_state_result(
        input_drawing_path=input_drawing_path,
        dxf_path=dxf_path,
        plane_json_path=plane_json_path,
        start_k=start_k,
        end_k=end_k,
        bounds=bounds,
        png_path=png_path,
        pgw_path=pgw_path,
        mask_path=mask_path,
        foreground_ratio=foreground_ratio,
    )


# ======================================================
# 7. LangChain Agent 工具封装
# ======================================================

@tool
def drawing_crop_and_mask_tool(
    input_drawing_path: str = None,
    plane_json_path: str = None,
    start_k: str = None,
    end_k: str = None,
    output_dir: str = None,
    config_file: str = None,
    checkpoint_file: str = None,
    device: str = "cpu",
    converter_path: str = None,
    dxf_output_dir: str = None,
    swap_xy: bool = True,
    buffer: float = 200.0,
    resolution: float = 0.5,
    max_image_pixels: int = 20000,
    expand_inserts: bool = False,
    patch_size: int = 256,
    stride: int = 256,
    background_id: int = 0,
    layer_config_json: str = None,
    design_state_json: str = None,
) -> dict:
    """
    图纸预处理工具。

    功能：
    1. 如果输入为 DWG，则调用 ODA File Converter 转为 DXF；
    2. 根据 plane.json 和设计桩号区间裁剪 DXF；
    3. 输出裁剪后的 PNG 和 PGW；
    4. 调用 MMSegmentation 模型生成障碍物 mask。

    输入：
        input_drawing_path:
            原始 DWG 或 DXF 图纸路径。
        plane_json_path:
            可选。平曲线 JSON 文件路径，需包含“平曲线结构”。正式智能体流程中优先从 design_state_json / data_loader_tool 输出中获取。
        start_k:
            起点桩号，例如 K26+300。
        end_k:
            终点桩号，例如 K27+700。
        output_dir:
            输出目录。
        config_file:
            MMSegmentation 模型配置文件路径。
        checkpoint_file:
            MMSegmentation 权重文件路径。
        device:
            cuda:0 或 cpu。
        converter_path:
            ODA File Converter 可执行文件路径。输入 DWG 时必须提供。
        dxf_output_dir:
            DXF 输出目录。为空时默认 output_dir/dxf。
        swap_xy:
            是否交换 plane.json 中的 X/Y。
        buffer:
            裁剪缓冲范围，单位 m。
        resolution:
            PNG 输出分辨率，单位 m/px。
        max_image_pixels:
            单边最大像素限制。
        expand_inserts:
            是否展开 INSERT 图块。
        patch_size:
            分割切片尺寸。
        stride:
            分割滑动步长。
        background_id:
            背景类别 ID。
        layer_config_json:
            可选，图层配置 JSON 字符串。不提供则使用默认 SXSS/DGX/DLSS/JMD。
        design_state_json:
            可选。上游工具状态或总 AgentState，可以是 JSON 字符串，也可以是 JSON 文件路径；其中应包含 data_loader_tool 生成的 plane_json_path。

    返回：
        {
          "success": true,
          "tool_name": "drawing_crop_and_mask_tool",
          "input_files": {"input_drawing_path": "", "dxf_path": "", "plane_json_path": ""},
          "output_files": {"png_path": "", "pgw_path": "", "mask_path": ""},
          "foreground_ratio": 0.0
        }
    """
    try:
        layer_config = None

        if layer_config_json:
            layer_config = json.loads(layer_config_json)

        design_state = _json_loads_or_dict(design_state_json)

        return run_drawing_crop_and_mask(
            input_drawing_path=input_drawing_path,
            plane_json_path=plane_json_path,
            start_k=start_k,
            end_k=end_k,
            output_dir=output_dir,
            config_file=config_file,
            checkpoint_file=checkpoint_file,
            device=device,
            converter_path=converter_path,
            dxf_output_dir=dxf_output_dir,
            swap_xy=swap_xy,
            buffer=buffer,
            resolution=resolution,
            max_image_pixels=max_image_pixels,
            expand_inserts=expand_inserts,
            patch_size=patch_size,
            stride=stride,
            background_id=background_id,
            layer_config=layer_config,
            design_state=design_state,
        )

    except Exception as e:
        return {
            "success": False,
            "tool_name": "drawing_crop_and_mask_tool",
            "error": str(e)
        }
