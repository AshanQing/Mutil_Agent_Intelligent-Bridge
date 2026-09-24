# tools/data_loader_tool.py
# ======================================================
# 功能：
# 1. 读取纬地线路数据文件：.pm / .WID / .zdm / .dmx
# 2. 可选读取既有障碍物/构造物 JSON（正式流程中障碍物应由图纸预处理与语义提取得到）
# 3. 读取 PM 后自动生成可复用 plane.json
# 4. 支持整体式单线路读取
# 5. 支持分离式 K / Z / Z1 / Z2 多线路读取
# 6. 支持按桩号范围裁剪
# 7. 作为 LangChain Tool 被 agent.py 调用
# ======================================================

from __future__ import annotations

import json
import re
import math
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from langchain_core.tools import tool


# ======================================================
# 日志配置
# ======================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ======================================================
# 1. 基础通用工具与桩号解析
# ======================================================

def read_float_tokens(s: str) -> List[float]:
    """提取文本行中的所有浮点数。"""
    return [
        float(x)
        for x in s.split()
        if re.match(r"^[\-\+]?(\d+(\.\d*)?|\.\d+)([eE][\-\+]?\d+)?$", x)
    ]


def format_k_station(value: Optional[float]) -> str:
    """
    将数值桩号格式化为 K 桩号。

    示例：
        12200.0 -> K12+200
        -2224.064 -> K-2+224.064
    """
    if value is None:
        return ""

    sign = "-" if value < 0 else ""
    val_abs = abs(float(value))
    km = int(val_abs // 1000)
    m = val_abs % 1000

    if abs(m - int(m)) < 1e-9:
        m_str = f"{int(m)}"
    else:
        m_str = f"{m:07.3f}".rstrip("0").rstrip(".")

    return f"K{sign}{km}+{m_str}"


def station_to_float(station: Union[str, int, float, None]) -> float:
    """
    通用桩号解析器。

    支持：
        K12+200
        12+200
        12200
        K-2+224.064
        -2224.064
        ZK12+200
    """
    if isinstance(station, (int, float)):
        return float(station)

    if station is None:
        return 0.0

    s = str(station).strip().upper()
    if not s:
        return 0.0

    # 特殊处理 K-2+224 这类负桩号
    if s.startswith("K-"):
        temp = s[2:]
        try:
            if "+" in temp:
                km_part, m_part = temp.split("+", 1)
                return -(float(km_part) * 1000.0 + float(m_part))
            return -float(temp)
        except Exception:
            return 0.0

    # 去掉开头线路字母，只保留可能的 K、数字、正负号、小数点、+
    # 例如 ZK12+200 -> K12+200
    k_index = s.find("K")
    if k_index >= 0:
        s = s[k_index + 1:]

    clean_s = re.sub(r"[^\d\.\-\+]", "", s)

    try:
        if "+" in clean_s:
            km_part, m_part = clean_s.split("+", 1)
            return float(km_part) * 1000.0 + float(m_part)
        return float(clean_s)
    except Exception:
        return 0.0


def parse_station_range(range_text: str) -> Optional[tuple[float, float]]:
    """
    从字符串中解析桩号范围。

    支持：
        K12+200 - K12+500
        K12+200 ~ K12+500
        K12+200至K12+500
        K-1+578 ~ K-1+583
    """
    if not range_text:
        return None

    text = str(range_text).strip()

    # 常见分隔符统一
    for sep in ["～", "~", "—", "–", "至", "到"]:
        text = text.replace(sep, "-")

    parts = [p.strip() for p in text.split("-") if p.strip()]

    # 注意：负桩号 K-1+578 中也含有 -
    # 上面简单 split 会破坏负桩号，所以用正则兜底
    if len(parts) < 2:
        pattern = r"[A-Z]*K?-?\d+\+\d+(?:\.\d+)?|[A-Z]*K?-?\d+(?:\.\d+)?"
        matches = re.findall(pattern, str(range_text).upper())
        if len(matches) >= 2:
            s_val = station_to_float(matches[0])
            e_val = station_to_float(matches[1])
            return (min(s_val, e_val), max(s_val, e_val))
        return None

    # 若 split 后异常，仍用正则优先
    pattern = r"[A-Z]*K?-?\d+\+\d+(?:\.\d+)?|[A-Z]*K?-?\d+(?:\.\d+)?"
    matches = re.findall(pattern, str(range_text).upper())
    if len(matches) >= 2:
        s_val = station_to_float(matches[0])
        e_val = station_to_float(matches[1])
        return (min(s_val, e_val), max(s_val, e_val))

    try:
        s_val = station_to_float(parts[0])
        e_val = station_to_float(parts[1])
        return (min(s_val, e_val), max(s_val, e_val))
    except Exception:
        return None


def range_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """判断两个桩号区间是否重叠。"""
    return not (a_end < b_start or a_start > b_end)


def deg(rad: Optional[float]) -> Optional[float]:
    """弧度转角度。"""
    if rad is None:
        return None
    return round(rad * 180.0 / math.pi, 6)


def norm_delta(a1: Optional[float], a2: Optional[float]) -> Optional[float]:
    """计算两个方位角之间的归一化夹角，单位为度。"""
    if a1 is None or a2 is None:
        return None

    d = a2 - a1
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi

    return round(abs(d * 180.0 / math.pi), 4)


def clean_radius(v: Optional[float]) -> Optional[float]:
    """清理纬地中常见的无穷大半径值。"""
    if v is None:
        return None
    if abs(v) >= 99999:
        return None
    return v


# ======================================================
# 2. 平曲线 PM 文件解析
#    已按 PM_reader.py 的修正版逻辑重构
# ======================================================

def _is_pm_header_line(s: str) -> bool:
    """
    判断是否为纬地 PM 线元头行。

    纬地标准头行特征：
        至少 7 个数；
        最后一个数为线元类型 TypeID。
    """
    toks = read_float_tokens(s)
    return len(toks) >= 7 and abs(toks[-1]) in (1, 3, 21, 22, 23, 24, 25, 26)


def parse_hintcad_pm(path: Union[str, Path]) -> Dict[str, Any]:
    """
    解析纬地 PM 平曲线文件。

    输出：
        {
          "平曲线结构": [
            {
              "序号": 1,
              "曲线类型": "直线 / 圆曲线 / 缓和曲线（前缓） / 缓和曲线（后缓）",
              "桩号范围": "Kxx+xxx - Kxx+xxx",
              "起点": {
                "桩号": "",
                "X": 0.0,
                "Y": 0.0,
                "方位角(°)": 0.0
              },
              "终点": {
                "桩号": "",
                "X": 0.0,
                "Y": 0.0,
                "方位角(°)": 0.0
              },
              "半径R": null,
              "缓和曲线长Ls": null,
              "转角Δ(°)": null,
              "圆心坐标": null,
              "备注": null
            }
          ]
        }
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.strip() for ln in f if ln.strip()]

    i = 0
    seq = 1
    segs: List[Dict[str, Any]] = []

    while i < len(raw):
        if not _is_pm_header_line(raw[i]):
            i += 1
            continue

        head = read_float_tokens(raw[i])

        # 纬地头行常见格式：
        # Dir Reserved Param1 Param2 R_Start R_End TypeID
        A, B, C, D = head[2], head[3], head[4], head[5]
        type_code = int(round(head[-1]))

        # 读取坐标行
        coords: List[float] = []
        j = i + 1

        while j < len(raw):
            toks = read_float_tokens(raw[j])

            # 修正逻辑：
            # 当已经读到足够坐标，并遇到 4 个数的行时，认为它是“起点桩号、终点桩号、起点方位角、终点方位角”
            if len(toks) == 4 and len(coords) >= 6:
                break

            if _is_pm_header_line(raw[j]):
                break

            coords += toks
            j += 1

            if j < len(raw) and _is_pm_header_line(raw[j]):
                break

        # 读取桩号与方位角行
        stake_start = stake_end = az_start = az_end = None

        while j < len(raw):
            toks = read_float_tokens(raw[j])
            if len(toks) >= 4:
                stake_start, stake_end, az_start, az_end = toks[:4]
                j += 1
                break
            j += 1

        # 纬地坐标行标准顺序：
        # [0],[1]：起点 X,Y
        # [2],[3]：交点 X,Y
        # [4],[5]：终点 X,Y
        # [6],[7]：圆心 X,Y
        x1 = y1 = x2 = y2 = None
        center = None

        if len(coords) >= 2:
            x1, y1 = coords[0], coords[1]

        if len(coords) >= 6:
            x2, y2 = coords[4], coords[5]

        if len(coords) >= 8:
            cx, cy = coords[6], coords[7]
            if abs(cx) > 0.001 or abs(cy) > 0.001:
                center = {"X": round(cx, 3), "Y": round(cy, 3)}

        curve_type = "未知"
        R = None
        Ls = None
        extra: Dict[str, Any] = {}

        if type_code == 1:
            curve_type = "直线"
            extra["直线长度"] = abs(A)

        elif type_code in (21, 23, 25):
            curve_type = "缓和曲线（前缓）"
            Ls = abs(A)
            R = clean_radius(D)

        elif type_code == 3:
            curve_type = "圆曲线"
            R = clean_radius(D)
            extra["圆曲弧长"] = abs(A)

        elif type_code in (22, 24, 26):
            curve_type = "缓和曲线（后缓）"
            Ls = abs(A)
            R = clean_radius(C)

        start_str = format_k_station(stake_start)
        end_str = format_k_station(stake_end)

        segs.append({
            "序号": seq,
            "曲线类型": curve_type,
            "桩号范围": f"{start_str} - {end_str}",
            "起点": {
                "桩号": start_str,
                "X": round(x1, 3) if x1 is not None else None,
                "Y": round(y1, 3) if y1 is not None else None,
                "方位角(°)": deg(az_start)
            },
            "终点": {
                "桩号": end_str,
                "X": round(x2, 3) if x2 is not None else None,
                "Y": round(y2, 3) if y2 is not None else None,
                "方位角(°)": deg(az_end)
            },
            "半径R": round(R, 3) if R is not None else None,
            "缓和曲线长Ls": round(Ls, 3) if Ls is not None else None,
            "转角Δ(°)": norm_delta(az_start, az_end),
            "圆心坐标": center,
            "备注": extra or None
        })

        seq += 1
        i = j

    return {"平曲线结构": segs}


# ======================================================
# 3. 横断面 WID 文件解析
# ======================================================

def extract_wid(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    解析纬地 WID 横断面文件。

    输出：
        {
          "横断面信息": [
            {
              "桩号范围": "Kxx+xxx - Kxx+xxx",
              "Left": {...},
              "Right": {...}
            }
          ]
        }
    """

    def parse_wid_line(line: str):
        parts = line.strip().split()
        if len(parts) < 6:
            return None

        try:
            station = station_to_float(parts[0])
            median_half = float(parts[1])
            lane_width = float(parts[2])
            sidewalk_width = float(parts[4])
            shoulder_width = float(parts[5])
            return station, median_half, lane_width, sidewalk_width, shoulder_width
        except Exception:
            return None

    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    left_data = []
    right_data = []
    current_side = None

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        if "[LEFT]" in clean_line or "zzzz" in clean_line.lower():
            current_side = "LEFT"
            continue

        if "[RIGHT]" in clean_line or "yyyy" in clean_line.lower():
            current_side = "RIGHT"
            continue

        parsed = parse_wid_line(clean_line)

        if parsed:
            if current_side == "LEFT":
                left_data.append(parsed)
            elif current_side == "RIGHT":
                right_data.append(parsed)

    segments: List[Dict[str, Any]] = []
    n = len(right_data)

    for i in range(n - 1):
        s_curr, med_r, lane_r, walk_r, soil_r = right_data[i]
        s_next = right_data[i + 1][0]

        match_l = next((x for x in left_data if abs(x[0] - s_curr) < 0.1), None)

        if match_l is None:
            med_l = lane_l = walk_l = soil_l = 0.0
        else:
            _, med_l, lane_l, walk_l, soil_l = match_l

        segments.append({
            "桩号范围": f"{format_k_station(s_curr)} - {format_k_station(s_next)}",
            "Left": {
                "行车道宽度": [lane_l],
                "人行道宽度": [walk_l],
                "土路肩宽度": [soil_l],
                "内侧护栏": 0.5
            },
            "Right": {
                "行车道宽度": [lane_r],
                "人行道宽度": [walk_r],
                "土路肩宽度": [soil_r],
                "内侧护栏": 0.5
            }
        })

    return {"横断面信息": segments}


# ======================================================
# 4. 纵断面 ZDM / DMX 文件解析
# ======================================================

def _read_vertical_file(path: Union[str, Path], is_zdm: bool = True) -> List[Dict[str, Any]]:
    """
    读取纵断面文件。

    is_zdm=True:
        设计线高程，可能包含 R
    is_zdm=False:
        地形线高程
    """
    data: List[Dict[str, Any]] = []
    path = Path(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if re.match(r"^\s*$", line):
                continue
            if "HINTCAD" in line:
                continue
            if is_zdm and "ZDM" in line:
                continue
            if (not is_zdm) and "DMX" in line:
                continue

            parts = line.strip().split()
            if len(parts) < 2:
                continue

            try:
                stake = station_to_float(parts[0])
                item = {
                    "桩号": stake,
                    "高程": float(parts[1])
                }

                if is_zdm and len(parts) >= 3:
                    try:
                        item["R"] = float(parts[2])
                    except Exception:
                        item["R"] = None

                data.append(item)

            except Exception:
                continue

    return data

def _interpolate_vertical_item(
    seq: List[Dict[str, Any]],
    station: float,
    *,
    source_label: str = "区间边界插值"
) -> Optional[Dict[str, Any]]:
    """
    在纵断面高程序列中按桩号线性插值。

    说明：
        1. 当前用于保证裁剪区间边界具备设计线/地形线高程；
        2. 对 ZDM 设计线，这里先采用相邻控制点线性插值；
        3. 若后续需要严格考虑竖曲线，可在此函数中替换为竖曲线高程计算。
    """
    if not seq:
        return None

    sorted_seq = sorted(
        [d for d in seq if d.get("桩号") is not None and d.get("高程") is not None],
        key=lambda x: float(x["桩号"])
    )

    if not sorted_seq:
        return None

    # station 正好落在已有点上
    for item in sorted_seq:
        if abs(float(item["桩号"]) - station) < 1e-6:
            copied = dict(item)
            copied.setdefault("来源", "原始控制点")
            return copied

    # 超出数据范围时，不外推
    if station < float(sorted_seq[0]["桩号"]) or station > float(sorted_seq[-1]["桩号"]):
        return None

    for i in range(len(sorted_seq) - 1):
        p1 = sorted_seq[i]
        p2 = sorted_seq[i + 1]

        s1 = float(p1["桩号"])
        s2 = float(p2["桩号"])

        if s1 <= station <= s2:
            h1 = float(p1["高程"])
            h2 = float(p2["高程"])

            if abs(s2 - s1) < 1e-9:
                h = h1
            else:
                ratio = (station - s1) / (s2 - s1)
                h = h1 + ratio * (h2 - h1)

            return {
                "桩号": round(station, 3),
                "高程": round(h, 3),
                "来源": source_label,
                "插值依据": {
                    "前点": {
                        "桩号": s1,
                        "高程": h1
                    },
                    "后点": {
                        "桩号": s2,
                        "高程": h2
                    }
                }
            }

    return None


def crop_vertical_sequence_with_boundary_points(
    seq: List[Dict[str, Any]],
    start_station: float,
    end_station: float
) -> List[Dict[str, Any]]:
    """
    裁剪纵断面高程序列，并强制补充起终点插值高程。

    输出结果包含：
        1. start_station 插值点；
        2. 区间内部原始控制点；
        3. end_station 插值点。

    目的：
        避免裁剪区间内只有 0 个或 1 个控制点，导致后续无法计算纵断面趋势。
    """
    if not seq:
        return []

    s_val = float(start_station)
    e_val = float(end_station)

    if s_val > e_val:
        s_val, e_val = e_val, s_val

    result: List[Dict[str, Any]] = []

    start_item = _interpolate_vertical_item(
        seq,
        s_val,
        source_label="区间起点插值"
    )
    if start_item is not None:
        result.append(start_item)

    inner_items = []
    for item in seq:
        try:
            sta = float(item.get("桩号", 0.0))
        except Exception:
            continue

        if s_val < sta < e_val:
            copied = dict(item)
            copied.setdefault("来源", "原始控制点")
            inner_items.append(copied)

    result.extend(inner_items)

    end_item = _interpolate_vertical_item(
        seq,
        e_val,
        source_label="区间终点插值"
    )
    if end_item is not None:
        result.append(end_item)

    # 去重并按桩号排序，避免起终点刚好等于原始控制点时重复
    unique: Dict[float, Dict[str, Any]] = {}
    for item in result:
        sta = round(float(item["桩号"]), 6)
        if sta not in unique:
            unique[sta] = item
        else:
            # 原始控制点优先于插值点
            if unique[sta].get("来源") != "原始控制点" and item.get("来源") == "原始控制点":
                unique[sta] = item

    return [unique[k] for k in sorted(unique.keys())]

def crop_vertical_sequence_with_context_points(
    seq: List[Dict[str, Any]],
    start_station: float,
    end_station: float
) -> List[Dict[str, Any]]:
    """
    裁剪纵断面高程序列，并保留区间边界所在坡段/竖曲线段的真实控制点。

    目的：
        1. 不人为插值生成设计线控制点；
        2. 保留 start_station 所在段的前后真实控制点；
        3. 保留 end_station 所在段的前后真实控制点；
        4. 保证裁剪后的设计线仍可用于坡段趋势、纵坡和高差计算。

    示例：
        原始控制点: 20190, 20710, 22310
        裁剪区间: 20500 ~ 22100
        输出: 20190, 20710, 22310
    """
    if not seq:
        return []

    s_val = float(start_station)
    e_val = float(end_station)

    if s_val > e_val:
        s_val, e_val = e_val, s_val

    sorted_seq = sorted(
        [
            d for d in seq
            if d.get("桩号") is not None and d.get("高程") is not None
        ],
        key=lambda x: float(x["桩号"])
    )

    if not sorted_seq:
        return []

    keep_indices = set()

    # 1. 保留区间内部真实控制点
    for i, item in enumerate(sorted_seq):
        sta = float(item["桩号"])
        if s_val <= sta <= e_val:
            keep_indices.add(i)

    # 2. 保留 start_station 所在段的前后真实控制点
    for i in range(len(sorted_seq) - 1):
        sta1 = float(sorted_seq[i]["桩号"])
        sta2 = float(sorted_seq[i + 1]["桩号"])

        if sta1 <= s_val <= sta2:
            keep_indices.add(i)
            keep_indices.add(i + 1)
            break

    # 3. 保留 end_station 所在段的前后真实控制点
    for i in range(len(sorted_seq) - 1):
        sta1 = float(sorted_seq[i]["桩号"])
        sta2 = float(sorted_seq[i + 1]["桩号"])

        if sta1 <= e_val <= sta2:
            keep_indices.add(i)
            keep_indices.add(i + 1)
            break

    result = []
    for i in sorted(keep_indices):
        copied = dict(sorted_seq[i])
        copied.setdefault("来源", "原始纵断面控制点")
        result.append(copied)

    return result

def build_vertical_profile(zdm_path: Union[str, Path], dmx_path: Union[str, Path]) -> Dict[str, Any]:
    """
    构建设计线与地形线高程序列。
    """
    d_line = _read_vertical_file(zdm_path, is_zdm=True)
    t_line = _read_vertical_file(dmx_path, is_zdm=False)

    if d_line:
        stake_range = f"{format_k_station(d_line[0]['桩号'])} ~ {format_k_station(d_line[-1]['桩号'])}"
    else:
        stake_range = "未知"

    return {
        "纵断面结构": {
            "桩号范围": stake_range,
            "设计线高程序列": d_line,
            "地形线高程序列": t_line
        }
    }


# ======================================================
# 5. 障碍物 / 构造物路径与裁剪
# ======================================================

def resolve_structures_path(base_dir: str, structures_filename: Optional[str]) -> Optional[Path]:
    """
    解析构造物/障碍物文件路径。

    支持：
        1. 绝对路径
        2. base_dir 下相对路径
        3. 当前工作目录相对路径
    """
    if not structures_filename:
        return None

    p = Path(structures_filename)

    if p.is_absolute():
        return p

    p_base = Path(base_dir) / p
    if p_base.exists():
        return p_base

    return p.resolve()


def load_structures_json(base_dir: str, structures_filename: Optional[str]) -> tuple[List[Dict[str, Any]], str]:
    """
    读取障碍物 / 构造物 JSON。
    """
    if not structures_filename:
        return [], "未提供 structures_filename"

    struct_path = resolve_structures_path(base_dir, structures_filename)

    logger.info(f"尝试读取障碍物文件: {struct_path}")

    if not struct_path or not struct_path.exists():
        return [], f"文件缺失: {struct_path}"

    try:
        with open(struct_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容几种结构：
        # 1. list
        # 2. {"障碍物信息": [...]}
        # 3. {"构造物信息": [...]}
        if isinstance(data, list):
            return data, "成功"

        if isinstance(data, dict):
            if "障碍物信息" in data and isinstance(data["障碍物信息"], list):
                return data["障碍物信息"], "成功"
            if "构造物信息" in data and isinstance(data["构造物信息"], list):
                return data["构造物信息"], "成功"

            # 如果是分区段存储：{"Section_xxx": [...]}
            merged = []
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            copied = dict(item)
                            copied.setdefault("来源分区", key)
                            merged.append(copied)
            if merged:
                return merged, "成功"

        return [], "解析成功但未识别到列表结构"

    except Exception as e:
        return [], f"解析失败: {e}"


def get_obstacle_station_range(obs: Dict[str, Any]) -> Optional[tuple[float, float]]:
    """
    从障碍物对象中提取桩号范围。

    兼容字段：
        桩号
        中心桩号
        station
        range
        桩号范围
        起点桩号 + 终点桩号
        start_station + end_station
    """
    # 区间字段
    for key in ["桩号范围", "range", "影响范围", "覆盖范围"]:
        if key in obs and obs.get(key):
            parsed = parse_station_range(str(obs.get(key)))
            if parsed:
                return parsed

    # 起终点字段
    start_keys = ["起点桩号", "start_station", "start", "起点"]
    end_keys = ["终点桩号", "end_station", "end", "终点"]

    start_val = None
    end_val = None

    for key in start_keys:
        if key in obs and obs.get(key):
            start_val = obs.get(key)
            break

    for key in end_keys:
        if key in obs and obs.get(key):
            end_val = obs.get(key)
            break

    if start_val is not None and end_val is not None:
        s = station_to_float(start_val)
        e = station_to_float(end_val)
        return min(s, e), max(s, e)

    # 单点字段
    for key in ["桩号", "中心桩号", "station"]:
        if key in obs and obs.get(key):
            val = station_to_float(obs.get(key))
            return val, val

    return None


def crop_obstacles(
    obstacles: List[Dict[str, Any]],
    start_station: Union[str, float],
    end_station: Union[str, float]
) -> List[Dict[str, Any]]:
    """
    裁剪障碍物，支持单点桩号和区间型障碍物。
    """
    s_val = station_to_float(start_station)
    e_val = station_to_float(end_station)

    if s_val > e_val:
        s_val, e_val = e_val, s_val

    result = []

    for obs in obstacles:
        if not isinstance(obs, dict):
            continue

        obs_range = get_obstacle_station_range(obs)
        if not obs_range:
            continue

        o_start, o_end = obs_range

        if range_overlap(s_val, e_val, o_start, o_end):
            copied = dict(obs)
            copied.setdefault("解析桩号范围", f"{format_k_station(o_start)} ~ {format_k_station(o_end)}")
            result.append(copied)

    return result


# ======================================================
# 6. 单线路数据读取与裁剪
# ======================================================

def build_single_route_full_data(
    base_dir: str,
    file_prefix: str,
    route_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    读取单条线路的完整数据。

    适用于：
        整体式 K
        分离式 K
        分离式 Z
        分离式 Z1
        分离式 Z2
    """
    base = Path(base_dir)

    pm_file = base / f"{file_prefix}.pm"
    wid_file = base / f"{file_prefix}.WID"
    zdm_file = base / f"{file_prefix}.zdm"
    dmx_file = base / f"{file_prefix}.dmx"

    # 大小写兜底
    if not wid_file.exists():
        wid_alt = base / f"{file_prefix}.wid"
        if wid_alt.exists():
            wid_file = wid_alt

    full_data: Dict[str, Any] = {
        "文件前缀": file_prefix,
        "数据目录": str(base),
        "解析状态": {}
    }

    # 平曲线
    if pm_file.exists():
        try:
            full_data.update(parse_hintcad_pm(pm_file))
            full_data["解析状态"]["平曲线"] = "成功"
        except Exception as e:
            full_data["解析状态"]["平曲线"] = f"解析失败: {e}"
            logger.error(f"平曲线解析失败: {pm_file}, {e}", exc_info=True)
    else:
        full_data["解析状态"]["平曲线"] = f"文件缺失: {pm_file}"

    # 横断面
    if wid_file.exists():
        try:
            full_data.update(extract_wid(wid_file))
            full_data["解析状态"]["横断面"] = "成功"
            logger.info(f"{route_id or file_prefix} 断面解析完成: {len(full_data.get('横断面信息', []))} 个区间")
        except Exception as e:
            full_data["解析状态"]["横断面"] = f"解析失败: {e}"
            logger.error(f"横断面解析失败: {wid_file}, {e}", exc_info=True)
    else:
        full_data["解析状态"]["横断面"] = f"文件缺失: {wid_file}"

    # 纵断面
    if zdm_file.exists() and dmx_file.exists():
        try:
            full_data.update(build_vertical_profile(zdm_file, dmx_file))
            full_data["解析状态"]["纵断面"] = "成功"
        except Exception as e:
            full_data["解析状态"]["纵断面"] = f"解析失败: {e}"
            logger.error(f"纵断面解析失败: {zdm_file}, {dmx_file}, {e}", exc_info=True)
    else:
        full_data["解析状态"]["纵断面"] = f"文件缺失: zdm={zdm_file.exists()}, dmx={dmx_file.exists()}"

    return full_data


def crop_hintcad_data(
    full_data: Dict[str, Any],
    start_station: Union[str, float],
    end_station: Union[str, float],
    obstacles: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    根据桩号范围裁剪单线路数据。
    """
    s_val = station_to_float(start_station)
    e_val = station_to_float(end_station)

    if s_val > e_val:
        s_val, e_val = e_val, s_val

    logger.info(f"执行裁剪: {start_station} ~ {end_station} ({s_val} ~ {e_val})")

    cropped: Dict[str, Any] = {
        "线路名称": full_data.get("线路名称", "未知线路"),
        "线路编号": full_data.get("线路编号", ""),
        "文件前缀": full_data.get("文件前缀", ""),
        "截取范围": f"{format_k_station(s_val)} ~ {format_k_station(e_val)}",
        "解析状态": full_data.get("解析状态", {})
    }

    # 1. 平曲线裁剪
    if "平曲线结构" in full_data:
        p_seq = []
        for c in full_data.get("平曲线结构", []):
            try:
                c_start = station_to_float(c.get("起点", {}).get("桩号"))
                c_end = station_to_float(c.get("终点", {}).get("桩号"))

                if c_start > c_end:
                    c_start, c_end = c_end, c_start

                if range_overlap(s_val, e_val, c_start, c_end):
                    p_seq.append(c)

            except Exception:
                continue

        cropped["平曲线结构"] = p_seq

    # 2. 横断面裁剪
    if "横断面信息" in full_data:
        c_seq = []
        for seg in full_data.get("横断面信息", []):
            range_str = seg.get("桩号范围", "")
            parsed = parse_station_range(range_str)

            if not parsed:
                c_seq.append(seg)
                continue

            r_start, r_end = parsed
            if range_overlap(s_val - 1.0, e_val + 1.0, r_start, r_end):
                c_seq.append(seg)

        cropped["横断面信息"] = c_seq

    # 3. 纵断面裁剪
    if "纵断面结构" in full_data:
        vert = full_data["纵断面结构"]

        # 设计线：保留目标区间涉及坡段/竖曲线段的真实控制点，不做人为插值
        d_seq = crop_vertical_sequence_with_context_points(
            vert.get("设计线高程序列", []),
            s_val,
            e_val
        )

        # 地面线：仍按目标区间严格裁剪，用于限定 prompt 中实际分析范围
        t_seq = [
            d for d in vert.get("地形线高程序列", [])
            if s_val <= float(d.get("桩号", 0.0)) <= e_val
        ]

        cropped["纵断面结构"] = {
            "桩号范围": cropped["截取范围"],
            "设计线高程序列": d_seq,
            "地形线高程序列": t_seq,
            "实际分析范围": {
                "起点桩号": s_val,
                "终点桩号": e_val,
                "起点桩号文本": format_k_station(s_val),
                "终点桩号文本": format_k_station(e_val)
            },
            "说明": (
                "设计线高程序列保留实际分析范围所涉及坡段或竖曲线段的原始控制点，"
                "可能包含区间外控制点；地形线高程序列按实际分析范围严格裁剪。"
            )
        }

        cropped.setdefault("解析状态", {})
        if len(d_seq) < 2:
            cropped["解析状态"]["设计线裁剪"] = (
                f"设计线真实控制点不足: {len(d_seq)}，无法可靠构成设计线"
            )
        else:
            cropped["解析状态"]["设计线裁剪"] = "成功"

    # 4. 障碍物裁剪
    obs_source = obstacles if obstacles is not None else full_data.get("障碍物信息", [])
    if obs_source:
        cropped["障碍物信息"] = crop_obstacles(obs_source, s_val, e_val)
    else:
        cropped["障碍物信息"] = []

    return cropped


def load_single_route_data(
    base_dir: str,
    file_prefix: str,
    route_id: Optional[str] = None,
    start_station: Optional[str] = None,
    end_station: Optional[str] = None,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
    generate_plane_json: bool = True
) -> Dict[str, Any]:
    """
    读取并可选裁剪单条线路数据。
    """
    full_data = build_single_route_full_data(
        base_dir=base_dir,
        file_prefix=file_prefix,
        route_id=route_id
    )

    rid = route_id or file_prefix

    if start_station and end_station:
        data = crop_hintcad_data(
            full_data=full_data,
            start_station=start_station,
            end_station=end_station,
            obstacles=obstacles
        )
    else:
        full_data["截取范围"] = "全线"
        if obstacles is not None:
            full_data["障碍物信息"] = obstacles
        data = full_data

    if generate_plane_json:
        plane_output_dir = get_default_preprocess_dir(base_dir, output_dir) / "plane"
        attach_plane_path_to_route_data(
            route_data=data,
            plane_output_dir=plane_output_dir,
            route_id=rid
        )

        # 单线路结果也提供 plane_paths，便于后续工具统一读取
        data["plane_paths"] = {str(rid).upper(): data.get("plane_path")}
        data.setdefault("空间基准文件", {})
        data["空间基准文件"]["plane_paths"] = data["plane_paths"]

    return data


# ======================================================
# 7. 多线路数据读取与裁剪
# ======================================================

def normalize_route_configs(route_configs: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    规范化多线路配置。

    支持传入：
    1. list[dict]
    2. JSON 字符串
    3. dict，例如：
       {
         "K": {"base_dir": "...", "file_prefix": "..."},
         "Z1": {"base_dir": "...", "file_prefix": "..."}
       }
    """
    if isinstance(route_configs, str):
        route_configs = json.loads(route_configs)

    if isinstance(route_configs, dict):
        configs = []
        for route_id, cfg in route_configs.items():
            if isinstance(cfg, dict):
                item = dict(cfg)
                item.setdefault("route_id", route_id)
                configs.append(item)
        return configs

    if isinstance(route_configs, list):
        return route_configs

    raise ValueError("route_configs 必须是 list、dict 或 JSON 字符串")


def load_multi_route_data(
    route_configs: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    route_type: str,
    start_station: Optional[str] = None,
    end_station: Optional[str] = None,
    structures_filename: Optional[str] = None,
    structures_base_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    generate_plane_json: bool = True
) -> Dict[str, Any]:
    """
    读取分离式或多线路数据。

    route_configs 示例：
    [
      {
        "route_id": "K",
        "base_dir": r"D:/project/K",
        "file_prefix": "K"
      },
      {
        "route_id": "Z1",
        "base_dir": r"D:/project/Z1",
        "file_prefix": "Z1"
      },
      {
        "route_id": "Z2",
        "base_dir": r"D:/project/Z2",
        "file_prefix": "Z2"
      }
    ]
    """
    configs = normalize_route_configs(route_configs)

    if not configs:
        raise ValueError("route_configs 为空，无法读取多线路数据")

    # 障碍物一般是项目级公共文件
    # 若未指定 structures_base_dir，则默认用第一个 route 的 base_dir
    obs_base_dir = structures_base_dir or configs[0].get("base_dir", "")
    obstacles, obs_status = load_structures_json(obs_base_dir, structures_filename)

    result: Dict[str, Any] = {
        "线路类型": route_type,
        "截取范围": (
            f"{start_station} ~ {end_station}"
            if start_station and end_station
            else "全线"
        ),
        "解析状态": {
            "障碍物": obs_status
        },
        "线路数据": {},
        "障碍物信息": [],
        "plane_paths": {},
        "空间基准文件": {
            "plane_paths": {}
        }
    }

    # 项目级障碍物裁剪
    if start_station and end_station and obstacles:
        result["障碍物信息"] = crop_obstacles(obstacles, start_station, end_station)
    else:
        result["障碍物信息"] = obstacles

    for cfg in configs:
        route_id = cfg.get("route_id") or cfg.get("线路编号") or cfg.get("name")
        base_dir = cfg.get("base_dir")
        file_prefix = cfg.get("file_prefix")

        if not route_id:
            raise ValueError(f"route_config 缺少 route_id: {cfg}")
        if not base_dir:
            raise ValueError(f"route_config 缺少 base_dir: {cfg}")
        if not file_prefix:
            raise ValueError(f"route_config 缺少 file_prefix: {cfg}")

        logger.info(f"读取线路 {route_id}: base_dir={base_dir}, file_prefix={file_prefix}")

        route_data = load_single_route_data(
            base_dir=base_dir,
            file_prefix=file_prefix,
            route_id=route_id,
            start_station=start_station,
            end_station=end_station,
            obstacles=None,
            output_dir=output_dir,
            generate_plane_json=generate_plane_json
        )

        result["线路数据"][route_id] = route_data
        result["解析状态"][route_id] = route_data.get("解析状态", {})

        rid = str(route_id).upper()
        result["plane_paths"][rid] = route_data.get("plane_path")
        result["空间基准文件"]["plane_paths"][rid] = route_data.get("plane_path")

    return result


# ======================================================
# 8. 文件输出辅助函数，可选使用
# ======================================================

def save_json(data: Dict[str, Any], output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sanitize_filename(name: str) -> str:
    """
    文件名安全化。

    用途：
        生成 plane.json、区段缓存文件等中间成果文件名。
    """
    safe = "".join(
        ch if ch.isalnum() or ch in ["_", "-", "."] else "_"
        for ch in str(name)
    )
    while "__" in safe:
        safe = safe.replace("__", "_")
    return (safe.strip("_") or "route")[:120]


def get_default_preprocess_dir(base_dir: str, output_dir: Optional[str] = None) -> Path:
    """
    获取预处理成果根目录。

    若 agent 显式传入 output_dir，则使用 output_dir/preprocess；
    否则使用 base_dir/preprocess。
    """
    root = Path(output_dir) if output_dir else Path(base_dir)
    return root / "preprocess"


def write_plane_json_from_route_data(
    route_data: Dict[str, Any],
    plane_output_dir: Union[str, Path],
    route_id: str
) -> Optional[str]:
    """
    从路线数据中的“平曲线结构”写出可复用 plane.json。

    注意：
        该 plane.json 是后续图纸裁剪、障碍物提取、碰撞检测共同使用的空间基准文件。
    """
    if not isinstance(route_data, dict):
        raise TypeError("route_data 必须是 dict")

    plane_segments = route_data.get("平曲线结构", [])
    if not plane_segments:
        logger.warning(f"线路 {route_id} 未找到平曲线结构，无法生成 plane.json")
        return None

    plane_output_dir = Path(plane_output_dir)
    plane_output_dir.mkdir(parents=True, exist_ok=True)

    plane_path = plane_output_dir / f"{sanitize_filename(route_id)}_plane.json"

    with open(plane_path, "w", encoding="utf-8") as f:
        json.dump({"平曲线结构": plane_segments}, f, ensure_ascii=False, indent=2)

    logger.info(f"线路 {route_id} plane.json 已生成: {plane_path}")
    return str(plane_path)


def attach_plane_path_to_route_data(
    route_data: Dict[str, Any],
    plane_output_dir: Union[str, Path],
    route_id: str
) -> Dict[str, Any]:
    """
    为单条路线数据生成 plane.json，并把路径写回 route_data。

    写入字段：
        route_data["plane_path"]
        route_data["空间基准文件"]["plane_path"]
    """
    plane_path = write_plane_json_from_route_data(
        route_data=route_data,
        plane_output_dir=plane_output_dir,
        route_id=route_id
    )

    route_data.setdefault("空间基准文件", {})
    route_data["空间基准文件"]["plane_path"] = plane_path
    route_data["plane_path"] = plane_path

    route_data.setdefault("解析状态", {})
    route_data["解析状态"]["plane_json"] = "成功" if plane_path else "未生成"

    return route_data


def attach_plane_paths_to_project_data(
    project_data: Dict[str, Any],
    plane_output_dir: Union[str, Path]
) -> Dict[str, Any]:
    """
    为单线路或多线路数据统一生成 plane_paths。

    单线路输入：
        {
          "线路编号": "K",
          "平曲线结构": [...]
        }

    多线路输入：
        {
          "线路数据": {
            "K": {...},
            "Z1": {...}
          }
        }

    输出新增：
        project_data["plane_paths"]
        project_data["空间基准文件"]["plane_paths"]
    """
    plane_paths: Dict[str, Optional[str]] = {}

    if "线路数据" in project_data and isinstance(project_data.get("线路数据"), dict):
        for route_id, route_data in project_data["线路数据"].items():
            rid = str(route_id).upper()
            attach_plane_path_to_route_data(
                route_data=route_data,
                plane_output_dir=plane_output_dir,
                route_id=rid
            )
            plane_paths[rid] = route_data.get("plane_path")

    else:
        rid = str(
            project_data.get("线路编号")
            or project_data.get("route_id")
            or "K"
        ).upper()

        attach_plane_path_to_route_data(
            route_data=project_data,
            plane_output_dir=plane_output_dir,
            route_id=rid
        )
        plane_paths[rid] = project_data.get("plane_path")

    project_data["plane_paths"] = plane_paths
    project_data.setdefault("空间基准文件", {})
    project_data["空间基准文件"]["plane_paths"] = plane_paths

    return project_data


def process_and_crop_project(
    base_dir: str,
    prefix: str,
    start_station: Optional[str] = None,
    end_station: Optional[str] = None,
    structures_filename: Optional[str] = None,
    output_json: bool = True,
    output_dir: Optional[str] = None,
    generate_plane_json: bool = True
) -> Dict[str, Any]:
    """
    单线路脚本式执行入口。

    主要用于本地测试，也可被外部脚本调用。
    """
    obstacles, obs_status = load_structures_json(base_dir, structures_filename)

    full_data = build_single_route_full_data(
        base_dir=base_dir,
        file_prefix=prefix,
        route_id=prefix
    )
    full_data["障碍物信息"] = obstacles
    full_data["解析状态"]["障碍物"] = obs_status

    base = Path(base_dir)

    if output_json:
        full_output_json = base / f"{prefix}_全线整合.json"
        save_json(full_data, full_output_json)
        logger.info(f"全线整合数据已生成: {full_output_json}")

    if start_station and end_station:
        cropped = crop_hintcad_data(
            full_data=full_data,
            start_station=start_station,
            end_station=end_station,
            obstacles=obstacles
        )

        if generate_plane_json:
            plane_output_dir = get_default_preprocess_dir(base_dir, output_dir) / "plane"
            attach_plane_path_to_route_data(
                route_data=cropped,
                plane_output_dir=plane_output_dir,
                route_id=prefix
            )
            cropped["plane_paths"] = {str(prefix).upper(): cropped.get("plane_path")}
            cropped.setdefault("空间基准文件", {})
            cropped["空间基准文件"]["plane_paths"] = cropped["plane_paths"]

        if output_json:
            safe_range = f"{start_station}_{end_station}".replace("+", "_").replace("-", "m")
            crop_output_path = base / f"{prefix}_截取范围_{safe_range}.json"
            save_json(cropped, crop_output_path)
            logger.info(f"局部截取数据已生成: {crop_output_path}")

        return cropped

    full_data["截取范围"] = "全线"

    if generate_plane_json:
        plane_output_dir = get_default_preprocess_dir(base_dir, output_dir) / "plane"
        attach_plane_path_to_route_data(
            route_data=full_data,
            plane_output_dir=plane_output_dir,
            route_id=prefix
        )
        full_data["plane_paths"] = {str(prefix).upper(): full_data.get("plane_path")}
        full_data.setdefault("空间基准文件", {})
        full_data["空间基准文件"]["plane_paths"] = full_data["plane_paths"]

    return full_data


# ======================================================
# 9. LangChain 工具封装：整体式 / 单线路
# ======================================================

@tool
def load_and_crop_project_data(
    base_dir: str,
    file_prefix: str,
    start_station: str = None,
    end_station: str = None,
    structures_filename: str = None,
    route_id: str = "K",
    output_dir: str = None,
    generate_plane_json: bool = True
) -> dict:
    """
    读取单线路纬地数据文件（.pm, .WID, .zdm, .dmx）以及可选障碍物 JSON，
    并根据给定起止桩号裁剪范围内数据。

    适用场景：
        1. 整体式路基；
        2. 只读取一条线路；
        3. 分离式中的单独 K 线、Z 线、Z1 或 Z2 线。

    参数:
        base_dir:
            数据文件所在目录。
        file_prefix:
            文件前缀，例如 "示例施工图-K"，对应：
                示例施工图-K.pm
                示例施工图-K.WID
                示例施工图-K.zdm
                示例施工图-K.dmx
        route_name:
            线路名称。
        start_station:
            起点桩号。
        end_station:
            终点桩号。
        structures_filename:
            障碍物 JSON 文件路径，可为绝对路径或相对路径。

    返回:
        {
          "线路名称": "",
          "线路编号": "",
          "截取范围": "",
          "解析状态": {},
          "平曲线结构": [],
          "横断面信息": [],
          "纵断面结构": {},
          "障碍物信息": [],
          "plane_paths": {"K": ""},
          "空间基准文件": {"plane_paths": {"K": ""}}
        }
    """
    try:
        obstacles, obs_status = load_structures_json(base_dir, structures_filename)

        data = load_single_route_data(
            base_dir=base_dir,
            file_prefix=file_prefix,
            route_id=route_id or file_prefix,
            start_station=start_station,
            end_station=end_station,
            obstacles=obstacles,
            output_dir=output_dir,
            generate_plane_json=generate_plane_json
        )

        data.setdefault("解析状态", {})
        data["解析状态"]["障碍物"] = obs_status

        logger.info(f"工具返回数据，范围：{data.get('截取范围')}")
        legacy_obstacles = data.get("障碍物信息", [])
        if legacy_obstacles:
           logger.info(f"可选既有构造物/障碍物记录数量：{len(legacy_obstacles)}")
        else:
           logger.info("未读取既有构造物/障碍物文件；正式障碍物信息将由 obstacle_semantic_extractor_tool 提取")

        return data

    except Exception as e:
        logger.error(f"数据加载失败: {e}", exc_info=True)
        return {"error": f"加载失败: {str(e)}"}


# ======================================================
# 10. LangChain 工具封装：分离式 / 多线路
# ======================================================

@tool
def load_and_crop_multi_route_project_data(
    route_configs: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    route_type: str,
    start_station: str = None,
    end_station: str = None,
    structures_filename: str = None,
    structures_base_dir: str = None,
    output_dir: str = None,
    generate_plane_json: bool = True
) -> dict:
    """
    读取多线路纬地数据，适配分离式路基 K / Z / Z1 / Z2。

    参数:
        route_configs:
            多线路配置，可传 JSON 字符串、list 或 dict。

            推荐 list 格式：
            [
              {
                "route_id": "K",
                "base_dir": "D:/project/K",
                "file_prefix": "K"
              },
              {
                "route_id": "Z1",
                "base_dir": "D:/project/Z1",
                "file_prefix": "Z1"
              },
              {
                "route_id": "Z2",
                "base_dir": "D:/project/Z2",
                "file_prefix": "Z2"
              }
            ]

            也支持 dict 格式：
            {
              "K": {
                "base_dir": "D:/project/K",
                "file_prefix": "K"
              },
              "Z1": {
                "base_dir": "D:/project/Z1",
                "file_prefix": "Z1"
              }
            }

        route_type:
            线路类型，例如 "分离式"。
        route_name:
            项目或线路名称。
        start_station:
            起点桩号。
        end_station:
            终点桩号。
        structures_filename:
            障碍物 JSON 文件路径。
        structures_base_dir:
            障碍物 JSON 的基准目录。若为空，默认采用第一条线路的 base_dir。

    返回:
        {
          "线路类型": "分离式",
          "线路名称": "",
          "截取范围": "",
          "解析状态": {},
          "线路数据": {
            "K": {...},
            "Z1": {...},
            "Z2": {...}
          },
          "障碍物信息": []
        }
    """
    try:
        data = load_multi_route_data(
            route_configs=route_configs,
            route_type=route_type,
            start_station=start_station,
            end_station=end_station,
            structures_filename=structures_filename,
            structures_base_dir=structures_base_dir,
            output_dir=output_dir,
            generate_plane_json=generate_plane_json
        )

        logger.info(f"多线路工具返回数据，线路数：{len(data.get('线路数据', {}))}")

        legacy_obstacles = data.get("障碍物信息", [])
        if legacy_obstacles:
            logger.info(f"可选既有构造物/障碍物记录数量：{len(legacy_obstacles)}")
        else:
            logger.info("未读取既有构造物/障碍物文件；正式障碍物信息将由 obstacle_semantic_extractor_tool 提取")

        return data

    except Exception as e:
        logger.error(f"多线路数据加载失败: {e}", exc_info=True)
        return {"error": f"多线路加载失败: {str(e)}"}
