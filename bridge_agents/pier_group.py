from __future__ import annotations

import math
import re
from typing import Any, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# 桩号与高程
# ---------------------------------------------------------------------------

def parse_station(value: Any) -> Optional[float]:
    """解析桩号文本为数值（米），例如 K12+373.500 -> 12373.5。"""
    text = str(value or "").strip().upper().replace(" ", "")
    match = re.search(r"[KZ]?(\d+)\+(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1)) * 1000.0 + float(match.group(2))
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def interpolate_elevation(
    points: Sequence[Mapping[str, Any]],
    station: float,
) -> Optional[float]:
    """按桩号线性插值高程；桩号越界返回 None。"""
    cleaned: List[tuple[float, float]] = []
    for item in points:
        if not isinstance(item, Mapping):
            continue
        sta = item.get("桩号")
        elev = item.get("高程")
        try:
            sta_f = float(sta)
            elev_f = float(elev)
        except (TypeError, ValueError):
            continue
        if math.isfinite(sta_f) and math.isfinite(elev_f):
            cleaned.append((sta_f, elev_f))
    if len(cleaned) < 2:
        return None
    cleaned.sort(key=lambda pair: pair[0])
    if station < cleaned[0][0] or station > cleaned[-1][0]:
        return None
    for (s1, h1), (s2, h2) in zip(cleaned, cleaned[1:]):
        if s1 <= station <= s2:
            if math.isclose(s1, s2):
                return h1
            t = (station - s1) / (s2 - s1)
            return h1 + (h2 - h1) * t
    return None


# ---------------------------------------------------------------------------
# 上部结构高度经验表（假设，需记录来源）
# ---------------------------------------------------------------------------

_T_BEAM_HEIGHTS = {20.0: 1.8, 25.0: 2.2, 30.0: 2.5, 40.0: 3.0}
_BOX_GIRDER_HEIGHTS = {20.0: 1.8, 30.0: 2.2, 40.0: 2.5}


def _interpolate_span_table(table: Mapping[float, float], span_m: float) -> float:
    spans = sorted(table)
    if span_m <= spans[0]:
        return table[spans[0]]
    if span_m >= spans[-1]:
        return table[spans[-1]]
    for lo, hi in zip(spans, spans[1:]):
        if lo <= span_m <= hi:
            t = (span_m - lo) / (hi - lo)
            return table[lo] + (table[hi] - table[lo]) * t
    return table[spans[-1]]


def superstructure_height_m(bridge_type: Any, span_m: float) -> float:
    """按桥型与跨径返回上部结构总高度（假设值，单位为米）。"""
    text = str(bridge_type or "")
    table = _BOX_GIRDER_HEIGHTS if "箱梁" in text else _T_BEAM_HEIGHTS
    return _interpolate_span_table(table, float(span_m))


# ---------------------------------------------------------------------------
# 盖梁变截面与柱位高度
# ---------------------------------------------------------------------------

def cap_height_at_column_x(
    h_mid: float,
    h_end: float,
    cap_length: float,
    cantilever: float,
    column_x: float,
) -> float:
    """对称变截面盖梁在柱位横向坐标处的截面高度。"""
    half = cap_length / 2.0
    inner_half = half - cantilever
    abs_x = abs(column_x)
    if abs_x <= inner_half + 1e-9:
        return h_mid
    t = (half - abs_x) / cantilever if cantilever > 0 else 0.0
    return h_end + (h_mid - h_end) * t


# ---------------------------------------------------------------------------
# 墩高计算
# ---------------------------------------------------------------------------

def compute_layout_pier_height(
    design_elev: float,
    ground_elev: float,
    super_height: float,
) -> float:
    """布跨墩高 = |设计高程 - 地面高程| - 上部结构高度。"""
    return abs(float(design_elev) - float(ground_elev)) - float(super_height)


# ---------------------------------------------------------------------------
# 布跨结果墩行提取与墩高核实
# ---------------------------------------------------------------------------

def parse_pier_no(value: Any) -> str:
    """从「0 (桥台)」这类标注中提取纯数字墩号。"""
    text = str(value or "").strip()
    match = re.search(r"\d+", text)
    return match.group(0) if match else text


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _unwrap_layout_payload(layout_result: Mapping[str, Any]) -> Mapping[str, Any]:
    payload: Any = layout_result
    for key in ("layout_result", "design_result", "result", "output"):
        if isinstance(payload, Mapping) and isinstance(payload.get(key), Mapping):
            payload = payload[key]
            break
    return payload if isinstance(payload, Mapping) else {}


def _iter_layout_sides(bridge: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    sides: List[Mapping[str, Any]] = []
    unified = bridge.get("统一布跨方案")
    if isinstance(unified, Mapping) and unified:
        sides.append(unified)
    for side in bridge.get("分幅布跨方案列表", []) or []:
        if isinstance(side, Mapping):
            sides.append(side)
    return sides


def _pick_layout_height(item: Mapping[str, Any], side_label: str) -> float:
    """按幅别选择墩高字段，兼容「左/右幅墩高(m)」与旧「本幅墩高(m)」两种命名。"""
    if "右" in side_label:
        candidates = ("右幅墩高(m)", "本幅墩高(m)", "左幅墩高(m)", "墩高(m)")
    else:
        candidates = ("左幅墩高(m)", "本幅墩高(m)", "右幅墩高(m)", "墩高(m)")
    for key in candidates:
        value = item.get(key)
        if value is not None and str(value).strip() != "":
            return _to_float(value, 0.0)
    return 0.0


def extract_pier_rows(layout_result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """从布跨结果提取所有「墩位与墩高」行，附带宽幅与桥型上下文。"""
    payload = _unwrap_layout_payload(layout_result)
    rows: List[Dict[str, Any]] = []
    for bridge in payload.get("桥位列表", []) or []:
        if not isinstance(bridge, Mapping):
            continue
        bridge_id = str(bridge.get("桥位编号") or bridge.get("桥梁编号") or "").strip()
        for side in _iter_layout_sides(bridge):
            side_label = str(side.get("幅别") or side.get("侧别") or "").strip()
            bridge_type = (side.get("设桥信息") or {}).get("桥型")
            for item in side.get("墩位与墩高", []) or []:
                if not isinstance(item, Mapping):
                    continue
                rows.append({
                    "bridge_id": bridge_id,
                    "side": side_label,
                    "bridge_type": bridge_type,
                    "pier_raw": item.get("墩号"),
                    "pier_no": parse_pier_no(item.get("墩号")),
                    "station_text": item.get("桩号"),
                    "station": parse_station(item.get("桩号")),
                    "pier_type": item.get("墩位类型"),
                    "layout_height_m": _pick_layout_height(item, side_label),
                    "check_note": item.get("墩位校核说明"),
                })
    return rows


def verify_pier_height(
    row: Mapping[str, Any],
    vertical_profile: Mapping[str, Any],
    super_height: float,
    tolerance: float = 0.5,
) -> Dict[str, Any]:
    """确定性核实单个墩的布跨墩高，返回高程、重算值与差值。"""
    station = row.get("station")
    base = {
        "bridge_id": row.get("bridge_id"),
        "side": row.get("side"),
        "pier_no": row.get("pier_no"),
        "station": station,
        "layout_height_m": _to_float(row.get("layout_height_m"), 0.0),
    }
    if station is None:
        return {**base, "status": "station_missing", "height_mismatch": False}

    design_elev = interpolate_elevation(
        vertical_profile.get("设计线高程序列", []) or [], station
    )
    ground_elev = interpolate_elevation(
        vertical_profile.get("地形线高程序列", []) or [], station
    )
    if design_elev is None or ground_elev is None:
        return {**base, "status": "elevation_missing", "height_mismatch": False}

    computed = compute_layout_pier_height(design_elev, ground_elev, super_height)
    layout_height = base["layout_height_m"]
    delta = abs(computed - layout_height)
    return {
        **base,
        "design_elevation": design_elev,
        "ground_elevation": ground_elev,
        "computed_layout_height_m": computed,
        "height_delta_m": delta,
        "height_mismatch": delta > tolerance,
        "status": "verified",
    }


# ---------------------------------------------------------------------------
# 设计组确定性归并
# ---------------------------------------------------------------------------

def _parse_span_combo(value: Any) -> List[float]:
    """解析跨径组合，例如 40+40 或 6×30。"""
    text = str(value or "").replace("×", "x").replace("X", "x").replace("*", "x").replace(" ", "")
    spans: List[float] = []
    for part in text.split("+"):
        if not part:
            continue
        match = re.match(r"^(\d+)\s*x\s*(\d+(?:\.\d+)?)$", part)
        if match:
            spans.extend([float(match.group(2))] * int(match.group(1)))
            continue
        match = re.match(r"^(\d+(?:\.\d+)?)$", part)
        if match:
            spans.append(float(match.group(1)))
    return spans


def _freeze(value: Any) -> Any:
    """把嵌套结构转换为可哈希的稳定形式。"""
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(v) for v in value))
    return value


def _deck_width_key(deck: Mapping[str, Any]) -> Any:
    return _freeze({
        "宽度类型": deck.get("宽度类型"),
        "起点宽度": deck.get("起点宽度"),
        "终点宽度": deck.get("终点宽度"),
    })


def _pier_sort_key(pier_no: str) -> tuple:
    try:
        return (0, int(pier_no))
    except (TypeError, ValueError):
        return (1, str(pier_no))


def _find_fallback_group(
    by_key: Dict[Any, Dict[str, Any]],
    info: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """为缺少尺寸分组的桥墩选择同单元相邻分组（同角色优先）。"""
    if not by_key:
        return None
    for bucket in by_key.values():
        if bucket["info"]["pier_role"] == info.get("pier_role"):
            return bucket
    for bucket in by_key.values():
        if bucket["info"]["pier_role"] != "桥台":
            return bucket
    return next(iter(by_key.values()))


def _unwrap_design_units_root(design_units: Mapping[str, Any]) -> Mapping[str, Any]:
    """兼容三种 design_units 布局，返回含“桥梁列表”的任务1结果体。

    - extract 工具返回信封：{"success": ..., "design_units_result": {"任务1_设计单元提取结果": {...}}, ...}；
    - 落盘文件与历史断点 state：{"任务1_设计单元提取结果": {...}}；
    - 极端情况已是 {"桥梁列表": [...]} 的任务1内容体。
    与 tools/dimension_design_tool.py 的信封解包保持一致，避免“dimension 正常但
    pier_group 归并出 0 分组”这类同一数据两种读法的断裂。
    """
    if not isinstance(design_units, Mapping):
        return {}
    payload = design_units.get("design_units_result", design_units)
    if not isinstance(payload, Mapping):
        return {}
    root = payload.get("任务1_设计单元提取结果", payload)
    return root if isinstance(root, Mapping) else {}


def _collect_pier_units(
    design_units: Mapping[str, Any],
) -> tuple[Dict[tuple, Dict[str, Any]], Dict[tuple, str]]:
    root = _unwrap_design_units_root(design_units)
    units: Dict[tuple, Dict[str, Any]] = {}
    first_unit: Dict[tuple, str] = {}
    for bridge in root.get("桥梁列表", []) or []:
        if not isinstance(bridge, Mapping):
            continue
        bridge_id = str(bridge.get("桥梁编号") or "").strip()
        bridge_type = bridge.get("桥型")
        for unit in bridge.get("设计单元列表", []) or []:
            if not isinstance(unit, Mapping):
                continue
            unit_id = str(unit.get("单元编号") or "").strip()
            spans = _parse_span_combo((unit.get("本联信息") or {}).get("跨径组合"))
            deck = unit.get("桥面宽度信息") or {}
            for pier in unit.get("桥墩信息", []) or []:
                if not isinstance(pier, Mapping):
                    continue
                pier_no = parse_pier_no(pier.get("原始墩号"))
                first_key = (bridge_id, pier_no)
                first_unit.setdefault(first_key, unit_id)
                units[(bridge_id, unit_id, pier_no)] = {
                    "bridge_id": bridge_id,
                    "unit_id": unit_id,
                    "pier_no": pier_no,
                    "pier_role": pier.get("墩位角色"),
                    "is_connection_pier": bool(pier.get("是否连接墩")),
                    "bridge_type": bridge_type,
                    "spans": spans,
                    "deck_width": deck,
                }
    return units, first_unit


def _collect_pier_dimensions(
    dimension_design_result: Mapping[str, Any],
) -> Dict[tuple, Dict[str, Any]]:
    # dimension_design_tool 返回 {"success": ..., "dimension_design_result": {...}}，
    # 先解掉这层包装，与 _extract_dimension_unit_items / _extract_dimension_groups 保持一致。
    nested = dimension_design_result.get("dimension_design_result")
    if isinstance(nested, Mapping):
        dimension_design_result = nested
    root = dimension_design_result.get("任务2_下部结构尺寸设计结果", dimension_design_result)
    if not isinstance(root, Mapping):
        root = {}
    dims: Dict[tuple, Dict[str, Any]] = {}
    for bridge in root.get("桥梁列表", []) or []:
        if not isinstance(bridge, Mapping):
            continue
        bridge_id = str(bridge.get("桥梁编号") or "").strip()
        for unit in bridge.get("单元尺寸设计结果", []) or []:
            if not isinstance(unit, Mapping):
                continue
            unit_id = str(unit.get("单元编号") or "").strip()
            groups = {
                str(g.get("分组编号") or ""): g
                for g in unit.get("分组尺寸设计结果", []) or []
                if isinstance(g, Mapping)
            }
            for mapping in unit.get("桥墩尺寸映射关系", []) or []:
                if not isinstance(mapping, Mapping):
                    continue
                pier_no = parse_pier_no(mapping.get("桥墩号"))
                group = groups.get(str(mapping.get("所属分组编号") or ""))
                if group is not None:
                    dims[(bridge_id, unit_id, pier_no)] = group
    return dims


def _merge_fields(
    info: Mapping[str, Any],
    dim: Mapping[str, Any],
    material_grade: str,
) -> Dict[str, Any]:
    return {
        "bridge_type": info.get("bridge_type"),
        "pier_role": info.get("pier_role"),
        "spans": info.get("spans"),
        "deck_width": _deck_width_key(info.get("deck_width") or {}),
        "is_double_support": dim.get("是否双排支座"),
        "t_beam_layout": dim.get("T梁布置"),
        "盖梁尺寸": dim.get("盖梁尺寸"),
        "pier_type": dim.get("墩柱类型"),
        "墩柱尺寸": dim.get("墩柱尺寸"),
        "material_grade": material_grade,
    }


def build_design_groups(
    design_units: Mapping[str, Any],
    dimension_design_result: Mapping[str, Any],
    material_grade: str = "C40+HRB400",
) -> Dict[str, Any]:
    """单联内确定性设计组归并，连接墩只归属其首次出现的联。"""
    pier_units, first_unit = _collect_pier_units(design_units)
    pier_dims = _collect_pier_dimensions(dimension_design_result)

    buckets: Dict[tuple, List[tuple[Dict[str, Any], Optional[Dict[str, Any]]]]] = {}
    for key, info in pier_units.items():
        bridge_id, unit_id, pier_no = key
        if info["is_connection_pier"] and first_unit.get((bridge_id, pier_no)) != unit_id:
            # 连接墩默认只归属其首次出现的联；但当本联的尺寸设计明确为该墩
            # 分配了尺寸分组（如 2-2 单元的 G2 含墩1）时，必须在本联独立成组，
            # 否则该分组的净高/配筋将没有数据来源，绘图侧会因柱高缺失被排除。
            if (bridge_id, unit_id, pier_no) not in pier_dims:
                continue
        buckets.setdefault((bridge_id, unit_id), []).append((info, pier_dims.get(key)))

    design_groups: List[Dict[str, Any]] = []
    unmapped_piers: List[Dict[str, Any]] = []
    for (bridge_id, unit_id) in sorted(buckets):
        records = sorted(buckets[(bridge_id, unit_id)], key=lambda r: _pier_sort_key(r[0]["pier_no"]))
        by_key: Dict[Any, Dict[str, Any]] = {}
        for info, dim in records:
            if dim is None:
                if info["pier_role"] != "桥台":
                    fallback = _find_fallback_group(by_key, info)
                    if fallback is not None:
                        fallback["member_piers"].append(info["pier_no"])
                        fallback.setdefault("assumed_member_piers", []).append(info["pier_no"])
                        continue
                unmapped_piers.append({
                    "bridge_id": bridge_id,
                    "unit_id": unit_id,
                    "pier_no": info["pier_no"],
                    "reason": "dimension_missing",
                })
                continue
            key = _freeze(_merge_fields(info, dim, material_grade))
            bucket = by_key.setdefault(key, {
                "info": info,
                "dim": dim,
                "member_piers": [],
            })
            bucket["member_piers"].append(info["pier_no"])

        for bucket in by_key.values():
            members = sorted(bucket["member_piers"], key=_pier_sort_key)
            design_groups.append({
                "design_group_id": f"{bridge_id}-{unit_id}-{len(design_groups) + 1}",
                "bridge_id": bridge_id,
                "unit_id": unit_id,
                "pier_role": bucket["info"]["pier_role"],
                "member_piers": members,
                "assumed_member_piers": sorted(
                    bucket.get("assumed_member_piers", []),
                    key=_pier_sort_key,
                ),
                "merge_fields": _merge_fields(bucket["info"], bucket["dim"], material_grade),
                "controlling_pier_id": None,
            })

    return {"design_groups": design_groups, "unmapped_piers": unmapped_piers}


# ---------------------------------------------------------------------------
# 净高计算与顶层整合
# ---------------------------------------------------------------------------

def cap_height_at_pier(dim: Mapping[str, Any]) -> float:
    """墩位处盖梁高度：多柱取各柱位盖梁高度的最小值（净高最不利）。"""
    cap = dim.get("盖梁尺寸") or {}
    col = dim.get("墩柱尺寸") or {}
    h_mid = _to_float(cap.get("中高"), 0.0)
    h_end = _to_float(cap.get("端高"), h_mid)
    cap_length = _to_float(cap.get("长度"), 0.0)
    cantilever = _to_float(cap.get("悬臂"), 0.0)
    count = int(_to_float(col.get("数量"), 0.0))
    spacing = _to_float(col.get("中心间距"), 0.0)
    if count <= 0 or cap_length <= 0:
        return 0.0
    heights = []
    for i in range(1, count + 1):
        x = (i - (count + 1) / 2.0) * spacing
        heights.append(cap_height_at_column_x(h_mid, h_end, cap_length, cantilever, x))
    return min(heights)


def compute_net_height(
    layout_height_m: float,
    cap_height: float,
    foundation_offset: float,
) -> float:
    """墩柱净高 = 布跨墩高 - 柱位处盖梁高度 - 基础顶面附加偏移。"""
    return float(layout_height_m) - float(cap_height) - float(foundation_offset)


def compute_pier_groups(
    layout_result: Mapping[str, Any],
    design_units: Mapping[str, Any],
    dimension_design_result: Mapping[str, Any],
    vertical_profiles: Mapping[str, Mapping[str, Any]],
    super_height: Optional[float] = None,
    foundation_top_offset_m: float = 0.0,
    material_grade: str = "C40+HRB400",
) -> Dict[str, Any]:
    """整合墩高核实、柱位盖梁高度、净高与审计，形成桥墩结构设计组成果。"""
    grouped = build_design_groups(design_units, dimension_design_result, material_grade)
    design_groups = grouped["design_groups"]
    design_bridge_ids = {group["bridge_id"] for group in design_groups}

    pier_rows = extract_pier_rows(layout_result)
    rows_by_pier: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in pier_rows:
        key_bridge = resolve_design_bridge_id(
            row["bridge_id"], row["side"], design_bridge_ids
        )
        rows_by_pier.setdefault((key_bridge, row["pier_no"]), []).append(row)

    height_audit: List[Dict[str, Any]] = []

    for group in design_groups:
        bridge_id = group["bridge_id"]
        bridge_type = group["merge_fields"].get("bridge_type")
        spans = group["merge_fields"].get("spans") or []
        member_heights: Dict[str, float] = {}
        for pier_no in group["member_piers"]:
            rows = rows_by_pier.get((bridge_id, pier_no), [])
            row = rows[0] if rows else None
            if row is None:
                height_audit.append({
                    "bridge_id": bridge_id,
                    "pier_no": pier_no,
                    "status": "row_missing",
                })
                continue
            profile = vertical_profiles.get(row.get("side")) or {}
            sh = (
                super_height
                if super_height is not None
                else superstructure_height_m(bridge_type, max(spans) if spans else 0.0)
            )
            verified = verify_pier_height(row, profile, sh)
            cap_h = cap_height_at_pier(group["merge_fields"])
            layout_height = verified.get("layout_height_m", 0.0)
            net = compute_net_height(layout_height, cap_h, foundation_top_offset_m)
            member_heights[pier_no] = net
            height_audit.append({
                "bridge_id": bridge_id,
                "side": row.get("side"),
                "pier_no": pier_no,
                "station_text": row.get("station_text"),
                "design_elevation": verified.get("design_elevation"),
                "ground_elevation": verified.get("ground_elevation"),
                "layout_height_m": layout_height,
                "computed_layout_height_m": verified.get("computed_layout_height_m"),
                "height_mismatch": verified.get("height_mismatch"),
                "cap_height_at_pier_m": cap_h,
                "net_height_m": net,
                "status": verified.get("status"),
            })

        group["member_net_heights_m"] = member_heights
        if member_heights:
            controlling = max(member_heights, key=lambda k: member_heights[k])
            group["controlling_pier_id"] = controlling
            group["controlling_net_height_m"] = member_heights[controlling]
        else:
            group["controlling_pier_id"] = None
            group["controlling_net_height_m"] = None

    return {
        "design_groups": design_groups,
        "unmapped_piers": grouped["unmapped_piers"],
        "height_audit": height_audit,
    }


# ---------------------------------------------------------------------------
# 纵断面提取与幅别映射
# ---------------------------------------------------------------------------

def _extract_profile(route: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(route, Mapping):
        return None
    vertical = route.get("纵断面结构")
    if isinstance(vertical, Mapping):
        design = vertical.get("设计线高程序列")
        ground = vertical.get("地形线高程序列")
        if design or ground:
            return {"设计线高程序列": design or [], "地形线高程序列": ground or []}
    design = route.get("设计线高程序列")
    ground = route.get("地形线高程序列")
    if design or ground:
        return {"设计线高程序列": design or [], "地形线高程序列": ground or []}
    return None


def collect_vertical_profiles(cropped_data: Any) -> Dict[str, Dict[str, Any]]:
    """从裁剪数据提取「线路 key -> 纵断面结构」映射。"""
    profiles: Dict[str, Dict[str, Any]] = {}
    if not isinstance(cropped_data, Mapping):
        return profiles

    route_map = cropped_data.get("线路数据")
    if isinstance(route_map, Mapping):
        for key, route in route_map.items():
            profile = _extract_profile(route)
            if profile:
                profiles[str(key)] = profile

    for key in ("K", "Z", "Z1", "Z2"):
        if key in cropped_data and key not in profiles:
            profile = _extract_profile(cropped_data[key])
            if profile:
                profiles[key] = profile

    if not profiles:
        profile = _extract_profile(cropped_data)
        if profile:
            profiles["K"] = profile
    return profiles


def build_side_route_map(
    layout_result: Mapping[str, Any],
    vertical_profiles: Mapping[str, Any],
) -> Dict[str, str]:
    """建立「幅别 -> 线路 key」映射；缺省约定右幅为 K、左幅为 Z。"""
    route_map: Dict[str, str] = {}
    payload = _unwrap_layout_payload(layout_result)
    for bridge in payload.get("桥位列表", []) or []:
        if not isinstance(bridge, Mapping):
            continue
        for side in _iter_layout_sides(bridge):
            side_label = str(side.get("幅别") or side.get("侧别") or "").strip()
            line = str(side.get("线路") or "").strip().replace("线", "")
            if side_label and line and line in vertical_profiles:
                route_map[side_label] = line

    for side_label in ("右幅", "左幅"):
        if side_label in route_map:
            continue
        key = "K" if "右" in side_label else "Z"
        if key in vertical_profiles:
            route_map[side_label] = key
    return route_map


def resolve_design_bridge_id(
    layout_bridge_id: Any,
    side: Any,
    design_bridge_ids: Any,
) -> str:
    """把布跨桥位编号映射到设计单元桥梁编号（分幅时加 -R/-L）。"""
    known = {str(value) for value in (design_bridge_ids or [])}
    plain = str(layout_bridge_id)
    if plain in known:
        return plain
    side_text = str(side or "")
    suffix = "R" if "右" in side_text else "L" if "左" in side_text else ""
    if suffix:
        candidate = f"{plain}-{suffix}"
        if candidate in known:
            return candidate
    return plain
