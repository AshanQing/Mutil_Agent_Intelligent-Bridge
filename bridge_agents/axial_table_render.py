"""墩柱轴压验算汇总的表格化成果（PNG 预览图 + CSV 表格）。

把配筋批次落盘的 axial_check_summary.json 渲染成：
  - axial_check_summary.png：成果中心可直接预览的表格位图；
  - axial_check_summary.csv：Excel/外部程序打开的表格。

模块不依赖 Tkinter；PIL 延迟导入，缺失或渲染失败时上层可退回
"外部打开 JSON"（不影响成果中心其它功能）。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",  # 黑体
    r"C:\Windows\Fonts\simsun.ttc",  # 宋体
)

_VERDICT_LABELS = {
    "passed": "通过",
    "failed": "不满足",
    "manual_review": "需人工复核",
    "not_applicable": "不适用(无柱身)",
    "missing": "缺失",
}
_VERDICT_COLORS = {
    "passed": (18, 141, 63),
    "failed": (217, 48, 37),
    "manual_review": (199, 120, 0),
    "not_applicable": (100, 116, 139),
    "missing": (142, 142, 147),
}

# (列头, 字典键, 列宽px, 值格式化)
_COLUMNS = [
    ("设计组", "task_id", 150, lambda v: str(v or "")),
    ("关联桥墩", "member_piers", 170, lambda v: "、".join(str(x) for x in (v or []))),
    ("柱径/mm", "column_diameter_mm", 90, lambda v: f"{float(v):.0f}" if isinstance(v, (int, float)) else "-"),
    ("净高/m", "controlling_net_height_m", 90, lambda v: f"{float(v):.2f}" if isinstance(v, (int, float)) else "-"),
    ("k", "effective_length_factor", 56, lambda v: f"{float(v):g}" if isinstance(v, (int, float)) else "-"),
    ("λ(l0/i)", "slenderness_ratio", 96, lambda v: f"{float(v):.1f}" if isinstance(v, (int, float)) else "-"),
    ("φ", "stability_factor_phi", 72, lambda v: f"{float(v):.3f}" if isinstance(v, (int, float)) else "-"),
    ("利用率", "utilization", 84, lambda v: f"{float(v):.3f}" if isinstance(v, (int, float)) else "-"),
    ("结论", "verdict", 130, lambda v: _VERDICT_LABELS.get(str(v or ""), str(v or "-"))),
    ("说明", "message", 430, lambda v: _clip(str(v or ""), 56)),
]


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _load_font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def write_axial_check_summary_csv(summary: Dict[str, Any], csv_path: str | Path) -> None:
    """把汇总表写成 CSV（表头中文，结论/状态列保留原文+中文）。"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["设计组", "关联桥墩", "柱径/mm", "净高/m", "k", "λ(l0/i)",
                         "φ", "利用率", "状态", "结论", "说明"])
        for row in summary.get("groups") or []:
            verdict = str(row.get("verdict") or "")
            writer.writerow([
                row.get("task_id") or "",
                "、".join(str(x) for x in (row.get("member_piers") or [])),
                _fmt_num(row.get("column_diameter_mm"), 0),
                _fmt_num(row.get("controlling_net_height_m"), 2),
                _fmt_num(row.get("effective_length_factor"), 3),
                _fmt_num(row.get("slenderness_ratio"), 1),
                _fmt_num(row.get("stability_factor_phi"), 3),
                _fmt_num(row.get("utilization"), 3),
                row.get("status") or "-",
                _VERDICT_LABELS.get(verdict, verdict or "-"),
                _clip(row.get("message") or "", 200),
            ])


def _fmt_num(value: Any, digits: int) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.{digits}f}"


def render_axial_check_summary_table(
    summary: Dict[str, Any],
    png_path: str | Path,
) -> bool:
    """把汇总表渲染成 PNG 表格图；返回是否成功（PIL/字体不可用或渲染异常返回 False）。"""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False
    try:
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        groups = summary.get("groups") or []
        rows = min(len(groups), 200)
        scale = 2  # 2x 抗锯齿再缩小，保证文字清晰
        pad_x, pad_y = 12, 10
        header_h = 34
        row_h = 30
        title_h = 46
        foot_h = 40
        width = sum(col[2] for col in _COLUMNS) + pad_x * 2
        height = title_h + header_h + row_h * rows + foot_h + pad_y * 2

        image = Image.new("RGB", (width * scale, height * scale), "white")
        draw = ImageDraw.Draw(image)

        font_title = _load_font(20 * scale)
        font_header = _load_font(15 * scale)
        font_cell = _load_font(14 * scale)
        font_small = _load_font(12 * scale)

        y0 = pad_y * scale
        draw.text((pad_x * scale, y0), "墩柱轴压验算汇总", font=font_title, fill=(17, 24, 39))
        counts = summary.get("verdict_counts") or {}
        count_text = (
            f"共 {summary.get('expected_task_count', 0)} 组 · "
            f"通过 {counts.get('passed', 0)} · "
            f"不满足 {counts.get('failed', 0)} · "
            f"需人工复核 {counts.get('manual_review', 0)} · "
            f"不适用(无柱身) {counts.get('not_applicable', 0)} · "
            f"缺失 {counts.get('missing', 0)} · "
            "表5.3.1 稳定系数，λ 为回转半径口径 l0/i"
        )
        draw.text((pad_x * scale, (title_h - 20) * scale), count_text, font=font_small, fill=(80, 90, 101))

        top = title_h * scale
        # 表头
        x = pad_x * scale
        for label, _key, col_w, _fmt in _COLUMNS:
            draw.rectangle([x, top, x + col_w * scale, top + header_h * scale], fill=(242, 243, 247),
                           outline=(200, 202, 207))
            draw.text((x + 6, top + (header_h * scale - 20 * scale) / 2), label,
                      font=font_header, fill=(38, 44, 52))
            x += col_w * scale
        # 数据行
        y = top + header_h * scale
        for row in groups[:rows]:
            verdict = str(row.get("verdict") or "")
            verdict_color = _VERDICT_COLORS.get(verdict, (51, 51, 51))
            x = pad_x * scale
            for _label, key, col_w, fmt in _COLUMNS:
                draw.rectangle([x, y, x + col_w * scale, y + row_h * scale], fill="white",
                               outline=(225, 228, 233))
                text = fmt(row.get(key))
                color = verdict_color if key == "verdict" else (17, 24, 39)
                # message 列纵向居中按实际文本高度
                try:
                    bbox = draw.textbbox((0, 0), text, font=font_cell)
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_h = 14 * scale
                draw.text((x + 6, y + (row_h * scale - text_h) / 2 - 2), text, font=font_cell, fill=color)
                x += col_w * scale
            y += row_h * scale
        # 脚注
        note = "说明：passed=轴压通过；failed=轴压承载力不足；manual_review=长细比超登记范围/需人工；missing=无验算记录。k=1 为缺少墩顶/墩底约束刚度时的保守假定。"
        draw.text((pad_x * scale, y + 4 * scale), _clip(note, 150), font=font_small, fill=(120, 126, 134))

        image = image.resize((width, height), Image.LANCZOS)
        image.save(png_path)
        return True
    except Exception:
        return False


def ensure_axial_check_views(summary_path: str | Path) -> Tuple[str, str]:
    """幂等生成 summary 对应的 PNG/CSV；返回 (png_path or "", csv_path or "")。

    仅当 PNG 缺失或早于 JSON 时重渲染，避免成果中心反复刷新重复生成。
    CSV 每次比对 mtime 重写（开销小、内容小）。
    """
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        return "", ""
    try:
        import json

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return "", ""
    except Exception:
        return "", ""

    csv_path = summary_path.with_name(summary_path.stem + ".csv")
    png_path = summary_path.with_name(summary_path.stem + ".png")

    try:
        write_axial_check_summary_csv(payload, csv_path)
    except Exception:
        pass

    try:
        stale = not png_path.is_file() or os.path.getmtime(png_path) < os.path.getmtime(summary_path)
        if stale:
            render_axial_check_summary_table(payload, png_path)
    except Exception:
        pass
    return (str(png_path) if png_path.is_file() else "", str(csv_path) if csv_path.is_file() else "")
