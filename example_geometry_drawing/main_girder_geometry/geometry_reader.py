from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree as ET
import zipfile

from .geometry_config import get_reader_config
from .geometry_models import DesignParameters


_reader_cfg = get_reader_config()
_SHEET_LAYOUT = {
    name: {"section": sheet.section, "symbol_column": sheet.symbol_column, "value_column": sheet.value_column}
    for name, sheet in _reader_cfg.sheets.items()
}
_GLOBAL_SHEET_LAYOUT = {
    "sheet_name": _reader_cfg.global_sheet.sheet_name,
    "symbol_column": _reader_cfg.global_sheet.symbol_column,
    "value_column": _reader_cfg.global_sheet.value_column,
}

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
PROFILE_KEYS = {"profile_top_L", "profile_top_R"}
EXCEL_ERROR_TOKENS = {
    "#NULL!",
    "#DIV/0!",
    "#VALUE!",
    "#REF!",
    "#NAME?",
    "#NUM!",
    "#N/A",
    "#GETTING_DATA",
    "#SPILL!",
    "#CALC!",
}
CELL_REF_PATTERN = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")
PAIR_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[×xX*]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
REPEATED_NUMBER_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:\s*[×xX*]\s*([1-9][0-9]*))?\s*$"
)
REPEATED_SPAN_PATTERN = re.compile(
    r"^\s*([1-9][0-9]*)\s*[×xX*]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)


class ParameterReadError(ValueError):
    pass


def load_design_parameters(path: str | Path) -> DesignParameters:
    source_path = Path(path).expanduser().resolve()
    suffix = source_path.suffix.lower()
    if suffix == ".xlsx":
        return _load_xlsx(source_path)
    if suffix == ".json":
        return _load_json(source_path)
    raise ParameterReadError(f"不支持的输入格式：{source_path.suffix}")


def _load_json(path: Path) -> DesignParameters:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParameterReadError(f"JSON读取失败：{exc}") from exc

    output = document.get("output", document) if isinstance(document, dict) else None
    if not isinstance(output, dict):
        raise ParameterReadError("JSON顶层必须是对象，并包含output或直接包含参数分区。")
    longitudinal = output.get("longitudinal")
    cross_section = output.get("cross_section")
    if not isinstance(longitudinal, dict) or not isinstance(cross_section, dict):
        raise ParameterReadError("JSON缺少output.longitudinal或output.cross_section对象。")

    global_parameters = document.get("input", {})
    if not isinstance(global_parameters, dict):
        raise ParameterReadError("JSON的input必须是对象。")

    return DesignParameters(
        longitudinal=dict(longitudinal),
        cross_section=dict(cross_section),
        source={"type": "json", "path": str(path)},
        global_parameters=dict(global_parameters),
    )


def _load_xlsx(path: Path) -> DesignParameters:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            shared_strings = _read_shared_strings(archive)
            sheet_paths = _read_sheet_paths(archive)
            sections: dict[str, dict[str, Any]] = {
                "longitudinal": {},
                "cross_section": {},
            }
            global_parameters: dict[str, Any] = {}
            for sheet_name, layout in _SHEET_LAYOUT.items():
                member = sheet_paths.get(sheet_name)
                if member is None:
                    raise ParameterReadError(f"工作簿缺少工作表：{sheet_name}")
                cells = _read_sheet_cells(archive, member, shared_strings)
                symbol_col = layout["symbol_column"]
                value_col = layout["value_column"]
                for row_number in _row_numbers(cells, symbol_col):
                    symbol = cells.get(f"{symbol_col}{row_number}")
                    if symbol is None:
                        continue
                    if not isinstance(symbol, str) or not symbol.strip():
                        continue
                    symbol = symbol.strip()
                    value_ref = f"{value_col}{row_number}"
                    value = _normalize_parameter_value(
                        symbol,
                        cells.get(value_ref),
                    )
                    sections[layout["section"]][symbol] = value

            global_member = sheet_paths.get(_GLOBAL_SHEET_LAYOUT["sheet_name"])
            if global_member is not None:
                cells = _read_sheet_cells(archive, global_member, shared_strings)
                symbol_col = _GLOBAL_SHEET_LAYOUT["symbol_column"]
                value_col = _GLOBAL_SHEET_LAYOUT["value_column"]
                for row_number in _row_numbers(cells, symbol_col):
                    symbol = cells.get(f"{symbol_col}{row_number}")
                    if not isinstance(symbol, str) or not symbol.strip():
                        continue
                    value = cells.get(f"{value_col}{row_number}")
                    global_parameters[symbol.strip()] = _normalize_global_value(
                        symbol.strip(),
                        value,
                    )
    except ParameterReadError:
        raise
    except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise ParameterReadError(f"XLSX读取失败：{exc}") from exc

    if not sections["longitudinal"] or not sections["cross_section"]:
        raise ParameterReadError("未从工作簿中读取到纵向参数或横断面参数。")
    return DesignParameters(
        longitudinal=sections["longitudinal"],
        cross_section=sections["cross_section"],
        source={"type": "xlsx", "path": str(path)},
        global_parameters=global_parameters,
    )


def _normalize_global_value(symbol: str, value: Any) -> Any:
    if symbol != "span_layout" or not isinstance(value, str):
        return value
    repeated = REPEATED_SPAN_PATTERN.match(value)
    if repeated:
        span = _parse_number(repeated.group(2))
        if float(span) <= 0:
            raise ParameterReadError("参数span_layout中的跨度必须大于0。")
        return [span] * int(repeated.group(1))
    tokens = [item.strip() for item in value.split("+")]
    if not tokens or any(not item for item in tokens):
        raise ParameterReadError("参数span_layout必须是以+分隔的跨度序列。")
    try:
        spans = [_parse_number(item) for item in tokens]
    except ValueError as exc:
        raise ParameterReadError("参数span_layout包含无效跨度。") from exc
    if any(float(item) <= 0 for item in spans):
        raise ParameterReadError("参数span_layout中的跨度必须大于0。")
    return spans


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: list[str] = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        strings.append(
            "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
        )
    return strings


def _read_sheet_paths(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relations.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    result: dict[str, str] = {}
    sheets = workbook.find(f"{{{MAIN_NS}}}sheets")
    if sheets is None:
        return result
    for sheet in sheets:
        name = sheet.attrib.get("name")
        rel_id = sheet.attrib.get(f"{{{DOC_REL_NS}}}id")
        target = rel_targets.get(rel_id or "")
        if not name or not target:
            continue
        normalized = target.replace("\\", "/").lstrip("/")
        if not normalized.startswith("xl/"):
            normalized = f"xl/{normalized}"
        result[name] = normalized
    return result


def _read_sheet_cells(
    archive: zipfile.ZipFile,
    member: str,
    shared_strings: list[str],
) -> dict[str, Any]:
    root = ET.fromstring(archive.read(member))
    cells: dict[str, Any] = {}
    for cell in root.iter(f"{{{MAIN_NS}}}c"):
        reference = cell.attrib.get("r")
        if not reference:
            continue
        cell_type = cell.attrib.get("t")
        value_node = cell.find(f"{{{MAIN_NS}}}v")
        raw = value_node.text if value_node is not None else None

        if cell_type == "inlineStr":
            inline = cell.find(f"{{{MAIN_NS}}}is")
            value: Any = (
                "".join(node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t"))
                if inline is not None
                else ""
            )
        elif cell_type == "s":
            value = shared_strings[int(raw)] if raw is not None else ""
        elif cell_type == "b":
            value = raw == "1"
        elif cell_type == "e":
            raise ParameterReadError(
                f"单元格{reference}缓存为Excel错误：{raw or '未知错误'}"
            )
        elif cell_type == "str":
            value = raw or ""
        elif raw is None:
            value = None
        else:
            value = _parse_number(raw)

        if isinstance(value, str) and value.strip().upper() in EXCEL_ERROR_TOKENS:
            raise ParameterReadError(
                f"单元格{reference}缓存为Excel错误：{value.strip()}"
            )
        cells[reference] = value
    return cells


def _row_numbers(cells: dict[str, Any], column: str) -> list[int]:
    result: list[int] = []
    for reference in cells:
        match = CELL_REF_PATTERN.match(reference)
        if match and match.group(1) == column:
            result.append(int(match.group(2)))
    return sorted(result)


def _normalize_parameter_value(symbol: str, value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.lower() == "null":
        return None
    if text.upper() in EXCEL_ERROR_TOKENS:
        raise ParameterReadError(f"参数{symbol}缓存为Excel错误：{text}")

    if symbol in PROFILE_KEYS:
        profile: list[dict[str, float | int]] = []
        for line in text.splitlines():
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != 2:
                raise ParameterReadError(f"参数{symbol}的横坡行无法解析：{line!r}")
            profile.append(
                {
                    "length_mm": _parse_number(tokens[0]),
                    "slope_percent": _parse_number(tokens[1]),
                }
            )
        return profile

    pair_match = PAIR_PATTERN.match(text)
    if pair_match:
        return [
            _parse_number(pair_match.group(1)),
            _parse_number(pair_match.group(2)),
        ]
    if text.startswith("[") and text.endswith("]"):
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            candidate = _parse_repeated_number_sequence(symbol, text)
        return candidate
    try:
        return _parse_number(text)
    except ValueError:
        return text


def _parse_number(value: str) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"非有限数值：{value}")
    return int(number) if number.is_integer() else number


def _parse_repeated_number_sequence(
    symbol: str,
    text: str,
) -> list[int | float]:
    body = text[1:-1].strip()
    if not body:
        return []
    result: list[int | float] = []
    for token in re.split(r"[,，]", body):
        match = REPEATED_NUMBER_PATTERN.match(token)
        if match is None:
            raise ParameterReadError(f"参数{symbol}的数组无法解析：{text}")
        value = _parse_number(match.group(1))
        repeat = int(match.group(2) or "1")
        result.extend([value] * repeat)
    return result
