"""Configuration loader for geometry_drawing.

Reads ``data/geometry_config.json`` and provides typed access to every
configurable value.  If the file is missing or invalid the module prints a
warning and returns hard-coded defaults that exactly match the pre-config
behaviour — the pipeline therefore works with zero configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


# ── typed containers ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerStyle:
    """Visual properties of one AutoCAD / SVG layer."""

    linetype: str
    color: str
    lineweight: str
    dash: str | None
    description: str = ""


@dataclass
class SvgTextStyle:
    font_family: str = "Microsoft YaHei, sans-serif"
    font_size: int = 12
    font_weight: str = "normal"
    color: str = "#172033"


@dataclass
class SvgPadding:
    left: float = 70.0
    right: float = 70.0
    top: float = 90.0
    bottom: float = 70.0


@dataclass
class SvgHatchConfig:
    tile_width: int = 14
    tile_height: int = 14
    rotation: int = 35
    background: str = "#dbe4ee"
    line_color: str = "#8aa0b8"
    line_width: str = "2"


@dataclass
class SvgOriginConfig:
    dot_radius: float = 4.0
    dot_color: str = "#21885b"
    label_color: str = "#21885b"


@dataclass
class SvgAxisConfig:
    stroke: str = "#35a06f"
    stroke_width: str = "1.2"
    dash: str = "8 6"


@dataclass
class SvgConfig:
    width_px: int = 1400
    height_px: int = 820
    arc_step_deg: float = 4.0
    padding: SvgPadding = field(default_factory=SvgPadding)
    background_color: str = "#f7f9fc"
    title: SvgTextStyle = field(default_factory=lambda: SvgTextStyle(font_size=20, font_weight="600", color="#172033"))
    subtitle: SvgTextStyle = field(default_factory=lambda: SvgTextStyle(font_size=13, color="#5f6b7a"))
    legend_text: SvgTextStyle = field(default_factory=lambda: SvgTextStyle(color="#364152"))
    mono_text: SvgTextStyle = field(default_factory=lambda: SvgTextStyle(font_family="Consolas, monospace", color="#5f6b7a"))
    origin: SvgOriginConfig = field(default_factory=SvgOriginConfig)
    axis: SvgAxisConfig = field(default_factory=SvgAxisConfig)
    hatch: SvgHatchConfig = field(default_factory=SvgHatchConfig)
    view_labels: dict[str, str] = field(default_factory=lambda: {"cross_section": "横断面", "elevation": "立面", "plan": "平面"})


@dataclass
class CadConfig:
    layout_gap_mm: float = 5000.0
    layout_origin_x: float = 0.0
    layout_origin_y: float = 0.0
    combined_view_name: str = "cad_layout"
    combined_section_id: str = "COMPLETE-GIRDER-LAYOUT"
    format_precision: str = ".12g"
    fallback_layer: str = "0"
    default_linetype: str = "BYLAYER"
    output_linetype_commands: bool = True


@dataclass
class GeometryConfig:
    closure_tolerance_mm: float = 0.01
    epsilon: float = 1e-9
    round_digits: int = 6
    schema_version: str = "2.0.0"
    units: str = "mm"
    angle_unit: str = "degree"
    section_specs_filename: str = "geometry_section_specs.json"
    key_section_ids: list[str] = field(default_factory=lambda: ["END-DIAPHRAGM", "CONSTANT-SECTION", "MID-DIAPHRAGM", "PIER-SECTION"])
    entity_prefixes: dict[str, str] = field(default_factory=lambda: {"cross_section": "xs_{section_id}", "elevation": "el_half_girder_elevation", "plan": "pl_half_girder_plan"})
    station_id_patterns: dict[str, str] = field(default_factory=lambda: {"elevation": "el_station_{index:03d}", "plan": "pl_station_{index:03d}"})


@dataclass
class ReaderSheet:
    section: str = ""
    symbol_column: str = "E"
    value_column: str = "G"


@dataclass
class ReaderGlobalSheet:
    sheet_name: str = ""
    symbol_column: str = "C"
    value_column: str = "D"


@dataclass
class ReaderConfig:
    sheets: dict[str, ReaderSheet] = field(default_factory=dict)
    global_sheet: ReaderGlobalSheet = field(default_factory=ReaderGlobalSheet)


# ── internal defaults ─────────────────────────────────────────────────────────

# Every value below matches the pre-config hard-coded behaviour exactly.
# When the JSON file is missing or a key is absent these are used verbatim.

_DEFAULT_LAYERS: dict[str, LayerStyle] = {
    "XS-OUTLINE":     LayerStyle("BYLAYER", "#E7E9F0", "0.50", None, "横断面外轮廓"),
    "XS-VOID":        LayerStyle("BYLAYER", "#172033", "0.35", None, "横断面箱室孔洞"),
    "XS-MANHOLE":     LayerStyle("BYLAYER", "#172033", "0.35", None, "横断面人洞"),
    "XS-MANHOLE-REF": LayerStyle("BYLAYER", "#f41212", "0.35", "7 5", "横断面人洞参考线"),
    "XS-CONCRETE":    LayerStyle("BYLAYER", "#8aa0b8", "0.25", None, "横断面混凝土面域"),
    "EL-OUTLINE":     LayerStyle("BYLAYER", "#172033", "0.50", None, "立面外轮廓"),
    "EL-VOID":        LayerStyle("BYLAYER", "#172033", "0.35", None, "立面箱室及人洞空腔"),
    "EL-CUT":         LayerStyle("BYLAYER", "#607d8b", "0.30", "6 4", "立面截断线"),
    "EL-CONCRETE":    LayerStyle("BYLAYER", "#8aa0b8", "0.25", None, "立面混凝土面域"),
    "PL-OUTLINE":     LayerStyle("BYLAYER", "#172033", "0.50", None, "平面外轮廓"),
    "PL-VOID":        LayerStyle("BYLAYER", "#172033", "0.35", None, "平面箱室空腔"),
    "PL-CUT":         LayerStyle("BYLAYER", "#607d8b", "0.30", "6 4", "平面截断线"),
    "PL-CONCRETE":    LayerStyle("BYLAYER", "#8aa0b8", "0.25", None, "平面混凝土面域"),
}

# ── singleton loader ──────────────────────────────────────────────────────────

_config: dict[str, Any] | None = None


def _config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "geometry_config.json"


def _load_config() -> dict[str, Any]:
    """Return the parsed JSON config, or an empty dict if unavailable."""
    path = _config_path()
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        print(f"[geometry_config] 配置文件不存在，使用内置默认值：{path}")
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[geometry_config] 配置文件读取失败，使用内置默认值：{exc}")
    return {}


def _init() -> None:
    global _config
    if _config is None:
        _config = _load_config()


def _section(name: str) -> dict[str, Any]:
    """Return a top-level section of the config, falling back to {}."""
    _init()
    assert _config is not None
    value = _config.get(name)
    return value if isinstance(value, dict) else {}


# ── public accessors ──────────────────────────────────────────────────────────


def get_layer_style(layer: str) -> LayerStyle:
    """Return the visual style for *layer*.

    If the layer is unknown it returns a safe fallback (Continuous, black).
    """
    section_data = _section("layers")
    if layer in section_data and isinstance(section_data[layer], dict):
        return _layer_from_entry(section_data[layer], layer)
    default = _DEFAULT_LAYERS.get(layer)
    return default if default is not None else LayerStyle("Continuous", "#000000", "0.25", None, layer)


def _layer_from_entry(entry: dict[str, Any], name: str) -> LayerStyle:
    default = _DEFAULT_LAYERS.get(name, LayerStyle("BYLAYER", "#000000", "0.25", None, name))
    return LayerStyle(
        linetype=str(entry.get("linetype", default.linetype)),
        color=str(entry.get("color", default.color)),
        lineweight=str(entry.get("lineweight", default.lineweight)),
        dash=entry.get("dash", default.dash) if entry.get("dash", default.dash) is not None else None,
        description=str(entry.get("description", default.description)),
    )


def get_svg_config() -> SvgConfig:
    """Return the full SVG configuration."""
    section_data = _section("svg")

    def _text(key: str, default: SvgTextStyle) -> SvgTextStyle:
        entry = section_data.get(key)
        if isinstance(entry, dict):
            return SvgTextStyle(
                font_family=str(entry.get("font_family", default.font_family)),
                font_size=int(entry.get("font_size", default.font_size)),
                font_weight=str(entry.get("font_weight", default.font_weight)),
                color=str(entry.get("color", default.color)),
            )
        return default

    padding_data = section_data.get("padding")
    if isinstance(padding_data, dict):
        padding = SvgPadding(
            left=float(padding_data.get("left", 70)),
            right=float(padding_data.get("right", 70)),
            top=float(padding_data.get("top", 90)),
            bottom=float(padding_data.get("bottom", 70)),
        )
    else:
        padding = SvgPadding()

    hatch_data = section_data.get("hatch")
    if isinstance(hatch_data, dict):
        hatch = SvgHatchConfig(
            tile_width=int(hatch_data.get("tile_width", 14)),
            tile_height=int(hatch_data.get("tile_height", 14)),
            rotation=int(hatch_data.get("rotation", 35)),
            background=str(hatch_data.get("background", "#dbe4ee")),
            line_color=str(hatch_data.get("line_color", "#8aa0b8")),
            line_width=str(hatch_data.get("line_width", "2")),
        )
    else:
        hatch = SvgHatchConfig()

    origin_data = section_data.get("origin")
    if isinstance(origin_data, dict):
        origin = SvgOriginConfig(
            dot_radius=float(origin_data.get("dot_radius", 4)),
            dot_color=str(origin_data.get("dot_color", "#21885b")),
            label_color=str(origin_data.get("label_color", "#21885b")),
        )
    else:
        origin = SvgOriginConfig()

    axis_data = section_data.get("axis")
    if isinstance(axis_data, dict):
        axis = SvgAxisConfig(
            stroke=str(axis_data.get("stroke", "#35a06f")),
            stroke_width=str(axis_data.get("stroke_width", "1.2")),
            dash=str(axis_data.get("dash", "8 6")),
        )
    else:
        axis = SvgAxisConfig()

    view_labels_data = section_data.get("view_labels")
    if isinstance(view_labels_data, dict):
        view_labels = {str(k): str(v) for k, v in view_labels_data.items()}
    else:
        view_labels = {"cross_section": "横断面", "elevation": "立面", "plan": "平面"}

    return SvgConfig(
        width_px=int(section_data.get("width_px", 1400)),
        height_px=int(section_data.get("height_px", 820)),
        arc_step_deg=float(section_data.get("arc_step_deg", 4.0)),
        padding=padding,
        background_color=str(section_data.get("background_color", "#f7f9fc")),
        title=_text("title", SvgTextStyle(font_size=20, font_weight="600", color="#172033")),
        subtitle=_text("subtitle", SvgTextStyle(font_size=13, color="#5f6b7a")),
        legend_text=_text("legend_text", SvgTextStyle(color="#364152")),
        mono_text=_text("mono_text", SvgTextStyle(font_family="Consolas, monospace", color="#5f6b7a")),
        origin=origin,
        axis=axis,
        hatch=hatch,
        view_labels=view_labels,
    )


def get_cad_config() -> CadConfig:
    """Return CAD export settings."""
    section_data = _section("cad")
    return CadConfig(
        layout_gap_mm=float(section_data.get("layout_gap_mm", 5000.0)),
        layout_origin_x=float(section_data.get("layout_origin_x", 0.0)),
        layout_origin_y=float(section_data.get("layout_origin_y", 0.0)),
        combined_view_name=str(section_data.get("combined_view_name", "cad_layout")),
        combined_section_id=str(section_data.get("combined_section_id", "COMPLETE-GIRDER-LAYOUT")),
        format_precision=str(section_data.get("format_precision", ".12g")),
        fallback_layer=str(section_data.get("fallback_layer", "0")),
        default_linetype=str(section_data.get("default_linetype", "BYLAYER")),
        output_linetype_commands=bool(section_data.get("output_linetype_commands", True)),
    )


def get_geometry_config() -> GeometryConfig:
    """Return geometry-engine constants."""
    section_data = _section("geometry")

    prefixes_data = section_data.get("entity_prefixes")
    if isinstance(prefixes_data, dict):
        prefixes = {str(k): str(v) for k, v in prefixes_data.items()}
    else:
        prefixes = {"cross_section": "xs_{section_id}", "elevation": "el_half_girder_elevation", "plan": "pl_half_girder_plan"}

    station_data = section_data.get("station_id_patterns")
    if isinstance(station_data, dict):
        station_patterns = {str(k): str(v) for k, v in station_data.items()}
    else:
        station_patterns = {"elevation": "el_station_{index:03d}", "plan": "pl_station_{index:03d}"}

    key_ids = section_data.get("key_section_ids")
    if not isinstance(key_ids, list) or not key_ids:
        key_ids = ["END-DIAPHRAGM", "CONSTANT-SECTION", "MID-DIAPHRAGM", "PIER-SECTION"]

    return GeometryConfig(
        closure_tolerance_mm=float(section_data.get("closure_tolerance_mm", 0.01)),
        epsilon=float(section_data.get("epsilon", 1e-9)),
        round_digits=int(section_data.get("round_digits", 6)),
        schema_version=str(section_data.get("schema_version", "2.0.0")),
        units=str(section_data.get("units", "mm")),
        angle_unit=str(section_data.get("angle_unit", "degree")),
        section_specs_filename=str(section_data.get("section_specs_filename", "geometry_section_specs.json")),
        key_section_ids=[str(s) for s in key_ids],
        entity_prefixes=prefixes,
        station_id_patterns=station_patterns,
    )


def get_reader_config() -> ReaderConfig:
    """Return Excel-reader sheet layout."""
    section_data = _section("reader")

    sheets: dict[str, ReaderSheet] = {}
    sheets_data = section_data.get("sheets")
    if isinstance(sheets_data, dict):
        for name, entry in sheets_data.items():
            if isinstance(entry, dict):
                sheets[str(name)] = ReaderSheet(
                    section=str(entry.get("section", "")),
                    symbol_column=str(entry.get("symbol_column", "E")),
                    value_column=str(entry.get("value_column", "G")),
                )

    global_data = section_data.get("global_sheet")
    if isinstance(global_data, dict):
        global_sheet = ReaderGlobalSheet(
            sheet_name=str(global_data.get("sheet_name", "")),
            symbol_column=str(global_data.get("symbol_column", "C")),
            value_column=str(global_data.get("value_column", "D")),
        )
    else:
        global_sheet = ReaderGlobalSheet()

    # If no sheets defined in config, fall back to the hard-coded layout
    if not sheets:
        sheets = {
            "纵向参数": ReaderSheet(section="longitudinal", symbol_column="E", value_column="G"),
            "横断面参数": ReaderSheet(section="cross_section", symbol_column="E", value_column="G"),
        }
    if not global_sheet.sheet_name:
        global_sheet = ReaderGlobalSheet(
            sheet_name="增补全局参数-主要的QA(Q PART)", symbol_column="C", value_column="D"
        )

    return ReaderConfig(sheets=sheets, global_sheet=global_sheet)


def get_entity_prefix(view_name: str, section_id: str = "") -> str:
    """Return the entity-ID prefix for *view_name*.

    For cross-section views the *section_id* (e.g. ``"END-DIAPHRAGM"``) is
    substituted into the template.
    """
    geo = get_geometry_config()
    template = geo.entity_prefixes.get(view_name, "")
    if not template:
        return view_name
    safe_id = section_id.lower().replace("-", "_")
    return template.format(section_id=safe_id)
