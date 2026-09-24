"""桥梁设计成果的确定性绘图表达层。"""

from .group_sources import (
    DrawingSourceConflictError,
    DrawingSourceError,
    DrawingSourceMissingError,
    collect_drawing_group_sources,
)
from .config import DrawingConfig, DrawingConfigError, load_drawing_config
from .layout import Bounds, build_combined_layout, entity_bounds, view_bounds
from .models import (
    ArcEntity,
    CircleEntity,
    DimensionEntity,
    DrawingDocument,
    DrawingEntity,
    DrawingGroupSource,
    DrawingLayer,
    EntityRef,
    GeometryView,
    LeaderEntity,
    LineEntity,
    Point2D,
    Region,
    SectionGeometry,
    TextEntity,
)
from .package import (
    build_final_deliverables,
    DrawingPackageResult,
    UnverifiedDrawingSourceError,
    generate_drawing_package,
)
from .reinforcement_adapter import (
    BarScheduleRow,
    CapGeometry,
    ColumnGeometry,
    DrawingExpressionError,
    adapt_cap_geometry,
    adapt_column_geometry,
    build_bar_schedule,
    evaluate_drawing_expression,
)
from .views import (
    build_cap_elevation,
    build_cap_section,
    build_column_elevation,
    build_column_section,
    build_reinforcement_drawing,
)
from .scr_exporter import format_point, render_autocad_script
from .svg_exporter import render_svg
from .validation import (
    GeometryDiagnostic,
    GeometryValidationResult,
    validate_geometry,
)

__all__ = [
    "DrawingGroupSource",
    "Point2D",
    "DrawingLayer",
    "LineEntity",
    "ArcEntity",
    "CircleEntity",
    "TextEntity",
    "LeaderEntity",
    "DimensionEntity",
    "EntityRef",
    "Region",
    "SectionGeometry",
    "GeometryView",
    "DrawingDocument",
    "DrawingEntity",
    "DrawingSourceConflictError",
    "DrawingSourceError",
    "DrawingSourceMissingError",
    "collect_drawing_group_sources",
    "DrawingExpressionError",
    "CapGeometry",
    "ColumnGeometry",
    "BarScheduleRow",
    "evaluate_drawing_expression",
    "adapt_cap_geometry",
    "adapt_column_geometry",
    "build_bar_schedule",
    "build_cap_elevation",
    "build_cap_section",
    "build_column_elevation",
    "build_column_section",
    "build_reinforcement_drawing",
    "DrawingConfig",
    "DrawingConfigError",
    "load_drawing_config",
    "Bounds",
    "entity_bounds",
    "view_bounds",
    "build_combined_layout",
    "GeometryDiagnostic",
    "GeometryValidationResult",
    "validate_geometry",
    "format_point",
    "render_autocad_script",
    "render_svg",
    "DrawingPackageResult",
    "UnverifiedDrawingSourceError",
    "generate_drawing_package",
    "build_final_deliverables",
]
