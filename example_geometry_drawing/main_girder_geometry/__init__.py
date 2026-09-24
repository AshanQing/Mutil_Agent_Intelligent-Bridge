"""Main-girder geometry conversion and drawing package."""

__version__ = "0.2.0"

from .geometry_config import (
    get_cad_config,
    get_layer_style,
    get_reader_config,
)
from .geometry_generator import (
    generate_cross_section,
    generate_geometry,
    generate_key_sections,
    validate_geometry,
)
from .geometry_preview import write_geometry_preview
from .geometry_reader import load_design_parameters
from .plan_generator import (
    generate_plan,
    generate_plan_stations,
    validate_plan_parameters,
)

__all__ = [
    "generate_cross_section",
    "generate_geometry",
    "generate_key_sections",
    "generate_plan",
    "generate_plan_stations",
    "get_cad_config",
    "get_layer_style",
    "get_reader_config",
    "load_design_parameters",
    "validate_plan_parameters",
    "validate_geometry",
    "write_geometry_preview",
]
