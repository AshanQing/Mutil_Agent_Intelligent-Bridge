from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from .reinforcement_resolution import ResolvedColumn, ResolvedColumnZone


class ResolvedColumnReinforcementDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_column_count: int = Field(gt=0)
    representative_member_count: int = Field(default=1, gt=0)
    diameter_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)
    longitudinal_mark: str
    longitudinal_diameter_mm: float = Field(gt=0)
    longitudinal_centers: list[tuple[float, float]]
    angular_spacing_deg: float = Field(gt=0)
    longitudinal_y_to_mm: float
    spiral_mark: str
    spiral_centerline_radius_mm: float = Field(gt=0)
    spiral_zones: list[ResolvedColumnZone]
    ordinary_hoop_mark: str
    ordinary_hoop_positions_mm: list[float]
    strengthening_mark: str
    strengthening_radius_mm: float = Field(gt=0)
    strengthening_positions_mm: list[float]
    strengthening_semantic_review_required: bool


def compile_column_reinforcement(
    column: ResolvedColumn,
) -> ResolvedColumnReinforcementDetail:
    count = column.longitudinal.count or 0
    centerline_diameter = column.longitudinal.centerline_diameter_mm or 0.0
    centerline_radius = centerline_diameter / 2.0
    angular_spacing = 360.0 / count
    centers = [
        (
            centerline_radius * math.cos(math.radians(index * angular_spacing)),
            centerline_radius * math.sin(math.radians(index * angular_spacing)),
        )
        for index in range(count)
    ]
    return ResolvedColumnReinforcementDetail(
        source_column_count=column.count,
        diameter_mm=column.diameter_mm,
        height_mm=column.height_mm,
        longitudinal_mark=column.longitudinal.mark,
        longitudinal_diameter_mm=column.longitudinal.diameter_mm,
        longitudinal_centers=centers,
        angular_spacing_deg=angular_spacing,
        longitudinal_y_to_mm=column.longitudinal.y_to_mm or column.height_mm,
        spiral_mark=column.spiral.mark,
        spiral_centerline_radius_mm=(
            column.spiral.centerline_diameter_mm or 0.0
        )
        / 2.0,
        spiral_zones=column.spiral_zones,
        ordinary_hoop_mark=column.ordinary_hoop.bar.mark,
        ordinary_hoop_positions_mm=column.ordinary_hoop.y_positions_mm,
        strengthening_mark=column.strengthening.bar.mark,
        strengthening_radius_mm=column.strengthening.hoop_diameter_mm / 2.0,
        strengthening_positions_mm=column.strengthening.y_positions_mm,
        strengthening_semantic_review_required=(
            column.strengthening.semantic_review_required
        ),
    )

