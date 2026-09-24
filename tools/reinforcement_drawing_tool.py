from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from bridge_agents.drawing.group_sources import collect_drawing_group_sources
from bridge_agents.drawing.package import generate_drawing_package


def reinforcement_drawing_tool(
    *,
    pier_group_result: Mapping[str, Any],
    dimension_design_result: Mapping[str, Any],
    reinforcement_design_result: Mapping[str, Any],
    capacity_check_result: Mapping[str, Any] | None,
    output_dir: str | Path,
    allow_unverified: bool = False,
    accepted_risks: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    sources = collect_drawing_group_sources(
        pier_group_result=pier_group_result,
        dimension_design_result=dimension_design_result,
        reinforcement_design_result=reinforcement_design_result,
        capacity_check_result=capacity_check_result,
    )
    result = generate_drawing_package(
        sources=sources,
        output_dir=output_dir,
        allow_unverified=allow_unverified,
        accepted_risks=accepted_risks,
    )
    return result.model_dump(mode="json")
