from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from bridge_agents.result_catalog import ResultArtifact, STAGE_TITLES


@dataclass(frozen=True)
class ArtifactRow:
    stage: str
    stage_title: str
    kind: str
    title: str
    file_path: str
    preview_path: str
    status: str
    design_group: str
    source: str
    diagnostics_text: str


def build_artifact_rows(artifacts: Iterable[ResultArtifact]) -> tuple[ArtifactRow, ...]:
    return tuple(
        ArtifactRow(
            stage=item.stage,
            stage_title=STAGE_TITLES.get(item.stage, item.stage),
            kind=item.kind,
            title=item.title,
            file_path=item.file_path,
            preview_path=item.preview_path,
            status=item.status,
            design_group=item.design_group,
            source=item.source,
            diagnostics_text="\n".join(f"{key}：{value}" for key, value in item.diagnostics.items()),
        )
        for item in artifacts
    )
