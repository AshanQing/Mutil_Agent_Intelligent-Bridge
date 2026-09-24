from __future__ import annotations

import json
import math
import re
from typing import Annotated, Any, Literal, Self, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DrawingGroupSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    design_group_id: str
    task_id: str
    member_piers: list[str]
    dimension: dict[str, Any]
    reinforcement: dict[str, Any]
    check_status: Literal["passed", "failed", "missing"]
    design_context: dict[str, Any] = Field(default_factory=dict)
    source_paths: dict[str, str] = Field(default_factory=dict)
    source_hashes: dict[str, str] = Field(default_factory=dict)

    @field_validator("design_group_id", "task_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("标识不能为空")
        return normalized

    @field_validator("member_piers")
    @classmethod
    def validate_member_piers(cls, value: list[str]) -> list[str]:
        normalized = [str(item).strip() for item in value]
        if not normalized or any(not item for item in normalized):
            raise ValueError("member_piers 必须包含非空桥墩编号")
        if len(set(normalized)) != len(normalized):
            raise ValueError("member_piers 不允许重复桥墩编号")
        return normalized

    @field_validator("source_hashes")
    @classmethod
    def validate_source_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        normalized = {str(key): str(digest).lower() for key, digest in value.items()}
        invalid = [
            key
            for key, digest in normalized.items()
            if not _SHA256_RE.fullmatch(digest)
        ]
        if invalid:
            raise ValueError(f"source_hashes 包含非法 SHA256: {invalid}")
        return normalized


class _DrawingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Point2D(_DrawingModel):
    x: float
    y: float

    @model_validator(mode="before")
    @classmethod
    def accept_coordinate_pair(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return {"x": value[0], "y": value[1]}
        return value


class DrawingLayer(_DrawingModel):
    name: str
    color_rgb: tuple[int, int, int] = (255, 255, 255)
    line_type: str = "Continuous"
    lineweight_mm: float = 0.25

    @field_validator("name", "line_type")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("图层名称和线型不能为空")
        return normalized

    @field_validator("color_rgb")
    @classmethod
    def validate_rgb(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(component < 0 or component > 255 for component in value):
            raise ValueError("RGB 分量必须位于 0 到 255")
        return value

    @field_validator("lineweight_mm")
    @classmethod
    def validate_lineweight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("线宽必须大于 0")
        return value


class _EntityBase(_DrawingModel):
    entity_id: str
    layer: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entity_id", "layer")
    @classmethod
    def validate_entity_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("图元 ID 和图层不能为空")
        return normalized


class LineEntity(_EntityBase):
    kind: Literal["line"] = "line"
    start: Point2D
    end: Point2D

    @model_validator(mode="after")
    def validate_nonzero_length(self) -> Self:
        if self.start == self.end:
            raise ValueError("线段必须具有非零长度")
        return self


class ArcEntity(_EntityBase):
    kind: Literal["arc"] = "arc"
    center: Point2D
    radius: float = Field(gt=0)
    start_angle_deg: float
    end_angle_deg: float
    clockwise: bool

    @model_validator(mode="after")
    def validate_nonzero_sweep(self) -> Self:
        sweep = (self.end_angle_deg - self.start_angle_deg) % 360.0
        if math.isclose(sweep, 0.0, abs_tol=1e-9):
            raise ValueError("圆弧扫角必须位于 0 到 360 度之间")
        return self


class CircleEntity(_EntityBase):
    kind: Literal["circle"] = "circle"
    center: Point2D
    radius: float = Field(gt=0)


class TextEntity(_EntityBase):
    kind: Literal["text"] = "text"
    insertion: Point2D
    text: str
    height: float = Field(gt=0)
    rotation_deg: float = 0.0

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("文字内容不能为空")
        return str(value)


class LeaderEntity(_EntityBase):
    kind: Literal["leader"] = "leader"
    points: list[Point2D] = Field(min_length=2)
    text: str
    text_height: float = Field(gt=0)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("引线文字不能为空")
        return str(value)


class DimensionEntity(_EntityBase):
    kind: Literal["dimension"] = "dimension"
    start: Point2D
    end: Point2D
    offset: float
    text_override: str | None = None

    @model_validator(mode="after")
    def validate_nonzero_measurement(self) -> Self:
        if self.start == self.end:
            raise ValueError("尺寸起止点必须不同")
        return self


DrawingEntity = Annotated[
    Union[
        LineEntity,
        ArcEntity,
        CircleEntity,
        TextEntity,
        LeaderEntity,
        DimensionEntity,
    ],
    Field(discriminator="kind"),
]


class EntityRef(_DrawingModel):
    entity_id: str
    direction: Literal["forward", "reverse"] = "forward"

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("引用图元 ID 不能为空")
        return normalized


class Region(_DrawingModel):
    region_id: str
    layer: str
    outer: list[EntityRef] = Field(min_length=1)
    holes: list[list[EntityRef]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("region_id", "layer")
    @classmethod
    def validate_region_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("区域 ID 和图层不能为空")
        return normalized

    @field_validator("holes")
    @classmethod
    def validate_holes(cls, value: list[list[EntityRef]]) -> list[list[EntityRef]]:
        if any(not loop for loop in value):
            raise ValueError("Region 孔洞引用环不能为空")
        return value


class SectionGeometry(_DrawingModel):
    section_id: str
    entities: list[DrawingEntity] = Field(min_length=1)
    regions: list[Region] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("section_id")
    @classmethod
    def validate_section_id(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("截面 ID 不能为空")
        return normalized

    @model_validator(mode="after")
    def validate_and_sort_entities(self) -> Self:
        entity_by_id: dict[str, DrawingEntity] = {}
        for entity in self.entities:
            if entity.entity_id in entity_by_id:
                raise ValueError(f"截面内图元 ID 重复: {entity.entity_id}")
            entity_by_id[entity.entity_id] = entity

        region_ids: set[str] = set()
        referenceable = {
            entity.entity_id
            for entity in self.entities
            if isinstance(entity, (LineEntity, ArcEntity, CircleEntity))
        }
        for region in self.regions:
            if region.region_id in region_ids:
                raise ValueError(f"截面内 Region ID 重复: {region.region_id}")
            region_ids.add(region.region_id)
            refs = [*region.outer, *(ref for hole in region.holes for ref in hole)]
            missing = sorted(
                {ref.entity_id for ref in refs if ref.entity_id not in referenceable}
            )
            if missing:
                raise ValueError(
                    f"Region {region.region_id} 引用了不存在或不可成边界的图元: {missing}"
                )

        self.entities = sorted(self.entities, key=lambda item: item.entity_id)
        self.regions = sorted(self.regions, key=lambda item: item.region_id)
        return self


class GeometryView(_DrawingModel):
    view_id: str
    sections: list[SectionGeometry] = Field(min_length=1)
    coordinate_system: dict[str, str] = Field(
        default_factory=lambda: {"origin": "(0, 0)", "x_axis": "+X", "y_axis": "+Y"}
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("view_id")
    @classmethod
    def validate_view_id(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("视图 ID 不能为空")
        return normalized

    @model_validator(mode="after")
    def validate_and_sort_sections(self) -> Self:
        section_ids = [section.section_id for section in self.sections]
        duplicates = sorted(
            {
                section_id
                for section_id in section_ids
                if section_ids.count(section_id) > 1
            }
        )
        if duplicates:
            raise ValueError(f"视图内截面 ID 重复: {duplicates}")
        self.sections = sorted(self.sections, key=lambda item: item.section_id)
        return self


class DrawingDocument(_DrawingModel):
    schema_version: Literal["drawing-ir-v1"] = "drawing-ir-v1"
    units: Literal["mm"] = "mm"
    angle_units: Literal["degrees"] = "degrees"
    design_group_id: str
    member_piers: list[str] = Field(min_length=1)
    layers: list[DrawingLayer] = Field(min_length=1)
    views: list[GeometryView] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("design_group_id")
    @classmethod
    def validate_design_group_id(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("设计组 ID 不能为空")
        return normalized

    @field_validator("member_piers")
    @classmethod
    def validate_member_piers(cls, value: list[str]) -> list[str]:
        normalized = [str(item).strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("member_piers 必须包含非空桥墩编号")
        if len(set(normalized)) != len(normalized):
            raise ValueError("member_piers 不允许重复桥墩编号")
        return normalized

    @model_validator(mode="after")
    def validate_document_references(self) -> Self:
        layer_names = [layer.name for layer in self.layers]
        duplicate_layers = sorted(
            {name for name in layer_names if layer_names.count(name) > 1}
        )
        if duplicate_layers:
            raise ValueError(f"图层名称重复: {duplicate_layers}")
        known_layers = set(layer_names)

        view_ids = [view.view_id for view in self.views]
        duplicate_views = sorted(
            {view_id for view_id in view_ids if view_ids.count(view_id) > 1}
        )
        if duplicate_views:
            raise ValueError(f"视图 ID 重复: {duplicate_views}")

        global_ids: set[str] = set()
        for view in self.views:
            for section in view.sections:
                for item in [*section.entities, *section.regions]:
                    item_id = (
                        item.entity_id
                        if isinstance(item, _EntityBase)
                        else item.region_id
                    )
                    if item_id in global_ids:
                        raise ValueError(f"DrawingDocument 图元 ID 重复: {item_id}")
                    global_ids.add(item_id)
                    if item.layer not in known_layers:
                        raise ValueError(
                            f"图元 {item_id} 引用了未声明图层: {item.layer}"
                        )

        self.layers = sorted(self.layers, key=lambda item: item.name)
        self.views = sorted(self.views, key=lambda item: item.view_id)
        return self

    def to_stable_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            indent=indent,
        )
