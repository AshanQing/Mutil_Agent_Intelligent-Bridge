from __future__ import annotations

import pytest
from pydantic import ValidationError

import bridge_agents.drawing as drawing
from bridge_agents.drawing.models import DrawingGroupSource
from bridge_agents.drawing.models import (
    ArcEntity,
    CircleEntity,
    DimensionEntity,
    DrawingDocument,
    DrawingEntity,
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


def test_drawing_group_source_rejects_duplicate_member_piers() -> None:
    with pytest.raises(ValidationError, match="member_piers"):
        DrawingGroupSource(
            design_group_id="B1-U1-1",
            task_id="B1-U1-G1",
            member_piers=["P1", "P1"],
            dimension={"盖梁尺寸": {"长度": 12000}},
            reinforcement={"reinforcement": {"pier_cap": {}, "pier_column": {}}},
            check_status="passed",
        )


def test_drawing_group_source_rejects_malformed_source_hash() -> None:
    with pytest.raises(ValidationError, match="source_hashes"):
        DrawingGroupSource(
            design_group_id="B1-U1-1",
            task_id="B1-U1-G1",
            member_piers=["P1"],
            dimension={"盖梁尺寸": {"长度": 12000}},
            reinforcement={"reinforcement": {"pier_cap": {}, "pier_column": {}}},
            check_status="passed",
            source_hashes={"reinforcement": "not-a-sha256"},
        )


def test_drawing_package_exposes_the_stable_source_contract() -> None:
    assert drawing.DrawingGroupSource is DrawingGroupSource
    assert callable(drawing.collect_drawing_group_sources)
    assert issubclass(drawing.DrawingSourceMissingError, ValueError)
    assert issubclass(drawing.DrawingSourceConflictError, ValueError)


def test_drawing_package_exposes_the_stable_ir_contract() -> None:
    expected = {
        "Point2D": Point2D,
        "DrawingLayer": DrawingLayer,
        "LineEntity": LineEntity,
        "ArcEntity": ArcEntity,
        "CircleEntity": CircleEntity,
        "TextEntity": TextEntity,
        "LeaderEntity": LeaderEntity,
        "DimensionEntity": DimensionEntity,
        "EntityRef": EntityRef,
        "Region": Region,
        "SectionGeometry": SectionGeometry,
        "GeometryView": GeometryView,
        "DrawingDocument": DrawingDocument,
        "DrawingEntity": DrawingEntity,
    }
    assert {name: getattr(drawing, name, None) for name in expected} == expected


def _document(*, entities, regions=None, layers=None) -> DrawingDocument:
    return DrawingDocument(
        design_group_id="B1-U1-1",
        member_piers=["P1", "P2"],
        layers=layers
        or [
            DrawingLayer(name="S-BEAM", color_rgb=(255, 255, 255)),
            DrawingLayer(name="A-TEXT", color_rgb=(255, 255, 0)),
        ],
        views=[
            GeometryView(
                view_id="cap-elevation",
                sections=[
                    SectionGeometry(
                        section_id="CAP-ELEVATION",
                        entities=entities,
                        regions=regions or [],
                    )
                ],
            )
        ],
    )


def test_drawing_document_rejects_entity_on_unknown_layer() -> None:
    with pytest.raises(ValidationError, match="R-MAIN"):
        _document(
            entities=[
                LineEntity(
                    entity_id="cap-main-001",
                    layer="R-MAIN",
                    start=(0, 0),
                    end=(1000, 0),
                )
            ]
        )


def test_line_rejects_zero_length_and_non_finite_coordinates() -> None:
    with pytest.raises(ValidationError, match="非零长度"):
        LineEntity(
            entity_id="line-zero",
            layer="S-BEAM",
            start=(1, 2),
            end=(1, 2),
        )
    with pytest.raises(ValidationError):
        Point2D(x=float("inf"), y=0)


def test_arc_and_circle_require_valid_radius_and_sweep() -> None:
    with pytest.raises(ValidationError):
        CircleEntity(
            entity_id="circle-zero",
            layer="S-BEAM",
            center=(0, 0),
            radius=0,
        )
    with pytest.raises(ValidationError, match="圆弧扫角"):
        ArcEntity(
            entity_id="arc-zero-sweep",
            layer="S-BEAM",
            center=(0, 0),
            radius=100,
            start_angle_deg=45,
            end_angle_deg=405,
            clockwise=False,
        )


def test_region_references_must_exist_in_the_same_section() -> None:
    with pytest.raises(ValidationError, match="missing-line"):
        _document(
            entities=[
                LineEntity(
                    entity_id="line-1",
                    layer="S-BEAM",
                    start=(0, 0),
                    end=(1000, 0),
                )
            ],
            regions=[
                Region(
                    region_id="cap-concrete",
                    layer="S-BEAM",
                    outer=[EntityRef(entity_id="missing-line")],
                )
            ],
        )


def test_entity_ids_are_unique_across_the_document() -> None:
    line = LineEntity(
        entity_id="duplicate-id",
        layer="S-BEAM",
        start=(0, 0),
        end=(1000, 0),
    )
    with pytest.raises(ValidationError, match="duplicate-id"):
        DrawingDocument(
            design_group_id="B1-U1-1",
            member_piers=["P1"],
            layers=[DrawingLayer(name="S-BEAM")],
            views=[
                GeometryView(
                    view_id="view-1",
                    sections=[SectionGeometry(section_id="section-1", entities=[line])],
                ),
                GeometryView(
                    view_id="view-2",
                    sections=[SectionGeometry(section_id="section-2", entities=[line])],
                ),
            ],
        )


def test_all_supported_entities_round_trip_with_canonical_order() -> None:
    document = _document(
        layers=[
            DrawingLayer(name="S-BEAM", color_rgb=(255, 255, 255)),
            DrawingLayer(name="A-TEXT", color_rgb=(255, 255, 0)),
        ],
        entities=[
            TextEntity(
                entity_id="text-006",
                layer="A-TEXT",
                insertion=(0, 500),
                text="盖梁立面",
                height=250,
            ),
            DimensionEntity(
                entity_id="dimension-005",
                layer="A-TEXT",
                start=(0, 0),
                end=(1000, 0),
                offset=300,
            ),
            LeaderEntity(
                entity_id="leader-004",
                layer="A-TEXT",
                points=[(0, 0), (200, 200)],
                text="N1",
                text_height=180,
            ),
            CircleEntity(
                entity_id="circle-003",
                layer="S-BEAM",
                center=(500, 500),
                radius=100,
            ),
            ArcEntity(
                entity_id="arc-002",
                layer="S-BEAM",
                center=(500, 0),
                radius=500,
                start_angle_deg=0,
                end_angle_deg=180,
                clockwise=False,
            ),
            LineEntity(
                entity_id="line-001",
                layer="S-BEAM",
                start=(0, 0),
                end=(1000, 0),
            ),
        ],
    )

    restored = DrawingDocument.model_validate_json(document.model_dump_json())

    assert restored == document
    entity_ids = [
        entity.entity_id
        for view in restored.views
        for section in view.sections
        for entity in section.entities
    ]
    assert entity_ids == [
        "arc-002",
        "circle-003",
        "dimension-005",
        "leader-004",
        "line-001",
        "text-006",
    ]


def test_stable_json_is_independent_of_metadata_insertion_order() -> None:
    line = LineEntity(
        entity_id="line-001",
        layer="S-BEAM",
        start=(0, 0),
        end=(1000, 0),
    )
    first = _document(entities=[line])
    first.metadata = {"source": "dimension", "revision": 2}
    second = _document(entities=[line])
    second.metadata = {"revision": 2, "source": "dimension"}

    assert first.to_stable_json() == second.to_stable_json()


@pytest.mark.parametrize(
    "layer",
    [
        {"name": "", "color_rgb": (255, 255, 255)},
        {"name": "S-BEAM", "line_type": ""},
        {"name": "S-BEAM", "color_rgb": (256, 0, 0)},
        {"name": "S-BEAM", "lineweight_mm": 0},
    ],
)
def test_layer_style_rejects_empty_name_invalid_rgb_and_nonpositive_weight(
    layer,
) -> None:
    with pytest.raises(ValidationError):
        DrawingLayer.model_validate(layer)


def test_entity_and_region_identifiers_cannot_be_blank() -> None:
    with pytest.raises(ValidationError):
        LineEntity(
            entity_id=" ",
            layer="S-BEAM",
            start=(0, 0),
            end=(1, 0),
        )
    with pytest.raises(ValidationError):
        EntityRef(entity_id=" ")
    with pytest.raises(ValidationError):
        Region(region_id=" ", layer="S-BEAM", outer=[EntityRef(entity_id="line-1")])
    valid_line = LineEntity(
        entity_id="line-1",
        layer="S-BEAM",
        start=(0, 0),
        end=(1, 0),
    )
    with pytest.raises(ValidationError):
        SectionGeometry(section_id=" ", entities=[valid_line])
    with pytest.raises(ValidationError):
        GeometryView(
            view_id=" ",
            sections=[SectionGeometry(section_id="section-1", entities=[valid_line])],
        )


def test_text_and_leader_require_visible_content_and_positive_size() -> None:
    with pytest.raises(ValidationError):
        TextEntity(
            entity_id="text-1",
            layer="A-TEXT",
            insertion=(0, 0),
            text=" ",
            height=0,
        )
    with pytest.raises(ValidationError):
        LeaderEntity(
            entity_id="leader-1",
            layer="A-TEXT",
            points=[(0, 0)],
            text="N1",
            text_height=0,
        )


def test_dimension_rejects_identical_measurement_points() -> None:
    with pytest.raises(ValidationError, match="尺寸起止点"):
        DimensionEntity(
            entity_id="dimension-zero",
            layer="A-TEXT",
            start=(1, 1),
            end=(1, 1),
            offset=300,
        )


def test_region_requires_nonempty_boundary_loops() -> None:
    with pytest.raises(ValidationError):
        Region(region_id="region-empty", layer="S-BEAM", outer=[])
    with pytest.raises(ValidationError, match="孔洞"):
        Region(
            region_id="region-empty-hole",
            layer="S-BEAM",
            outer=[EntityRef(entity_id="line-1")],
            holes=[[]],
        )


def test_document_rejects_empty_hierarchy_and_invalid_group_members() -> None:
    with pytest.raises(ValidationError):
        DrawingDocument(
            design_group_id=" ",
            member_piers=[],
            layers=[],
            views=[],
        )
    with pytest.raises(ValidationError, match="member_piers"):
        DrawingDocument(
            design_group_id="B1-U1-1",
            member_piers=["P1", "P1"],
            layers=[DrawingLayer(name="S-BEAM")],
            views=[
                GeometryView(
                    view_id="view-1",
                    sections=[
                        SectionGeometry(
                            section_id="section-1",
                            entities=[
                                LineEntity(
                                    entity_id="line-1",
                                    layer="S-BEAM",
                                    start=(0, 0),
                                    end=(1, 0),
                                )
                            ],
                        )
                    ],
                )
            ],
        )


def test_document_rejects_duplicate_layer_view_and_section_ids() -> None:
    line = LineEntity(
        entity_id="line-1",
        layer="S-BEAM",
        start=(0, 0),
        end=(1, 0),
    )
    section = SectionGeometry(section_id="section-1", entities=[line])
    with pytest.raises(ValidationError, match="图层名称重复"):
        DrawingDocument(
            design_group_id="B1-U1-1",
            member_piers=["P1"],
            layers=[DrawingLayer(name="S-BEAM"), DrawingLayer(name="S-BEAM")],
            views=[GeometryView(view_id="view-1", sections=[section])],
        )
    with pytest.raises(ValidationError, match="视图 ID 重复"):
        DrawingDocument(
            design_group_id="B1-U1-1",
            member_piers=["P1"],
            layers=[DrawingLayer(name="S-BEAM")],
            views=[
                GeometryView(view_id="view-1", sections=[section]),
                GeometryView(view_id="view-1", sections=[section]),
            ],
        )
    with pytest.raises(ValidationError, match="截面 ID 重复"):
        GeometryView(view_id="view-1", sections=[section, section])
