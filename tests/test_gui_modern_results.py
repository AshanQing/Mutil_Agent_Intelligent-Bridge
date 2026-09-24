from bridge_agents.gui_modern_results import build_artifact_rows
from bridge_agents.result_catalog import ResultArtifact


def test_artifact_rows_keep_stage_path_and_preview() -> None:
    rows = build_artifact_rows(
        [
            ResultArtifact(
                stage="initial_design",
                kind="terrain",
                title="原始地形栅格",
                file_path="a.png",
                preview_path="a.png",
                diagnostics={"尺寸": "100×80"},
            )
        ]
    )

    assert rows[0].stage_title == "初步设计"
    assert rows[0].preview_path == "a.png"
    assert rows[0].diagnostics_text == "尺寸：100×80"


def test_artifact_rows_preserve_stable_input_order() -> None:
    artifacts = [
        ResultArtifact(stage="layout_revision", kind="a", title="第一轮", file_path="1.json"),
        ResultArtifact(stage="layout_revision", kind="b", title="第二轮", file_path="2.json"),
    ]

    assert [row.title for row in build_artifact_rows(artifacts)] == ["第一轮", "第二轮"]
