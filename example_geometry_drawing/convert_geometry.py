from __future__ import annotations

import json
from pathlib import Path

if __package__:
    from .main_girder_geometry.cad_export import (
        COMBINED_LAYOUT_SECTION,
        COMBINED_LAYOUT_VIEW,
        build_combined_layout,
        export_geometry_to_cad,
    )
    from .main_girder_geometry.elevation_generator import generate_elevation
    from .main_girder_geometry.geometry_generator import generate_key_sections
    from .main_girder_geometry.geometry_preview import write_geometry_preview
    from .main_girder_geometry.geometry_reader import load_design_parameters
    from .main_girder_geometry.plan_generator import generate_plan
else:
    from main_girder_geometry.cad_export import (
        COMBINED_LAYOUT_SECTION,
        COMBINED_LAYOUT_VIEW,
        build_combined_layout,
        export_geometry_to_cad,
    )
    from main_girder_geometry.elevation_generator import generate_elevation
    from main_girder_geometry.geometry_generator import generate_key_sections
    from main_girder_geometry.geometry_preview import write_geometry_preview
    from main_girder_geometry.geometry_reader import load_design_parameters
    from main_girder_geometry.plan_generator import generate_plan


# =============================================================================
# 直接在VSCode中修改以下任务参数，然后点击“运行Python文件”
# =============================================================================

TASK_ROOT = Path(__file__).resolve().parent

INPUT_FILE = TASK_ROOT / "单箱_无砟_T构_2×88.xlsx"# 或使用绝对路径，INPUT_FILE = Path(r"路径")

OUTPUT_DIR = TASK_ROOT/ "runs" / "geometry_results_2×88"

# 转换完成后是否同时为全部断面、半联立面和半联平面生成SVG预览图
CREATE_PREVIEW = True

# 是否生成可由CAD的SCRIPT命令执行的SCR轮廓脚本
CREATE_CAD = True

PREVIEW_FILENAMES = {
    ("cross_section", "END-DIAPHRAGM"): "01_end_diaphragm.svg",
    ("cross_section", "CONSTANT-SECTION"): "02_constant_section.svg",
    ("cross_section", "MID-DIAPHRAGM"): "03_mid_diaphragm.svg",
    ("cross_section", "PIER-SECTION"): "04_pier_section.svg",
    ("elevation", "HALF-GIRDER-ELEVATION"): "04_half_girder_elevation.svg",
    ("plan", "HALF-GIRDER-PLAN"): "05_half_girder_plan.svg",
}

CAD_VIEW_DIRECTORIES = {
    "cross_section": "cross_sections",
    "elevation": "elevation",
    "plan": "plan",
}


def main() -> None:
    parameters = load_design_parameters(INPUT_FILE)
    documents = {
        "cross_sections.json": generate_key_sections(parameters).to_dict(),
        "elevation.json": generate_elevation(parameters).to_dict(),
        "plan.json": generate_plan(parameters).to_dict(),
    }
    layout_document = build_combined_layout(list(documents.values()))
    output_documents = {
        **documents,
        "complete_girder_layout.json": layout_document,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for filename, geometry_data in output_documents.items():
        output_path = OUTPUT_DIR / filename
        output_path.write_text(
            json.dumps(geometry_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_paths.append(output_path)

    preview_paths: list[Path] = []
    if CREATE_PREVIEW:
        for geometry_data in documents.values():
            for view_name, view in geometry_data["views"].items():
                for section in view["sections"]:
                    preview_paths.append(
                        write_geometry_preview(
                            geometry_data,
                            OUTPUT_DIR
                            / PREVIEW_FILENAMES[(view_name, section["id"])],
                            section_id=section["id"],
                            view_name=view_name,
                        )
                    )

    cad_paths: list[Path] = []
    if CREATE_CAD:
        for geometry_data in documents.values():
            for view_name in geometry_data["views"]:
                cad_paths.extend(
                    export_geometry_to_cad(
                        geometry_data,
                        OUTPUT_DIR / "cad" / CAD_VIEW_DIRECTORIES[view_name],
                        view_name=view_name,
                    )
                )
        cad_paths.extend(
            export_geometry_to_cad(
                layout_document,
                OUTPUT_DIR / "cad",
                section_id=COMBINED_LAYOUT_SECTION,
                view_name=COMBINED_LAYOUT_VIEW,
            )
        )

    print("转换完成")
    print(f"输入文件：{INPUT_FILE}")
    for output_path in output_paths:
        print(f"图元JSON：{output_path}")
    for preview_path in preview_paths:
        print(f"预览图：{preview_path}")
    for cad_path in cad_paths:
        print(f"CAD文件：{cad_path}")
    for geometry_data in documents.values():
        for view_name, view in geometry_data["views"].items():
            for section in view["sections"]:
                entities = section["entities"]
                print(
                    f"{view_name}/{section['id']}："
                    f"{len(entities['lines'])}条直线，"
                    f"{len(entities['arcs'])}条圆弧，"
                    f"{len(entities['regions'])}个区域"
                )


if __name__ == "__main__":
    main()
