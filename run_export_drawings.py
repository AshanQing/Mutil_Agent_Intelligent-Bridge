from __future__ import annotations

import argparse
import json

from bridge_agents.agent import _load_initial_state_from_settings
from tools.reinforcement_drawing_tool import reinforcement_drawing_tool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从已有设计成果导出桥墩配筋图。")
    parser.add_argument("--config", default="config/settings.yaml", help="项目配置路径")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="允许输出带有醒目标记的未验算草稿预览",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = _load_initial_state_from_settings(args.config)
    result = reinforcement_drawing_tool(
        pier_group_result=state.get("pier_group_result") or {},
        dimension_design_result=state.get("dimension_design_result") or {},
        reinforcement_design_result=state.get("reinforcement_design_result") or {},
        capacity_check_result=state.get("capacity_check_result"),
        output_dir=state.get("output_dir") or "outputs",
        allow_unverified=args.allow_unverified,
        accepted_risks=state.get("accepted_risks") or [],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
