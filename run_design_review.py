from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from bridge_agents.design_review import DesignReviewAgent


def _load_settings(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _configured_output_dir(settings: Dict[str, Any]) -> str | None:
    paths = settings.get("paths") if isinstance(settings.get("paths"), dict) else {}
    return paths.get("output_dir") or settings.get("output_dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对已完成的桥梁设计成果进行整体评估或专业问答。")
    parser.add_argument("--config", default="config/settings.yaml", help="项目配置文件路径")
    parser.add_argument("--output-dir", default=None, help="设计输出目录；省略时读取配置 paths.output_dir")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--assess", action="store_true", help="生成一次整体设计评估")
    mode.add_argument("--question", help="针对当前设计成果提出专业问题")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = _load_settings(args.config)
    output_dir = args.output_dir or _configured_output_dir(settings)
    if not output_dir:
        raise SystemExit("缺少输出目录：请传入 --output-dir 或配置 paths.output_dir。")
    agent = DesignReviewAgent()
    if args.assess:
        result = agent.assess(output_dir=output_dir, config_path=args.config)
    else:
        result = agent.answer(args.question, output_dir=output_dir, config_path=args.config)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
