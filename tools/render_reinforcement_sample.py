from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridge_agents.drawing.sample_rendering import render_reinforcement_sample_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用固定盖梁与墩柱 few-shot 样例生成三张配筋验证图。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "drawing_sample_validation",
        help="SVG、SCR、图元 IR、清单和诊断的输出目录。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = render_reinforcement_sample_set(
        output_dir=args.output_dir,
        repo_root=REPO_ROOT,
    )
    print(f"已生成 {len(result.sheet_ids)} 张验证图，未调用在线模型。")
    print(f"成果清单：{result.manifest_path}")
    print(f"诊断文件：{result.diagnostics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
