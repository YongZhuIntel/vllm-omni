# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a self-describing LingBot-VLA 2.0 model directory for vLLM-Omni."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vllm_omni.diffusion.models.lingbot_vla_v2 import LingbotVlaV2Config

EXAMPLE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = EXAMPLE_DIR / "deployment"
DEFAULT_QWEN3VL_PATH = DEPLOYMENT_DIR / "qwen3vl_base_config"
DEFAULT_ROBOT_CONFIG = DEPLOYMENT_DIR / "configs/robot_configs/robotwin.yaml"
DEFAULT_DATA_CONFIG = DEPLOYMENT_DIR / "configs/vla/robotwin/robotwin.yaml"
DEFAULT_NORM_STATS = DEPLOYMENT_DIR / "assets/norm_stats/robotwin.json"


def prepare(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    transformer_dir = output / "transformer"
    transformer_dir.mkdir(exist_ok=True)

    config = LingbotVlaV2Config.from_release_checkpoint(
        checkpoint.as_posix(),
        qwen3vl_path=Path(args.qwen3vl_path).resolve().as_posix(),
    )
    payload = asdict(config)
    payload.update(
        robot_config=Path(args.robot_config).resolve().as_posix(),
        data_config=Path(args.data_config).resolve().as_posix(),
        norm_stats=Path(args.norm_stats).resolve().as_posix() if args.norm_stats else None,
    )
    (output / "model_index.json").write_text(json.dumps({"_class_name": "LingbotVlaV2Pipeline"}, indent=2) + "\n")
    (transformer_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n")

    weights = sorted(checkpoint.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"no safetensors files found in {checkpoint}")
    for source in weights:
        target = output / source.name
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace {target}")
        target.symlink_to(source)

    print(f"prepared {output} with {len(weights)} weight link(s)")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen3vl-path", default=DEFAULT_QWEN3VL_PATH)
    parser.add_argument("--robot-config", default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--norm-stats", default=DEFAULT_NORM_STATS)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
