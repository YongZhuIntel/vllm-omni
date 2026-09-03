# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one RobotWin action chunk through the vLLM-Omni diffusion engine."""

from __future__ import annotations

import argparse
import time

import numpy as np

from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    robot_obs = {
        "images": {
            "observation.images.cam_high": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            "observation.images.cam_left_wrist": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            "observation.images.cam_right_wrist": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        },
        "state": np.zeros(14, dtype=np.float32),
        "prompt": args.prompt,
    }
    engine = OmniDiffusion(model=args.model, dtype=args.dtype)
    params = OmniDiffusionSamplingParams(
        seed=args.seed,
        extra_args={"robot_obs": robot_obs},
        save_output=False,
    )
    started = time.perf_counter()
    output = engine.generate(args.prompt, params)[0]
    elapsed = time.perf_counter() - started
    actions = output.multimodal_output["actions"]
    print(f"type={output.final_output_type} shape={actions.shape} dtype={actions.dtype} elapsed={elapsed:.3f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Prepared model directory")
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
