#!/usr/bin/env python3
"""Grade the vendored RobotWin processor against upstream FeatureTransform.

This uses the real RobotWin mapping, normalization statistics and Qwen3-VL
processor, but does not load the policy checkpoint. It compares all six kernel
inputs and the action conversion back to the robot's 14-dimensional command.

    python phase2_processor_parity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import bootstrap
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_INPUT_KEYS = (
    "images",
    "img_masks",
    "lang_tokens",
    "lang_masks",
    "state",
    "image_grid_thw",
)


def _upstream_configs(data_path: Path):
    payload = yaml.safe_load(data_path.read_text())
    data = payload["data"]
    data_config = SimpleNamespace(
        **{
            **data,
            "joints": [str(entry) for entry in data["joints"]],
            "norm_type": [str(entry) for entry in data["norm_type"]],
        }
    )
    model_config = SimpleNamespace(
        max_state_dim=55,
        max_action_dim=55,
        return_image_grid_thw=True,
        tokenizer_max_length=72,
        use_qwen3_chat_template=True,
        qwen3vl_use_vision_boundaries=True,
    )
    return data_config, model_config


def _observation(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    source_keys = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    frames = {key: rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8) for key in source_keys}
    state = rng.standard_normal(14).astype(np.float32)
    prompt = "pick up the object"
    upstream = {
        **{key: torch.from_numpy(frame).permute(2, 0, 1) for key, frame in frames.items()},
        "observation.state": torch.from_numpy(state),
        "task": prompt,
    }
    vendored = {"images": frames, "state": state, "prompt": prompt}
    return upstream, vendored


def _compare(name: str, upstream: torch.Tensor, vendored: torch.Tensor) -> bool:
    upstream = upstream.detach().cpu()
    vendored = vendored.detach().cpu()
    exact = upstream.dtype == vendored.dtype and torch.equal(upstream, vendored)
    if upstream.shape != vendored.shape:
        print(f"FAIL {name:<16} shape {tuple(upstream.shape)} != {tuple(vendored.shape)}")
        return False
    if upstream.is_floating_point():
        upstream_float = upstream.to(torch.float64)
        vendored_float = vendored.to(torch.float64)
        diff = (upstream_float - vendored_float).abs()
        max_abs = float(diff.max()) if diff.numel() else 0.0
        passed = torch.allclose(upstream_float, vendored_float, rtol=0.0, atol=1e-6)
        print(f"{'PASS' if passed else 'FAIL'} {name:<16} shape={tuple(upstream.shape)!s:<20} max|d|={max_abs:.3e}")
        return passed
    print(f"{'PASS' if exact else 'FAIL'} {name:<16} shape={tuple(upstream.shape)} exact={exact}")
    return exact


def run(args: argparse.Namespace) -> int:
    bootstrap.setup(verbose=False)

    from lingbotvla.data.vla_data.utils import FeatureTransform
    from transformers import AutoProcessor

    from vllm_omni.diffusion.models.lingbot_vla_v2 import (
        LingbotVlaV2Config,
        LingbotVlaV2Processor,
        RobotSpec,
    )

    root = Path(args.lingbot_root)
    robot_path = root / "configs/robot_configs/robotwin.yaml"
    data_path = root / "configs/vla/robotwin/robotwin.yaml"
    stats_path = root / "assets/norm_stats/robotwin.json"
    qwen_path = Path(args.qwen3vl_path)

    hf_processor = AutoProcessor.from_pretrained(qwen_path.as_posix(), padding_side="right")
    data_config, upstream_model_config = _upstream_configs(data_path)
    upstream_processor = FeatureTransform(
        robot_path.as_posix(),
        data_config,
        upstream_model_config,
        hf_processor,
        chunk_size=50,
        norm_stats_path=stats_path.as_posix(),
        use_future_image=False,
    )

    config = LingbotVlaV2Config(qwen3vl_path=qwen_path.as_posix())
    spec = RobotSpec.from_files(robot_path, data_path, stats_path)
    vendored_processor = LingbotVlaV2Processor(
        spec,
        config,
        tokenizer=hf_processor.tokenizer,
        image_processor=hf_processor.image_processor,
    )

    upstream_observation, vendored_observation = _observation(args.seed)
    upstream_features = upstream_processor.apply(upstream_observation, policy_eval=True)
    vendored_features = vendored_processor.preprocess(vendored_observation)

    passed = True
    vendored_inputs = vendored_features.model_inputs()
    for key in MODEL_INPUT_KEYS:
        passed &= _compare(key, upstream_features[key], vendored_inputs[key][0])

    actions = torch.linspace(-1.25, 1.25, 50 * 55, dtype=torch.float32).reshape(50, 55)
    upstream_output = upstream_processor.unapply(
        {
            "state": upstream_features["state"],
            "actions": actions,
            "state_joint_mask": upstream_features["state_joint_mask"],
            "action_joint_mask": upstream_features["action_joint_mask"],
        }
    )
    vendored_output = vendored_processor.postprocess(actions, vendored_features)
    passed &= _compare(
        "action",
        upstream_output["action"],
        torch.from_numpy(vendored_output["action"]),
    )

    print("\nPASS - processor agrees with upstream." if passed else "\nFAIL - processor parity mismatch.")
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-root", default=bootstrap.LINGBOT_SRC.as_posix())
    parser.add_argument("--qwen3vl-path", default=bootstrap.QWEN3VL_PATH.as_posix())
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
