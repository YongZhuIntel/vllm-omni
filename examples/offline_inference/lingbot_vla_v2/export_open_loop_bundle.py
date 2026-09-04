# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export held-out RobotWin episodes to the portable open-loop NPZ format.

Run this script in an environment containing the upstream LingBot-VLA source,
LeRobot, and its video decoder. It reuses upstream ``FeatureTransform`` for
ground-truth action mapping and unnormalization, but does not load model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

try:
    import torch
except ImportError:  # Allows --help outside the upstream training environment.
    torch = None

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def to_numpy(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def image_to_uint8_hwc(value: Any) -> np.ndarray:
    """Convert LeRobot CHW/HWC images to the policy's HWC uint8 contract."""
    image = to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"expected a 3D image, got {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] not in (1, 3):
        raise ValueError(f"expected channel-first or channel-last image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating) and float(image.max(initial=0)) <= 2.0:
        image = np.rint(image * 255.0)
    return np.clip(image, 0, 255).astype(np.uint8)


def valid_action_steps(sample: dict[str, Any], horizon: int, episode_remaining: int) -> int:
    padding = sample.get("action_is_pad")
    if padding is None:
        return min(horizon, episode_remaining)
    padding = to_numpy(padding).astype(bool).reshape(-1)[:horizon]
    return int(np.count_nonzero(~padding))


def transform_ground_truth(
    sample: dict[str, Any], feature_transform: Any, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map through upstream normalization and back to raw RobotWin units."""
    transformed = feature_transform.apply(dict(sample))
    restored = feature_transform.unapply(transformed)
    state = to_numpy(restored[STATE_KEY]).astype(np.float32).reshape(-1)
    actions = to_numpy(restored[ACTION_KEY]).astype(np.float32)[:horizon]
    if state.shape != (14,):
        raise ValueError(f"RobotWin state must be [14], got {state.shape}")
    if actions.shape != (horizon, 14):
        raise ValueError(f"RobotWin actions must be [{horizon},14], got {actions.shape}")
    return state, actions


def _data_namespace(data_config: dict[str, Any]) -> SimpleNamespace:
    values = dict(data_config)
    values["joints"] = [repr(entry) for entry in values["joints"]]
    values["norm_type"] = [repr(entry) for entry in values["norm_type"]]
    return SimpleNamespace(**values)


def _episode_bounds(dataset: Any, api_version: str, episode_id: int) -> tuple[int, int]:
    if api_version == "v2":
        return (
            int(dataset.episode_data_index["from"][episode_id]),
            int(dataset.episode_data_index["to"][episode_id]),
        )
    episode = dataset.meta.episodes[episode_id]
    return int(episode["dataset_from_index"]), int(episode["dataset_to_index"])


def export(args: argparse.Namespace) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("export requires PyTorch and the upstream LingBot training environment")
    if args.horizon != 50:
        raise ValueError("the vLLM-Omni open-loop bundle schema currently requires horizon=50")
    lingbot_root = Path(args.lingbot_root).expanduser().resolve()
    sys.path.insert(0, str(lingbot_root))

    try:
        from lingbotvla.data.vla_data.base_dataset import (
            LEROBOT_DATASET_API,
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
        from lingbotvla.data.vla_data.utils import FeatureTransform
    except ImportError as exc:
        raise RuntimeError(
            "export requires the upstream LingBot environment with lerobot==0.4.2 and its video dependencies"
        ) from exc

    robot_config = lingbot_root / args.robot_config
    data_config_path = lingbot_root / args.data_config
    norm_stats = lingbot_root / args.norm_stats
    data_payload = yaml.safe_load(data_config_path.read_text())["data"]
    data_config = _data_namespace(data_payload)
    model_config = SimpleNamespace(max_state_dim=55, max_action_dim=55)
    feature_transform = FeatureTransform(
        robot_config.as_posix(),
        data_config,
        model_config,
        processor=None,
        disabled_image_features=True,
        chunk_size=args.horizon,
        return_item_befor_padding=True,
        norm_stats_path=norm_stats.as_posix(),
        use_future_image=False,
    )

    data_path = Path(args.data_path).expanduser()
    if data_path.exists():
        repo_id, root = data_path.name, data_path.resolve()
    else:
        repo_id, root = args.data_path, None
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = {
        action_key: [step / metadata.fps for step in range(args.horizon)]
        for action_key in feature_transform.org_features["actions"]
    }
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)

    images, states, actions, prompts = [], [], [], []
    episode_ids, frame_indices, valid_steps = [], [], []
    for episode_id in args.episodes:
        start, end = _episode_bounds(dataset, LEROBOT_DATASET_API, episode_id)
        for chunk_index, data_index in enumerate(range(start, end, args.horizon)):
            if args.max_chunks_per_episode is not None and chunk_index >= args.max_chunks_per_episode:
                break
            sample = dataset[data_index]
            missing_cameras = set(CAMERA_KEYS) - set(sample)
            if missing_cameras:
                raise KeyError(f"sample {data_index} lacks cameras: {sorted(missing_cameras)}")
            state, action = transform_ground_truth(sample, feature_transform, args.horizon)
            images.append(np.stack([image_to_uint8_hwc(sample[key]) for key in CAMERA_KEYS]))
            states.append(state)
            actions.append(action)
            prompts.append(str(sample["task"]))
            episode_ids.append(episode_id)
            frame_indices.append(int(to_numpy(sample.get("frame_index", data_index))))
            valid_steps.append(valid_action_steps(sample, args.horizon, end - data_index))

    if not states:
        raise ValueError("no samples exported")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "images": np.stack(images),
        "states": np.stack(states).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "prompts": np.asarray(prompts, dtype=np.str_),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "valid_steps": np.asarray(valid_steps, dtype=np.int64),
    }
    np.savez_compressed(output, **arrays)
    manifest = {
        "schema_version": 1,
        "output": output.as_posix(),
        "data_path": args.data_path,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "num_samples": len(states),
        "lerobot_api": LEROBOT_DATASET_API,
        "camera_keys": list(CAMERA_KEYS),
        "state_key": STATE_KEY,
        "action_key": ACTION_KEY,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-root", required=True, help="upstream lingbot-vla-v2 source checkout")
    parser.add_argument("--data-path", required=True, help="local LeRobot root or Hugging Face repo ID")
    parser.add_argument("--output", required=True, help="output .npz path")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--max-chunks-per-episode", type=int, default=10)
    parser.add_argument("--robot-config", default="configs/robot_configs/robotwin.yaml")
    parser.add_argument("--data-config", default="configs/vla/robotwin/robotwin.yaml")
    parser.add_argument("--norm-stats", default="assets/norm_stats/robotwin.json")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(0 if export(parse_args()) else 1)
