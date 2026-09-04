# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate LingBot-VLA 2.0 action chunks against held-out RobotWin data.

The input is a portable NPZ bundle so the vLLM runtime does not need LeRobot or
a video decoder. See the example README for the schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.array([6, 13])
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass(frozen=True)
class OpenLoopBundle:
    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    prompts: np.ndarray
    episode_ids: np.ndarray
    frame_indices: np.ndarray
    valid_steps: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.states.shape[0])


def load_bundle(path: str | Path) -> OpenLoopBundle:
    with np.load(path, allow_pickle=False) as data:
        required = {"images", "states", "actions", "prompts", "episode_ids", "frame_indices"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"open-loop bundle lacks keys: {sorted(missing)}")
        values = {key: data[key] for key in required}
        values["valid_steps"] = (
            data["valid_steps"]
            if "valid_steps" in data.files
            else np.full(values["states"].shape[0], values["actions"].shape[1], dtype=np.int64)
        )
        bundle = OpenLoopBundle(**values)
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: OpenLoopBundle) -> None:
    count = bundle.num_samples
    if bundle.images.ndim != 5 or bundle.images.shape[1] != 3 or bundle.images.shape[-1] != 3:
        raise ValueError(f"images must have shape [N,3,H,W,3], got {bundle.images.shape}")
    if bundle.images.dtype != np.uint8:
        raise ValueError(f"images must be uint8, got {bundle.images.dtype}")
    if bundle.states.shape != (count, 14):
        raise ValueError(f"states must have shape [N,14], got {bundle.states.shape}")
    if bundle.actions.shape != (count, 50, 14):
        raise ValueError(f"actions must have shape [N,50,14], got {bundle.actions.shape}")
    for name, values in (
        ("prompts", bundle.prompts),
        ("episode_ids", bundle.episode_ids),
        ("frame_indices", bundle.frame_indices),
        ("valid_steps", bundle.valid_steps),
    ):
        if values.shape != (count,):
            raise ValueError(f"{name} must have shape [N], got {values.shape}")
    if not np.isfinite(bundle.states).all() or not np.isfinite(bundle.actions).all():
        raise ValueError("states and actions must contain only finite values")
    if np.any(bundle.valid_steps < 1) or np.any(bundle.valid_steps > bundle.actions.shape[1]):
        raise ValueError(f"valid_steps must be in [1,{bundle.actions.shape[1]}]")


def prediction_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    valid_steps: np.ndarray | None = None,
) -> dict[str, float]:
    if ground_truth.shape != prediction.shape or ground_truth.shape[-1] != 14:
        raise ValueError(
            f"ground truth and prediction must share [...,14], got {ground_truth.shape} and {prediction.shape}"
        )
    error = prediction.astype(np.float64) - ground_truth.astype(np.float64)
    if valid_steps is not None:
        if error.ndim != 3 or valid_steps.shape != (error.shape[0],):
            raise ValueError("valid_steps requires [N,H,14] actions and shape [N]")
        valid_mask = np.arange(error.shape[1])[None, :] < valid_steps[:, None]
        error = error[valid_mask]

    def summarize(values: np.ndarray) -> tuple[float, float]:
        return float(np.mean(values**2)), float(np.mean(np.abs(values)))

    mse, mae = summarize(error)
    mse_joint, mae_joint = summarize(error[..., JOINT_INDICES])
    mse_gripper, mae_gripper = summarize(error[..., GRIPPER_INDICES])
    return {
        "mse": mse,
        "mae": mae,
        "mse_joint": mse_joint,
        "mae_joint": mae_joint,
        "mse_gripper": mse_gripper,
        "mae_gripper": mae_gripper,
    }


def aggregate_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    episode_ids: np.ndarray,
    valid_steps: np.ndarray | None = None,
) -> dict[str, Any]:
    if valid_steps is None:
        valid_steps = np.full(ground_truth.shape[0], ground_truth.shape[1], dtype=np.int64)
    episodes = []
    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        episodes.append(
            {
                "episode_id": int(episode_id),
                "num_chunks": int(mask.sum()),
                "num_valid_steps": int(valid_steps[mask].sum()),
                **prediction_metrics(ground_truth[mask], prediction[mask], valid_steps[mask]),
            }
        )
    macro_keys = tuple(prediction_metrics(ground_truth[:1], prediction[:1], valid_steps[:1]))
    return {
        "num_samples": int(ground_truth.shape[0]),
        "num_episodes": len(episodes),
        "num_valid_steps": int(valid_steps.sum()),
        "micro": prediction_metrics(ground_truth, prediction, valid_steps),
        "macro": {key: float(np.mean([episode[key] for episode in episodes])) for key in macro_keys},
        "episodes": episodes,
    }


def make_noise(seed: int, sample_index: int) -> tuple[np.ndarray, str]:
    noise = np.random.default_rng(seed + sample_index).standard_normal((1, 50, 55)).astype(np.float32)
    return noise, hashlib.sha256(noise.tobytes()).hexdigest()


def write_episode_plots(
    output_dir: Path,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    episode_ids: np.ndarray,
    valid_steps: np.ndarray,
) -> None:
    """Write one 14-axis GT/prediction plot per episode."""
    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plots requires matplotlib") from exc

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        gt_rows = [chunk[:steps] for chunk, steps in zip(ground_truth[mask], valid_steps[mask], strict=True)]
        pred_rows = [chunk[:steps] for chunk, steps in zip(prediction[mask], valid_steps[mask], strict=True)]
        gt_episode = np.concatenate(gt_rows)
        pred_episode = np.concatenate(pred_rows)
        figure, axes = plt.subplots(14, 1, figsize=(10, 32), sharex=True)
        for action_index, axis in enumerate(axes):
            axis.plot(gt_episode[:, action_index], label="ground truth")
            axis.plot(pred_episode[:, action_index], label="prediction")
            axis.set_ylabel(f"a{action_index}")
            if action_index == 0:
                axis.legend()
        axes[-1].set_xlabel("valid action timestep")
        figure.suptitle(f"RobotWin episode {int(episode_id)}")
        figure.tight_layout()
        figure.savefig(plots_dir / f"episode_{int(episode_id)}.png")
        plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_bundle(args.dataset)
    selected = np.arange(bundle.num_samples)
    if args.episodes:
        requested = np.array(args.episodes, dtype=bundle.episode_ids.dtype)
        selected = selected[np.isin(bundle.episode_ids, requested)]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    if selected.size == 0:
        raise ValueError("no samples selected for evaluation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        summary = {
            "dry_run": True,
            "dataset": str(Path(args.dataset).resolve()),
            "num_samples": int(selected.size),
            "episodes": sorted(int(value) for value in np.unique(bundle.episode_ids[selected])),
        }
        (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return summary

    from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine = OmniDiffusion(model=args.model, dtype=args.dtype)
    predictions = []
    noise_seeds = []
    noise_hashes = []
    try:
        for ordinal, index in enumerate(selected):
            prompt = str(bundle.prompts[index])
            observation = {
                "images": {key: bundle.images[index, camera] for camera, key in enumerate(CAMERA_KEYS)},
                "state": bundle.states[index].astype(np.float32, copy=False),
                "prompt": prompt,
            }
            noise, noise_hash = make_noise(args.seed, int(index))
            params = OmniDiffusionSamplingParams(
                extra_args={"robot_obs": observation, "noise": noise},
                save_output=False,
            )
            output = engine.generate(prompt, params)[0]
            prediction = np.asarray(output.multimodal_output["actions"], dtype=np.float32)
            if prediction.shape != (50, 14):
                raise ValueError(f"model returned {prediction.shape}, expected (50, 14)")
            predictions.append(prediction)
            noise_seeds.append(args.seed + int(index))
            noise_hashes.append(noise_hash)
            print(
                f"sample={ordinal + 1}/{selected.size} episode={bundle.episode_ids[index]} "
                f"frame={bundle.frame_indices[index]}"
            )
    finally:
        engine.close()

    prediction_array = np.stack(predictions)
    ground_truth = bundle.actions[selected].astype(np.float32, copy=False)
    summary = {
        "dry_run": False,
        "dataset": str(Path(args.dataset).resolve()),
        "model": str(Path(args.model).resolve()),
        "dtype": args.dtype,
        "seed": args.seed,
        "selected_indices": selected.tolist(),
        **aggregate_metrics(
            ground_truth,
            prediction_array,
            bundle.episode_ids[selected],
            bundle.valid_steps[selected],
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "predictions.npz",
        predictions=prediction_array,
        ground_truth=ground_truth,
        states=bundle.states[selected],
        episode_ids=bundle.episode_ids[selected],
        frame_indices=bundle.frame_indices[selected],
        valid_steps=bundle.valid_steps[selected],
        noise_seeds=np.asarray(noise_seeds, dtype=np.int64),
        noise_hashes=np.asarray(noise_hashes),
    )
    if args.plots:
        write_episode_plots(
            output_dir,
            ground_truth,
            prediction_array,
            bundle.episode_ids[selected],
            bundle.valid_steps[selected],
        )
    print(json.dumps(summary["micro"], indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="prepared LingBot model directory")
    parser.add_argument("--dataset", required=True, help="open-loop NPZ bundle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--plots", action="store_true", help="write per-episode GT/prediction PNGs")
    parser.add_argument("--dry-run", action="store_true", help="validate/select data without loading the model")
    args = parser.parse_args()
    if not args.dry_run and not args.model:
        parser.error("--model is required unless --dry-run is used")
    return args


if __name__ == "__main__":
    raise SystemExit(0 if evaluate(parse_args()) else 1)
