# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Why is the open-loop MAE 0.6? Is the model conditioned on its inputs at all?

``run_open_loop_eval.log`` reports mse 0.665 / mae 0.615 against RobotWin
ground truth. Two very different things produce that number:

* a **port bug** -- the observation never reaches the model, so the chunk is a
  plausible-looking but unconditioned sample from the action prior; or
* a **model/data mismatch** -- the plumbing is right and this checkpoint simply
  does not predict this dataset well.

They are distinguishable without any ground truth. A conditioned policy's output
must *move* when its inputs move, and its first action must land on the state it
was given (the GT chunks satisfy ``action[0] == state`` to 5 decimal places).
This runs the same chunk under perturbed inputs and reports both properties.

    PYTHONPATH=. python spikes/lingbot_vla_v2/open_loop_conditioning_probe.py \
        --model /tmp/lingbot-open-loop-eager \
        --dataset /llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase5_latency import build  # noqa: E402

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def make_noise(seed: int, sample_index: int) -> np.ndarray:
    """Byte-identical to ``open_loop_eval.make_noise`` so runs are comparable."""
    return np.random.default_rng(seed + sample_index).standard_normal((1, 50, 55)).astype(np.float32)


def predict(processor, model, obs, noise, device, dtype) -> np.ndarray:
    """One chunk, through the same preprocess/postprocess the pipeline uses."""
    features = processor.preprocess(obs).to(device=device, dtype=dtype)
    noise_t = torch.as_tensor(noise, device=device, dtype=dtype)
    with torch.no_grad():
        actions = model.sample_actions(**features.model_inputs(), noise=noise_t)
    out = processor.postprocess(actions, features)
    return np.asarray(next(iter(out.values())), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = build(Path(args.model), device, dtype, None, None)

    bundle = np.load(args.dataset)
    images, states, actions = bundle["images"], bundle["states"], bundle["actions"]
    prompts = bundle["prompts"]

    def observation(sample: int, *, state_from: int | None = None,
                    images_from: int | None = None, prompt: str | None = None) -> dict:
        return {
            "images": {
                key: images[sample if images_from is None else images_from, camera]
                for camera, key in enumerate(CAMERA_KEYS)
            },
            "state": states[sample if state_from is None else state_from].astype(np.float32),
            "prompt": str(prompts[sample]) if prompt is None else prompt,
        }

    noise = make_noise(args.seed, 0)
    base = predict(processor, model, observation(0), noise, device, dtype)
    gt = actions[0].astype(np.float32)

    print(f"\nsample 0, prompt={str(prompts[0])!r}")
    print(f"GT   action[0][:7] = {np.round(gt[0, :7], 3)}")
    print(f"state       [:7]   = {np.round(states[0, :7], 3)}")
    print(f"PRED action[0][:7] = {np.round(base[0, :7], 3)}")
    print(f"\n|GT action[0] - state|   = {np.abs(gt[0, :7] - states[0, :7]).mean():.4f}   "
          "<- a conditioned chunk starts at the current pose")
    print(f"|PRED action[0] - state| = {np.abs(base[0, :7] - states[0, :7]).mean():.4f}")

    variants = {
        "repeat (determinism)": observation(0),
        "state <- sample 5": observation(0, state_from=5),
        "images <- sample 5": observation(0, images_from=5),
        "prompt <- unrelated": observation(0, prompt="close the drawer and step back"),
    }
    print(f"\n{'perturbation':24s} {'mean |d|':>10s} {'max |d|':>10s}  (vs base prediction)")
    print("-" * 60)
    for label, obs in variants.items():
        other = predict(processor, model, obs, noise, device, dtype)
        delta = np.abs(other - base)
        print(f"{label:24s} {delta.mean():10.4f} {delta.max():10.4f}")

    # A different noise draw shows how much of the output is just the prior.
    other = predict(processor, model, observation(0), make_noise(args.seed, 99), device, dtype)
    delta = np.abs(other - base)
    print(f"{'noise <- different draw':24s} {delta.mean():10.4f} {delta.max():10.4f}")

    print(f"\nfor scale: GT chunk std = {gt.std():.4f}, "
          f"|GT - prediction| mean = {np.abs(gt - base).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
