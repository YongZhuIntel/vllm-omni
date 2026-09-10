# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Is the open-loop chunk under-integrated, or just wrong?

``open_loop_conditioning_probe.py`` established that the model *is* conditioned
on its inputs, and that the plumbing (normalization, slot packing, chat template,
image size, bundle export) all check out against upstream. What is left is the
shape of the failure: the predicted chunk jitters with a temporal std of ~0.25
across every sample, where the ground truth holds still at ~0.06. A real robot
trajectory is smooth; residual noise is not.

That is what a flow-matching sample looks like when the Euler integration has not
converged. This sweeps ``num_steps`` and reports, per setting:

* **MAE vs ground truth** -- does more integration get closer to the answer?
* **jerk** (mean |second difference| along time) -- a scale-free smoothness
  measure. Residual noise shows up here long before it shows up in MAE.

If jerk falls towards the ground truth's as steps rise, the chunk was
under-integrated and ``num_steps=10`` is simply too few. If jerk is flat in
``num_steps``, the velocity field itself is wrong and more steps cannot help.

    PYTHONPATH=. python spikes/lingbot_vla_v2/open_loop_steps_sweep.py \
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

from open_loop_conditioning_probe import CAMERA_KEYS, make_noise  # noqa: E402
from phase5_latency import build  # noqa: E402


def jerk(chunk: np.ndarray) -> float:
    """Mean |second difference| along time -- 0 for a straight line, high for noise."""
    return float(np.abs(np.diff(chunk, n=2, axis=-2)).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, nargs="*", default=[1, 2, 5, 10, 20, 50, 100])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = build(Path(args.model), device, dtype, None, None)

    bundle = np.load(args.dataset)
    images, states, actions, prompts = (
        bundle["images"], bundle["states"], bundle["actions"], bundle["prompts"])
    gt = actions.astype(np.float32)

    print(f"\nground truth: MAE 0 by definition, jerk {jerk(gt[..., :7]):.4f}")
    hold = np.repeat(states[:, None, :], 50, axis=1).astype(np.float32)
    print(f"hold-state baseline: MAE {np.abs(hold[..., :7] - gt[..., :7]).mean():.4f}, "
          f"jerk {jerk(hold[..., :7]):.4f}")

    print(f"\n{'steps':>6s} {'MAE(0-6)':>10s} {'MAE(all)':>10s} {'jerk(0-6)':>11s}")
    print("-" * 42)
    for num_steps in args.steps:
        preds = []
        for index in range(len(states)):
            obs = {
                "images": {k: images[index, c] for c, k in enumerate(CAMERA_KEYS)},
                "state": states[index].astype(np.float32),
                "prompt": str(prompts[index]),
            }
            features = processor.preprocess(obs).to(device=device, dtype=dtype)
            noise = torch.as_tensor(make_noise(args.seed, index), device=device, dtype=dtype)
            with torch.no_grad():
                out = model.sample_actions(
                    **features.model_inputs(), noise=noise, num_steps=num_steps)
            preds.append(np.asarray(
                next(iter(processor.postprocess(out, features).values())), dtype=np.float32))
        pred = np.stack(preds)
        print(f"{num_steps:6d} {np.abs(pred[..., :7] - gt[..., :7]).mean():10.4f} "
              f"{np.abs(pred - gt).mean():10.4f} {jerk(pred[..., :7]):11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
