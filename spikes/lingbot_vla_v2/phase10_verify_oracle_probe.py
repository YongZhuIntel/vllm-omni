#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 10 gate 3 — what can FLASH's verify accept, at best, on this model?

The speculative round replaces ten Euler steps with **one** near-terminal
`predict_velocity` call per verify timestep, and accepts the draft wherever

    x0_hat_k = x_t_k - t_k * v(x_t_k, t_k),   x_t_k = t_k*noise + (1-t_k)*x0_draft

lands within `tau` of the draft. Two error sources are folded into that: the
draft's own error, and the fact that a **single** step at `t=0.1` is not the same
answer as ten steps from `t=1`. This probe isolates the second by feeding the
verify a *perfect* draft -- the teacher's own 10-step output -- and measuring
what comes back.

That gives the ceiling. If a perfect draft is not accepted, no trained draft can
be, and Phase 10 stops here for the cost of one afternoon rather than after the
cache build and the training run.

It then sweeps a **synthetic draft error**: `x0_draft = x0_teacher + eps*N(0,1)`
on the real action dims. Because the accept distance is an RMS over dims, `eps`
is in the same units as `tau`, so the output table reads directly as
"a draft with per-dim RMS error `eps` gets `n` steps accepted at threshold
`tau`" -- which is the training target Phase 10.1 has to hit, and the only way
to know it before training rather than after.

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_verify_oracle_probe.py \
        --model /tmp/lingbot-open-loop \
        --dataset ~/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz

The strided 6-chunk bundle is fine here -- unlike the staleness probe, this one
needs no consecutive frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from open_loop_conditioning_probe import CAMERA_KEYS, make_noise  # noqa: E402
from phase10_kv_staleness_probe import fill_prefix  # noqa: E402
from phase10_spec_common import (  # noqa: E402
    build_x_t,
    gripper_dims,
    pose_dims,
    radius_prefix_acceptance,
    stitch_prefix,
    x0_from_velocity,
)
from phase5_latency import build  # noqa: E402


def verify_once(
    model: Any,
    prefix: tuple[torch.Tensor, torch.Tensor, list],
    state: torch.Tensor,
    noise: torch.Tensor,
    x0_draft: torch.Tensor,
    t_list: list[float],
) -> torch.Tensor:
    """K near-terminal steps -> ``x0_hat`` stacked as ``[B,K,H,D]``.

    Run **sequentially**, one `predict_velocity` per timestep, rather than as one
    `B*K` batch. FLASH batches (`expand_past_key_values:311`), but that
    materialises the 42.2 MB prefix KV cache K times on a device where F2 showed
    the loop is already memory-bound on weight streaming. At K=2 the sequential
    cost is 2x21.3 ms and needs no change to the attention path; batching is a
    Phase 10.2 optimisation to measure, not an assumption to build on.
    """
    pad_masks, position_ids, past_key_values = prefix
    bsize = int(state.shape[0])
    hats = []
    for t in t_list:
        x_t = build_x_t(noise, x0_draft, t)
        timestep = torch.full((bsize,), float(t), device=x_t.device, dtype=x_t.dtype)
        v_t = model.predict_velocity(
            state=state,
            prefix_pad_masks=pad_masks,
            prefix_position_ids=position_ids,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )
        hats.append(x0_from_velocity(x_t, t, v_t))
    return torch.stack(hats, dim=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=10, help="teacher's Euler steps")
    parser.add_argument(
        "--t-list",
        type=float,
        nargs="*",
        default=[0.10, 0.05],
        help="verify timesteps; FLASH's default is (0.10, 0.05)",
    )
    parser.add_argument("--taus", type=float, nargs="*", default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
    parser.add_argument(
        "--draft-errors",
        type=float,
        nargs="*",
        default=[0.0, 0.02, 0.05, 0.10, 0.20],
        help="synthetic per-dim RMS draft error, in the same units as --taus",
    )
    parser.add_argument("--max-exec-steps", type=int, default=12, help="steps executed before replanning")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    with np.load(Path(args.dataset).expanduser(), allow_pickle=False) as data:
        bundle = {key: data[key] for key in data.files}
    processor, model = build(Path(args.model), device, dtype, args.num_steps)

    dims = pose_dims(processor)
    grippers = gripper_dims(processor)
    print(f"[spec] accept radius over {len(dims)} pose dims {dims}")
    print(f"[spec] gripper dims {grippers}")
    print(f"[spec] verify timesteps {args.t_list}, eval horizon {args.max_exec_steps}")

    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for row in range(len(bundle["states"])):
            obs = {
                "images": {key: bundle["images"][row, camera] for camera, key in enumerate(CAMERA_KEYS)},
                "state": bundle["states"][row].astype(np.float32),
                "prompt": str(bundle["prompts"][row]),
            }
            features = processor.preprocess(obs).to(device=device, dtype=dtype)
            inputs = features.model_inputs()
            prefix = fill_prefix(model, inputs)
            noise = torch.as_tensor(make_noise(args.seed, row), device=device, dtype=dtype)

            # The reference the draft is supposed to reproduce: the full loop.
            x0_teacher = model.denoise_actions(
                state=inputs["state"],
                prefix_pad_masks=prefix[0],
                prefix_position_ids=prefix[1],
                past_key_values=prefix[2],
                noise=noise,
                num_steps=args.num_steps,
            )

            real_dims = torch.as_tensor(sorted(dims + grippers), device=device, dtype=torch.long)
            for eps in args.draft_errors:
                x0_draft = x0_teacher.clone()
                if eps > 0.0:
                    jitter = torch.as_tensor(
                        rng.standard_normal((1, x0_draft.shape[1], len(real_dims))) * eps,
                        device=device,
                        dtype=x0_draft.dtype,
                    )
                    x0_draft[:, :, real_dims] += jitter

                x0_hat = verify_once(model, prefix, inputs["state"], noise, x0_draft, args.t_list)
                x0_tail = x0_hat.mean(dim=1)

                for tau in args.taus:
                    accepted, dist = radius_prefix_acceptance(
                        x0_draft, x0_hat, tau=tau, dims=dims, eval_h=args.max_exec_steps
                    )
                    stitched = stitch_prefix(x0_draft, x0_tail, accepted)
                    records.append(
                        {
                            "row": row,
                            "draft_error": eps,
                            "tau": tau,
                            "accepted": int(accepted.item()),
                            "dist_mean": float(dist.mean().item()),
                            "dist_p90": float(dist.flatten().quantile(0.9).item()),
                            # How far the executed chunk lands from the teacher's,
                            # in model space, over the steps that get executed.
                            "mae_vs_teacher": float(
                                (stitched - x0_teacher)[:, : args.max_exec_steps, real_dims].abs().mean().item()
                            ),
                        }
                    )

    eval_h = args.max_exec_steps
    oracle = [row for row in records if row["draft_error"] == 0.0]
    if oracle:
        print(
            f"\n[oracle] perfect draft, distance between the 1-step estimate and the {args.num_steps}-step answer:"
            f"  mean {np.mean([r['dist_mean'] for r in oracle]):.4f}"
            f"  p90 {np.mean([r['dist_p90'] for r in oracle]):.4f}"
        )
        print("[oracle] any tau below that p90 cannot accept a full prefix even from a perfect draft.")

    print(f"\naccepted steps out of {eval_h}, mean over {len(bundle['states'])} samples")
    header = "  eps  " + "".join(f"{f'tau={tau:g}':>12s}" for tau in args.taus)
    print(header)
    print("-" * len(header))
    for eps in args.draft_errors:
        cells = []
        for tau in args.taus:
            rows = [r["accepted"] for r in records if r["draft_error"] == eps and r["tau"] == tau]
            cells.append(f"{np.mean(rows):12.2f}" if rows else f"{'-':>12s}")
        print(f"{eps:6.3f}" + "".join(cells))

    print(f"\nMAE vs teacher over the {eval_h} executed steps (model space)")
    print(header)
    print("-" * len(header))
    for eps in args.draft_errors:
        cells = []
        for tau in args.taus:
            rows = [r["mae_vs_teacher"] for r in records if r["draft_error"] == eps and r["tau"] == tau]
            cells.append(f"{np.mean(rows):12.5f}" if rows else f"{'-':>12s}")
        print(f"{eps:6.3f}" + "".join(cells))

    print(
        "\nRead: the eps=0 row is the ceiling. Pick the tau where accepted steps stay high\n"
        "while MAE-vs-teacher stays low, then read off the largest eps that still clears it --\n"
        "that is the per-dim RMS accuracy the draft head has to reach in Phase 10.1."
    )

    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"args": vars(args), "records": records}, indent=2) + "\n")
        print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
