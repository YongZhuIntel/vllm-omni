#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 10 — is `phase10_spec_common` really FLASH's algorithm?

Two questions, and they pull in opposite directions:

1. **Where the port claims to be faithful, is it?** `radius_prefix_acceptance`,
   `stitch_prefix` and `truncate_on_gripper_switch` are supposed to be
   `spec_pi0_pytorch.py`'s `_compute_radius_prefix_acceptance:126`,
   `_stitch_radius_prefix_output:160` and
   `_truncate_accepted_prefix_on_gripper_switch:50` with the hardcoded LIBERO
   dimensions lifted out. Configured back to LIBERO's layout (7-DoF, pose 0..5,
   gripper 6) they must agree **exactly**, not approximately. The FLASH
   originals are inlined below verbatim so this file is self-contained and does
   not need the openpi environment importable.

2. **Where it claims to diverge, does the divergence matter?** FLASH hardcodes
   action index 6 as the gripper. This prints what the real `RobotSpec` says.

Also checks the flow-matching identity `x0_hat = x_t - t*v_t` that the entire
verify rests on, in fp32 and in the deployed fp16 -- the fp16 figure is a floor
on how small `tau` can usefully be, which is not obvious and is cheap to know
before the tau sweep in `phase10_verify_oracle_probe.py`.

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_port_exactness.py

No model weights, no device: this is pure algebra and one YAML parse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase10_spec_common import (  # noqa: E402
    action_group_slots,
    build_x_t,
    gripper_dims,
    pose_dims,
    radius_prefix_acceptance,
    stitch_prefix,
    truncate_on_gripper_switch,
    x0_from_velocity,
)

DEPLOY = REPO_ROOT / "examples/offline_inference/lingbot_vla_v2/deployment"


# -- FLASH originals, verbatim from spec_pi0_pytorch.py --------------------


def flash_accept(*, x0_draft, x0_hat, tau_radius, dist_dims, eval_h):
    b, h, d = x0_draft.shape
    eval_h2 = int(min(h, max(1, int(eval_h))))
    eval_d = int(min(d, int(dist_dims)))
    if d >= 7:
        eval_d = int(min(eval_d, 6))
    diff = x0_hat[:, :, :eval_h2, :eval_d] - x0_draft[:, None, :eval_h2, :eval_d]
    norm_d = torch.tensor(float(eval_d), dtype=torch.float32).sqrt().clamp_min(1.0)
    dist = torch.linalg.vector_norm(diff, ord=2, dim=3).to(dtype=torch.float32) / norm_d
    prefix_mask = (dist <= float(tau_radius)).to(dtype=torch.int64).cumprod(dim=2)
    return prefix_mask.sum(dim=2).min(dim=1).values.to(torch.int64), dist


def flash_stitch(*, x0_draft, x0_tail, accepted_prefix_len):
    idx = torch.arange(int(x0_draft.shape[1]), dtype=torch.int64)[None, :]
    return torch.where((idx < accepted_prefix_len[:, None])[:, :, None], x0_draft, x0_tail)


def flash_truncate(*, x0_out, accepted_prefix_len, gripper_prev, gripper_switch_threshold):
    _, h, _ = x0_out.shape
    prev = torch.cat([gripper_prev.to(torch.float32)[:, None], x0_out[:, :-1, 6].to(torch.float32)], dim=1)
    curr = x0_out[:, :, 6].to(torch.float32)
    t = float(gripper_switch_threshold)
    switch = ((prev < t) & (curr >= t)) | ((prev >= t) & (curr < t))
    switch = switch & (torch.arange(h, dtype=torch.int64)[None, :] < accepted_prefix_len[:, None])
    cut = switch.any(dim=1)
    return torch.where(cut, switch.to(torch.int64).argmax(dim=1), accepted_prefix_len).to(torch.int64), cut


# -- checks ----------------------------------------------------------------


def check_equivalence(trials: int, seed: int) -> int:
    """Random-data equivalence against the originals, in LIBERO's configuration."""
    torch.manual_seed(seed)
    batch, k, horizon, dim = 4, 2, 50, 7
    failures = 0
    for trial in range(trials):
        x0_draft = torch.randn(batch, horizon, dim)
        x0_hat = x0_draft[:, None] + 0.15 * torch.randn(batch, k, horizon, dim)
        tau = float(torch.empty(1).uniform_(0.05, 0.4))
        eval_h = int(torch.randint(1, horizon + 1, (1,)))

        want_len, want_dist = flash_accept(
            x0_draft=x0_draft, x0_hat=x0_hat, tau_radius=tau, dist_dims=7, eval_h=eval_h
        )
        got_len, got_dist = radius_prefix_acceptance(
            x0_draft, x0_hat, tau=tau, dims=list(range(6)), eval_h=eval_h
        )
        if not (torch.equal(want_len, got_len) and torch.equal(want_dist, got_dist)):
            failures += 1
            print(f"  accept mismatch at trial {trial}")

        tail = x0_hat.mean(dim=1)
        want_stitch = flash_stitch(x0_draft=x0_draft, x0_tail=tail, accepted_prefix_len=want_len)
        got_stitch = stitch_prefix(x0_draft, tail, got_len)
        if not torch.equal(want_stitch, got_stitch):
            failures += 1
            print(f"  stitch mismatch at trial {trial}")

        previous = torch.randn(batch)
        want_cut = flash_truncate(
            x0_out=want_stitch,
            accepted_prefix_len=want_len,
            gripper_prev=previous,
            gripper_switch_threshold=0.0,
        )
        got_cut = truncate_on_gripper_switch(
            got_stitch, got_len, gripper_prev=previous[:, None], dims=[6], threshold=0.0
        )
        if not all(torch.equal(a, b) for a, b in zip(want_cut, got_cut, strict=True)):
            failures += 1
            print(f"  truncate mismatch at trial {trial}")

    verdict = "ALL EQUAL" if failures == 0 else f"{failures} MISMATCH"
    print(f"[port] {trials} trials x 3 functions vs the FLASH originals: {verdict}")
    return failures


def check_identity() -> None:
    """``x0_hat = x_t - t*v_t`` must return x0 exactly for the linear path."""
    for dtype, label in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        worst = 0.0
        for t in (0.30, 0.10, 0.05, 0.01):
            x0 = torch.randn(2, 50, 55).to(dtype)
            noise = torch.randn(2, 50, 55).to(dtype)
            velocity = (noise - x0).to(dtype)
            recovered = x0_from_velocity(build_x_t(noise, x0, t), t, velocity)
            worst = max(worst, float((recovered.float() - x0.float()).abs().max()))
        print(f"[identity] max|x0_hat - x0| over t in (0.3,0.1,0.05,0.01), {label}: {worst:.3e}")
    print("[identity] the fp16 figure is in the same units as tau -- it is a floor on a useful tau.")


def check_dims() -> None:
    """What the real RobotSpec says about where the grippers actually are."""
    from vllm_omni.diffusion.models.lingbot_vla_v2.processor import RobotSpec

    spec = RobotSpec.from_files(
        DEPLOY / "configs/robot_configs/robotwin.yaml",
        DEPLOY / "configs/vla/robotwin/robotwin.yaml",
        DEPLOY / "assets/norm_stats/robotwin.json",
    )
    processor = SimpleNamespace(spec=spec)

    print("\n[dims] joint slots in the model's action space, in slot order:")
    offset = 0
    for joint in spec.joints:
        carries_action = "yes" if joint.name in spec.action_slices else "no"
        print(f"  [{offset:2d}:{offset + joint.max_dim:2d}) {joint.name:<20s} max_dim={joint.max_dim:<3d} action={carries_action}")
        offset += joint.max_dim
    print(f"  {offset} packed slots, then zero-padded to max_action_dim (processor.py:591)")

    pose, grippers = pose_dims(processor), gripper_dims(processor)
    print(f"[dims] action_group_slots: {action_group_slots(processor)}")
    print(f"[dims] pose_dims    ({len(pose)}): {pose}")
    print(f"[dims] gripper_dims ({len(grippers)}): {grippers}")
    if 6 in grippers:
        print("[dims] FLASH's hardcoded index 6 would have been right here.")
    else:
        print("[dims] FLASH's hardcoded index 6 is an ARM joint here, not a gripper -- it would have")
        print(f"[dims] watched the wrong dimension silently. The grippers are at {grippers}.")

    # A switch on either arm's gripper must cut the accepted prefix.
    chunk = torch.zeros(1, 6, 55)
    chunk[:, :, grippers] = -1.0
    chunk[:, 3:, grippers[-1]] = 1.0
    previous = torch.full((1, len(grippers)), -1.0)
    length, cut = truncate_on_gripper_switch(
        chunk, torch.tensor([6]), gripper_prev=previous, dims=grippers, threshold=0.0
    )
    ok = int(length[0]) == 3 and bool(cut[0])
    print(f"[dims] flip on dim {grippers[-1]} at step 3 -> len {int(length[0])} (expect 3), "
          f"cut {bool(cut[0])} (expect True): {'OK' if ok else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    failures = check_equivalence(args.trials, args.seed)
    check_identity()
    check_dims()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
