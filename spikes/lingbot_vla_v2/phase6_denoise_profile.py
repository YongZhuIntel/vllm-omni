# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M6 step 3 — where do the denoise loop's 614 ms actually go?

``phase6_moe_micro.py`` established that the routed-MoE einsums are not the
problem: at the action expert's exact shapes they run at 13.06 TFLOPS, which is
*faster* than the OpenVINO reference's whole-loop rate, and account for only
~106 ms of the 614 ms loop. So ~508 ms is spent on everything else -- attention,
norms, the router, the shared expert, and the cost of dispatching all of it.

The same file measured the dispatch floor on this stack: ~5.1 us of submit-side
cost per operator, ~6.4 us amortised. At 36 layers x 10 steps, every operator in
a decoder layer costs ~2.3 ms of wall time across the request, whatever it does.
Three hundred small operators per layer-step would account for the entire budget
on their own.

This script settles it by profiling the real model: it counts the operators the
denoise loop actually dispatches and ranks them by self time on the device, so
the remaining 508 ms gets an address instead of a hypothesis.

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase6_denoise_profile.py \
        --model /tmp/lingbot-vla-v2-perf
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from phase5_latency import DEFAULT_MODEL, build, observation


def _sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()


def _model_inputs(processor, model, obs, device: torch.device, dtype: torch.dtype):
    """The same construction ``phase5_latency.one_request`` performs, untimed.

    The processor is not callable as a whole; its private builders are what the
    pipeline's ``preprocess`` actually calls, so this uses them directly rather
    than an approximation that might dispatch different operators.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.processor import (
        RobotFeatures,
        _action_key,
        _as_float_tensor,
        _state_key,
    )

    config = model.config
    with torch.device("cpu"):
        raw = {"observation.state": _as_float_tensor(obs["state"])}
        state, state_mask = processor._build_vector(
            raw, processor.spec.state_slices, _state_key, config.max_state_dim
        )
        _, action_mask = processor._build_vector(
            raw, processor.spec.action_slices, _action_key, config.max_action_dim, values=False
        )
        images, img_masks, grid = processor._build_images(obs["images"])
        lang_tokens, lang_masks = processor._build_language(obs["prompt"])

    features = RobotFeatures(
        images=images,
        img_masks=img_masks,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=state.unsqueeze(0),
        image_grid_thw=grid,
        state_mask=state_mask,
        action_mask=action_mask,
    )
    return features.to(device=device, dtype=dtype).model_inputs()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = build(Path(args.model), device, dtype, None, None)
    obs = observation(processor.spec, 0)

    inputs = _model_inputs(processor, model, obs, device, dtype)

    # Warm up: the first call pays lazy kernel setup that would swamp the profile.
    for _ in range(2):
        model.sample_actions(**inputs)
    _sync(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "xpu" and hasattr(ProfilerActivity, "XPU"):
        activities.append(ProfilerActivity.XPU)

    with profile(activities=activities, record_shapes=False) as prof:
        model.sample_actions(**inputs)
        _sync(device)

    events = prof.key_averages()
    # Leaf operators only: anything with children double-counts its own subtree.
    leaves = [e for e in events if e.key.startswith("aten::")]
    total_dev_us = sum(_self_device_us(e) for e in leaves)
    total_cpu_us = sum(_self_cpu_us(e) for e in leaves)
    total_calls = sum(e.count for e in leaves)

    print(f"\ndevice={device} dtype={args.dtype}  one full 10-step denoise + prefix")
    print(f"aten operators dispatched: {total_calls}")
    print(f"summed self DEVICE time:   {total_dev_us / 1e3:.1f} ms  "
          "(what the card actually spent)")
    print(f"summed self CPU time:      {total_cpu_us / 1e3:.1f} ms  "
          "(what the host spent dispatching)")
    print(f"implied per-op floor at 5.1 us submit cost: {total_calls * 5.1 / 1e3:.1f} ms\n")

    key = _self_cpu_us if total_dev_us == 0 else _self_device_us
    print(f"{'operator':34s} {'calls':>8s} {'dev ms':>9s} {'cpu ms':>9s} {'cpu us/call':>12s}")
    print("-" * 76)
    for event in sorted(leaves, key=lambda e: -key(e))[:args.top]:
        dev_us, cpu_us = _self_device_us(event), _self_cpu_us(event)
        print(f"{event.key[:34]:34s} {event.count:8d} {dev_us / 1e3:9.2f} "
              f"{cpu_us / 1e3:9.2f} {cpu_us / max(event.count, 1):12.1f}")

    # Group by "is this real arithmetic or glue?" -- the distinction that decides
    # whether the fix is a better kernel or fewer operators.
    gemm_keys = ("mm", "bmm", "matmul", "einsum", "linear", "addmm", "baddbmm", "scaled_dot")
    buckets: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    for event in leaves:
        name = event.key.lower()
        bucket = "gemm" if any(k in name for k in gemm_keys) else "glue"
        buckets[bucket][0] += event.count
        buckets[bucket][1] += _self_device_us(event)
        buckets[bucket][2] += _self_cpu_us(event)
    print("\n{:10s} {:>8s} {:>9s} {:>9s} {:>8s}".format(
        "bucket", "calls", "dev ms", "cpu ms", "cpu share"))
    print("-" * 50)
    for bucket, (calls, dev_us, cpu_us) in sorted(buckets.items(), key=lambda kv: -kv[1][2]):
        share = cpu_us / total_cpu_us * 100 if total_cpu_us else 0.0
        print(f"{bucket:10s} {int(calls):8d} {dev_us / 1e3:9.2f} {cpu_us / 1e3:9.2f} "
              f"{share:7.1f}%")
    return 0


def _self_device_us(event) -> float:
    """Self time on the accelerator, or 0.0 if the backend reports none.

    Deliberately does *not* fall back to CPU time: the whole question is whether
    the loop is device-bound or dispatch-bound, and a silent fallback answers it
    wrongly by relabelling host time as device time.
    """
    for attr in ("self_device_time_total", "self_xpu_time_total", "self_cuda_time_total"):
        value = getattr(event, attr, 0) or 0
        if value:
            return float(value)
    return 0.0


def _self_cpu_us(event) -> float:
    """Host-side self time -- dispatch, shape logic, allocator, submit."""
    return float(getattr(event, "self_cpu_time_total", 0) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
