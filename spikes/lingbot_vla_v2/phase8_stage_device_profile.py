#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 8 step F — how does the compiled fp16 path's 294 ms split host vs device?

`phase6_denoise_profile.py` answered this for the *eager* path and its answer was
misread. It reports ``summed self DEVICE time: 51.0 ms``, but that number is a
**floor**: ``mm``/``einsum`` report zero device time on this backend, so the
profiler's per-operator device attribution silently omits most of the arithmetic.
`PHASE5_PERF.md` says so at the point of measurement and estimates real device
work in the loop at ~150-250 ms. Two conclusions in `PHASE8_LATENCY_PARITY.md`
were nevertheless built on treating 51.0 ms as a total and had to be retracted.

So this script does not ask the profiler. It uses the issue/drain test, which
needs no per-operator device time at all:

    sync; t0;  run stage (no sync);  t_issued;  sync;  t_done

    host_issue = t_issued - t0    how long the host took to submit the work
    drain      = t_done - t_issued how long the device kept working after that
    wall       = t_done - t0

``drain`` is a hard **lower bound** on device work: the device was demonstrably
busy for that long with the host no longer feeding it. A large ``drain`` means
device-bound. A near-zero ``drain`` with ``host_issue == wall`` means the device
kept up with submission, i.e. dispatch-bound.

The single shot alone can be fooled two ways, so each stage is also run N times
back-to-back under one sync. In steady state that costs
``N * max(host_per_iter, device_per_iter)``, which pins the binding constraint
even when the driver throttles submission because the command queue is full (in
which case ``host_issue`` inflates toward ``wall`` and the single shot would
wrongly read as dispatch-bound).

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase8_stage_device_profile.py \
        --model /tmp/lingbot-vla-v2-perf

Defaults are the deployed configuration: fp16, ``compile_denoise_step`` on,
eager prefix attention. That is the 294 ms this is trying to account for, so the
flags follow `run_openpi_server.sh` rather than the older bf16 spike defaults.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from phase5_latency import DEFAULT_MODEL, build, observation


def _sync(device: torch.device) -> None:
    if device.type in ("xpu", "cuda"):
        torch.accelerator.synchronize()


def measure(fn: Callable[[], Any], device: torch.device, repeats: int) -> dict[str, float]:
    """Issue/drain timings for one stage, plus the back-to-back steady state.

    ``fn`` must not synchronise internally; if it does, ``drain`` collapses to
    zero and the stage will be reported as dispatch-bound whatever the device is
    really doing. That failure mode is ruled out separately: on this stack 117
    queued 4096x4096 fp16 matmuls (~298 ms of device work) submit in 1.8 ms and
    leave 168 ms of drain outstanding, so submission is genuinely asynchronous
    and a near-zero drain here is a real result.

    Each stage is warmed individually before timing. Calling a compiled callable
    with a differently-guarded argument triggers a fresh Inductor graph, and
    without this the first measured ``wall`` is compilation: an earlier run of
    this script reported 62856.9 ms for one denoise step for exactly that reason.
    """
    fn()
    _sync(device)

    _sync(device)
    t0 = time.perf_counter()
    fn()
    t_issued = time.perf_counter()
    _sync(device)
    t_done = time.perf_counter()

    _sync(device)
    r0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    r_done = time.perf_counter()

    return {
        "wall": (t_done - t0) * 1e3,
        "host_issue": (t_issued - t0) * 1e3,
        "drain": (t_done - t_issued) * 1e3,
        "repeat_per_iter": (r_done - r0) / repeats * 1e3,
    }


def verdict(m: dict[str, float]) -> str:
    """Name the binding constraint, or refuse to when the evidence is ambiguous.

    The thresholds are deliberately wide. This test distinguishes "the device is
    the constraint" from "submission is the constraint"; it does not pretend to
    apportion a percentage between them.
    """
    drain_share = m["drain"] / m["wall"] if m["wall"] else 0.0
    # The steady state is the authority: if repeating costs materially more than
    # the host needed to submit one iteration, the device is what we are waiting
    # for, whatever the single shot's issue time looked like.
    device_led = m["repeat_per_iter"] > m["host_issue"] * 1.25
    if drain_share > 0.5 or device_led:
        return "device-bound"
    if drain_share < 0.1 and not device_led:
        return "dispatch-bound"
    return "mixed / inconclusive"


@torch.inference_mode()
def profile(args: argparse.Namespace) -> int:
    """Everything after argument parsing, under the deployed inference mode.

    `pipeline_lingbot_vla_v2.py:141` decorates `forward` with
    `@torch.inference_mode()` and `phase5_latency.py:467` wraps its timing loop in
    it. Profiling outside it pays autograd bookkeeping on every op, which for a
    dispatch-bound path lands entirely in the number being measured.
    """
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = build(Path(args.model), device, dtype, None, None)
    model.qwenvl_with_expert.attention_backend = args.attention_backend
    model.qwenvl_with_expert.attention_precision = args.attention_precision
    obs = observation(processor.spec, 0)
    features = processor.preprocess(obs)
    inputs = features.to(device=device, dtype=dtype).model_inputs()

    compiled = not args.no_compile_denoise_step
    if compiled:
        from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import denoise_compile_options

        # Exactly what `pipeline_lingbot_vla_v2.py:59` does in production.
        print("[compile] torch.compile(predict_velocity, inductor, dynamic=False, fullgraph=True)")
        model.predict_velocity = torch.compile(
            model.predict_velocity,
            backend="inductor",
            dynamic=False,
            fullgraph=True,
            options=denoise_compile_options(),
        )

    # Warm up through the full path: the first calls pay Inductor compilation and
    # lazy kernel setup, either of which would land entirely in stage one.
    for _ in range(args.warmup):
        model.sample_actions(**inputs)
    _sync(device)

    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import make_att_2d_masks

    # Stage inputs, captured once. Re-deriving them per repeat would time the
    # predecessor stage inside its successor.
    prefix_args = (
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["image_grid_thw"],
    )
    embs, pad_masks, att_masks, position_ids, visual_masks, deepstack = model.embed_prefix(*prefix_args)
    att_2d = make_att_2d_masks(pad_masks, att_masks)
    prefix_forward = getattr(model, "_benchmark_compiled_prefix", model.qwenvl_with_expert.forward)

    def run_prefix_forward():
        return prefix_forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            inputs_embeds=[embs, None],
            past_key_values=None,
            fill_kv_cache=True,
            visual_pos_masks=visual_masks,
            deepstack_visual_embeds=deepstack,
        )

    _, past_key_values = run_prefix_forward()
    _sync(device)
    noise = torch.randn(
        (inputs["state"].shape[0], model.config.chunk_size, model.config.max_action_dim),
        device=device,
        dtype=dtype,
    )
    denoise_kwargs = dict(
        state=inputs["state"],
        prefix_pad_masks=pad_masks,
        prefix_position_ids=position_ids,
        past_key_values=past_key_values,
        noise=noise,
    )

    stages: list[tuple[str, Callable[[], Any]]] = [
        ("embed_prefix", lambda: model.embed_prefix(*prefix_args)),
        ("prefix_forward", run_prefix_forward),
        (
            "predict_velocity (1 step)",
            lambda: model.predict_velocity(
                state=inputs["state"],
                prefix_pad_masks=pad_masks,
                prefix_position_ids=position_ids,
                past_key_values=past_key_values,
                x_t=noise,
                timestep=torch.ones(noise.shape[0], device=device, dtype=dtype),
            ),
        ),
        (
            f"denoise_actions ({model.config.num_steps} steps)",
            lambda: model.denoise_actions(num_steps=model.config.num_steps, **denoise_kwargs),
        ),
        ("sample_actions (whole request)", lambda: model.sample_actions(**inputs)),
    ]

    print(
        f"\ndevice={device} dtype={args.dtype} "
        f"compile_denoise_step={compiled} moe={model.config.moe_implementation} "
        f"attention={args.attention_backend}/{args.attention_precision} "
        f"inference_mode={not torch.is_grad_enabled()} repeats={args.repeats}"
    )
    print(f"host load1={open('/proc/loadavg').read().split()[0]}\n")
    header = f"{'stage':32s} {'wall':>9s} {'host issue':>11s} {'drain':>9s} {'repeat/it':>10s}  verdict"
    print(header)
    print("-" * len(header))

    results: dict[str, dict[str, float]] = {}
    for name, fn in stages:
        m = measure(fn, device, args.repeats)
        results[name] = m
        print(
            f"{name:32s} {m['wall']:8.1f}m {m['host_issue']:10.1f}m {m['drain']:8.1f}m "
            f"{m['repeat_per_iter']:9.1f}m  {verdict(m)}"
        )

    print(
        "\nwall = one cold-queue call end to end. host issue = time to submit it.\n"
        "drain = device still working after submission returned; a hard LOWER bound\n"
        "on device time. repeat/it = steady state over the repeats, which equals\n"
        "max(host, device) per iteration and is the authority on which one binds.\n"
        "All times in milliseconds."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="xpu")
    # fp16, compiled denoise step, eager/fp16 attention: the deployed
    # configuration, and the same flags `run_openvino_comparison.sh` passes to
    # `phase5_latency.py`. Matching it exactly is the point — these numbers are
    # only comparable to the recorded 294 ms if every knob agrees.
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--no-compile-denoise-step", action="store_true")
    # `LingbotVlaV2WithExpertModel.__init__` hardcodes fp32 attention
    # (`modeling_lingbot_vla_v2.py:942-943`); the deployed value comes from the
    # config via `pipeline_lingbot_vla_v2.py:61`, and `phase5_latency.py:460-462`
    # mirrors that. Forgetting it silently profiles fp32 attention: the first run
    # of this script reported prefix_forward at 67.7 ms for exactly that reason,
    # against 57.6 ms for the deployed fp16 path.
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--attention-precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    return profile(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
