#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tick latency of the shipped speculative path, as a function of acceptance.

Driven by ``run_openvino_comparison.sh --spec-decode [--accept-rate ...]``. Unlike
`phase10_spec_runtime.py`, which was the spike that proved the mechanism, this
runs the **production** objects -- ``SpecDecoder`` and ``IGpuDraftClient`` from
``vllm_omni.diffusion.models.lingbot_vla_v2`` -- so what it reports is what the
server does.

Why acceptance has to be *simulated* to be measured: a draft head that has not
been trained is rejected every time, and FLASH's rule is that a rejected round
forces the next tick to re-ground with a full round. So real acceptance today is
0, every tick becomes a full round, and the schedule is never exercised.
``spec_force_accept_rate`` overrides the accept decision while still running the
K verify passes, so the **latency is real** at each rate; the actions are not,
because an accepted chunk is the draft's own. That is the trade this script
makes, and it is the only way to price the schedule before Phase 10.1.

Acceptance changes performance through exactly one mechanism -- whether the next
tick has to be a full round -- so the sweep traces the curve between the two ends
that are already known: p=0 is a full round every tick (today's 294 ms) and p=1
is the periodic schedule (95.8 ms at K=2, ``spec_full_every=4``).

    ZE_AFFINITY_MASK=0 PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_spec_accept_sweep.py \
        --model /tmp/lingbot-vla-v2-perf --compile-denoise-step \
        --accept-rate 0,0.5,0.9,1 --worker-cpu 11

``--worker-cpu`` is not optional in practice: the draft worker's oneCCL recv
hard-spins a core, and unreserved that collides with this process's OpenMP pool
and adds ~215 ms to every full round (PHASE10_SPECULATIVE.md §11.4). It inflates
the speedup, because the baseline arm slows down too.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase5_latency import build, compile_denoise_step, observation  # noqa: E402


def sync(device: torch.device) -> None:
    if device.type in ("xpu", "cuda"):
        torch.accelerator.synchronize()


def summarise(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    p90 = statistics.quantiles(samples, n=10)[-1] if len(samples) >= 10 else max(samples)
    return statistics.median(samples), p90


def run_baseline(model: Any, processor: Any, obs: dict, device, dtype, args) -> list[float]:
    """Today's path: preprocess plus a full ``sample_actions``, per tick.

    The preprocess has to be inside the timed region, because it is inside the
    served request and inside the speculative arm's full rounds. Hoisting it out
    made this arm 10 ms faster than the same work measured through the decoder --
    a 3.5% error in the wrong direction, all of it in the comparison's favour.
    """
    samples: list[float] = []
    for tick in range(args.warmup + args.ticks):
        sync(device)
        start = time.perf_counter()
        features = processor.preprocess(obs).to(device=device, dtype=dtype)
        model.sample_actions(**features.model_inputs(), noise=None, num_steps=args.num_steps)
        sync(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if tick >= args.warmup:
            samples.append(elapsed)
    return samples


def run_rate(decoder: Any, obs: dict, device, args) -> dict[str, Any]:
    """One arm: ``args.ticks`` control ticks through the production decoder."""
    per_kind: dict[str, list[float]] = {"full": [], "spec": []}
    accepted: list[int] = []
    for tick in range(args.warmup + args.ticks):
        sync(device)
        start = time.perf_counter()
        result = decoder.decode(
            obs, session_id="sweep", reset=tick == 0, noise=None, num_steps=args.num_steps
        )
        sync(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if tick >= args.warmup:
            per_kind[result.stats.kind].append(elapsed)
            if result.stats.kind == "spec":
                accepted.append(result.stats.accepted)
    total = per_kind["full"] + per_kind["spec"]
    full_median, _ = summarise(per_kind["full"])
    spec_median, spec_p90 = summarise(per_kind["spec"])
    return {
        "full_rounds": len(per_kind["full"]),
        "spec_rounds": len(per_kind["spec"]),
        "full_ms_median": full_median,
        "spec_ms_median": spec_median,
        "spec_ms_p90": spec_p90,
        "per_tick_ms": sum(total) / len(total) if total else float("nan"),
        # Realised, not requested: the §11.4 lesson is to report what happened.
        "realised_accept_rate": (
            sum(1 for value in accepted if value > 0) / len(accepted) if accepted else float("nan")
        ),
        "mean_accepted_steps": (sum(accepted) / len(accepted)) if accepted else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="prepared model directory")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--attention-precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--compile-denoise-step", action="store_true")
    parser.add_argument("--compile-max-relative-error", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=10, help="full round's Euler steps")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--ticks", type=int, default=20, help="keep a multiple of --full-every + 1")
    parser.add_argument(
        "--accept-rate",
        default="0,0.5,1",
        help="comma-separated forced acceptance rates; 'measured' uses the real accept rule "
             "(which is 0 until a draft head is trained)",
    )
    parser.add_argument("--full-every", type=int, default=4)
    parser.add_argument("--t-list", type=float, nargs="*", default=[0.10, 0.05], help="verify timesteps (K = len)")
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--max-exec-steps", type=int, default=12)
    parser.add_argument("--worker-cpu", default=None, help="CPUs reserved for the draft worker, e.g. 11")
    parser.add_argument("--draft-path", default=None, help="trained draft head; random weights if unset")
    parser.add_argument("--no-baseline", action="store_true", help="skip the non-speculative arm")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    rates: list[float | None] = []
    for token in str(args.accept_rate).split(","):
        token = token.strip()
        if not token:
            continue
        rates.append(None if token == "measured" else float(token))
    if not rates:
        rates = [None]

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = build(Path(args.model), device, dtype, args.num_steps)
    model.qwenvl_with_expert.attention_backend = args.attention_backend
    model.qwenvl_with_expert.attention_precision = args.attention_precision
    obs = observation(processor.spec, seed=0)
    if args.compile_denoise_step:
        compile_denoise_step(
            processor, model, obs, device, dtype, "inductor", False, False, args.compile_max_relative_error
        )

    from vllm_omni.diffusion.models.lingbot_vla_v2.draft_igpu import IGpuDraftClient
    from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import SpecDecoder, prefix_shape

    # One config for the fields the draft client reads; the forced rate varies
    # per arm and is the only thing rebuilt.
    base = dataclasses.replace(
        model.config,
        spec_decode=True,
        spec_t_list=list(args.t_list),
        spec_tau=args.tau,
        spec_full_every=args.full_every,
        spec_max_exec_steps=args.max_exec_steps,
        spec_worker_cpu=args.worker_cpu,
        spec_draft_path=args.draft_path,
    )
    prefix_len, prefix_width = prefix_shape(model, processor, obs, device=device, dtype=dtype)
    draft = IGpuDraftClient(
        config=base, prefix_len=prefix_len, prefix_width=prefix_width, device=device, dtype=dtype
    )

    results: dict[str, Any] = {}
    try:
        baseline = None
        if not args.no_baseline:
            samples = run_baseline(model, processor, obs, device, dtype, args)
            baseline = statistics.median(samples)
            results["baseline"] = {"per_tick_ms": baseline, "ticks": len(samples)}
            print(f"[sweep] baseline (no speculation): {baseline:.1f} ms per tick", flush=True)

        for rate in rates:
            decoder = SpecDecoder(
                transformer=model,
                processor=processor,
                config=dataclasses.replace(base, spec_force_accept_rate=rate),
                device=device,
                dtype=dtype,
                draft=draft,
            )
            entry = run_rate(decoder, obs, device, args)
            entry["requested_accept_rate"] = rate
            results["measured" if rate is None else f"{rate:g}"] = entry
            print(
                f"[sweep] accept {'measured' if rate is None else f'{rate:g}':>8s}: "
                f"per tick {entry['per_tick_ms']:7.1f} ms  "
                f"full {entry['full_rounds']:3d} x {entry['full_ms_median']:6.1f}  "
                f"spec {entry['spec_rounds']:3d} x {entry['spec_ms_median']:5.1f}",
                flush=True,
            )
            decoder.sessions.clear()
    finally:
        draft.close()

    print(f"\n{'accept rate':>12s}{'per tick ms':>13s}{'vs baseline':>13s}{'full':>7s}{'spec':>7s}"
          f"{'spec ms':>9s}{'realised':>10s}")
    print("-" * 71)
    for key, entry in results.items():
        if key == "baseline":
            print(f"{'none':>12s}{entry['per_tick_ms']:13.1f}{'1.00x':>13s}"
                  f"{entry['ticks']:7d}{0:7d}{'-':>9s}{'-':>10s}")
            continue
        speedup = f"{baseline / entry['per_tick_ms']:.2f}x" if baseline else "-"
        print(f"{key:>12s}{entry['per_tick_ms']:13.1f}{speedup:>13s}"
              f"{entry['full_rounds']:7d}{entry['spec_rounds']:7d}"
              f"{entry['spec_ms_median']:9.1f}{entry['realised_accept_rate']:10.2f}")

    print(
        "\nRead: the forced rate changes only *the schedule* -- a rejected round makes the next\n"
        "tick a full round. The verify passes run at every rate, so the speculative-round cost\n"
        "is measured, not modelled. Actions under a forced rate are NOT valid: an accepted\n"
        "chunk is the untrained draft's own."
    )
    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2) + "\n")
        print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
