#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tick latency of the shipped speculative path, over scheme x K x acceptance.

Driven by ``spikes/lingbot_vla_v2/test_spec_oneccl.sh``, and by
``run_openvino_comparison.sh --spec-decode`` for the single-arm case. Unlike
`phase10_spec_runtime.py`, which was the spike that proved the mechanism, this
runs the **production** objects -- ``SpecDecoder`` and ``IGpuDraftClient`` from
``vllm_omni.diffusion.models.lingbot_vla_v2`` -- so what it reports is what the
server does.

Three axes, because the answer is a surface and not a number:

* ``--scheme cached`` is the shipped schedule: reuse the last full round's prefix
  KV, and on rejection defer a full round to the next tick. Grounding is
  amortised over ``--full-every`` rounds and a rejection costs nothing *now*.
* ``--scheme reground`` is ``config.spec_reground``: ground every speculative
  round on its own observation, draft off the fresh embeddings, and absorb a
  rejection with the Euler loop inside the rejecting tick. Nothing is stale and
  no tick executes a one-step guess off a rejected draft -- but grounding is paid
  every tick and a rejection is paid on the spot.
* ``--k-list`` is the verify count. ``spec_verify_batched`` runs those K
  timesteps as one ``B*K`` forward, so K is much cheaper than it looks
  (``phase10_verify_batch_sweep.py``) -- but ``radius_prefix_acceptance`` takes
  ``min`` over K, a conjunction, so a larger K *lowers* real acceptance. That
  second effect is invisible here, because acceptance is forced. **This script
  prices K; it cannot tell you which K to want.**

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
        --scheme cached,reground --k-list 1,2,4 \
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

from phase10_verify_batch_sweep import t_list_for  # noqa: E402
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
    """One arm: ``args.ticks`` control ticks through the production decoder.

    Speculative rounds are split three ways, because under ``spec_reground`` they
    are no longer one population: a round that accepted is cheap, a round that
    fell back ran the Euler loop in its own tick and costs more than a full round
    did. Reporting only the median over both would hide the shape that decides
    which scheme wins.
    """
    per_kind: dict[str, list[float]] = {"full": [], "spec": [], "fallback": []}
    accepted: list[int] = []
    regrounded = 0
    for tick in range(args.warmup + args.ticks):
        sync(device)
        start = time.perf_counter()
        result = decoder.decode(
            obs, session_id="sweep", reset=tick == 0, noise=None, num_steps=args.num_steps
        )
        sync(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if tick < args.warmup:
            continue
        stats = result.stats
        per_kind["fallback" if stats.fell_back and stats.kind == "spec" else stats.kind].append(elapsed)
        if stats.kind == "spec":
            accepted.append(stats.accepted)
            regrounded += int(stats.regrounded)
    total = per_kind["full"] + per_kind["spec"] + per_kind["fallback"]
    full_median, _ = summarise(per_kind["full"])
    spec_median, spec_p90 = summarise(per_kind["spec"])
    fallback_median, _ = summarise(per_kind["fallback"])
    spec_rounds = len(per_kind["spec"]) + len(per_kind["fallback"])
    # The same per-tick cost rebuilt from the per-class medians and counts. It
    # should track ``per_tick_ms``, which is a plain mean; where it does not, the
    # arm caught something transient -- a recompile that escaped warmup, or the
    # draft worker contending for a core -- and the mean is carrying it. This is
    # the cross-check, not a second result.
    mix = sum(
        len(per_kind[kind]) * median
        for kind, median in (("full", full_median), ("spec", spec_median), ("fallback", fallback_median))
        if per_kind[kind]
    )
    return {
        "full_rounds": len(per_kind["full"]),
        "spec_rounds": spec_rounds,
        # Speculative rounds that did *not* fall back, which is the population
        # ``spec_ms_median`` describes. Under ``reground`` at a low accept rate
        # this is 0 and the median is nan; that is the arm, not a bug.
        "accepting_rounds": len(per_kind["spec"]),
        "fallback_rounds": len(per_kind["fallback"]),
        "full_ms_median": full_median,
        # Accepting speculative rounds only; the fallbacks are the column beside it.
        "spec_ms_median": spec_median,
        "spec_ms_p90": spec_p90,
        "fallback_ms_median": fallback_median,
        "per_tick_ms": sum(total) / len(total) if total else float("nan"),
        "per_tick_ms_from_medians": mix / len(total) if total else float("nan"),
        # Realised, not requested: the §11.4 lesson is to report what happened.
        "realised_accept_rate": (
            sum(1 for value in accepted if value > 0) / len(accepted) if accepted else float("nan")
        ),
        "mean_accepted_steps": (sum(accepted) / len(accepted)) if accepted else float("nan"),
        # Tripwire, not decoration: if a --scheme reground arm reports 0 here, the
        # config knob did not reach the decoder and the whole column is the other
        # scheme measured twice.
        "regrounded_rounds": regrounded,
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
    parser.add_argument(
        "--scheme",
        default="cached",
        help="comma-separated: 'cached' (reuse the last full round's prefix KV) and/or "
             "'reground' (config.spec_reground: fresh prefix every speculative round, "
             "rejection denoises in the same tick)",
    )
    parser.add_argument(
        "--k-list",
        default=None,
        help="comma-separated verify counts, e.g. 1,2,4. Timesteps come from "
             "phase10_verify_batch_sweep.t_list_for, which pins K=1 and K=2 to the shipped "
             "values. Overrides --t-list when given",
    )
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

    schemes = [token.strip() for token in str(args.scheme).split(",") if token.strip()]
    unknown = [scheme for scheme in schemes if scheme not in ("cached", "reground")]
    if unknown or not schemes:
        parser.error(f"--scheme takes 'cached' and/or 'reground'; got {args.scheme!r}")

    if args.k_list:
        ks = [int(token) for token in str(args.k_list).split(",") if token.strip()]
        if any(k < 1 for k in ks):
            parser.error("--k-list takes verify counts, so every entry must be >= 1")
    else:
        ks = [len(args.t_list)]

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    # Every distinct verify batch size is its own dynamo specialisation under
    # ``dynamic=False``, and a K sweep asks for more of them than the default
    # cache allows -- past the limit dynamo silently falls back to eager and the
    # arm reads slow for a reason that has nothing to do with the scheme.
    torch._dynamo.config.cache_size_limit = max(64, 8 * len(ks) + 8)

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

    baseline: float | None = None
    arms: list[dict[str, Any]] = []
    try:
        if not args.no_baseline:
            samples = run_baseline(model, processor, obs, device, dtype, args)
            baseline = statistics.median(samples)
            print(f"[sweep] baseline (no speculation): {baseline:.1f} ms per tick", flush=True)

        # Scheme outermost, then K, then rate: the arms that share a compiled
        # verify batch size stay adjacent, so each specialisation is warmed once
        # and the sweep does not pay for recompiles it could have avoided.
        for scheme in schemes:
            for k in ks:
                for rate in rates:
                    config = dataclasses.replace(
                        base,
                        spec_t_list=t_list_for(k) if args.k_list else list(args.t_list),
                        spec_reground=scheme == "reground",
                        spec_force_accept_rate=rate,
                    )
                    decoder = SpecDecoder(
                        transformer=model, processor=processor, config=config,
                        device=device, dtype=dtype, draft=draft,
                    )
                    entry = run_rate(decoder, obs, device, args)
                    entry.update(scheme=scheme, k=k, requested_accept_rate=rate)
                    arms.append(entry)
                    label = "measured" if rate is None else f"{rate:g}"
                    print(
                        f"[sweep] {scheme:>8s} K={k} accept {label:>8s}: "
                        f"per tick {entry['per_tick_ms']:7.1f} ms "
                        f"(mix {entry['per_tick_ms_from_medians']:6.1f})  "
                        f"full {entry['full_rounds']:3d} x {entry['full_ms_median']:6.1f}  "
                        f"spec {entry['accepting_rounds']:3d} x {entry['spec_ms_median']:6.1f}  "
                        f"fallback {entry['fallback_rounds']:3d} x {entry['fallback_ms_median']:6.1f}",
                        flush=True,
                    )
                    decoder.sessions.clear()
    finally:
        draft.close()

    report(arms, schemes, ks, rates, baseline)
    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"args": vars(args), "baseline_ms": baseline, "arms": arms}, indent=2) + "\n"
        )
        print(f"[out] {out}")
    return 0


def rate_label(rate: float | None) -> str:
    return "measured" if rate is None else f"{rate:g}"


def report(
    arms: list[dict[str, Any]],
    schemes: list[str],
    ks: list[int],
    rates: list[float | None],
    baseline: float | None,
) -> None:
    """One per-tick matrix per scheme, then the head-to-head if both ran."""
    cells = {(arm["scheme"], arm["k"], rate_label(arm["requested_accept_rate"])): arm for arm in arms}
    labels = [rate_label(rate) for rate in rates]

    for scheme in schemes:
        print(f"\n== per-tick ms, scheme={scheme} -- rows K, columns forced accept rate ==")
        print("  K " + "".join(f"{label:>10s}" for label in labels))
        for k in ks:
            row = "".join(
                f"{cells[(scheme, k, label)]['per_tick_ms']:10.1f}" if (scheme, k, label) in cells
                else f"{'-':>10s}"
                for label in labels
            )
            print(f"{k:>3d} " + row)
        if baseline:
            print(f"    (baseline, no speculation: {baseline:.1f} ms per tick)")

    if len(schemes) == 2:
        print("\n== reground / cached, same K and rate -- below 1.00 means re-grounding wins ==")
        print("  K " + "".join(f"{label:>10s}" for label in labels))
        for k in ks:
            row = ""
            for label in labels:
                left, right = cells.get(("reground", k, label)), cells.get(("cached", k, label))
                row += (
                    f"{left['per_tick_ms'] / right['per_tick_ms']:9.2f}x"
                    if left and right else f"{'-':>10s}"
                )
            print(f"{k:>3d} " + row)

    print(
        "\nRead, in order:\n"
        "1. A forced rate changes only *the schedule*. The verify passes run at every rate, so\n"
        "   the speculative-round cost is measured; the actions are NOT valid, because an\n"
        "   accepted chunk is the untrained draft's own.\n"
        "2. `cached` and `reground` are not the same decode at a different price. `cached`\n"
        "   verifies against a teacher reading an older frame, and a rejected tick still\n"
        "   executes `x0_tail` -- a one-step estimate off the draft it just rejected.\n"
        "   `reground` has neither property. Compare the columns knowing that.\n"
        "3. K is priced here, not chosen: acceptance is forced, so the one thing a larger K\n"
        "   really does -- lower the accept rate, `min` over K being a conjunction -- cannot\n"
        "   show up in these numbers."
    )


if __name__ == "__main__":
    raise SystemExit(main())
