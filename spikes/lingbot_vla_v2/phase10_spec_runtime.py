#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 10.2/10.3 — the speculative control loop, end to end, three arms.

Everything before this file measured one piece in isolation: gate 3 the verify
algebra, gate 4 the draft's shape, §9 the transport, `phase10_draft_worker.py`
the protocol. This assembles them into an actual tick loop over the real 6B
verifier and reports **per-tick latency**, which is the number the whole of
Phase 10 exists to move.

    baseline   full round every tick                         -- today's path
    local      draft on the dGPU next to the verifier        -- Phase 10.2
    igpu       draft on the iGPU in a second process         -- Phase 10.3

A full round is what the model does today: preprocess, `embed_prefix`,
`prefix_fill`, ten Euler steps. A speculative round keeps the prefix KV cache
from the last full round, normalises the new state, asks the draft for a whole
chunk, and spends **K** `predict_velocity` calls checking it
(`phase10_verify_oracle_probe.verify_once`, the same code gate 3 measured).

## Why an untrained draft is the right subject for this run

Per-tick latency does not depend on the draft's weights: the draft's shape is
fixed, the verify cost is K forward passes whatever comes back, and the
transport moves the same bytes. So the plumbing can be proven, and the schedule
priced, before the head is trained -- and after training nothing about the
latency changes, only the acceptance.

Acceptance, on the other hand, is **zero** with random weights, by construction.
That has one consequence worth understanding before reading any table below:
FLASH forces a full round whenever nothing was accepted
(`_should_schedule_full_fallback:181`), so leaving that rule on would turn every
tick into a full round and measure nothing. `--fallback off` (the default here)
follows the *intended* schedule -- `--full-every n` -- regardless of acceptance,
which is what makes the latency number meaningful. `--fallback on` reproduces
FLASH's behaviour and is the right setting once a trained head exists.

Acceptance is reported either way, and with random weights it should read 0.
If it does not, something is wrong with the accept rule, not with the draft.

## What the numbers are and are not

Latency is measured on the real model on this host and is a real result.
**Accuracy is not measured here**: the committed 6-chunk bundle strides by 50
frames, so the cached prefix a speculative round reuses belongs to an
observation 50 frames old rather than 1, and the draft is untrained. Staleness
is therefore reported in **rounds**, not frames; the frame-level question is
gate 2's (`phase10_kv_staleness_probe.py`, still unrun) and it needs the dense
bundle. `--score` will compute a teacher chunk alongside each speculative round
if you want the MAE anyway, but with random weights it measures the
mean-of-verify fallback path, not a draft.

    I=/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install
    export LD_LIBRARY_PATH=$I/lib:$I/opt/mpi/lib:$LD_LIBRARY_PATH CCL_PLUGIN=ONECCL_IGPU
    ZE_AFFINITY_MASK=0 PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_spec_runtime.py \
        --model /tmp/lingbot-open-loop \
        --dataset /llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz \
        --reps 3 --ticks 20 --compile-denoise-step --worker-cpu 11

`ZE_AFFINITY_MASK=0` matters: it pins this process to the dGPU and lets the
worker have the iGPU as its own `xpu:0` (PHASE8 §K -- `device_count()` is 1 per
process on this host, which is why the iGPU arm is a second process at all).
`--worker-cpu` matters at least as much: without it the resident worker's
spinning `recv` collides with this process's OpenMP pool and adds ~215 ms to
every full round's `preprocess` -- which inflates the speculative speedup,
because it slows the baseline arm too. See `phase10_draft_worker.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from open_loop_conditioning_probe import make_noise  # noqa: E402
from phase10_draft_worker import (  # noqa: E402
    LocalDraft,
    add_draft_arguments,
    build_head,
    connect_worker,
)
from phase10_kv_staleness_probe import embed_prefix_inputs, fill_prefix, observation  # noqa: E402
from phase10_spec_common import (  # noqa: E402
    gripper_dims,
    pose_dims,
    radius_prefix_acceptance,
    stitch_prefix,
    truncate_on_gripper_switch,
)
from phase10_verify_oracle_probe import verify_once  # noqa: E402
from phase5_latency import Stopwatch, build, compile_denoise_step  # noqa: E402

ARMS = ("baseline", "local", "igpu")


def normalize_state(processor: Any, raw_state: np.ndarray) -> torch.Tensor:
    """The state half of `preprocess`, without touching the cameras.

    A speculative round has no use for the images: the draft reads the projected
    prefix it already holds and the verifier reads the cached KV. Running the
    Qwen3-VL image processor anyway would put ~10 ms of host work into a ~22 ms
    tick. Going through `_build_vector` keeps the normalisation identical to the
    full round's; when this moves into `vllm_omni/` that should become a public
    `preprocess_state` rather than a private call.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.config import OBS_STATE
    from vllm_omni.diffusion.models.lingbot_vla_v2.processor import _state_key

    raw = {OBS_STATE: torch.as_tensor(np.asarray(raw_state, dtype=np.float32))}
    vector, _ = processor._build_vector(
        raw, processor.spec.state_slices, _state_key, processor.config.max_state_dim
    )
    return vector[None]


@dataclass
class SpecSession:
    """What survives between ticks. The serving seam for this is `session_id`.

    `pipeline_lingbot_vla_v2.forward()` is stateless today; `extra_args` already
    carries `session_id` and `reset` (`entrypoints/openpi/serving.py:153`), so
    Phase 10.4 indexes one of these per session instead of adding a hook.
    """

    pad_masks: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: list
    features: Any  # the full round's RobotFeatures -- postprocess needs its masks
    anchor_tick: int
    anchor_row: int
    rounds_since_full: int = 0
    pending_full: bool = False
    gripper_prev: torch.Tensor | None = None


@dataclass
class TickRecord:
    arm: str
    rep: int
    tick: int
    kind: str
    row: int
    total_ms: float
    stages: dict[str, float]
    accepted: int | None = None
    eval_h: int = 0
    gripper_cut: bool = False
    age_rounds: int = 0
    anchor_row: int = -1  # which observation filled the KV cache this round reused
    dist_mean: float | None = None
    draft_rms: float | None = None
    mae_vs_teacher: float | None = None


@dataclass
class Plan:
    """One arm's configuration, resolved once so the tick loop stays readable."""

    arm: str
    backend: Any | None
    full_every: int
    fallback: bool
    t_list: list[float]
    tau: float
    max_exec_steps: int
    num_steps: int
    dims: list[int]
    grippers: list[int]
    gripper_threshold: float
    score: bool
    real_dims: torch.Tensor = field(repr=False, default=None)


def full_round(
    model: Any, processor: Any, plan: Plan, obs: dict, noise: torch.Tensor, device, dtype, stopwatch: Stopwatch
) -> tuple[torch.Tensor, SpecSession, torch.Tensor | None]:
    """Today's path, plus the draft's prefix refresh. Returns ``(x0, session, x0_draft)``."""
    stopwatch.start()
    features = processor.preprocess(obs).to(device=device, dtype=dtype)
    inputs = features.model_inputs()
    stopwatch.lap("preprocess")

    embedded = embed_prefix_inputs(model, inputs)
    stopwatch.lap("embed_prefix")
    pad_masks, position_ids, past_key_values = fill_prefix(model, inputs, embedded=embedded)
    stopwatch.lap("prefix_fill")

    x0_draft = None
    if plan.backend is not None:
        # `embedded[0]` is `prefix_embs` [1,286,2560]. The projection to 512 runs
        # here on the dGPU either way; only the iGPU arm puts it on the wire.
        x0_draft = plan.backend.refresh(embedded[0], inputs["state"])
    stopwatch.lap("draft_refresh")

    x0 = model.denoise_actions(
        state=inputs["state"],
        prefix_pad_masks=pad_masks,
        prefix_position_ids=position_ids,
        past_key_values=past_key_values,
        noise=noise,
        num_steps=plan.num_steps,
    )
    stopwatch.lap("denoise")

    session = SpecSession(
        pad_masks=pad_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        features=features,
        anchor_tick=0,
        anchor_row=0,
    )
    return x0, session, x0_draft


def spec_round(
    model: Any, processor: Any, plan: Plan, session: SpecSession, raw_state: np.ndarray,
    noise: torch.Tensor, device, dtype, stopwatch: Stopwatch,
) -> tuple[torch.Tensor, int, bool, float]:
    """Draft once, verify K times, accept a prefix. Returns ``(x0_out, accepted, cut, dist)``."""
    stopwatch.start()
    state = normalize_state(processor, raw_state).to(device=device, dtype=dtype)
    stopwatch.lap("state")

    x0_draft = plan.backend.draft(state)
    stopwatch.lap("draft")

    # The cached prefix is the whole point: no `embed_prefix`, no `prefix_fill`.
    x0_hat = verify_once(
        model,
        (session.pad_masks, session.position_ids, session.past_key_values),
        state,
        noise,
        x0_draft,
        plan.t_list,
    )
    stopwatch.lap("verify")

    accepted, dist = radius_prefix_acceptance(
        x0_draft, x0_hat, tau=plan.tau, dims=plan.dims, eval_h=plan.max_exec_steps
    )
    x0_tail = x0_hat.mean(dim=1)
    x0_out = stitch_prefix(x0_draft, x0_tail, accepted)
    # FLASH truncates the *accepted length* on a speculated gripper transition
    # and leaves the stitched chunk alone (`_action_stage_impl:745`): fewer steps
    # execute, so the next replan comes sooner. Note that FLASH's other gripper
    # guard, the any-K pre-verify stop, is **not** ported into
    # `phase10_spec_common` yet -- see PHASE10_SPECULATIVE.md.
    accepted, cut = truncate_on_gripper_switch(
        x0_out,
        accepted,
        gripper_prev=session.gripper_prev,
        dims=plan.grippers,
        threshold=plan.gripper_threshold,
    )
    stopwatch.lap("accept")
    return x0_out, int(accepted.item()), bool(cut.any().item()), float(dist.mean().item())


def run_arm(
    arm: str, rep: int, model: Any, processor: Any, bundle: dict, plan: Plan, args, device, dtype
) -> list[TickRecord]:
    records: list[TickRecord] = []
    session: SpecSession | None = None
    rows = len(bundle["states"])

    for tick in range(args.ticks):
        row = tick % rows
        noise = torch.as_tensor(make_noise(args.seed, tick), device=device, dtype=dtype)
        stopwatch = Stopwatch(device)
        want_full = (
            arm == "baseline"
            or session is None
            or session.pending_full
            or session.rounds_since_full >= plan.full_every
        )

        if want_full:
            x0, session, x0_draft = full_round(
                model, processor, plan, observation(bundle, row), noise, device, dtype, stopwatch
            )
            session.anchor_tick, session.anchor_row = tick, row
            record = TickRecord(
                arm=arm, rep=rep, tick=tick, kind="full", row=row,
                total_ms=sum(stopwatch.laps.values()) * 1000.0,
                stages={name: value * 1000.0 for name, value in stopwatch.laps.items()},
                eval_h=plan.max_exec_steps,
            )
            if x0_draft is not None:
                # Free accuracy signal: the draft and the teacher answered the
                # same frame. This is the quantity gate 3 reads as `eps`.
                delta = (x0_draft - x0)[:, : plan.max_exec_steps, plan.real_dims].to(torch.float32)
                record.draft_rms = float(delta.pow(2).mean().sqrt().item())
        else:
            x0, accepted, cut, dist = spec_round(
                model, processor, plan, session, bundle["states"][row].astype(np.float32),
                noise, device, dtype, stopwatch,
            )
            session.rounds_since_full += 1
            record = TickRecord(
                arm=arm, rep=rep, tick=tick, kind="spec", row=row,
                total_ms=sum(stopwatch.laps.values()) * 1000.0,
                stages={name: value * 1000.0 for name, value in stopwatch.laps.items()},
                accepted=accepted, eval_h=plan.max_exec_steps, gripper_cut=cut,
                age_rounds=session.rounds_since_full, anchor_row=session.anchor_row, dist_mean=dist,
            )
            if plan.fallback and (accepted == 0 or cut):
                session.pending_full = True
            if plan.score:
                _, _, teacher = _teacher_reference(model, processor, plan, bundle, row, noise, device, dtype)
                record.mae_vs_teacher = float(
                    (x0 - teacher)[:, : plan.max_exec_steps, plan.real_dims].abs().mean().item()
                )

        # The last executed gripper command, for the next round's switch guard.
        # This loop replans every tick, so exactly one step of each chunk is
        # executed and that step is index 0. A loop that executed
        # `max_exec_steps` before replanning would read the last of those.
        session.gripper_prev = x0[:, 0, :].index_select(-1, torch.as_tensor(
            plan.grippers, device=x0.device, dtype=torch.long)).detach()
        records.append(record)

    return records


def _teacher_reference(model, processor, plan, bundle, row, noise, device, dtype):
    """A fresh full round for `--score`. Doubles the tick; off by default."""
    features = processor.preprocess(observation(bundle, row)).to(device=device, dtype=dtype)
    inputs = features.model_inputs()
    pad_masks, position_ids, past_key_values = fill_prefix(model, inputs)
    x0 = model.denoise_actions(
        state=inputs["state"], prefix_pad_masks=pad_masks, prefix_position_ids=position_ids,
        past_key_values=past_key_values, noise=noise, num_steps=plan.num_steps,
    )
    return features, inputs, x0


# -- reporting -------------------------------------------------------------


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def report(records: list[TickRecord], args) -> dict[str, Any]:
    """Per-arm table, then the amortised comparison the plan's arithmetic lives on."""
    summary: dict[str, Any] = {}
    print(f"\n{'arm':>9s} {'rep':>4s} {'full ms':>9s} {'spec ms':>9s} {'spec p90':>9s} "
          f"{'per tick':>9s} {'ticks':>6s} {'accept':>7s}")
    print("-" * 74)
    for arm in ARMS:
        for rep in range(args.reps):
            rows = [r for r in records if r.arm == arm and r.rep == rep]
            if not rows:
                continue
            full = [r.total_ms for r in rows if r.kind == "full"]
            spec = [r.total_ms for r in rows if r.kind == "spec"]
            accepted = [r.accepted for r in rows if r.accepted is not None]
            per_tick = sum(r.total_ms for r in rows) / len(rows)
            summary.setdefault(arm, []).append(per_tick)
            p90 = statistics.quantiles(spec, n=10)[-1] if len(spec) >= 10 else (max(spec) if spec else float("nan"))
            print(f"{arm:>9s} {rep:>4d} {median(full):9.1f} {median(spec):9.1f} {p90:9.1f} "
                  f"{per_tick:9.1f} {len(rows):6d} {np.mean(accepted) if accepted else float('nan'):7.2f}")

    print(f"\n{'arm':>9s} {'per-tick median of reps':>25s} {'vs baseline':>12s}")
    print("-" * 49)
    baseline = median(summary.get("baseline", []))
    for arm in ARMS:
        if arm not in summary:
            continue
        value = median(summary[arm])
        speedup = f"{baseline / value:.2f}x" if value and not np.isnan(baseline) else "-"
        print(f"{arm:>9s} {value:25.1f} {speedup:>12s}")

    # The measured mean above only equals the schedule's amortised cost when the
    # tick count is a whole number of periods; this derives it from the measured
    # medians instead, and extends it to the `n` values PHASE10's table quotes.
    schedule = [n for n in sorted({1, 2, args.full_every, 9}) if n > 0]
    print(f"\namortised (full + n*spec)/(n+1) from the measured medians, baseline {baseline:.1f} ms")
    print(f"{'arm':>9s}" + "".join(f"{f'n={n}':>16s}" for n in schedule))
    print("-" * (9 + 16 * len(schedule)))
    for arm in ARMS:
        rows = [r for r in records if r.arm == arm]
        spec = [r.total_ms for r in rows if r.kind == "spec"]
        full = [r.total_ms for r in rows if r.kind == "full"]
        if not spec or not full:
            continue
        cells = []
        for n in schedule:
            value = (median(full) + n * median(spec)) / (n + 1)
            ratio = f"{baseline / value:4.1f}x" if not np.isnan(baseline) else ""
            cells.append(f"{value:9.1f} {ratio}")
        print(f"{arm:>9s}" + "".join(f"{cell:>16s}" for cell in cells))

    print("\nstage medians, ms  (full rounds, then speculative rounds)")
    for kind in ("full", "spec"):
        for arm in ARMS:
            rows = [r for r in records if r.arm == arm and r.kind == kind]
            if not rows:
                continue
            names = list(rows[0].stages)
            cells = "  ".join(f"{name} {median([r.stages.get(name, 0.0) for r in rows]):.1f}" for name in names)
            print(f"  {kind:<5s} {arm:<9s} {cells}")

    spec_rows = [r for r in records if r.kind == "spec"]
    if spec_rows:
        accepted = [r.accepted for r in spec_rows]
        print(
            f"\nacceptance over {len(spec_rows)} speculative rounds: mean {np.mean(accepted):.2f}"
            f" of {spec_rows[0].eval_h}, zero in {sum(1 for a in accepted if a == 0)}"
            f", gripper cuts {sum(1 for r in spec_rows if r.gripper_cut)}"
            f", mean accept distance {np.mean([r.dist_mean for r in spec_rows]):.3f}"
        )
        print(f"staleness at execution: max {max(r.age_rounds for r in spec_rows)} rounds since the full round"
              f" (in frames this bundle strides by 50 -- see the module docstring)")
    drafts = [r.draft_rms for r in records if r.draft_rms is not None]
    if drafts:
        print(f"draft vs teacher, per-dim RMS on full rounds: mean {np.mean(drafts):.3f}"
              f"  (gate 3 wants <= 0.02 for tau >= 0.05)")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=10, help="teacher's Euler steps in a full round")
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    parser.add_argument("--reps", type=int, default=3, help="alternating repeats, the P1/P4 method")
    parser.add_argument("--ticks", type=int, default=20,
                        help="control ticks per arm per rep; keep it a multiple of --full-every + 1")
    parser.add_argument("--full-every", type=int, default=4, help="speculative rounds between full rounds")
    parser.add_argument("--fallback", choices=("off", "on"), default="off",
                        help="force a full round after a rejected draft; see the docstring before turning it on")
    parser.add_argument("--t-list", type=float, nargs="*", default=[0.10, 0.05], help="verify timesteps (K = len)")
    parser.add_argument("--tau", type=float, default=0.15, help="accept radius, per-dim RMS (gate 3: ~1.5x eps)")
    parser.add_argument("--max-exec-steps", type=int, default=12)
    parser.add_argument("--gripper-threshold", type=float, default=0.0)
    # These three exist so the baseline arm reproduces the 299.6 ms / 21.3 ms
    # per step figure the rest of Phase 10 quotes. That number comes from
    # `phase5_latency.py --compile-denoise-step`, whose own defaults are
    # eager/fp16 attention -- and the *model's* defaults are eager/fp32, which
    # costs 2.3x per step. Leaving them out would have produced a self-consistent
    # set of ratios next to absolute numbers nobody could reconcile.
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--attention-precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--compile-denoise-step", action="store_true",
                        help="compile predict_velocity; verify and denoise both go through it")
    parser.add_argument("--compile-backend", default="inductor")
    # phase5's own gate is 1e-3, which the compiled step misses at fp16 by a hair
    # (1.34e-3 measured here). `phase10_port_exactness.py` measured the verify
    # reconstruction `x0_hat = x_t - t*v_t` at **3.9e-3** in fp16, so a 1e-3 gate
    # on one step is stricter than the arithmetic it feeds.
    parser.add_argument("--compile-max-relative-error", type=float, default=2e-3)
    parser.add_argument("--score", action="store_true", help="also run a teacher round per spec tick for MAE")
    parser.add_argument("--json-out", default=None)
    add_draft_arguments(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # The worker first: the rendezvous has to happen before the 6B load, or the
    # iGPU process sits in a connect timeout while this one reads safetensors.
    remote = None
    if "igpu" in args.arms:
        remote = connect_worker(args, device=device, dtype=dtype, log=args.worker_log)
        print(f"[runtime] iGPU draft worker connected over {args.transport}; log {args.worker_log}")

    try:
        with np.load(Path(args.dataset).expanduser(), allow_pickle=False) as data:
            bundle = {key: data[key] for key in data.files}
        processor, model = build(Path(args.model), device, dtype, args.num_steps)
        model.qwenvl_with_expert.attention_backend = args.attention_backend
        model.qwenvl_with_expert.attention_precision = args.attention_precision
        print(f"[runtime] attention {args.attention_backend}/{args.attention_precision}")
        if args.compile_denoise_step:
            # Patches `model.predict_velocity`, so the verify path inherits it
            # without knowing: one compiled fixed-shape step serves both the
            # full round's ten and the speculative round's K.
            compile_denoise_step(
                processor, model, observation(bundle, 0), device, dtype, args.compile_backend,
                False, False, args.compile_max_relative_error,
            )
        dims, grippers = pose_dims(processor), gripper_dims(processor)
        real_dims = torch.as_tensor(sorted(dims + grippers), device=device, dtype=torch.long)
        print(f"[runtime] accept radius over {len(dims)} pose dims, gripper dims {grippers}")
        print(f"[runtime] K={len(args.t_list)} verify steps at {args.t_list}, tau {args.tau}, "
              f"full every {args.full_every} rounds, fallback {args.fallback}")
        if args.ticks % (args.full_every + 1):
            print(f"[warn] {args.ticks} ticks is not a whole number of {args.full_every + 1}-tick periods; "
                  "the measured per-tick mean will over-weight full rounds. Read the amortised table instead.")
        if args.draft_checkpoint is None:
            print("[runtime] draft head is RANDOM (--draft-checkpoint unset): latency is real, "
                  "acceptance is 0 by construction")

        local = None
        if "local" in args.arms:
            local = LocalDraft(
                build_head(device, dtype, d_model=args.d_model, seed=args.draft_seed,
                           checkpoint=args.draft_checkpoint),
                device,
            )

        def plan_for(arm: str) -> Plan:
            return Plan(
                arm=arm,
                backend={"baseline": None, "local": local, "igpu": remote}[arm],
                full_every=args.full_every,
                fallback=args.fallback == "on",
                t_list=list(args.t_list),
                tau=args.tau,
                max_exec_steps=args.max_exec_steps,
                num_steps=args.num_steps,
                dims=dims,
                grippers=grippers,
                gripper_threshold=args.gripper_threshold,
                score=args.score,
                real_dims=real_dims,
            )

        records: list[TickRecord] = []
        with torch.no_grad():
            # Warm up every arm outside the measurement: first-call allocation
            # and kernel autotuning would otherwise land entirely on whichever
            # arm runs first, which is exactly what the alternating reps exist
            # to rule out.
            for arm in args.arms:
                run_arm(arm, -1, model, processor, bundle, plan_for(arm), _replace_ticks(args, 2), device, dtype)
            for rep in range(args.reps):
                for arm in args.arms:  # arms alternate inside each rep, not across
                    started = time.perf_counter()
                    records.extend(run_arm(arm, rep, model, processor, bundle, plan_for(arm), args, device, dtype))
                    print(f"[runtime] rep {rep} arm {arm:<9s} {time.perf_counter() - started:6.1f} s", flush=True)

        report(records, args)
        if args.json_out:
            out = Path(args.json_out).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"args": vars(args), "records": [vars(r) for r in records]}, indent=2) + "\n")
            print(f"[out] {out}")
    finally:
        if remote is not None:
            remote.shutdown()
    return 0


def _replace_ticks(args, ticks: int):
    """A shallow copy of `args` with fewer ticks, for the warm-up pass."""
    clone = argparse.Namespace(**vars(args))
    clone.ticks = ticks
    return clone


if __name__ == "__main__":
    raise SystemExit(main())
