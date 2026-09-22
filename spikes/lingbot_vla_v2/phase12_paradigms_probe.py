#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 12 — ParaDiGMS (Picard parallel sampling), on one card and on two.

Phase 10 removes the ten sequential Euler steps by **guessing the endpoint**: a
draft head proposes ``x0`` and the teacher checks it at K near-terminal ``t`` in
one batched forward (30.6 ms at K=2 against the loop's 213.5 ms).

ParaDiGMS (Shih et al., *Parallel Sampling of Diffusion Models*) removes them a
different way and guesses nothing: hold **all ten points of the trajectory at
once** and refine them together by Picard iteration. With ``x_0`` fixed at the
noise and ``dt = -1/P``,

    x_{j+1}^{m+1} = x_0 + dt * sum_{i<=j} v(x_i^m, t_i)

Every ``v`` in a sweep is independent, so one sweep is one batched forward at
B=P. Sweep m leaves ``x_0..x_m`` exactly equal to the sequential solution -- a
proof by induction on j, and this probe's self-check, which holds to 2-6e-4 for
every ``m < P``. So P sweeps always converge and the entire question is **how
many sweeps it actually takes**. (At ``m = P`` the check reads 1.3e-2 instead:
that is not a failure, it is the two integrators' accumulation orders parting
company in fp16, and §M §2 unpicks it.)

That number is the one thing the existing spikes never measured, and it decides
the scheme on its own:

    one sweep at P=10, batched     90.5 ms   (config.spec_verify_batched table)
    ten sequential Euler steps    213.5 ms   (§11 full round 294.3 - grounding 80.8)
    -> break-even at              2.36 sweeps

So ParaDiGMS has to converge in **two sweeps** to be worth anything at all, and
in one to be interesting. Arm ``picard`` measures it against the shipped model.

The second half of the question is the user's: **split the P points across the
iGPU and the dGPU**. Three facts already on file bracket it, and none of them
settles it:

* §L: the iGPU runs the real MoE layer-step at 3.1687 ms, so 36 layers x 10
  steps is ~1140 ms of MoE alone against the dGPU's 98 -- but that is MoE only,
  not a whole ``predict_velocity``, so it is a **lower bound** on the iGPU's
  share cost. Arm ``step-cost`` measures the whole thing, once per device.
* §K: a sustained iGPU load costs the dGPU request **1.74-1.76x**. Splitting a
  sweep across two cards *requires* them to run concurrently, so this tax is not
  avoidable overhead, it is the scheme's structural cost. But §K measured it
  with a 2048^2 matmul, which is EU-bound, and denoise is memory-bound -- so the
  rate has to be re-measured at this shape. Arm ``contention`` does that.
* §K1: torch-xpu enumerates one Level-Zero platform per process, so "split
  across two cards" is a process boundary, exactly as in Phase 10's draft.

**Results are in PHASE8_LATENCY_PARITY.md §M, and both "no"s are decisive:**
Picard converges in 9 sweeps against a 2.19 break-even (0.24x), so it loses on
the dGPU alone; and the iGPU's 233 ms for one point exceeds the dGPU's 153 ms
for all ten, so the optimal split is zero points on the iGPU. The contention tax
that was expected to decide it measures **1.01x** -- §K's 1.74-1.76x is an
EU-occupancy effect and a memory-bound iGPU workload does not trigger it.

Arms, and the device each wants::

    ZE_AFFINITY_MASK=0 python phase12_paradigms_probe.py --arm picard     --model ...
    ZE_AFFINITY_MASK=0 python phase12_paradigms_probe.py --arm step-cost  --model ...
    ZE_AFFINITY_MASK=1 python phase12_paradigms_probe.py --arm step-cost  --model ...
    ZE_AFFINITY_MASK=0 python phase12_paradigms_probe.py --arm contention --model ...

``contention`` spawns its own iGPU-side load generator (``--arm load``), so it
is the only arm that touches both cards.

Host state matters (Rules #2): ``load < 2.0`` and no other container holding a
GPU. §L's absolute numbers swung 2.4x under a busy host and had to be thrown
away; every arm here prints the load average it started at for the same reason.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase5_latency import build, compile_denoise_step, observation  # noqa: E402

# PHASE10_SPECULATIVE.md §10: the fp16 noise floor of the x0 reconstruction, in
# per-dim RMS units. Nothing in this model is meaningful below it, so it is the
# convergence threshold for "Picard has reproduced the sequential answer".
FP16_FLOOR = 3.9e-3
# config.spec_tau -- the accept radius Phase 10 ships. A looser, task-level
# threshold: "close enough that the accept rule would not have noticed".
SPEC_TAU = 0.15


def sync(device: torch.device) -> None:
    if device.type in ("xpu", "cuda"):
        torch.accelerator.synchronize()


def load_average() -> float:
    return os.getloadavg()[0]


class ZeroDraft:
    """A ``DraftBackend`` that is free and constant -- no arm here drafts."""

    def __init__(self, chunk: torch.Tensor) -> None:
        self.chunk = chunk

    def refresh(self, prefix_embs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.chunk

    def draft(self, state: torch.Tensor) -> torch.Tensor:
        return self.chunk

    def close(self) -> None:
        pass


def ground(model: Any, processor: Any, obs: dict, device: torch.device, dtype: torch.dtype):
    """One real full round, so the session holds a real prefix KV cache.

    Returns ``(decoder, session, state)``. Uses the shipped ``SpecDecoder``
    rather than a hand-rolled prefix fill so the KV under test is the one the
    product builds -- same reason ``phase10_verify_batch_sweep.py`` does.
    """
    import dataclasses

    from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import SpecDecoder

    config = dataclasses.replace(model.config, spec_decode=True)
    chunk = torch.zeros((1, config.chunk_size, config.max_action_dim), device=device, dtype=dtype)
    decoder = SpecDecoder(
        transformer=model,
        processor=processor,
        config=config,
        device=device,
        dtype=dtype,
        draft=ZeroDraft(chunk),
    )
    decoder.decode(obs, session_id="bench", reset=True, noise=None, num_steps=config.num_steps)
    state = processor.preprocess_state(obs).to(device=device, dtype=dtype)
    return decoder, decoder.sessions["bench"], state


def real_dims(processor: Any, device: torch.device) -> torch.Tensor:
    """Every real action slot -- pose and gripper, padding excluded.

    Matches ``SpecDecoder._draft_rms`` so an RMS here is in the same units as
    every accuracy number already recorded in PHASE10_SPECULATIVE.md.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import gripper_dims, pose_dims

    dims = sorted(pose_dims(processor) + gripper_dims(processor))
    return torch.as_tensor(dims, device=device, dtype=torch.long)


def rms(a: torch.Tensor, b: torch.Tensor, dims: torch.Tensor, horizon: int) -> float:
    """Per-dim RMS over the executed horizon, in ``spec_tau``'s units."""
    delta = (a - b)[:, :horizon].index_select(-1, dims).to(torch.float32)
    return float(delta.pow(2).mean().sqrt().item())


# ---------------------------------------------------------------------------
# The two integrators
# ---------------------------------------------------------------------------
def sequential_euler(
    model: Any, session: Any, state: torch.Tensor, noise: torch.Tensor, num_steps: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """``denoise_actions``, opened up so the whole trajectory is observable.

    Reproduces its arithmetic exactly -- ``dt`` and ``time`` as tensors in the
    model dtype, ``time`` accumulated by repeated addition rather than recomputed
    -- because Picard has to be scored against the trajectory the product
    actually produces, fp16 accumulation included.

    Returns ``(trajectory, timesteps)``; the trajectory has ``num_steps + 1``
    entries and ``timesteps`` the ``num_steps`` values the loop passed in.
    """
    dtype, device, bsize = state.dtype, state.device, int(state.shape[0])
    dt = torch.tensor(-1.0 / num_steps, dtype=dtype, device=device)
    now = torch.tensor(1.0, dtype=dtype, device=device)

    x_t = noise
    traj, times = [x_t], []
    for _ in range(num_steps):
        times.append(now.clone())
        v_t = model.predict_velocity(
            state=state,
            prefix_pad_masks=session.pad_masks,
            prefix_position_ids=session.position_ids,
            past_key_values=session.past_key_values,
            x_t=x_t,
            timestep=now.expand(bsize),
        )
        x_t = x_t + dt * v_t
        traj.append(x_t)
        now = now + dt
    return traj, times


def expand_session(session: Any, state: torch.Tensor, p: int) -> dict[str, Any]:
    """Conditioning for a B=P sweep, built **once** and reused across sweeps.

    The expansion copies the prefix KV P times (42.2 MB * P, so 422 MB at P=10),
    and a real ParaDiGMS implementation would hoist it out of the sweep loop just
    like this. Hoisting is the scheme's best case, which is what it should be
    measured at.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import _expand_rows

    return {
        "state": _expand_rows(state, p),
        "prefix_pad_masks": _expand_rows(session.pad_masks, p),
        "prefix_position_ids": _expand_rows(session.position_ids, p, dim=1),
        "past_key_values": [
            (_expand_rows(key, p), _expand_rows(value, p)) for key, value in session.past_key_values
        ],
    }


def picard_sweep(
    model: Any, cond: dict[str, Any], xs: list[torch.Tensor], times: torch.Tensor, dt: torch.Tensor
) -> list[torch.Tensor]:
    """One ParaDiGMS sweep: P velocities in one forward, then a prefix sum.

    ``xs`` is ``x_0..x_{P-1}`` (the points a velocity is needed at); the return
    is the refreshed ``x_1..x_P``, with ``x_0`` unchanged by definition.

    The prefix sum runs in fp32. The sequential loop accumulates ``x`` step by
    step in fp16 while this accumulates ``v`` and multiplies once, so the two
    round differently no matter what; doing the reduction in fp32 keeps that
    difference at the fp16 storage floor instead of adding to it.
    """
    p = len(xs)
    x_batch = torch.cat(xs, dim=0)
    v = model.predict_velocity(**cond, x_t=x_batch, timestep=times)
    cum = torch.cumsum(v.to(torch.float32), dim=0)
    tail = xs[0].to(torch.float32) + dt.to(torch.float32) * cum
    return [tail[j : j + 1].to(xs[0].dtype) for j in range(p)]


# ---------------------------------------------------------------------------
# Arm: picard
# ---------------------------------------------------------------------------
def arm_picard(args: argparse.Namespace, processor: Any, model: Any, obs: dict,
               device: torch.device, dtype: torch.dtype) -> dict:
    p = args.num_steps or model.config.num_steps
    decoder, session, state = ground(model, processor, obs, device, dtype)
    dims = real_dims(processor, device)
    horizon = model.config.spec_max_exec_steps

    noise = torch.randn((1, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype)
    dt = torch.tensor(-1.0 / p, dtype=dtype, device=device)

    # --- correctness ------------------------------------------------------
    ref_traj, ref_times = sequential_euler(model, session, state, noise, p)
    ref = ref_traj[-1]
    times = torch.cat([t.reshape(1) for t in ref_times])

    cond = expand_session(session, state, p)

    # Convergence is measured **against the previous iterate**, not against the
    # shipped trajectory, and two earlier attempts at a reference got it wrong
    # in ways worth recording:
    #
    #   * scored against the fp16 Euler answer, Picard's error falls to 1.3e-3
    #     at sweep 7 and then *rises* to 1.3e-2 and sits there. That looked like
    #     divergence; it is the iterate passing near the Euler answer on its way
    #     to its own fixed point.
    #   * one sweep off the exact trajectory is not that fixed point either.
    #     Picard's iterates are fp16 points reached by an fp32 prefix sum, the
    #     Euler loop's are fp16 points reached by step-by-step fp16 addition, so
    #     the two self-consistent solutions genuinely differ.
    #
    # ``delta_rms`` needs no reference and is the standard criterion for a
    # fixed-point iteration. ``final_rms`` stays as the *quality* column: how far
    # the converged Picard answer lands from what the robot gets today.
    xs = [noise.clone() for _ in range(p)]  # trivial init: every point at the noise
    rows, previous = [], None
    for sweep in range(1, args.max_sweeps + 1):
        tail = picard_sweep(model, cond, xs, times, dt)
        xs = [noise] + tail[: p - 1]
        final_err = rms(tail[-1], ref, dims, horizon)
        delta = float("inf") if previous is None else rms(tail[-1], previous, dims, horizon)
        previous = tail[-1]
        # The induction property, this arm's self-check: after sweep m, x_m is
        # the sequential x_m. Holds for m < P; at m = P the two integrators'
        # accumulation orders have diverged by more than the fp16 floor, which
        # is the 1.3e-2 the note above is about.
        exact_err = rms(tail[sweep - 1], ref_traj[sweep], dims, horizon) if sweep <= p else float("nan")
        rows.append({"sweep": sweep, "delta_rms": delta, "final_rms": final_err, "induction_rms": exact_err})
        print(f"[picard] sweep {sweep:>2}  delta_rms={delta:.3e}  "
              f"final_rms={final_err:.3e}  induction_rms={exact_err:.3e}")

    def sweeps_to(threshold: float, key: str = "final_rms") -> int | None:
        for row in rows:
            if row[key] <= threshold:
                return row["sweep"]
        return None

    # --- cost -------------------------------------------------------------
    def time_sweep() -> float:
        local = [noise.clone() for _ in range(p)]
        sync(device)
        start = time.perf_counter()
        picard_sweep(model, cond, local, times, dt)
        sync(device)
        return (time.perf_counter() - start) * 1e3

    def time_sequential() -> float:
        sync(device)
        start = time.perf_counter()
        sequential_euler(model, session, state, noise, p)
        sync(device)
        return (time.perf_counter() - start) * 1e3

    for _ in range(args.warmup):
        time_sweep()
        time_sequential()

    sweep_ms, seq_ms = [], []
    for _ in range(args.iters):  # interleaved, so drift cannot become the effect
        sweep_ms.append(time_sweep())
        seq_ms.append(time_sequential())
    sweep_med, seq_med = statistics.median(sweep_ms), statistics.median(seq_ms)
    breakeven = seq_med / sweep_med

    need_floor = sweeps_to(FP16_FLOOR, "delta_rms")
    need_tau = sweeps_to(SPEC_TAU)
    quality = next((r["final_rms"] for r in rows if r["sweep"] == need_floor), float("nan"))
    print()
    print(f"one Picard sweep (B={p}, batched)   {sweep_med:8.2f} ms")
    print(f"{p} sequential Euler steps          {seq_med:8.2f} ms")
    print(f"break-even                         {breakeven:8.2f} sweeps")
    print(f"sweeps to converged  (delta_rms <= {FP16_FLOOR:.1e})  {need_floor}")
    print(f"sweeps to spec_tau   (final_rms <= {SPEC_TAU:.2f})  {need_tau}")
    print(f"converged answer vs shipped trajectory: {quality:.3e} "
          f"({'inside' if quality <= SPEC_TAU else 'OUTSIDE'} spec_tau)")
    for label, need in (("converged", need_floor), ("spec_tau", need_tau)):
        if need is None:
            print(f"  -> {label}: not reached in {args.max_sweeps} sweeps")
        else:
            total = need * sweep_med
            verdict = "FASTER" if total < seq_med else "SLOWER"
            print(f"  -> {label}: {need} x {sweep_med:.1f} = {total:.1f} ms, {seq_med / total:.2f}x -- {verdict}")

    decoder.close()
    return {
        "arm": "picard", "p": p, "rows": rows,
        "converged_vs_shipped_rms": quality,
        "sweep_ms": sweep_med, "sequential_ms": seq_med, "breakeven_sweeps": breakeven,
        "sweeps_to_converged": need_floor, "sweeps_to_spec_tau": need_tau,
    }


# ---------------------------------------------------------------------------
# Arm: step-cost  (run once per device)
# ---------------------------------------------------------------------------
def arm_step_cost(args: argparse.Namespace, processor: Any, model: Any, obs: dict,
                  device: torch.device, dtype: torch.dtype) -> dict:
    """Eager ``predict_velocity`` at a few batch sizes, on whichever card is visible.

    Eager on purpose: the iGPU arm is the comparison, and compiling there is
    neither cheap nor obviously representative. Run the same command under
    ``ZE_AFFINITY_MASK=0`` and ``=1`` and read the ratio, the way §L's probe does.
    """
    decoder, session, state = ground(model, processor, obs, device, dtype)
    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    rows = []
    for b in batches:
        cond = expand_session(session, state, b)
        x_t = torch.randn(
            (b, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype
        )
        timestep = torch.full((b,), 0.5, device=device, dtype=dtype)

        def once() -> None:
            model.predict_velocity(**cond, x_t=x_t, timestep=timestep)

        for _ in range(args.warmup):
            once()
        sync(device)
        samples = []
        for _ in range(args.iters):
            sync(device)
            start = time.perf_counter()
            once()
            sync(device)
            samples.append((time.perf_counter() - start) * 1e3)
        median = statistics.median(samples)
        rows.append({"batch": b, "ms": median, "per_point_ms": median / b})
        print(f"[step-cost] B={b:>3}  {median:8.2f} ms  ({median / b:7.2f} ms/point)")

    decoder.close()
    return {"arm": "step-cost", "affinity": os.environ.get("ZE_AFFINITY_MASK", "unset"), "rows": rows}


# ---------------------------------------------------------------------------
# Arm: load  (internal -- the iGPU side of `contention`)
# ---------------------------------------------------------------------------
def arm_load(args: argparse.Namespace, processor: Any, model: Any, obs: dict,
             device: torch.device, dtype: torch.dtype) -> dict:
    """Run real denoise on this card until killed. Signals readiness by file.

    The load has to be the **real** workload: §K measured contention with a
    2048^2 matmul, which saturates the EU array, and established that the tax is
    a function of sustained occupancy. Denoise is memory-bound, so whether it
    taxes the dGPU the same way is exactly what is unknown.
    """
    decoder, session, state = ground(model, processor, obs, device, dtype)
    noise = torch.randn((1, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype)
    p = args.num_steps or model.config.num_steps

    Path(args.ready_file).write_text(str(os.getpid()))
    print(f"[load] ready on affinity={os.environ.get('ZE_AFFINITY_MASK', 'unset')}, denoising until killed")
    while True:
        sequential_euler(model, session, state, noise, p)
        sync(device)


# ---------------------------------------------------------------------------
# Arm: contention
# ---------------------------------------------------------------------------
def arm_contention(args: argparse.Namespace, processor: Any, model: Any, obs: dict,
                   device: torch.device, dtype: torch.dtype) -> dict:
    """dGPU sweep cost with the iGPU idle vs running real denoise.

    Blocks are interleaved idle/loaded/idle rather than measured back to back,
    so a thermal or host drift shows up as idle-vs-idle disagreement instead of
    being attributed to the load.
    """
    p = args.num_steps or model.config.num_steps
    decoder, session, state = ground(model, processor, obs, device, dtype)
    noise = torch.randn((1, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype)
    dt = torch.tensor(-1.0 / p, dtype=dtype, device=device)
    _, ref_times = sequential_euler(model, session, state, noise, p)
    times = torch.cat([t.reshape(1) for t in ref_times])
    cond = expand_session(session, state, p)

    def time_sweep() -> float:
        local = [noise.clone() for _ in range(p)]
        sync(device)
        start = time.perf_counter()
        picard_sweep(model, cond, local, times, dt)
        sync(device)
        return (time.perf_counter() - start) * 1e3

    def block() -> float:
        for _ in range(args.warmup):
            time_sweep()
        return statistics.median([time_sweep() for _ in range(args.iters)])

    ready = Path(args.ready_file)
    ready.unlink(missing_ok=True)

    idle_before = block()
    print(f"[contention] iGPU idle      {idle_before:8.2f} ms")

    env = dict(os.environ, ZE_AFFINITY_MASK=args.load_affinity)
    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--arm", "load",
         "--model", str(args.model), "--dtype", args.dtype, "--ready-file", str(ready)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + args.load_timeout
        while not ready.exists():
            if child.poll() is not None:
                raise RuntimeError(f"load generator exited early with code {child.returncode}")
            if time.time() > deadline:
                raise RuntimeError(f"load generator not ready after {args.load_timeout}s")
            time.sleep(1.0)
        print(f"[contention] load generator up on affinity={args.load_affinity}")
        loaded = block()
        print(f"[contention] iGPU denoising {loaded:8.2f} ms")
    finally:
        child.terminate()
        child.wait(timeout=30)
        ready.unlink(missing_ok=True)

    idle_after = block()
    print(f"[contention] iGPU idle again{idle_after:8.2f} ms")

    idle = statistics.median([idle_before, idle_after])
    drift = abs(idle_after - idle_before) / idle_before
    print()
    print(f"tax {loaded / idle:.2f}x   (idle drift between the two idle blocks: {drift:.1%})")
    if drift > 0.05:
        print("WARNING: the two idle blocks disagree by >5%. The host moved; do not quote this tax.")

    decoder.close()
    return {
        "arm": "contention", "idle_before_ms": idle_before, "loaded_ms": loaded,
        "idle_after_ms": idle_after, "tax": loaded / idle, "idle_drift": drift,
    }


ARMS = {"picard": arm_picard, "step-cost": arm_step_cost, "contention": arm_contention, "load": arm_load}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num-steps", type=int, default=None, help="P; defaults to config.num_steps")
    parser.add_argument("--max-sweeps", type=int, default=10, help="picard: P sweeps always converge")
    parser.add_argument("--batches", default="1,2,4,10", help="step-cost: batch sizes to price")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-denoise-step", action="store_true")
    parser.add_argument("--load-affinity", default="1", help="contention: the load generator's card")
    parser.add_argument("--load-timeout", type=float, default=900.0)
    parser.add_argument("--ready-file", default="/tmp/lingbot-paradigms-load.ready")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    print(f"[setup] arm={args.arm} affinity={os.environ.get('ZE_AFFINITY_MASK', 'unset')} "
          f"device={device} dtype={args.dtype} load1m={load_average():.2f}")
    if load_average() > 2.0 and args.arm != "load":
        print("WARNING: load average above 2.0. Rules #2 -- absolute numbers from this run are suspect.")

    torch.manual_seed(args.seed)
    processor, model = build(args.model, device, dtype, num_steps=args.num_steps)
    obs = observation(processor.spec, args.seed)
    if args.compile_denoise_step:
        # B=1 and B=P are two dynamic=False specialisations; give dynamo room.
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        compile_denoise_step(processor, model, obs, device, dtype, "inductor", False, False, 2e-2)

    with torch.no_grad():
        result = ARMS[args.arm](args, processor, model, obs, device, dtype)

    result["load1m"] = load_average()
    result["dtype"] = args.dtype
    result["compiled"] = args.compile_denoise_step
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"\n[json] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
