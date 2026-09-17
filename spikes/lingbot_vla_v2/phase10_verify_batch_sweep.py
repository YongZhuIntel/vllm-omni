#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What does a K-timestep verify actually cost, batched vs sequential?

§10.1 shipped the sequential verify and deferred the batched one to §10.2 with an
estimate and no measurement. The estimate was argued from the MoE roofline: at
``M = 51`` suffix tokens the expert GEMMs read 75.5 MB of weights and do 51 rows
of arithmetic against them, 51 FLOP/byte against this device's measured balance
of 207 (93.1 TFLOPS fp16 / 449 GB/s, ``phase8_moe_roofline.py``), so the compute
units idle ~75% of every layer-step and batching should spend that bubble for
free until ``M = 51*K`` crosses 207 at K=4. ``phase8_moe_gemm_probe.py`` measured
the GEMMs alone at 0.251 / 0.298 / 0.438 / 0.700 ms for M = 51 / 102 / 204 / 408.

This measures the whole thing -- ``SpecDecoder._verify``, the shipped object, on
the real model -- because the GEMMs are only 44% of a layer-step
(PHASE9_FLASHRT.md §1) and attention, the fused q/k/v, the router and the norms
scale differently. Reports per-round and per-timestep cost so the point where
batching stops paying is visible rather than predicted.

    ZE_AFFINITY_MASK=0 PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_verify_batch_sweep.py \
        --model /tmp/lingbot-vla-v2-perf --compile-denoise-step --k 1,2,4,8,10

**No iGPU and no draft worker needed**: the draft's content cannot change verify
cost, only its shape can, so the backend here is a stub returning zeros. Draft
cost is priced separately in ``phase10_igpu_draft_cost_probe.py``.

Two arms interleaved within each repetition, following P1/P4, so drift cannot
masquerade as the effect. ``--compile-denoise-step`` compiles
``predict_velocity`` ``dynamic=False``, so every distinct batch size is its own
specialisation: the sweep raises dynamo's cache limit and warms each arm
separately, and compile time is excluded from the reported medians.

Caveat to carry with any number out of this: K is a **conjunction** in
``radius_prefix_acceptance`` (``min`` over K), so a larger K lowers the accept
rate. This script prices K; it cannot tell you which K to want.
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

# Machine constants, both measured on this host; see the module docstring.
SUFFIX_TOKENS = 51
MACHINE_BALANCE = 207


def sync(device: torch.device) -> None:
    if device.type in ("xpu", "cuda"):
        torch.accelerator.synchronize()


class ZeroDraft:
    """A ``DraftBackend`` that is free and constant. Shape is all that matters here."""

    def __init__(self, chunk: torch.Tensor) -> None:
        self.chunk = chunk

    def refresh(self, prefix_embs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.chunk

    def draft(self, state: torch.Tensor) -> torch.Tensor:
        return self.chunk

    def close(self) -> None:
        pass


def t_list_for(k: int) -> list[float]:
    """K verify timesteps, spread over the near-terminal band.

    **Cost depends only on the count**, never on the values -- one
    ``predict_velocity`` at ``t=0.09`` costs what one at ``t=0.03`` costs -- so
    these are spaced, not tuned, and this helper prices K rather than choosing
    it. K=1 and K=2 return the shipped values so those two rows stay directly
    comparable to every number already recorded in PHASE10_SPECULATIVE.md.

    Imported by ``phase10_spec_accept_sweep.py`` so the two sweeps cannot end up
    pricing K at different timesteps.
    """
    if k == 1:
        return [0.05]
    if k == 2:
        return [0.10, 0.05]
    return [0.10 - index * (0.10 - 0.02) / (k - 1) for index in range(k)]


def build_decoder(model: Any, processor: Any, config: Any, device: torch.device, dtype: torch.dtype):
    from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import SpecDecoder

    chunk = torch.zeros((1, config.chunk_size, config.max_action_dim), device=device, dtype=dtype)
    return SpecDecoder(
        transformer=model,
        processor=processor,
        config=config,
        device=device,
        dtype=dtype,
        draft=ZeroDraft(chunk),
    )


def time_verify(
    decoder: Any, session: Any, state: torch.Tensor, noise: torch.Tensor, draft: torch.Tensor,
    *, device: torch.device, iters: int, warmup: int,
) -> list[float]:
    """Wall time of ``_verify`` alone, one sample per iteration."""
    for _ in range(warmup):
        decoder._verify(session, state, noise, draft)
    sync(device)

    samples = []
    for _ in range(iters):
        sync(device)
        start = time.perf_counter()
        decoder._verify(session, state, noise, draft)
        sync(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--k", default="1,2,4,8,10", help="verify timestep counts to price")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3, help="interleaved repetitions of every arm")
    parser.add_argument("--compile-denoise-step", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    ks = [int(value) for value in args.k.split(",") if value.strip()]
    if any(k < 1 for k in ks):
        parser.error("--k takes verify timestep counts, so every entry must be >= 1")
    # K=1 is the reference every reported ratio is against, and it doubles as the
    # per-step number, so it is measured whether or not it was asked for.
    ks = sorted(set(ks) | {1})
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)

    # Every distinct batch size is its own dynamo specialisation under
    # dynamic=False, and the default cache limit is smaller than this sweep.
    torch._dynamo.config.cache_size_limit = max(64, 4 * len(ks) + 8)

    processor, model = build(args.model, device, dtype, num_steps=None)
    obs = observation(processor.spec, args.seed)
    if args.compile_denoise_step:
        compile_denoise_step(processor, model, obs, device, dtype, "inductor", False, False, 2e-2)

    # One reference point outside the sweep: a single B=1 predict_velocity, which
    # is one Euler step, so every number below can be read in step units.
    arms: dict[tuple[int, bool], list[float]] = {}
    step_samples: list[float] = []

    print(f"[setup] device={device} dtype={args.dtype} compiled={args.compile_denoise_step}")
    print(f"[setup] iters={args.iters} warmup={args.warmup} repeats={args.repeats} K={ks}")

    for repeat in range(args.repeats):
        for k in ks:
            for batched in (False, True):
                if k == 1 and batched:
                    continue  # the batched path falls through to sequential at K=1
                config = dataclasses.replace(
                    model.config,
                    spec_decode=True,
                    spec_t_list=t_list_for(k),
                    spec_verify_batched=batched,
                )
                decoder = build_decoder(model, processor, config, device, dtype)
                # A real full round, so the session holds a real prefix KV cache.
                decoder.decode(obs, session_id="bench", reset=True, noise=None, num_steps=config.num_steps)
                session = decoder.sessions["bench"]

                state = processor.preprocess_state(obs).to(device=device, dtype=dtype)
                noise = torch.randn(
                    (1, config.chunk_size, config.max_action_dim), device=device, dtype=dtype
                )
                draft = torch.zeros_like(noise)

                samples = time_verify(
                    decoder, session, state, noise, draft,
                    device=device, iters=args.iters, warmup=args.warmup,
                )
                arms.setdefault((k, batched), []).extend(samples)

                if k == 1 and repeat == 0:
                    # Same call, B=1: the per-step reference.
                    step_samples = samples
                decoder.close()
        print(f"[rep {repeat + 1}/{args.repeats}] done")

    step_ms = statistics.median(step_samples) if step_samples else float("nan")
    print()
    print(f"one B=1 predict_velocity (== one Euler step): {step_ms:.2f} ms")
    print()
    header = (
        f"{'K':>3} {'M=51K':>6} {'AI':>5} {'regime':>14} {'sequential':>11} "
        f"{'batched':>9} {'speedup':>8} {'per-t':>7} {'vs K=1':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    k1 = statistics.median(arms[(1, False)])
    for k in ks:
        sequential = statistics.median(arms[(k, False)])
        batched = statistics.median(arms[(k, True)]) if (k, True) in arms else sequential
        intensity = SUFFIX_TOKENS * k
        regime = "memory-bound" if intensity < MACHINE_BALANCE else "compute-bound"
        if abs(intensity - MACHINE_BALANCE) <= SUFFIX_TOKENS // 2:
            regime = "~crossover"
        rows.append(
            {
                "k": k, "M": intensity, "sequential_ms": sequential, "batched_ms": batched,
                "speedup": sequential / batched, "per_timestep_ms": batched / k,
                "vs_k1": batched / k1, "regime": regime,
            }
        )
        print(
            f"{k:>3} {intensity:>6} {intensity:>5} {regime:>14} "
            f"{sequential:>10.2f}m {batched:>8.2f}m {sequential / batched:>7.2f}x "
            f"{batched / k:>6.2f}m {batched / k1:>6.2f}x"
        )

    print()
    print("Read the last column, not the speedup: it is what a larger K costs against K=1,")
    print("which is the only comparison the schedule cares about. The `speedup` column only")
    print("says batching beat the loop it replaced.")
    print()
    print("Reminder: `radius_prefix_acceptance` takes min over K, so raising K lowers the")
    print("accept rate. A cheap K is not a free K. See config.spec_verify_batched.")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "device": str(device), "dtype": args.dtype, "compiled": args.compile_denoise_step,
                    "iters": args.iters, "repeats": args.repeats,
                    "one_step_ms": step_ms, "rows": rows,
                },
                indent=2,
            )
        )
        print(f"\n[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
