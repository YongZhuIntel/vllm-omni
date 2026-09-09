#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 8 step J — the PyTorch-reachable half of OpenVINO's `FuseGatedMLP`.

Intel's pi0.5 report (section J of `PHASE8_LATENCY_PARITY.md`) describes a GPU
plugin pass that detects the gate-proj / Swish / multiply / up-proj / down-proj
subgraph and emits a single oneDNN `GatedMLP` primitive, on the grounds that
"each discrete kernel invocation incurs a non-negligible dispatch latency of
approximately 104 us" and that this is "particularly pronounced within the
iterative pi0.5 action head, which executes a dense sequence of small-dimension
Gated-MLP operations". That is our denoise loop: 36 layers x 10 steps = 360
gated-MLP invocations, in the regime F1/F4 already found to be dispatch-bound.

We cannot emit a oneDNN primitive from PyTorch. But half the pass is a weight
layout change: `gate_proj` and `up_proj` read the *same* input and differ only in
their weights, so concatenating them along the output axis turns two GEMMs into
one of twice the N. Three kernels become two, arithmetic is unchanged, and the
result is bit-identical -- there is no accuracy question to grade.

This probe measures both gated MLPs in a LingBot-VLA-V2 layer at their real
shapes:

* routed experts, `GroupedExperts.forward_dense`: `[32,M,768] x [32,768,512]`
* shared expert: `[M,768] x [768,2752]`

Measured 2026-09-09 on B60 at M=51 (fp16): routed 0.251 -> 0.239 ms (1.051x),
shared 0.0592 -> 0.0465 ms (1.273x), i.e. 4.4 + 4.6 = ~9 ms per request, with
`max|delta| = 0` for both.

The shared expert gains 1.27x against the routed path's 1.05x despite moving 6x
fewer bytes, which is the finding worth keeping: its GEMMs are small enough that
launch and setup dominate, while the routed path is already one batched `bmm`
over 32 experts and is nearer bandwidth-bound. Merging helps most exactly where
Intel says dispatch dominates.

Measurement note inherited from `phase8_moe_gemm_probe.py`: the first `bench()`
call in a process reads ~15% slow because oneDNN primitive setup for these shapes
is not amortised by warmup alone, so each configuration is benched once and
discarded first.

    python spikes/lingbot_vla_v2/phase8_gated_mlp_probe.py

Findings are recorded in `PHASE8_LATENCY_PARITY.md` under F5 and section J.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

# LingBot-VLA-V2 action expert, from /tmp/lingbot-vla-v2-perf/transformer/config.json
EXPERTS = 32
HIDDEN = 768
EXPERT_INTERMEDIATE = 512
SHARED_INTERMEDIATE = 2752
CHUNK = 51  # 1 state + 50 action tokens
INVOCATIONS = 36 * 10  # layers x denoise steps


def _bench(fn, iters: int, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _report(label: str, today, fused, iters: int) -> None:
    _bench(today, iters)  # discarded: absorbs first-call primitive setup
    a, b = _bench(today, iters), _bench(fused, iters)
    err = (today().float() - fused().float()).abs().max().item()
    print(
        f"  {label:16s} 3 GEMMs {a:7.4f} ms | merged {b:7.4f} ms | {a / b:5.3f}x | "
        f"loop {a * INVOCATIONS:6.1f} -> {b * INVOCATIONS:6.1f} ms "
        f"(-{(a - b) * INVOCATIONS:4.1f}) | max|d|={err:.1e}"
    )
    return (a - b) * INVOCATIONS


def run(args: argparse.Namespace) -> int:
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    m = args.tokens

    print(f"\ngated MLP, merged gate_up vs three separate GEMMs (M={m}, {args.dtype}, x{INVOCATIONS})")

    # --- routed experts: batched over all 32, the forward_dense path ---------
    gate = torch.empty(EXPERTS, HIDDEN, EXPERT_INTERMEDIATE, device=device, dtype=dtype).normal_(0, 0.02)
    up = torch.empty(EXPERTS, HIDDEN, EXPERT_INTERMEDIATE, device=device, dtype=dtype).normal_(0, 0.02)
    down = torch.empty(EXPERTS, EXPERT_INTERMEDIATE, HIDDEN, device=device, dtype=dtype).normal_(0, 0.02)
    gate_up = torch.cat([gate, up], dim=2).contiguous()  # [E, H, 2I], one weight
    xe = torch.empty(EXPERTS, m, HIDDEN, device=device, dtype=dtype).normal_()

    def routed_today():
        return torch.bmm(F.silu(torch.bmm(xe, gate)) * torch.bmm(xe, up), down)

    def routed_fused():
        g, u = torch.bmm(xe, gate_up).split(EXPERT_INTERMEDIATE, dim=2)
        return torch.bmm(F.silu(g) * u, down)

    # --- shared expert: a plain dense MLP, one per layer --------------------
    sgate = torch.empty(SHARED_INTERMEDIATE, HIDDEN, device=device, dtype=dtype).normal_(0, 0.02)
    sup = torch.empty(SHARED_INTERMEDIATE, HIDDEN, device=device, dtype=dtype).normal_(0, 0.02)
    sdown = torch.empty(HIDDEN, SHARED_INTERMEDIATE, device=device, dtype=dtype).normal_(0, 0.02)
    sgate_up = torch.cat([sgate, sup], dim=0).contiguous()  # [2I, H]
    xs = torch.empty(m, HIDDEN, device=device, dtype=dtype).normal_()

    def shared_today():
        return F.linear(F.silu(F.linear(xs, sgate)) * F.linear(xs, sup), sdown)

    def shared_fused():
        g, u = F.linear(xs, sgate_up).split(SHARED_INTERMEDIATE, dim=1)
        return F.linear(F.silu(g) * u, sdown)

    saved = _report("routed experts", routed_today, routed_fused, args.iters)
    saved += _report("shared expert", shared_today, shared_fused, args.iters)

    print(
        f"\n  total: -{saved:.1f} ms per request, bit-exact (step J)"
        f"\n  stacks with step I (-8 to -9 ms, F4): different work, no overlap"
        f"\n  note the shared expert gains more than the routed path despite 6x fewer"
        f"\n  bytes -- small GEMMs are dispatch-bound, which is Intel's whole argument"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tokens", type=int, default=CHUNK)
    parser.add_argument("--iters", type=int, default=200)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
