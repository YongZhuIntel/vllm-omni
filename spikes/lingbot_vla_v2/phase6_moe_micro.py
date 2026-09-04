# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M6 step 3 — is the dense MoE slow because of the kernel or because of everything around it?

The OpenVINO reference runs the identical dense 32-expert math at 8.62 TFLOPS;
this port's denoise loop manages 3.02. That gap is either in the einsums
themselves or in the dispatch and glue around them, and those two have completely
different fixes. This isolates the einsums: no model, no weights, no loop --
just the three expert GEMMs at the exact shapes the action expert uses, timed
against the same operation expressed as plain matmuls.

``forward_dense`` writes the routed experts as

    gate = einsum("th,eih->tei", x, w_gate)     (51,768) x (32,512,768)
    up   = einsum("th,eih->tei", x, w_up)
    down = einsum("tei,ehi->teh", act, w_down)  (51,32,512) x (32,768,512)

The first two are one (51,768) @ (768, 32*512) GEMM if the weights are laid out
for it, which is a much friendlier shape than anything batched. Whether einsum
already finds that is the question.

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase6_moe_micro.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

# The action expert, from the prepared config: 36 layers, 51 suffix tokens,
# hidden 768, 32 routed experts of intermediate 512.
TOKENS = 51
HIDDEN = 768
EXPERTS = 32
INTERMEDIATE = 512
LAYERS = 36
STEPS = 10


def _sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device: torch.device, iters: int, warmup: int) -> float:
    """Median wall time of ``fn`` in milliseconds, synced on both sides."""
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(iters):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    x = torch.randn(TOKENS, HIDDEN, device=device, dtype=dtype)
    w_gate = torch.randn(EXPERTS, INTERMEDIATE, HIDDEN, device=device, dtype=dtype)
    w_up = torch.randn(EXPERTS, INTERMEDIATE, HIDDEN, device=device, dtype=dtype)
    w_down = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, device=device, dtype=dtype)
    routing = torch.rand(TOKENS, EXPERTS, device=device, dtype=dtype)

    # Same weights, laid out so the "eih" contraction is a single 2-D GEMM.
    w_gate_flat = w_gate.reshape(EXPERTS * INTERMEDIATE, HIDDEN).t().contiguous()
    w_up_flat = w_up.reshape(EXPERTS * INTERMEDIATE, HIDDEN).t().contiguous()
    # For the down projection the expert axis cannot be folded away -- each
    # expert contracts its own intermediate -- so the honest alternative is bmm.
    w_down_bmm = w_down.transpose(1, 2).contiguous()  # (E, I, H)

    def einsum_gate():
        return torch.einsum("th,eih->tei", x, w_gate)

    def gemm_gate():
        return (x @ w_gate_flat).view(TOKENS, EXPERTS, INTERMEDIATE)

    def einsum_down():
        act = torch.empty(TOKENS, EXPERTS, INTERMEDIATE, device=device, dtype=dtype)
        return torch.einsum("tei,ehi->teh", act, w_down)

    act = torch.randn(TOKENS, EXPERTS, INTERMEDIATE, device=device, dtype=dtype)

    def einsum_down_real():
        return torch.einsum("tei,ehi->teh", act, w_down)

    def bmm_down():
        return torch.bmm(act.transpose(0, 1), w_down_bmm).transpose(0, 1)

    def einsum_combine():
        y = torch.randn(1, device=device, dtype=dtype)  # placeholder, replaced below
        return y

    out = torch.randn(TOKENS, EXPERTS, HIDDEN, device=device, dtype=dtype)

    def combine_einsum():
        return torch.einsum("teh,te->th", out, routing)

    def combine_mul():
        return (out * routing.unsqueeze(-1)).sum(dim=1)

    def full_dense_einsum():
        g = torch.einsum("th,eih->tei", x, w_gate)
        u = torch.einsum("th,eih->tei", x, w_up)
        h = torch.nn.functional.silu(g) * u
        y = torch.einsum("tei,ehi->teh", h, w_down)
        return torch.einsum("teh,te->th", y, routing)

    def full_dense_gemm():
        g = (x @ w_gate_flat).view(TOKENS, EXPERTS, INTERMEDIATE)
        u = (x @ w_up_flat).view(TOKENS, EXPERTS, INTERMEDIATE)
        h = torch.nn.functional.silu(g) * u
        y = torch.bmm(h.transpose(0, 1), w_down_bmm).transpose(0, 1)
        return (y * routing.unsqueeze(-1)).sum(dim=1)

    del einsum_down, einsum_combine  # written for clarity, superseded above

    gate_flops = 2 * TOKENS * HIDDEN * EXPERTS * INTERMEDIATE
    down_flops = 2 * TOKENS * EXPERTS * INTERMEDIATE * HIDDEN
    full_flops = 3 * gate_flops  # gate + up + down; combine is negligible

    cases = [
        ("gate  einsum th,eih->tei", einsum_gate, gate_flops),
        ("gate  x @ W.T then view", gemm_gate, gate_flops),
        ("down  einsum tei,ehi->teh", einsum_down_real, down_flops),
        ("down  bmm", bmm_down, down_flops),
        ("comb  einsum teh,te->th", combine_einsum, 2 * TOKENS * EXPERTS * HIDDEN),
        ("comb  mul + sum", combine_mul, 2 * TOKENS * EXPERTS * HIDDEN),
        ("FULL  all-einsum (current)", full_dense_einsum, full_flops),
        ("FULL  gemm + bmm", full_dense_gemm, full_flops),
    ]

    print(f"device={device} dtype={args.dtype}  tokens={TOKENS} hidden={HIDDEN} "
          f"experts={EXPERTS} intermediate={INTERMEDIATE}")
    print(f"{'case':32s} {'ms':>9s} {'TFLOPS':>9s} {'x36x10 ms':>11s}")
    print("-" * 66)
    for label, fn, flops in cases:
        ms = bench(fn, device, args.iters, args.warmup)
        tflops = flops / (ms * 1e-3) / 1e12
        print(f"{label:32s} {ms:9.4f} {tflops:9.2f} {ms * LAYERS * STEPS:11.1f}")

    # First-principles correctness check: the two full paths must agree.
    a = full_dense_einsum().float()
    b = full_dense_gemm().float()
    print(f"\nmax |einsum - gemm| = {(a - b).abs().max().item():.3e} "
          f"(rel {(a - b).abs().max().item() / a.abs().max().item():.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
