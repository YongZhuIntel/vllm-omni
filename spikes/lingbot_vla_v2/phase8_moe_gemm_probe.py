# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Does a grouped top-4 MoE actually beat the dense batched GEMM at T=51?

The FLOP argument says top-4 is 2.95x less arithmetic. The counter-argument is
GEMM aspect ratio: dense gives every expert M=51 rows, grouped gives it
51*4/32 = 6.4. This measures both at the real shapes instead of arguing.

Per MoE layer per denoise step (E=32, H=768, I=512):
  gate/up: [E,M,H] x [E,H,I]     down: [E,M,I] x [E,I,H]
There are 36*10 = 360 such invocations per request.

The sweep also runs M above 51, which answers a second question: at batch B the
dense path sees M = 51*B, so the M=102/204/408 rows say what batching costs on
the arithmetic that dominates the loop.

Measurement note, learned by getting it wrong: the FIRST `bench()` call in a
process reads ~15% slow (M=51 measures 0.290 ms first, 0.264 ms after one prior
call, 0.252 ms once the sweep is warm) because oneDNN primitive setup for these
shapes is not fully amortised by five warmup iterations. A first pass at this
probe published the contaminated 0.290. Hence the discarded warm-up call below
and `iters=200`.
"""
import time, torch, torch.nn.functional as F

d, dt = torch.device("xpu"), torch.float16
E, H, I, T, K = 32, 768, 512, 51, 4
INV = 36 * 10

gate_w = torch.empty(E, H, I, device=d, dtype=dt).normal_(0, .02)
up_w   = torch.empty(E, H, I, device=d, dtype=dt).normal_(0, .02)
down_w = torch.empty(E, I, H, device=d, dtype=dt).normal_(0, .02)

def bench(m, iters=200):
    """One MoE layer-step at M rows per expert: gate, up, silu*, down."""
    x = torch.empty(E, m, H, device=d, dtype=dt).normal_()
    def step():
        g = torch.bmm(x, gate_w)
        u = torch.bmm(x, up_w)
        return torch.bmm(F.silu(g) * u, down_w)
    for _ in range(5):
        step()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        step()
    torch.xpu.synchronize()
    per = (time.perf_counter() - t) / iters
    flops = 2 * E * m * H * I * 3
    return per * 1e3, flops / per / 1e12, flops

print(f"{'M/expert':>9} {'ms/layer-step':>14} {'TFLOPS':>8} {'GFLOP':>7} {'x360 = loop ms':>15}")
print("-" * 60)
bench(51)  # discarded: absorbs first-call primitive setup, see module docstring
rows = {}
for m in (408, 204, 102, 51, 26, 16, 13, 8, 4, 2):
    ms, tf, fl = bench(m)
    rows[m] = ms
    tag = "  <- dense (all 32 experts see all 51 tokens)" if m == 51 else ""
    if m == 8:
        tag = "  <- grouped top-4, capacity 8 (avg need 6.4)"
    if m == 26:
        tag = "  <- half the chunk, e.g. one side of a dGPU/iGPU split"
    print(f"{m:>9} {ms:>14.3f} {tf:>8.2f} {fl/1e9:>7.2f} {ms*INV:>15.1f}{tag}")

print()
print(f"dense M=51 loop projection:        {rows[51]*INV:6.1f} ms")
print(f"grouped M=8  loop projection:      {rows[8]*INV:6.1f} ms  "
      f"({rows[51]/rows[8]:.2f}x faster on GEMMs alone)")
print(f"FLOP ratio 51/8:                   {51/8:.2f}x  <- what the FLOP argument promises")
print(f"measured full loop (F1):           201.4 ms   (GEMMs are only part of it)")
print(f"DRAM floor for MoE weights:         60.5 ms  <- fp16 roofline; the real target")
print(f"headroom to that floor:            {rows[51]*INV - 60.5:6.1f} ms  (step H, part 1)")
print()
print(f"batching, per request: B=1 {rows[51]:.3f}  B=2 {rows[102]/2:.3f}  "
      f"B=4 {rows[204]/4:.3f}  B=8 {rows[408]/8:.3f} ms/layer-step")
print(f"B=2 costs {rows[102]/rows[51]:.2f}x for 2x the tokens: memory-bound means batching is cheap")
print()
# The same flatness, read the other way, is why splitting the 50 action tokens
# across two devices cannot work (G5): weight bytes do not depend on how many
# rows read them, so half the rows is nowhere near half the time.
slope = (rows[51] - rows[13]) / (51 - 13)
print(f"half the chunk (M=26) costs {rows[26]/rows[51]*100:.0f}% of M=51, not 50%")
print(f"extrapolated M->0 intercept: {rows[51] - slope*51:.3f} ms = "
      f"{(rows[51] - slope*51)/rows[51]*100:.0f}% of the cost is token-count independent")
