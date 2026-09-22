# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 11 — can denoise run on the iGPU if the data is kept cache-resident?

Results and verdict: `PHASE8_LATENCY_PARITY.md` §L. Short version: the iGPU's
16 MiB cache is worth up to 10.5x and 265 GB/s against its 29 GB/s DRAM, and
the denoise loop can collect none of it — the layer-step working set is 4.5x
the cache, and tiling it creates no reuse to hold.


The premise under test: §K measured the iGPU's DRAM read bandwidth at 29 GB/s
(15.4x below the B60) and §F2 established the denoise loop is memory-bound on
expert weights, so if the working set could be held in the iGPU's own caches or
walked in small enough tiles, the 29 GB/s wall would stop mattering.

`clinfo` on this host reports the iGPU (Intel Graphics, device 0xB08F, 80 EU /
10 Xe3 subslices, Core Ultra 5 338H) with a **16 MiB** GPU-side cache, 256 B
line, 128 KiB SLM per work-group — against 108.7 MB of weights per MoE
layer-step and 3.91 GB per denoise step (§I).

Four measurements, because the premise has four separable parts:

  1. **Is there a cache cliff at all, and how big is the prize?**
     A single `x.sum()` streams the buffer once, so it is a cold DRAM read at
     every footprint — that is why `phase8_bandwidth_probe.py` reads 29 GB/s
     from 4 MB to 2.7 GB. To see the cache, reuse is needed *inside one
     launch*: `x[None].expand(R, n).sum()` reads the same bytes R times, so
     after the first pass the buffer is resident.
  2. **The device's fp16 GEMM ceiling**, to get its machine balance.
  3. **The real MoE shape swept over M**, to see where M=51 sits against that
     ceiling.
  4. **Hot vs cold weights at the real M=51**, footprint swept across the
     16 MiB cache. `cold` rotates a pool far larger than the cache (the loop's
     real reuse distance is 3.91 GB, so every layer-step is a cold read);
     `hot` reuses one buffer — the unreachable best case for cache blocking.
     The gap between them is the entire prize, and (4) is the measurement that
     decides the question.

torch-xpu enumerates one Level-Zero platform per process (§K1), so run once per
device:

    ONEAPI_DEVICE_SELECTOR=level_zero:1 python phase11_igpu_cache_residency_probe.py  # iGPU
    ONEAPI_DEVICE_SELECTOR=level_zero:0 python phase11_igpu_cache_residency_probe.py  # B60

Host state matters (Rules, #2): these want `load < 2.0` and no other container
holding a GPU.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

# The denoise loop's MoE shape, from §F2: 32 routed experts, gate/up/down,
# hidden 768, intermediate 512, M=51 rows (50 action tokens + 1 state), fp16.
N_EXPERTS = 32
M_ROWS = 51
HIDDEN = 768
INTER = 512
EXPERT_BYTES = 3 * HIDDEN * INTER * 2
LAYER_STEPS = 36 * 10  # 36 layers x 10 denoise steps


def best_of(fn, iters: int = 20, warmup: int = 5) -> float:
    """Best-of-N seconds, matching `phase8_bandwidth_probe.py`."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    b = float("inf")
    for _ in range(iters):
        torch.xpu.synchronize()
        t = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        b = min(b, time.perf_counter() - t)
    return b


def launch_floor(dev: str) -> float:
    tiny = torch.ones(1024, dtype=torch.float16, device=dev)
    return best_of(tiny.sum, iters=50, warmup=10)


def stream_bandwidth(dev: str, gb: float = 2.72) -> float:
    """Single-pass read rate, the same method as `phase8_bandwidth_probe.py`.

    Must not reuse the cache-cliff numbers for this: those read one buffer R
    times, so even at 256 MiB they are partly cache hits and overstate DRAM.
    """
    x = torch.empty(int(gb * 1e9) // 2, dtype=torch.float16, device=dev).normal_()
    t = best_of(x.sum, iters=10, warmup=3)
    bw = x.numel() * 2 / t / 1e9
    del x
    torch.xpu.empty_cache()
    return bw


def probe_cache_cliff(dev: str, total: int = 512 * 2**20) -> list[dict]:
    """Cache-resident read bandwidth vs footprint, one launch per sample.

    `total` bytes are read per sample regardless of footprint, so launch
    overhead is amortised equally and the only variable is whether the
    footprint fits the cache.
    """
    rows = []
    for mib in (0.5, 1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 128, 256):
        nb = int(mib * 2**20)
        reps = max(2, total // nb)
        x = torch.empty(nb // 2, dtype=torch.float16, device=dev).normal_()
        view = x[None].expand(reps, nb // 2)
        t = best_of(view.sum, iters=10, warmup=3)
        rows.append({"mib": mib, "reps": reps, "ms": t * 1e3,
                     "gbps": nb * reps / t / 1e9})
        del x, view
        torch.xpu.empty_cache()
    return rows


def probe_gemm_ceiling(dev: str) -> list[dict]:
    rows = []
    for s in (1024, 2048, 4096):
        a = torch.randn(s, s, dtype=torch.float16, device=dev)
        b = torch.randn(s, s, dtype=torch.float16, device=dev)
        t = best_of(lambda: a @ b, iters=10)
        rows.append({"n": s, "ms": t * 1e3, "tflops": 2 * s**3 / t / 1e12})
        del a, b
        torch.xpu.empty_cache()
    return rows


def _moe_step(x, ws):
    gate, up, down = ws
    return torch.bmm(torch.bmm(x, gate) * torch.bmm(x, up), down)


def probe_m_sweep(dev: str) -> list[dict]:
    """Full-width MoE layer-step, M swept. Shows how far M=51 is from the ceiling."""
    rows = []
    ws = (
        torch.randn(N_EXPERTS, HIDDEN, INTER, dtype=torch.float16, device=dev),
        torch.randn(N_EXPERTS, HIDDEN, INTER, dtype=torch.float16, device=dev),
        torch.randn(N_EXPERTS, INTER, HIDDEN, dtype=torch.float16, device=dev),
    )
    wbytes = N_EXPERTS * EXPERT_BYTES
    for m in (M_ROWS, 102, 204, 512, 1024, 2048):
        x = torch.randn(N_EXPERTS, m, HIDDEN, dtype=torch.float16, device=dev)
        t = best_of(lambda: _moe_step(x, ws), iters=10)
        flops = 3 * 2 * N_EXPERTS * m * HIDDEN * INTER
        rows.append({"m": m, "ms": t * 1e3, "tflops": flops / t / 1e12,
                     "weight_gbps": wbytes / t / 1e9, "ms_per_row": t * 1e3 / m})
        del x
        torch.xpu.empty_cache()
    del ws
    torch.xpu.empty_cache()
    return rows


def probe_hot_cold(
    dev: str, pool_bytes: int = 256 * 2**20, reps: int = 40
) -> list[dict]:
    """The decisive one: does making the weights cache-resident help at M=51?

    `reps` steps run back-to-back inside one timed region. This matters: timing
    a single step per sync makes the 3 `bmm` launches (0.062 ms each on the
    iGPU) dominate at small E, which floors cold and hot at the same value and
    hides the cache effect entirely. With the launches pipelined, the iGPU shows
    up to 9.6x.
    """
    rows = []
    for e in (2, 4, 6, 8, 12, N_EXPERTS):
        wbytes = e * EXPERT_BYTES
        x = torch.randn(e, M_ROWS, HIDDEN, dtype=torch.float16, device=dev)
        # the pool must exceed the cache by enough that a buffer is evicted
        # before it comes round again -- the loop's real reuse distance is
        # 3.91 GB, so every layer-step is a cold read
        pool = [
            (
                torch.randn(e, HIDDEN, INTER, dtype=torch.float16, device=dev),
                torch.randn(e, HIDDEN, INTER, dtype=torch.float16, device=dev),
                torch.randn(e, INTER, HIDDEN, dtype=torch.float16, device=dev),
            )
            for _ in range(max(reps, pool_bytes // wbytes))
        ]

        def cold():
            for i in range(reps):
                _moe_step(x, pool[i % len(pool)])

        hot_ws = pool[0]

        def hot():
            for _ in range(reps):
                _moe_step(x, hot_ws)

        t_cold = best_of(cold, iters=8, warmup=3) / reps
        t_hot = best_of(hot, iters=8, warmup=3) / reps
        rows.append({"experts": e, "weight_mb": wbytes / 1e6,
                     "fits_16mib": wbytes <= 16 * 2**20,
                     "cold_ms": t_cold * 1e3, "hot_ms": t_hot * 1e3,
                     "gain": t_cold / t_hot,
                     "cold_gbps": wbytes / t_cold / 1e9,
                     "hot_gbps": wbytes / t_hot / 1e9,
                     "pool": len(pool)})
        del pool, x
        torch.xpu.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    assert torch.xpu.is_available(), "no XPU visible"
    dev = "xpu:0"
    p = torch.xpu.get_device_properties(0)
    print(f"device: {p.name}  eu={p.gpu_eu_count} subslices={p.gpu_subslice_count} "
          f"mem={p.total_memory / 2**30:.1f} GiB  bus={p.memory_bus_width}-bit")
    print(f"torch {torch.__version__}")
    floor = launch_floor(dev)
    print(f"launch+sync floor: {floor * 1e3:.3f} ms\n")

    out: dict = {"device": p.name, "eu": p.gpu_eu_count,
                 "torch": torch.__version__, "launch_floor_ms": floor * 1e3}

    print("1. cache-resident read bandwidth vs footprint")
    print(f"   {'footprint':>11} {'reps':>5} {'ms':>9} {'GB/s':>8}")
    out["cache_cliff"] = probe_cache_cliff(dev)
    for r in out["cache_cliff"]:
        print(f"   {r['mib']:>9.1f}Mi {r['reps']:>5} {r['ms']:>9.3f} {r['gbps']:>8.1f}")

    print("\n2. fp16 GEMM ceiling")
    out["gemm"] = probe_gemm_ceiling(dev)
    for r in out["gemm"]:
        print(f"   {r['n']}^3  {r['ms']:>8.3f} ms  {r['tflops']:>7.2f} TFLOPS")
    peak = max(r["tflops"] for r in out["gemm"])

    print(f"\n3. MoE layer-step, E={N_EXPERTS} {HIDDEN}x{INTER} fp16 "
          f"({N_EXPERTS * EXPERT_BYTES / 1e6:.1f} MB weights), M swept")
    print(f"   {'M':>6} {'ms':>9} {'TFLOPS':>8} {'wt GB/s':>9} {'ms/row':>8}")
    out["m_sweep"] = probe_m_sweep(dev)
    for r in out["m_sweep"]:
        print(f"   {r['m']:>6} {r['ms']:>9.3f} {r['tflops']:>8.2f} "
              f"{r['weight_gbps']:>9.1f} {r['ms_per_row']:>8.4f}")

    print(f"\n4. hot vs cold weights at M={M_ROWS}, footprint across the 16 MiB cache")
    print(f"   {'E':>3} {'weights':>9} {'fits':>5} {'cold ms':>9} {'hot ms':>9} "
          f"{'gain':>6} {'cold GB/s':>10} {'hot GB/s':>9}")
    out["hot_cold"] = probe_hot_cold(dev)
    for r in out["hot_cold"]:
        print(f"   {r['experts']:>3} {r['weight_mb']:>7.1f}MB "
              f"{'yes' if r['fits_16mib'] else 'no':>5} {r['cold_ms']:>9.4f} "
              f"{r['hot_ms']:>9.4f} {r['gain']:>5.2f}x {r['cold_gbps']:>10.1f} "
              f"{r['hot_gbps']:>9.1f}")

    # roofline placement at the real shape, against ceilings measured in this
    # same process (both swing with host load -- see Rules #2)
    full = next(r for r in out["m_sweep"] if r["m"] == M_ROWS)
    stream = stream_bandwidth(dev)
    wbytes = N_EXPERTS * EXPERT_BYTES
    flops = 3 * 2 * N_EXPERTS * M_ROWS * HIDDEN * INTER
    mem_rl = wbytes / (stream * 1e9) * 1e3  # ms
    cmp_rl = flops / (peak * 1e12) * 1e3  # ms
    balance = peak * 1e12 / (stream * 1e9)
    resident = [r for r in out["hot_cold"] if r["fits_16mib"]]
    out["roofline"] = {"stream_gbps": stream, "peak_tflops": peak,
                       "balance": balance, "ai": M_ROWS,
                       "mem_roofline_ms": mem_rl,
                       "compute_roofline_ms": cmp_rl,
                       "measured_ms": full["ms"]}

    print("\n   verdict for this device")
    print(f"   {peak:.1f} TFLOPS fp16 / {stream:.0f} GB/s stream -> machine balance "
          f"{balance:.0f} FLOP/byte, vs workload AI = M = {M_ROWS} "
          f"-> memory-bound by {balance / M_ROWS:.1f}x")
    print(f"   layer-step {full['ms']:.3f} ms vs memory roofline {mem_rl:.3f} ms "
          f"({mem_rl / full['ms'] * 100:.0f}% of roofline), "
          f"compute roofline {cmp_rl:.3f} ms")
    print(f"   best hot/cold gain where the footprint fits the cache: "
          f"{max(r['gain'] for r in resident):.2f}x "
          f"(up to {max(r['hot_gbps'] for r in resident):.0f} GB/s, "
          f"vs {stream:.0f} GB/s from DRAM)")
    print(f"   layer-step x{LAYER_STEPS} = {full['ms'] * LAYER_STEPS:.0f} ms of "
          f"denoise-loop MoE (measured loop on B60: 201.4 ms, §F1)")
    out["projection_ms"] = full["ms"] * LAYER_STEPS

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
