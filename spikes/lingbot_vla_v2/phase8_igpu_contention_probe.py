#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Section K — what does a busy iGPU cost the dGPU request, and what causes it?

Sections G/G2/G4/G5 rejected the iGPU for the model path four times, each time
pricing *contention* with a number borrowed from the vendor's OpenVINO logs
(~21 ms) plus a guess that ours would be worse. This probe measures it, and the
answer is more useful than "worse": contention is not uniform. Measured on this
host against ``run_openvino_comparison.sh --no-prepare --repeat 20``:

    background load on the iGPU                  model path      delta
    none                                          293.7 ms          --
    one busy CPU thread, no GPU (control)          298.1          +  4.1
    2048^2 fp16 matmul loop  (24.0 MiB set)        511.0          +217    1.74x
     512^2 fp16 matmul loop  ( 1.5 MiB set)        515.7          +222    1.76x
     128^2 fp16 matmul loop  ( 0.1 MiB set)        296.2          +  2.5
    saturated DRAM, 2.72 GB reads at 29 GB/s       304.8          + 10.8
    10 ms of compute per 320 ms period (~3% duty)  294.2          +  0.2

So there is a budget and it is denominated in *sustained EU occupancy*: 128^2
kernels are too short to fill the device, so it idles between them and the
request is untouched at the same 100% host CPU and the same 18.4 W.

``--why`` and ``--dgpu-side`` are the mechanism probes, and what they establish
is mostly negative. Ruled out, each by measurement:

* **the dGPU getting slower** -- read bandwidth 448 vs 449 GB/s, 4096^2 matmul
  1.46 vs 1.46 ms, MoE layer-step at the real shape 0.249 vs 0.251 ms
* **driver submission serialising** -- 20k tiny dGPU kernels, 5.19 vs 5.22 us
* **package power / CPU frequency** -- 128^2 draws 18.4 W and is free while
  2048^2 draws 22.4 W and is ruinous; a *pinned* CPU-frequency proxy moves 4%
* **LLC / working-set pollution** -- 24.0 MiB and 1.5 MiB cost the same
* **CPU core placement**, the F6 mechanism -- pinning the load to the LPE island
  still leaves 464.3 ms, and pinning the victim to P-cores as well leaves 444.5
* **host DRAM bandwidth** -- saturating it costs 3.7%

A correction worth keeping, because it is the same trap twice: a first pass here
concluded "package power", on a pure-Python loop that read 80.6 ms idle and
102.0 ms under load. The 80.6 was a single unpinned outlier -- five idle re-runs
read 101-106 ms unpinned and 113-121 ms pinned to P-core 0, because the loop
lands on a different core class each time. That is exactly the hazard F6 was
about, and it invalidated the conclusion, not just the number. ``--why`` now
prints a warning about it rather than a baseline to compare against.

The remaining candidate, unproven and stated as one: the shared kernel-mode GPU
driver. The request issues thousands of *distinct* kernels with per-step
synchronisation; the 20k-identical-kernel loop that shows no slowdown does
neither. Separating those would need ``ze_tracer`` or ``xpu-smi dump`` on both
devices.

Usage. The iGPU is not visible to a default torch process on this host: both GPUs
appear in ``sycl-ls`` but they report different Level Zero driver versions
(20.1.0 vs 30.0.4), so they are separate SYCL platforms and
``torch.xpu.device_count()`` is 1 even under ``ONEAPI_DEVICE_SELECTOR=level_zero:*``.
Select it exclusively instead:

    # the load, on the iGPU, in its own process
    ONEAPI_DEVICE_SELECTOR=level_zero:1 OMP_NUM_THREADS=4 \\
        python spikes/lingbot_vla_v2/phase8_igpu_contention_probe.py \\
        --load igpu-compute --side 2048 --seconds 300 &

    # the victim, on the dGPU
    OMP_NUM_THREADS=4 bash examples/online_serving/lingbot_vla_v2/run_openvino_comparison.sh \\
        --no-prepare --repeat 20

    # mechanism: run each once idle and once with a load up
    OMP_NUM_THREADS=4 python spikes/lingbot_vla_v2/phase8_igpu_contention_probe.py --why
    OMP_NUM_THREADS=4 python spikes/lingbot_vla_v2/phase8_igpu_contention_probe.py --dgpu-side

Two cautions carried over from F6, both of which bit during this measurement:

* ``setsid`` execs into a *new* process group, so ``$!`` in the launching shell is
  not the load's pgid and ``kill -- -$!`` silently misses it. Three loads were
  left spinning at 100% during this session before the process table was checked.
  Reap by pid from ``ps -eo pid,args``, and verify.
* Verify the load is alive *during* the victim run, not before it.
"""

from __future__ import annotations

import argparse
import time

WEIGHT_ELEMS = 36 * 3 * 32 * 768 * 512  # 2.72 GB fp16: the per-step MoE footprint


def load_cpu_spin(deadline: float) -> None:
    """The control: one busy host thread, no GPU at all. Costs the request 4.1 ms."""
    x = 0
    while time.time() < deadline:
        for _ in range(1_000_000):
            x += 1


def load_igpu_compute(deadline: float, side: int) -> None:
    """Occupy the EU array with resident operands, so this is occupancy, not DRAM.

    ``side`` is the discriminator. 2048 and 512 both fill the device and both cost
    the request ~1.75x despite a 16x difference in working set; 128 leaves it
    idling between kernels and costs nothing. Pass all three to reproduce the
    footprint-is-irrelevant result.
    """
    import torch

    d = torch.device("xpu")
    a = torch.empty(side, side, device=d, dtype=torch.float16).normal_()
    b = torch.empty(side, side, device=d, dtype=torch.float16).normal_()
    print(f"{side}^2, working set {3 * side * side * 2 / 2**20:.1f} MiB", flush=True)
    while time.time() < deadline:
        for _ in range(50):
            a @ b
        torch.xpu.synchronize()


def load_igpu_bandwidth(deadline: float) -> None:
    """Saturate host DRAM instead: the traffic an iGPU denoise replica would make."""
    import torch

    x = torch.empty(WEIGHT_ELEMS, device=torch.device("xpu"), dtype=torch.float16).normal_()
    while time.time() < deadline:
        x.sum()
        torch.xpu.synchronize()


def load_igpu_duty(deadline: float, busy_ms: float, period_ms: float) -> None:
    """A realistic duty cycle rather than saturation -- the row that says a budget exists."""
    import torch

    d = torch.device("xpu")
    a = torch.empty(1024, 1024, device=d, dtype=torch.float16).normal_()
    b = torch.empty(1024, 1024, device=d, dtype=torch.float16).normal_()
    a @ b
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        a @ b
    torch.xpu.synchronize()
    per = (time.perf_counter() - start) / 10
    reps = max(1, round(busy_ms / 1e3 / per))
    print(f"one 1024^3 matmul = {per * 1e3:.2f} ms; {reps} per {period_ms:.0f} ms period", flush=True)
    while time.time() < deadline:
        t0 = time.perf_counter()
        for _ in range(reps):
            a @ b
        torch.xpu.synchronize()
        rest = period_ms / 1e3 - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)


def why() -> None:
    """Host-side micro-benchmarks: CPU frequency proxy, CPU GEMM, submission rate.

    Run idle and under load. Established here: submission does not slow, and the
    frequency proxy moves ~4% once it is pinned. Do not compare unpinned runs of
    ``py-loop`` -- see the module docstring.
    """
    import torch

    torch.set_num_threads(1)

    t = time.perf_counter()
    x = 0
    for _ in range(4_000_000):
        x += 1
    py_ms = (time.perf_counter() - t) * 1e3

    a = torch.empty(1024, 1024).normal_()
    b = torch.empty(1024, 1024).normal_()
    a @ b
    t = time.perf_counter()
    for _ in range(20):
        a @ b
    gemm_ms = (time.perf_counter() - t) * 1e3 / 20

    d = torch.device("xpu")
    y = torch.ones(8, device=d)
    for _ in range(100):
        y.add_(1.0)
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(20_000):
        y.add_(1.0)
    submit_us = (time.perf_counter() - t) * 1e6 / 20_000
    torch.xpu.synchronize()

    print(f"device            {torch.xpu.get_device_properties(0).name}")
    print(f"py-loop   {py_ms:8.1f} ms   4M iterations   RUN UNDER `taskset -c 0` OR IGNORE")
    print(f"cpu-gemm  {gemm_ms:8.2f} ms   1024^3 fp32, 1 thread                (idle: 16.34)")
    print(f"submit    {submit_us:8.2f} us   per tiny dGPU kernel, host side       (idle: 5.22)")
    print("\n  Unpinned py-loop varies 101-121 ms by core class on this hybrid CPU;")
    print("  it is only a frequency proxy when pinned. Cross-check power with")
    print("  cat /sys/class/powercap/intel-rapl:0/energy_uj")


def dgpu_side() -> None:
    """Device-side only: did the dGPU get slower, or just its host?

    Every number here is unchanged under a saturating iGPU load, which is what
    moves the penalty onto the host side.
    """
    import torch
    import torch.nn.functional as functional

    d = torch.device("xpu")
    print(torch.xpu.get_device_properties(0).name)

    x = torch.empty(WEIGHT_ELEMS, device=d, dtype=torch.float16).normal_()
    gb = x.numel() * 2 / 1e9
    for _ in range(3):
        x.sum()
    torch.xpu.synchronize()
    best = 1e9
    for _ in range(10):
        torch.xpu.synchronize()
        t = time.perf_counter()
        x.sum()
        torch.xpu.synchronize()
        best = min(best, time.perf_counter() - t)
    print(f"read bandwidth   {gb / best:6.0f} GB/s   (idle: 449)")
    del x

    a = torch.empty(4096, 4096, device=d, dtype=torch.float16).normal_()
    b = torch.empty(4096, 4096, device=d, dtype=torch.float16).normal_()
    for _ in range(10):
        a @ b
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(200):
        a @ b
    torch.xpu.synchronize()
    per = (time.perf_counter() - t) / 200
    print(f"4096^2 matmul    {per * 1e3:6.2f} ms   {2 * 4096**3 / per / 1e12:5.1f} TFLOPS   (idle: 1.46)")

    e, h, i, m = 32, 768, 512, 51
    gw = torch.empty(e, h, i, device=d, dtype=torch.float16).normal_(0, 0.02)
    uw = torch.empty(e, h, i, device=d, dtype=torch.float16).normal_(0, 0.02)
    dw = torch.empty(e, i, h, device=d, dtype=torch.float16).normal_(0, 0.02)
    xe = torch.empty(e, m, h, device=d, dtype=torch.float16).normal_()

    def step() -> object:
        g = torch.bmm(xe, gw)
        u = torch.bmm(xe, uw)
        return torch.bmm(functional.silu(g) * u, dw)

    for _ in range(20):
        step()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(200):
        step()
    torch.xpu.synchronize()
    per = (time.perf_counter() - t) / 200
    print(f"MoE layer-step   {per * 1e3:6.3f} ms   x360 = {per * 360 * 1e3:6.1f} ms loop   (idle: 0.251 / 90.4)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load",
        choices=("cpu-spin", "igpu-compute", "igpu-bandwidth", "igpu-duty"),
        help="run a background load until --seconds elapses",
    )
    parser.add_argument("--why", action="store_true", help="host-side micro-benchmarks")
    parser.add_argument("--dgpu-side", action="store_true", help="device-side micro-benchmarks")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--side", type=int, default=2048, help="igpu-compute: matmul side; try 2048, 512, 128")
    parser.add_argument("--busy-ms", type=float, default=10.0, help="igpu-duty: EU-busy time per period")
    parser.add_argument("--period-ms", type=float, default=320.0, help="igpu-duty: the request period")
    args = parser.parse_args()

    if args.why:
        why()
        return 0
    if args.dgpu_side:
        dgpu_side()
        return 0
    if not args.load:
        parser.error("pass --load, --why or --dgpu-side")

    deadline = time.time() + args.seconds
    if args.load == "cpu-spin":
        load_cpu_spin(deadline)
        return 0

    import torch

    print(f"load {args.load} on {torch.xpu.get_device_properties(0).name}", flush=True)
    if args.load == "igpu-compute":
        load_igpu_compute(deadline, args.side)
    elif args.load == "igpu-bandwidth":
        load_igpu_bandwidth(deadline)
    else:
        load_igpu_duty(deadline, args.busy_ms, args.period_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
