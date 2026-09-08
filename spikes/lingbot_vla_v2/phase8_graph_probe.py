#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 8 step C — is there a graph-capture escape from the host dispatch tax?

`phase8_stage_device_profile.py` established that the whole request is
dispatch-bound: 282.6 ms of the 286.5 ms is the host submitting work, and the
device drains in 3.8 ms. On CUDA the textbook answer is a graph: capture the
kernel sequence once, then replay it with a single submission. This probe asks
whether that answer is available on the B60, in three parts.

1. **API surface.** Does anything in torch/IPEX bind a graph object for XPU?
2. **Runtime capability.** Does the device itself advertise the SYCL graph
   extension? (It does; the two facts together are the whole finding.)
3. **Does it actually do anything?** A deliberately dispatch-bound chain of
   tiny matmuls is run under eager, `torch.compile`, `mode="reduce-overhead"`,
   forced `triton.cudagraphs`, and AOTInductor, with the issue/drain method from
   `phase8_stage_device_profile.py`. A knob that is accepted but changes no
   timing is a no-op, and only measurement distinguishes the two.

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase8_graph_probe.py

Findings are recorded in `PHASE8_LATENCY_PARITY.md` under F3.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import time
from typing import Callable

import torch
from torch._inductor import config as inductor_config

# Long enough that per-kernel submission dominates, small enough that the device
# work per kernel is negligible: the regime a graph is supposed to rescue.
CHAIN_LEN = 60
CHAIN_DIM = 64
CHAIN_BATCH = 8


def report_api_surface() -> None:
    """What torch and IPEX expose, named explicitly rather than summarised."""
    print(f"torch {torch.__version__}")
    names = [n for n in dir(torch.xpu) if "graph" in n.lower()]
    print(f"  torch.xpu attrs matching 'graph': {names or 'none'}")
    for attr in (
        "XPUGraph",
        "graph",
        "graph_pool_handle",
        "make_graphed_callables",
        "is_current_stream_capturing",
    ):
        print(f"    torch.xpu.{attr:30s} {hasattr(torch.xpu, attr)}")
    try:
        importlib.import_module("torch.xpu.graphs")
        print("    torch.xpu.graphs module        True")
    except ModuleNotFoundError:
        print("    torch.xpu.graphs module        False")
    print(f"    torch._C._CUDAGraph            {hasattr(torch._C, '_CUDAGraph')}")
    print(f"    torch._C._XPUGraph             {hasattr(torch._C, '_XPUGraph')}")

    try:
        ipex = importlib.import_module("intel_extension_for_pytorch")
        found = [n for n in dir(ipex) if "graph" in n.lower()]
        print(f"ipex {ipex.__version__}")
        print(f"  ipex attrs matching 'graph': {found or 'none'}")
    except ModuleNotFoundError:
        print("ipex: not installed")

    print(
        f"inductor: triton.cudagraphs={inductor_config.triton.cudagraphs} "
        f"cudagraph_trees={inductor_config.triton.cudagraph_trees}"
    )
    print(
        "  AOTInductor: "
        f"aoti_compile_and_package={hasattr(torch._inductor, 'aoti_compile_and_package')} "
        f"aoti_load_package={hasattr(torch._inductor, 'aoti_load_package')}"
    )


def report_device_capability() -> None:
    """Ask the SYCL runtime, not PyTorch, whether graphs exist on this device.

    This is the part that makes the finding actionable rather than a dead end:
    the capability is present in the driver and simply unbound in Python.
    """
    props = torch.xpu.get_device_properties(0)
    print(f"\ndevice: {props.name}  driver {getattr(props, 'driver_version', '?')}")
    try:
        out = subprocess.run(
            ["sycl-ls", "--verbose"], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  sycl-ls unavailable ({exc}); cannot check graph aspects")
        return
    aspects = [ln.strip() for ln in out.splitlines() if "aspect" in ln.lower() or "graph" in ln.lower()]
    hits = sorted({tok for ln in aspects for tok in ln.split() if "graph" in tok.lower()})
    print(f"  SYCL aspects mentioning graph: {hits or 'none'}")


def bench(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, label: str, reps: int) -> float:
    """Issue/drain plus back-to-back steady state, as in the stage profile."""
    for _ in range(5):
        fn(x)
    torch.xpu.synchronize()

    torch.xpu.synchronize()
    t0 = time.perf_counter()
    fn(x)
    t_issued = time.perf_counter()
    torch.xpu.synchronize()
    t_done = time.perf_counter()

    torch.xpu.synchronize()
    r0 = time.perf_counter()
    for _ in range(reps):
        fn(x)
    torch.xpu.synchronize()
    steady = (time.perf_counter() - r0) / reps * 1e3

    print(
        f"  {label:40s} issue {(t_issued - t0) * 1e3:6.2f}  "
        f"drain {(t_done - t_issued) * 1e3:6.2f}  steady {steady:6.2f} ms"
    )
    return steady


def run_chain(args: argparse.Namespace) -> None:
    device = torch.device("xpu")
    weights = [
        torch.empty(CHAIN_DIM, CHAIN_DIM, device=device, dtype=torch.float16).normal_(0, 0.1)
        for _ in range(CHAIN_LEN)
    ]

    class Chain(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for w in weights:
                x = torch.nn.functional.silu(x @ w)
            return x

    x = torch.empty(CHAIN_BATCH, CHAIN_DIM, device=device, dtype=torch.float16).normal_()
    print(f"\n{CHAIN_LEN} chained {CHAIN_BATCH}x{CHAIN_DIM} @ {CHAIN_DIM}x{CHAIN_DIM} fp16 matmuls + silu")

    compile_kwargs = dict(backend="inductor", dynamic=False, fullgraph=True)
    eager = bench(Chain(), x, "eager", args.repeats)
    compiled = bench(torch.compile(Chain(), **compile_kwargs), x, "torch.compile (default)", args.repeats)
    overhead = bench(
        torch.compile(Chain(), mode="reduce-overhead", **compile_kwargs),
        x,
        'torch.compile mode="reduce-overhead"',
        args.repeats,
    )

    # Setting the flag by hand rules out the possibility that "reduce-overhead"
    # merely failed to request graphs; if the flag is honoured anywhere, this is
    # where it would show.
    inductor_config.triton.cudagraphs = True
    forced = bench(
        torch.compile(Chain(), **compile_kwargs), x, "torch.compile + forced triton.cudagraphs", args.repeats
    )
    print(f"    (no error raised; triton.cudagraphs reads back {inductor_config.triton.cudagraphs})")
    inductor_config.triton.cudagraphs = False

    t0 = time.perf_counter()
    exported = torch.export.export(Chain(), (x,))
    package = torch._inductor.aoti_compile_and_package(exported, package_path=args.aoti_path)
    compile_s = time.perf_counter() - t0
    aoti = bench(torch._inductor.aoti_load_package(package), x, "AOTInductor (.pt2, ahead-of-time)", args.repeats)
    print(f"    (offline compile {compile_s:.1f}s, package {os.path.getsize(package) / 1e3:.0f} kB)")

    print(
        f"\n  reduce-overhead vs compile  : {compiled / overhead:.2f}x\n"
        f"  forced cudagraphs vs compile: {compiled / forced:.2f}x\n"
        f"  AOTInductor vs compile      : {compiled / aoti:.2f}x\n"
        f"  AOTInductor vs eager        : {eager / aoti:.2f}x"
    )
    graphs_work = max(compiled / overhead, compiled / forced) > 1.10
    print(
        "=> graph capture "
        + ("changes timing; investigate" if graphs_work else "is a NO-OP on this backend")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # 50 repeats is not enough: at that count the chain's steady state moves by
    # ~15% run to run, which is larger than every effect being tested here, and
    # a first pass at this measurement read a 1.20x AOTInductor gain out of that
    # noise. At 300 the five variants reproduce to within ~0.03 ms.
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--aoti-path", default="/tmp/phase8_graph_probe_chain.pt2")
    parser.add_argument("--skip-chain", action="store_true", help="report API/device only")
    args = parser.parse_args()
    report_api_surface()
    report_device_capability()
    if not args.skip_chain:
        run_chain(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
