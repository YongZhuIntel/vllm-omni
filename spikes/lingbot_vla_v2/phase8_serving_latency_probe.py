#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 8 step E — where does the served wall time go that the model path does not?

Step E ("the ~165 ms outside the model path") sat unattributed from 2026-09-07
because it was only ever expressed as `wall median - model median`. Two medians
subtracted hide the thing that actually mattered: the served latency on this host
is **bimodal**, not scattered. Measured over 40 warm requests on one connection:

    322 323 323 324 325 326 326 327 329 329 330 331 332 333 333 333   16 fast, mean 328
    384 410 410 434 442 452                                            6 between
    484 495 497 ... 508 512 512 514                                   18 slow, mean 502

A 174 ms gap on ~45% of requests, in no temporal pattern. The cause is OpenMP
oversubscription on a hybrid CPU: PyTorch defaults to one intra-op thread per
logical CPU (12 here), the pool spans P-cores (capacity 1024), E-cores (695) and
a low-power island with no L3 (637), and F1 already established that this request
is host-dispatch-bound -- 282.6 ms of 286.5 ms is CPU. Roughly half the time the
one thread that matters loses. `OMP_NUM_THREADS=4` removes it: median 0.486 ->
0.326 s, spread 181 -> 6 ms. Full workings, and the four hypotheses that were
ruled out first, in `PHASE8_LATENCY_PARITY.md` under F6.

This probe is the measurement that made it visible. It reports the whole
distribution and a bimodality split rather than a median, and it breaks each
request into pack / send / wait / unpack so a regression can be placed on the
client, the wire, or the server without a second tool.

Start a server first, then:

    OMP_NUM_THREADS=4 python -m vllm_omni.entrypoints.cli.main serve \\
        /tmp/lingbot-vla-v2-perf --omni --host 127.0.0.1 --port 8000 \\
        --dtype float16 --enforce-eager --disable-log-stats

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase8_serving_latency_probe.py

Two cautions, both learned the hard way here:

* **Check the process table before believing anything.** A first pass concluded
  that `taskset -c 0-3` was the fix; it was three orphaned servers from earlier
  attempts, two spinning at ~100% CPU. `--require-idle` (default) refuses to
  measure when stray python processes or `load1 > 2.0` are present, the same
  contract `run_perf_check.sh` enforces.
* **The model runs in a spawned child process.** Kill patterns matched against
  `sys.argv` set inside Python do not match the OS command line of that child.
  Reap by process group.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np
from websockets.sync.client import connect

from vllm_omni.entrypoints.openpi.connection import (
    MAX_OPENPI_PAYLOAD_BYTES,
    _pack,
    _unpack,
)

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _host_is_idle() -> tuple[bool, str]:
    """The same refusal `run_perf_check.sh` applies, for the same reason."""
    with open("/proc/loadavg") as handle:
        load1 = float(handle.read().split()[0])
    out = subprocess.run(
        ["ps", "-eo", "pid,args", "--no-headers"], capture_output=True, text=True, check=False
    ).stdout
    strays = [
        line for line in out.splitlines() if "python" in line and "vscode" not in line and "claude" not in line
    ]
    if load1 > 2.0:
        return False, f"load1={load1} > 2.0"
    if len(strays) > 1:  # this process is one of them
        return False, f"{len(strays) - 1} stray python processes:\n  " + "\n  ".join(s[:100] for s in strays)
    return True, f"load1={load1}, no strays"


def _describe(label: str, values: list[float], split: float) -> None:
    ordered = sorted(values)
    n = len(ordered)
    fast = [v for v in values if v < split]
    slow = [v for v in values if v >= split]
    print(
        f"\n{label}: n={n}  min={ordered[0]:.0f}  p50={ordered[n // 2]:.0f}  "
        f"p90={ordered[int(n * 0.9)]:.0f}  max={ordered[-1]:.0f}  spread={ordered[-1] - ordered[0]:.0f} ms"
    )
    print("  " + " ".join(f"{v:.0f}" for v in ordered))
    if fast and slow:
        fm, sm = statistics.mean(fast), statistics.mean(slow)
        print(
            f"  BIMODAL: {len(fast)} fast (mean {fm:.0f}) / {len(slow)} slow (mean {sm:.0f}), "
            f"gap {sm - fm:.0f} ms, ratio {sm / fm:.2f}"
        )
        print("  -> a median is the wrong summary of this. See F6; try OMP_NUM_THREADS=4.")
    else:
        print(f"  unimodal ({'all fast' if fast else 'all slow'}) -- this is what a healthy host looks like")


def run(args: argparse.Namespace) -> int:
    if args.require_idle:
        idle, why = _host_is_idle()
        print(f"host: {why}")
        if not idle:
            print("\nREFUSING to measure: absolute latencies here are not comparable to any", file=sys.stderr)
            print("recorded number. Reap the strays (by process group) and re-run.", file=sys.stderr)
            return 3

    rng = np.random.default_rng(args.seed)
    uri = f"ws://{args.host}:{args.port}/v1/realtime/robot/openpi"
    stages: list[tuple[float, float, float, float]] = []
    totals: list[float] = []

    with connect(uri, max_size=MAX_OPENPI_PAYLOAD_BYTES) as websocket:
        metadata = _unpack(websocket.recv())
        height, width = metadata["image_resolution"]
        observation = {
            "images": {key: rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for key in CAMERA_KEYS},
            "state": np.zeros(metadata["action_dim"], dtype=np.float32),
            "prompt": args.prompt,
            "session_id": str(uuid.uuid4()),
        }
        # The first request through a fresh connection pays lazy setup; a robot's
        # steady state is what this number is for.
        for i in range(args.requests + 1):
            if args.gap:
                time.sleep(args.gap)
            t0 = time.perf_counter()
            payload = _pack(observation)
            t1 = time.perf_counter()
            websocket.send(payload)
            t2 = time.perf_counter()
            raw = websocket.recv()
            t3 = time.perf_counter()
            _unpack(raw)
            t4 = time.perf_counter()
            if i:
                stages.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3))
                totals.append((t4 - t0) * 1e3)

    print(f"payload {len(payload) / 1024:.0f} KiB ({len(CAMERA_KEYS)} cameras at {height}x{width})")
    _describe("wall (client-observed)", totals, args.split)

    print("\nper-stage means (ms), split by mode:")
    print(f"  {'':>6} {'pack':>7} {'send':>7} {'wait':>8} {'unpack':>7}")
    for name, keep in (("fast", lambda t: t < args.split), ("slow", lambda t: t >= args.split)):
        group = [s for s, t in zip(stages, totals) if keep(t)]
        if group:
            means = [statistics.mean(c) for c in zip(*group)]
            print(f"  {name:>6} " + " ".join(f"{m:7.2f}" for m in means) + f"   n={len(group)}")
    print(
        "\n  `pack`/`unpack` are ~0.1 ms and `send` is mode-independent: when this is"
        "\n  bimodal, the whole gap sits in `wait`, i.e. server-side inside"
        "\n  `engine_client.generate()`, not in transport or queueing."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--gap", type=float, default=0.0, help="idle seconds between requests; 1.0 showed pacing is irrelevant")
    parser.add_argument("--split", type=float, default=400.0, help="fast/slow boundary in ms")
    parser.add_argument("--no-require-idle", dest="require_idle", action="store_false", default=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
