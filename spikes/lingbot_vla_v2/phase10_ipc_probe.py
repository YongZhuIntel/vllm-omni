#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 10.3 — what does one dGPU<->iGPU round trip cost at our payload sizes?

Gate 4 showed the narrow draft costs 0.72 ms on the iGPU against a 21.3 ms
verify step. Whether that survives depends entirely on the transport, and this
is the last unmeasured number in the speculative round.

The payloads are small and **fixed-size**, which is unusual and matters:

    DRAFT request    state          [1,55]   fp32   =    220 B   dGPU -> iGPU
    DRAFT reply      x0_draft    [1,50,55]   fp32   = 11.0 KiB   iGPU -> dGPU
    REFRESH          prefix_kv  [286,512]    fp16   =  293 KiB   dGPU -> iGPU  (full rounds only)

Two transports, because the simple one has to be ruled out before the complex
one is justified:

* **`shm`** — POSIX shared memory plus a spin flag. Device -> host -> shm ->
  host -> device, no special library. This is the baseline; if it is fast
  enough at these sizes, nothing else is needed.
* **`oneccl`** — the oneCCL v2 C API with the iGPU plugin
  (`libccl_igpu.so`), driven through a ctypes wrapper adapted from
  `vllm/distributed/device_communicators/oneccl_igpu_communicator.py`. Two
  details from that file drive the design here:
  - only the **iGPU** rank must stage through plugin-managed USM host memory
    (`_prepare_send_tensor:399`); the dGPU rank sends straight from
    `tensor.data_ptr()`. PHASE8 §G's "every hop does a USM host round-trip" is
    half right.
  - `onecclCommRegister` makes the plugin run its pt2pt fd handshake **once
    per buffer** and skip it thereafter (`:104-113`, `_send_packed:429`).
    §G/§K priced this path at "even an optimistic 50 us" per hop *without* the
    registered fast path. Our buffers are fixed-size and reused forever, which
    is precisely the case registration exists for, so that estimate needs
    re-measuring rather than inheriting.

Process layout follows vLLM's iGPU path, not §K's: **`ZE_AFFINITY_MASK`**, one
device per process (0 = Arc Pro B60 dGPU, 1 = iGPU -- verified on this host),
so each rank sees its own device as `xpu:0`. `torch.xpu.device_count()` is 1
either way, which is why this is two processes.

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_ipc_probe.py
    PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_ipc_probe.py --transport shm

Read the result against 21.3 ms: the transport is irrelevant below ~0.5 ms,
decides K below ~2 ms, and kills the two-process design above ~5 ms.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch

# The **dispatcher**, not `libccl_igpu.so` directly: the iGPU transport is
# selected by `CCL_PLUGIN=ONECCL_IGPU`, which only the dispatcher reads.
DEFAULT_CCL_LIB = "/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install/lib/libccl.so"

# The three real payloads, as (name, numel, dtype).
PAYLOADS = {
    "request_state": (55, torch.float32),
    "reply_x0_draft": (50 * 55, torch.float32),
    "refresh_prefix_kv": (286 * 512, torch.float16),
}

# onecclDataType_t / onecclResult_t, from oneapi/ccl/v2/types.h.
ONECCL_SUCCESS = 0
ONECCL_DTYPE = {torch.float16: 6, torch.float32: 7, torch.bfloat16: 9, torch.int8: 0, torch.uint8: 1}
UNIQUE_ID_BYTES = 4096


class OneCCLUniqueId(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char * UNIQUE_ID_BYTES)]


class OneCCL:
    """Minimal ctypes binding, adapted from vLLM's `OneCCL` (same C API)."""

    def __init__(self, lib_path: str) -> None:
        self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        self._comm = ctypes.c_void_p()
        lib = self.lib
        lib.onecclGetUniqueId.argtypes = [ctypes.POINTER(OneCCLUniqueId)]
        lib.onecclSetDevice.argtypes = [ctypes.c_uint]
        lib.onecclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, OneCCLUniqueId, ctypes.c_int,
        ]
        for name in ("onecclSend", "onecclRecv"):
            getattr(lib, name).argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
        lib.onecclMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.onecclMemFree.argtypes = [ctypes.c_void_p]
        lib.onecclCommRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.onecclCommDestroy.argtypes = [ctypes.c_void_p]

    @staticmethod
    def _check(fn: str, code: int) -> None:
        if code != ONECCL_SUCCESS:
            raise RuntimeError(f"{fn} failed with onecclResult_t={code}")

    def get_unique_id(self) -> bytes:
        uid = OneCCLUniqueId()
        self._check("onecclGetUniqueId", self.lib.onecclGetUniqueId(ctypes.byref(uid)))
        return bytes(uid.data)

    def init(self, *, nranks: int, rank: int, uid_bytes: bytes, device: int) -> None:
        self._check("onecclSetDevice", self.lib.onecclSetDevice(device))
        uid = OneCCLUniqueId()
        uid.data = uid_bytes.ljust(UNIQUE_ID_BYTES, b"\x00")[:UNIQUE_ID_BYTES]
        self._check(
            "onecclCommInitRank",
            self.lib.onecclCommInitRank(ctypes.byref(self._comm), nranks, uid, rank),
        )

    def mem_alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self._check("onecclMemAlloc", self.lib.onecclMemAlloc(ctypes.byref(ptr), nbytes))
        return ptr.value

    def register(self, ptr: int, nbytes: int) -> int:
        handle = ctypes.c_void_p()
        self._check(
            "onecclCommRegister",
            self.lib.onecclCommRegister(self._comm, ctypes.c_void_p(ptr), nbytes, ctypes.byref(handle)),
        )
        return handle.value

    def send(self, ptr: int, count: int, dtype: int, peer: int, stream: int) -> None:
        self._check("onecclSend", self.lib.onecclSend(
            ctypes.c_void_p(ptr), count, dtype, peer, self._comm, ctypes.c_void_p(stream)))

    def recv(self, ptr: int, count: int, dtype: int, peer: int, stream: int) -> None:
        self._check("onecclRecv", self.lib.onecclRecv(
            ctypes.c_void_p(ptr), count, dtype, peer, self._comm, ctypes.c_void_p(stream)))


@dataclass
class Endpoint:
    """One fixed-size, registered buffer plus the device tensor behind it."""

    tensor: torch.Tensor
    ptr: int
    count: int
    dtype_code: int
    staged: bool  # True on the iGPU rank: ptr is plugin USM host memory

    def load(self, value: torch.Tensor) -> None:
        self.tensor.copy_(value)
        if self.staged:
            host = self.tensor.detach().contiguous().cpu()
            ctypes.memmove(self.ptr, host.data_ptr(), host.numel() * host.element_size())

    def store(self) -> torch.Tensor:
        if self.staged:
            host = torch.empty(self.tensor.shape, dtype=self.tensor.dtype, device="cpu")
            ctypes.memmove(host.data_ptr(), self.ptr, host.numel() * host.element_size())
            self.tensor.copy_(host.to(self.tensor.device))
        return self.tensor


def make_endpoint(ccl: OneCCL, numel: int, dtype: torch.dtype, device: torch.device, *, staged: bool) -> Endpoint:
    tensor = torch.zeros(numel, dtype=dtype, device=device)
    nbytes = tensor.numel() * tensor.element_size()
    ptr = ccl.mem_alloc(nbytes) if staged else tensor.data_ptr()
    try:
        ccl.register(ptr, nbytes)
    except RuntimeError as exc:  # registration is an optimisation, not a requirement
        print(f"[rank] onecclCommRegister unavailable ({exc}); falling back to unregistered", flush=True)
    return Endpoint(tensor, ptr, tensor.numel(), ONECCL_DTYPE[dtype], staged)


# -- shared-memory baseline ------------------------------------------------


class ShmChannel:
    """One-directional fixed-size channel: 8-byte sequence header + payload."""

    def __init__(self, name: str, nbytes: int, *, create: bool) -> None:
        self.nbytes = nbytes
        if create:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes + 8)
            self.shm.buf[:8] = struct.pack("<Q", 0)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self.seen = 0

    def send(self, payload: memoryview, seq: int) -> None:
        self.shm.buf[8 : 8 + self.nbytes] = payload
        self.shm.buf[:8] = struct.pack("<Q", seq)

    def wait(self, seq: int, timeout: float = 30.0, spin: float = 0.05) -> memoryview:
        """Spin for `spin` seconds, then poll with short sleeps.

        Spinning is what keeps the measured number the transport's rather than
        the scheduler's, and at 0.15 ms per round trip the hot path never leaves
        it. The sleeping tail is for `phase10_draft_worker.py`, whose worker can
        sit here for minutes waiting for its first message while the client loads
        the 6B verifier -- burning a core next to that load is not free.
        """
        start = time.time()
        deadline = start + timeout
        while struct.unpack("<Q", bytes(self.shm.buf[:8]))[0] != seq:
            now = time.time()
            if now > deadline:
                raise TimeoutError(f"shm channel {self.shm.name} stalled at seq {seq}")
            if now - start > spin:
                time.sleep(5e-5)
        return self.shm.buf[8 : 8 + self.nbytes]

    def close(self, unlink: bool) -> None:
        self.shm.close()
        if unlink:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


# -- ranks -----------------------------------------------------------------


def payload_specs(names: list[str]) -> list[tuple[str, int, torch.dtype, int]]:
    out = []
    for name in names:
        numel, dtype = PAYLOADS[name]
        out.append((name, numel, dtype, numel * torch.empty(0, dtype=dtype).element_size()))
    return out


def run_rank_oneccl(args) -> None:
    device = torch.device("xpu:0")
    rank = args.rank
    ccl = OneCCL(args.ccl_lib)
    uid_path = Path(args.uid_file)

    if rank == 0:
        uid = ccl.get_unique_id()
        uid_path.write_bytes(uid)
    else:
        deadline = time.time() + 60
        while not uid_path.exists():
            if time.time() > deadline:
                raise TimeoutError("rank 0 never published the oneCCL unique id")
            time.sleep(0.01)
        time.sleep(0.2)
        uid = uid_path.read_bytes()

    ccl.init(nranks=2, rank=rank, uid_bytes=uid, device=args.ccl_device)
    stream = torch.xpu.current_stream().sycl_queue
    staged = rank == 1
    peer = 1 - rank
    properties = torch.xpu.get_device_properties(0)
    print(f"[rank {rank}] {properties.name}, ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')}, "
          f"ccl_device={args.ccl_device}, staged={staged}", flush=True)

    results = {}
    for name, numel, dtype, nbytes in payload_specs(args.payloads):
        down = make_endpoint(ccl, numel, dtype, device, staged=staged)
        up = make_endpoint(ccl, numel, dtype, device, staged=staged)
        source = torch.randn(numel, device=device).to(dtype)
        samples = []
        for index in range(args.warmup + args.iters):
            if rank == 0:
                down.load(source)
                torch.xpu.synchronize(device)
                start = time.perf_counter()
                ccl.send(down.ptr, down.count, down.dtype_code, peer, stream)
                ccl.recv(up.ptr, up.count, up.dtype_code, peer, stream)
                torch.xpu.synchronize(device)
                elapsed = (time.perf_counter() - start) * 1000.0
                up.store()
                if index >= args.warmup:
                    samples.append(elapsed)
            else:
                ccl.recv(down.ptr, down.count, down.dtype_code, peer, stream)
                torch.xpu.synchronize(device)
                down.store()
                up.load(down.tensor)  # stand-in for the draft's output
                ccl.send(up.ptr, up.count, up.dtype_code, peer, stream)
                torch.xpu.synchronize(device)
        if rank == 0:
            results[name] = {
                "bytes": nbytes,
                "round_trip_ms_median": statistics.median(samples),
                "round_trip_ms_min": min(samples),
                "round_trip_ms_p90": statistics.quantiles(samples, n=10)[-1] if len(samples) >= 10 else max(samples),
            }
            print(f"[rank 0] {name:<20s} {nbytes:8d} B  round trip "
                  f"median {results[name]['round_trip_ms_median']:.3f} ms", flush=True)

    if rank == 0:
        Path(args.result_file).write_text(json.dumps(results, indent=2) + "\n")


def run_rank_shm(args) -> None:
    device = torch.device("xpu:0")
    rank = args.rank
    properties = torch.xpu.get_device_properties(0)
    print(f"[rank {rank}] {properties.name}, ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')}", flush=True)

    results = {}
    for name, numel, dtype, nbytes in payload_specs(args.payloads):
        # rank 0 creates and owns the segments; rank 1 waits for them to appear.
        if rank == 1:
            deadline = time.time() + 60
            while not Path(f"/dev/shm/{args.shm_prefix}_{name}_up").exists():
                if time.time() > deadline:
                    raise TimeoutError(f"rank 0 never created {args.shm_prefix}_{name}")
                time.sleep(0.01)
        down = ShmChannel(f"{args.shm_prefix}_{name}_down", nbytes, create=rank == 0)
        up = ShmChannel(f"{args.shm_prefix}_{name}_up", nbytes, create=rank == 0)

        source = torch.randn(numel, device=device).to(dtype)
        staging = torch.empty(numel, dtype=dtype, device="cpu")
        # One stable uint8 view of the staging buffer, so the timed loop does
        # exactly two host copies per hop and no allocation.
        staging_u8 = staging.numpy().view("uint8")
        samples = []
        for index in range(args.warmup + args.iters):
            seq = index + 1
            if rank == 0:
                torch.xpu.synchronize(device)
                start = time.perf_counter()
                staging.copy_(source)
                down.send(staging_u8.data, seq)
                staging_u8[:] = np.frombuffer(up.wait(seq), dtype=np.uint8)
                back = staging.to(device)
                torch.xpu.synchronize(device)
                elapsed = (time.perf_counter() - start) * 1000.0
                del back
                if index >= args.warmup:
                    samples.append(elapsed)
            else:
                staging_u8[:] = np.frombuffer(down.wait(seq), dtype=np.uint8)
                on_device = staging.to(device)
                torch.xpu.synchronize(device)
                staging.copy_(on_device)
                up.send(staging_u8.data, seq)
        if rank == 0:
            results[name] = {
                "bytes": nbytes,
                "round_trip_ms_median": statistics.median(samples),
                "round_trip_ms_min": min(samples),
                "round_trip_ms_p90": statistics.quantiles(samples, n=10)[-1] if len(samples) >= 10 else max(samples),
            }
            print(f"[rank 0] {name:<20s} {nbytes:8d} B  round trip "
                  f"median {results[name]['round_trip_ms_median']:.3f} ms", flush=True)
        down.close(unlink=rank == 0)
        up.close(unlink=rank == 0)

    if rank == 0:
        Path(args.result_file).write_text(json.dumps(results, indent=2) + "\n")


def run_driver(args) -> int:
    uid_file = Path(args.uid_file)
    uid_file.unlink(missing_ok=True)
    Path(args.result_file).unlink(missing_ok=True)

    children = []
    for rank in (0, 1):
        env = dict(os.environ)
        env["ZE_AFFINITY_MASK"] = str(rank)  # 0 = B60 dGPU, 1 = iGPU
        env.pop("ONEAPI_DEVICE_SELECTOR", None)
        env["OMP_NUM_THREADS"] = "2"
        command = [
            sys.executable, "-u", __file__,
            "--role", "rank", "--rank", str(rank),
            "--transport", args.transport,
            "--ccl-lib", args.ccl_lib,
            "--ccl-device", str(rank if args.ccl_device is None else args.ccl_device),
            "--uid-file", args.uid_file,
            "--result-file", args.result_file,
            "--shm-prefix", args.shm_prefix,
            "--warmup", str(args.warmup), "--iters", str(args.iters),
            "--payloads", *args.payloads,
        ]
        children.append(subprocess.Popen(command, env=env))
        if rank == 0:
            time.sleep(2.0)  # let rank 0 publish the unique id / create the shm

    codes = [child.wait() for child in children]
    if any(codes):
        print(f"[driver] ranks exited with {codes}", file=sys.stderr)
        return 1

    results = json.loads(Path(args.result_file).read_text())
    print(f"\n{'payload':<22s}{'bytes':>9s}{'round trip ms':>15s}{'min':>9s}{'p90':>9s}"
          f"{'vs 21.3 ms step':>17s}")
    print("-" * 81)
    for name, entry in results.items():
        print(f"{name:<22s}{entry['bytes']:>9d}{entry['round_trip_ms_median']:>15.3f}"
              f"{entry['round_trip_ms_min']:>9.3f}{entry['round_trip_ms_p90']:>9.3f}"
              f"{entry['round_trip_ms_median'] / 21.3:>16.1%}")
    per_tick = results.get("request_state", {}).get("round_trip_ms_median")
    reply = results.get("reply_x0_draft", {}).get("round_trip_ms_median")
    if per_tick is not None and reply is not None:
        print(f"\nA speculative tick pays ONE round trip carrying the 11 KiB reply: "
              f"{reply:.3f} ms via {args.transport}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("driver", "rank"), default="driver")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--transport", choices=("oneccl", "shm"), default="oneccl")
    parser.add_argument("--ccl-lib", default=os.environ.get("VLLM_XPU_IGPU_LIB", DEFAULT_CCL_LIB))
    parser.add_argument("--ccl-device", type=int, default=None, help="oneCCL device index (default: rank)")
    parser.add_argument("--uid-file", default="/tmp/phase10_ipc_uid.bin")
    parser.add_argument("--result-file", default="/tmp/phase10_ipc_result.json")
    parser.add_argument("--shm-prefix", default="phase10ipc")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--payloads", nargs="*", default=list(PAYLOADS))
    args = parser.parse_args()

    if args.role == "driver":
        return run_driver(args)
    if args.transport == "oneccl":
        run_rank_oneccl(args)
    else:
        run_rank_shm(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
