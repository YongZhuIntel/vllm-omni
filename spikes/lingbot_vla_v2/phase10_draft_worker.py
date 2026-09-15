#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 10.3 — the resident iGPU draft worker, and the protocol that reaches it.

`torch.xpu.device_count()` is 1 on this host whatever the selector (PHASE8 §K:
the two cards are different Level Zero platforms), so "draft on the iGPU,
verifier on the dGPU" is not a `.to("xpu:1")` -- it is **two processes**, one
`ZE_AFFINITY_MASK` each, and a protocol between them. This file is that protocol
plus the process that sits behind it.

Three moving parts:

* **`NarrowDraftHead`** -- imported from `phase10_igpu_draft_cost_probe`, not
  re-declared, so the thing being run is literally the shape gate 4 priced at
  **0.72 ms on the iGPU** (2.25x the dGPU, 2.31 M parameters).
* **the transport** -- `OneCCL`/`Endpoint`/`ShmChannel` imported from
  `phase10_ipc_probe`, so the wire is literally the one §9 measured at
  **0.13-0.14 ms** per round trip for our payloads. Both transports are here
  because neither dominates: in this request/reply pattern shm is 0.075 ms
  *faster* than oneCCL, the opposite of §9's tight-loop ordering, and both are
  ~3% of a verify step.
* **two backends behind one interface** -- `LocalDraft` (draft on the dGPU, the
  10.2 single-device arm) and `RemoteDraft` (draft on the iGPU, the 10.3 arm).
  `phase10_spec_runtime.py` holds a `DraftBackend` and does not know which.

## The protocol

Fixed-size, registered-once buffers; one outstanding message; request/reply.

    slot      dtype    numel       bytes   direction
    req       fp32     2 + 55        228   dGPU -> iGPU   [opcode, seq, state...]
    prefix    fp16     286 x 512  292864   dGPU -> iGPU   projected prefix
    reply     fp32     50 x 55     11000   iGPU -> dGPU   x0_draft

    DRAFT     send req(opcode=1, state)                  -> recv reply
    REFRESH   send req(opcode=2, state) + send prefix     -> recv reply
    SHUTDOWN  send req(opcode=0)                           (nothing follows it)

Two decisions worth stating, because the first one was wrong on the first attempt:

1. **Every request is acknowledged**, including `REFRESH`, whose reply is the
   draft for the state it carried. The first version made `REFRESH`
   fire-and-forget, to let the iGPU rebuild its cache *inside* the dGPU's own
   214 ms denoise loop. oneCCL tolerated that -- pt2pt sends queue -- but the shm
   transport is a **single-slot mailbox**, so the `DRAFT` that followed
   overwrote the unread `REFRESH` and both sides waited forever. The reply is
   what makes one-message-outstanding true on *both* transports rather than only
   on the forgiving one. It costs the 0.3 ms of rebuild that used to be hidden,
   on a 300 ms full round, and it buys two things worth more than that: a live
   "the worker has the new prefix" signal, and a free `x0_draft` for the frame
   the full round is already computing the teacher's answer for -- which is
   exactly the pair that predicts acceptance once the head is trained.
2. **The reply is exactly `x0_draft`, no header.** Keeping it at the 11000 bytes
   §9 measured means the runtime's transport cost is comparable to that
   measurement. Integrity is checked on the request side (`seq`) and, much more
   strongly, by `--role selftest` comparing the two backends' outputs.

The prefix projection is split across the devices: `prefix_proj` (2560 -> 512)
runs on the **dGPU** inside the full round, and only its output crosses. The
iGPU turns that into k/v. That is why the wire payload is 293 KiB rather than
the 1.4 MiB of `prefix_embs`, and it is why `NarrowDraftHead.project` and
`.kv_from_projection` exist as separate methods.

Both processes build the head from the same `--draft-seed` on CPU before moving
it, so with no checkpoint the weights are *identical* on both devices. Latency
does not depend on the weights, which is what makes an untrained draft a
perfectly good subject for a latency and plumbing run -- and it makes
`--role selftest` a real differential test of the wire rather than a smoke test.

**Always pass `--worker-cpu`.** A resident worker idles inside `recv`, and
`onecclRecv` hard-spins: 1.00 core idle over oneCCL, 0.05 over shm. On this
12-core host that one spinner collides with the verifier process's OpenMP pool
and takes `processor.preprocess` from 3.9 ms to 218 ms -- fifty times what the
whole speculative round saves, and invisible in the speedup ratio because it
slows the baseline arm too. `reserve_cpus_for_worker` splits the cores;
`OMP_WAIT_POLICY=PASSIVE` is the other fix.

    # everything below wants these three, for the oneCCL transport
    I=/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install
    export LD_LIBRARY_PATH=$I/lib:$I/opt/mpi/lib:$LD_LIBRARY_PATH CCL_PLUGIN=ONECCL_IGPU

    # protocol + differential check + per-call timing, no 6B model needed
    PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_draft_worker.py \
        --role selftest --worker-cpu 11
    PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_draft_worker.py \
        --role selftest --transport shm --worker-cpu 11   # no oneCCL env needed

`--role worker` is what `phase10_spec_runtime.py` spawns; it is not normally run
by hand.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase10_igpu_draft_cost_probe import (  # noqa: E402
    ACTION_DIM,
    CHUNK,
    PREFIX_LEN,
    STATE_DIM,
    VLM_HIDDEN,
    NarrowDraftHead,
)
from phase10_ipc_probe import (  # noqa: E402
    DEFAULT_CCL_LIB,
    OneCCL,
    ShmChannel,
    make_endpoint,
)

OP_SHUTDOWN, OP_DRAFT, OP_REFRESH = 0, 1, 2

# req = [opcode, seq, state...]. A history window (the plan's optional
# `history_proj`) would widen this by `6 * ACTION_DIM` floats and nothing else;
# at 228 bytes the request is nowhere near the transport's size floor.
REQ_HEADER = 2
REQ_NUMEL = REQ_HEADER + STATE_DIM

DGPU_RANK, IGPU_RANK = 0, 1


@dataclass(frozen=True)
class Slot:
    name: str
    numel: int
    dtype: torch.dtype


def slot_table(d_model: int) -> tuple[Slot, ...]:
    """The three buffers, in the order both ranks must allocate and register them."""
    return (
        Slot("req", REQ_NUMEL, torch.float32),
        Slot("prefix", PREFIX_LEN * d_model, torch.float16),
        Slot("reply", CHUNK * ACTION_DIM, torch.float32),
    )


def build_head(
    device: torch.device, dtype: torch.dtype, *, d_model: int, seed: int, checkpoint: str | None
) -> NarrowDraftHead:
    """The draft head, identical in both processes.

    Built on CPU under a fixed seed and then moved, so the dGPU and iGPU copies
    agree bit for bit before any device kernel runs. `--role selftest` depends on
    that: a diff between the two backends is then a transport bug, not init noise.
    """
    with torch.device("cpu"):
        torch.manual_seed(seed)
        head = NarrowDraftHead(d_model=d_model)
    if checkpoint:
        state = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)
        head.load_state_dict(state.get("model", state))
    return head.to(device=device, dtype=dtype).eval()


# -- transports ------------------------------------------------------------


class Transport(Protocol):
    def send(self, slot: str, value: torch.Tensor) -> None: ...

    def recv(self, slot: str) -> torch.Tensor: ...

    def close(self) -> None: ...


class OneCCLTransport:
    """oneCCL v2 pt2pt over the iGPU plugin. §9's transport, made persistent.

    One registered buffer per slot per rank. Registration is the reason this is
    fast: `onecclCommRegister` makes the plugin run its pt2pt fd handshake once
    per buffer instead of once per message (PHASE8 §G priced the *unregistered*
    path at 50 us/hop and rejected the iGPU on it).
    """

    def __init__(
        self,
        *,
        rank: int,
        device: torch.device,
        slots: tuple[Slot, ...],
        lib: str,
        ccl_device: int,
        uid_file: str,
        connect_timeout: float,
    ) -> None:
        self.rank, self.peer = rank, 1 - rank
        self.ccl = OneCCL(lib)
        path = Path(uid_file)
        if rank == DGPU_RANK:
            path.write_bytes(self.ccl.get_unique_id())
        else:
            deadline = time.time() + connect_timeout
            while not path.exists():
                if time.time() > deadline:
                    raise TimeoutError(f"rank {DGPU_RANK} never published {path}")
                time.sleep(0.01)
            time.sleep(0.2)  # the write is not atomic; let it land
        self.ccl.init(nranks=2, rank=rank, uid_bytes=path.read_bytes(), device=ccl_device)
        self.stream = torch.xpu.current_stream().sycl_queue
        self.device = device
        # Only the iGPU rank stages through plugin-managed USM host memory
        # (`_prepare_send_tensor` in vLLM's communicator); the dGPU rank sends
        # straight from `data_ptr()`.
        self.endpoints = {
            slot.name: make_endpoint(self.ccl, slot.numel, slot.dtype, device, staged=rank == IGPU_RANK)
            for slot in slots
        }

    def send(self, slot: str, value: torch.Tensor) -> None:
        endpoint = self.endpoints[slot]
        endpoint.load(value.reshape(-1))
        self.ccl.send(endpoint.ptr, endpoint.count, endpoint.dtype_code, self.peer, self.stream)
        torch.xpu.synchronize(self.device)

    def recv(self, slot: str) -> torch.Tensor:
        endpoint = self.endpoints[slot]
        self.ccl.recv(endpoint.ptr, endpoint.count, endpoint.dtype_code, self.peer, self.stream)
        torch.xpu.synchronize(self.device)
        return endpoint.store()

    def close(self) -> None:
        pass  # the comm dies with the process; destroying it mid-teardown hangs


class ShmTransport:
    """POSIX shared memory plus a spin flag -- §9's baseline, and not slower here.

    Needs no `LD_LIBRARY_PATH`, no plugin and no unique-id rendezvous, which
    makes it the arm to reach for when a two-process result looks strange and the
    question is whether oneCCL is involved. It earned that twice already: it
    caught the fire-and-forget `REFRESH` bug oneCCL's queuing hid, and its
    sleeping idle wait is what made the oneCCL spin visible by contrast.
    """

    def __init__(self, *, rank: int, device: torch.device, slots: tuple[Slot, ...], prefix: str,
                 connect_timeout: float) -> None:
        self.rank, self.device, self.timeout = rank, device, connect_timeout
        create = rank == DGPU_RANK
        if not create:
            last = slots[-1].name
            deadline = time.time() + connect_timeout
            while not Path(f"/dev/shm/{prefix}_{last}").exists():
                if time.time() > deadline:
                    raise TimeoutError(f"rank {DGPU_RANK} never created /dev/shm/{prefix}_*")
                time.sleep(0.01)
        # Both staging buffers are allocated once. Doing the device copy straight
        # into them is what keeps this near §9's measured 0.15/0.29 ms: the first
        # version wrote `value.cpu()` and paid an allocation plus an extra host
        # copy per hop, which cost 6 ms on the 293 KiB refresh.
        self.channels, self.host, self.views, self.resident, self.seq = {}, {}, {}, {}, {}
        for slot in slots:
            nbytes = slot.numel * torch.empty(0, dtype=slot.dtype).element_size()
            self.channels[slot.name] = ShmChannel(f"{prefix}_{slot.name}", nbytes, create=create)
            host = torch.empty(slot.numel, dtype=slot.dtype, device="cpu")
            self.host[slot.name] = host
            self.views[slot.name] = host.numpy().view("uint8")
            self.resident[slot.name] = torch.empty(slot.numel, dtype=slot.dtype, device=device)
            self.seq[slot.name] = 0

    def send(self, slot: str, value: torch.Tensor) -> None:
        self.host[slot].copy_(value.reshape(-1))  # one D2H, no allocation
        self.seq[slot] += 1
        self.channels[slot].send(self.views[slot].data, self.seq[slot])

    def recv(self, slot: str) -> torch.Tensor:
        self.seq[slot] += 1
        # The timeout has to cover the worker's first wait, which spans the
        # client's 6B model load.
        payload = self.channels[slot].wait(self.seq[slot], timeout=self.timeout)
        self.views[slot][:] = np.frombuffer(payload, dtype=np.uint8)
        self.resident[slot].copy_(self.host[slot])  # one H2D, no allocation
        return self.resident[slot]

    def close(self) -> None:
        for channel in self.channels.values():
            channel.close(unlink=self.rank == DGPU_RANK)


def make_transport(args, *, rank: int, device: torch.device) -> Transport:
    slots = slot_table(args.d_model)
    if args.transport == "oneccl":
        return OneCCLTransport(
            rank=rank,
            device=device,
            slots=slots,
            lib=args.ccl_lib,
            ccl_device=rank,  # aligned with ZE_AFFINITY_MASK; see §9
            uid_file=args.uid_file,
            connect_timeout=args.connect_timeout,
        )
    return ShmTransport(
        rank=rank, device=device, slots=slots, prefix=args.shm_prefix, connect_timeout=args.connect_timeout
    )


# -- backends --------------------------------------------------------------


class DraftBackend(Protocol):
    """What the speculative runtime needs, with the device split hidden.

    `refresh` is called once per full round, `draft` once per speculative round;
    both return the draft's `[1, CHUNK, ACTION_DIM]` chunk on the caller's
    device. `refresh` returning a chunk is not decoration -- the full round
    computes the teacher's answer for that same frame, so the pair is the
    draft's accuracy, measured once per full round for free.
    """

    def refresh(self, prefix_embs: torch.Tensor, state: torch.Tensor) -> torch.Tensor: ...

    def draft(self, state: torch.Tensor) -> torch.Tensor: ...

    def shutdown(self) -> None: ...


class LocalDraft:
    """Draft on the same device as the verifier -- the 10.2 arm, and the control.

    Gate 4 measured this shape at 0.32 ms on the dGPU. It is the arm that tells
    you how much of the two-process arm's per-tick cost is the transport.
    """

    label = "local"

    def __init__(self, head: NarrowDraftHead, device: torch.device) -> None:
        self.head, self.device = head, device
        self.cached: tuple[torch.Tensor, torch.Tensor] | None = None

    @torch.no_grad()
    def refresh(self, prefix_embs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.cached = self.head.refresh(prefix_embs.to(self.head.prefix_proj.weight.dtype))
        return self.draft(state)  # same work the remote arm does, for parity

    @torch.no_grad()
    def draft(self, state: torch.Tensor) -> torch.Tensor:
        if self.cached is None:
            raise RuntimeError("draft() before the first refresh(): no prefix cache")
        dtype = self.head.action_out.weight.dtype
        return self.head(state.to(dtype), self.cached)

    def shutdown(self) -> None:
        pass


class RemoteDraft:
    """Draft on the iGPU, in another process. The dGPU keeps only `prefix_proj`.

    The `x0_draft` that comes back is fp32 on the wire and is returned in the
    verifier's dtype: `build_x_t` mixes it with fp16 noise, and PHASE10's
    `phase10_port_exactness` measured the fp16 reconstruction noise at 3.9e-3,
    so there is nothing to gain from keeping the draft in fp32 past this point.
    """

    label = "igpu"

    def __init__(
        self,
        transport: Transport,
        *,
        projector: NarrowDraftHead,
        device: torch.device,
        dtype: torch.dtype,
        process: subprocess.Popen | None = None,
    ) -> None:
        self.transport, self.projector = transport, projector
        self.device, self.dtype, self.process = device, dtype, process
        self.seq = 0
        self.request = torch.zeros(REQ_NUMEL, dtype=torch.float32, device=device)
        self.refreshes = 0

    def _request(self, opcode: int, state: torch.Tensor | None) -> None:
        self.seq += 1
        self.request[0] = float(opcode)
        self.request[1] = float(self.seq)
        if state is not None:
            self.request[REQ_HEADER:] = state.reshape(-1).to(dtype=torch.float32)
        self.transport.send("req", self.request)

    @torch.no_grad()
    def refresh(self, prefix_embs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Project on the dGPU (0.07 ms), ship 293 KiB, wait for the rebuilt draft."""
        hidden = self.projector.project(prefix_embs.to(self.projector.prefix_proj.weight.dtype))
        self._request(OP_REFRESH, state)
        self.transport.send("prefix", hidden.to(torch.float16))
        self.refreshes += 1
        return self._reply()

    @torch.no_grad()
    def draft(self, state: torch.Tensor) -> torch.Tensor:
        self._request(OP_DRAFT, state)
        return self._reply()

    def _reply(self) -> torch.Tensor:
        reply = self.transport.recv("reply")
        # `copy=True`: oneCCL's recv hands back the registered buffer itself, and
        # the caller keeps `x0_draft` alive across the verify steps.
        return reply.view(1, CHUNK, ACTION_DIM).to(self.dtype, copy=True)

    def shutdown(self) -> None:
        try:
            self._request(OP_SHUTDOWN, None)
        except Exception as exc:  # the worker may already be gone
            print(f"[draft] shutdown message failed: {exc}", flush=True)
        self.transport.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print("[draft] worker did not exit; terminating", flush=True)
                self.process.terminate()


def parse_cpu_list(spec: str) -> set[int]:
    """``"11"`` or ``"8-11"`` or ``"2,8-9"`` -> the set of CPU ids."""
    cpus: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            low, high = (int(value) for value in part.split("-", 1))
            cpus.update(range(low, high + 1))
        else:
            cpus.add(int(part))
    return cpus


def reserve_cpus_for_worker(spec: str) -> set[int]:
    """Hand the worker's cores to the worker and keep the rest for this process.

    Why this exists, measured on this host: `onecclRecv` **hard-spins**. An idle
    worker therefore holds a full core (0.05 cores under the shm transport, 1.00
    under oneCCL), and on 12 cores that one spinner collides with the verifier
    process's OpenMP pool, whose barriers active-wait by default. The visible
    symptom is not the transport at all -- it is `processor.preprocess` going
    from 3.9 ms to 218 ms, which is ten times everything a speculative round
    saves. Splitting the cores removes the collision; so does
    `OMP_WAIT_POLICY=PASSIVE`, at some cost to the pool's own latency.
    """
    reserved = parse_cpu_list(spec)
    available = sorted(set(os.sched_getaffinity(0)) - reserved)
    if not available:
        raise ValueError(f"--worker-cpu {spec} would leave this process no CPUs")
    os.sched_setaffinity(0, available)
    torch.set_num_threads(len(available))
    print(f"[cpu] worker pinned to {sorted(reserved)}, this process to {available} "
          f"({len(available)} torch threads)")
    return reserved


def spawn_worker(args, *, log: str | None = None) -> subprocess.Popen:
    """Start the iGPU process. `ZE_AFFINITY_MASK=1` is the iGPU on this host (§9)."""
    env = dict(os.environ)
    env["ZE_AFFINITY_MASK"] = str(IGPU_RANK)
    env.pop("ONEAPI_DEVICE_SELECTOR", None)  # would re-enumerate under the mask
    env["OMP_NUM_THREADS"] = "2"
    command = [
        sys.executable, "-u", __file__,
        "--role", "worker",
        "--transport", args.transport,
        "--ccl-lib", args.ccl_lib,
        "--uid-file", args.uid_file,
        "--shm-prefix", args.shm_prefix,
        "--d-model", str(args.d_model),
        "--dtype", args.dtype,
        "--draft-seed", str(args.draft_seed),
        "--connect-timeout", str(args.connect_timeout),
    ]
    if args.draft_checkpoint:
        command += ["--draft-checkpoint", args.draft_checkpoint]
    if args.worker_cpu:
        command = ["taskset", "-c", args.worker_cpu, *command]
    handle = open(log, "w") if log else None
    return subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)


def connect_worker(args, *, device: torch.device, dtype: torch.dtype, log: str | None = None) -> RemoteDraft:
    """Spawn the worker and rendezvous with it.

    Call this **before** loading the 6B verifier: the rendezvous is where the two
    processes agree on buffers, and doing it first means the worker is sitting in
    `recv` while the big model loads instead of timing out waiting for it.
    """
    Path(args.uid_file).unlink(missing_ok=True)
    for stale in Path("/dev/shm").glob(f"{args.shm_prefix}_*"):
        stale.unlink(missing_ok=True)  # a killed run leaves its segments behind
    if args.worker_cpu:
        reserve_cpus_for_worker(args.worker_cpu)
    elif args.transport == "oneccl" and os.environ.get("OMP_WAIT_POLICY", "").upper() != "PASSIVE":
        # This is the trap that cost the most time in 10.3, and it fails in the
        # flattering direction: the spinning worker slows the *baseline* arm too,
        # so the speculative speedup comes out **larger** than it is (3.7x
        # measured, 3.1x real). Either reserve a core or make the pool sleep.
        print(
            "[warn] oneCCL's recv hard-spins a core and no CPUs are reserved for the worker.\n"
            "       On a 12-core host that collides with this process's OpenMP pool and adds\n"
            "       ~215 ms to each full round's `preprocess`. Pass --worker-cpu 11, or export\n"
            "       OMP_WAIT_POLICY=PASSIVE, before believing any number from this run.",
            flush=True,
        )
    process = spawn_worker(args, log=log)
    try:
        transport = make_transport(args, rank=DGPU_RANK, device=device)
    except Exception:
        process.terminate()
        raise
    projector = build_head(
        device, dtype, d_model=args.d_model, seed=args.draft_seed, checkpoint=args.draft_checkpoint
    )
    return RemoteDraft(transport, projector=projector, device=device, dtype=dtype, process=process)


# -- the resident process --------------------------------------------------


def run_worker(args) -> int:
    if not torch.xpu.is_available():
        raise SystemExit("worker sees no XPU; it must be spawned with ZE_AFFINITY_MASK=1")
    device = torch.device("xpu:0")
    dtype = getattr(torch, args.dtype)
    properties = torch.xpu.get_device_properties(0)
    print(
        f"[worker] {properties.name}, {properties.total_memory / 2**30:.2f} GiB, "
        f"eu {getattr(properties, 'gpu_eu_count', '?')}, ZE_AFFINITY_MASK="
        f"{os.environ.get('ZE_AFFINITY_MASK')}, transport={args.transport}",
        flush=True,
    )

    head = build_head(device, dtype, d_model=args.d_model, seed=args.draft_seed, checkpoint=args.draft_checkpoint)
    transport = make_transport(args, rank=IGPU_RANK, device=device)
    print("[worker] ready", flush=True)

    cached: tuple[torch.Tensor, torch.Tensor] | None = None
    drafts: list[float] = []
    refreshes: list[float] = []
    expected_seq = 0
    while True:
        request = transport.recv("req").cpu()  # 228 B; the D2H is inside the numbers
        start = time.perf_counter()
        opcode, seq = int(request[0].item()), int(request[1].item())
        expected_seq += 1
        if seq != expected_seq:
            # Loud, because a dropped or duplicated message otherwise shows up as
            # a plausible latency number computed from the wrong state.
            raise RuntimeError(f"protocol desync: got seq {seq}, expected {expected_seq}")

        if opcode == OP_SHUTDOWN:
            break
        if opcode == OP_REFRESH:
            hidden = transport.recv("prefix").view(1, PREFIX_LEN, args.d_model)
            with torch.no_grad():
                cached = head.kv_from_projection(hidden.to(dtype))
        elif opcode != OP_DRAFT:
            raise RuntimeError(f"unknown opcode {opcode}")
        if cached is None:
            raise RuntimeError("DRAFT before any REFRESH: the session has no prefix cache")

        state = request[REQ_HEADER:].view(1, STATE_DIM).to(device=device, dtype=dtype)
        with torch.no_grad():
            x0_draft = head(state, cached)
        transport.send("reply", x0_draft.to(torch.float32))
        (refreshes if opcode == OP_REFRESH else drafts).append((time.perf_counter() - start) * 1000.0)

    def summary(name: str, samples: list[float]) -> str:
        if not samples:
            return f"{name}: none"
        p90 = statistics.quantiles(samples, n=10)[-1] if len(samples) >= 10 else max(samples)
        return f"{name}: {len(samples)}x median {statistics.median(samples):.3f} ms p90 {p90:.3f} ms"

    # Worker-side service time: recv-to-send, so it includes the reply hop's
    # local half but not the client's wait. The client's own number is the one
    # that matters; this one says whether a surprise there is compute or wire.
    print(f"[worker] {summary('draft', drafts)}", flush=True)
    print(f"[worker] {summary('refresh', refreshes)}", flush=True)
    transport.close()
    return 0


# -- selftest --------------------------------------------------------------


def run_selftest(args) -> int:
    """Protocol, values and per-call cost, without the 6B verifier.

    The differential check is the point: `RemoteDraft` and `LocalDraft` hold the
    same weights, so if the wire carries the wrong bytes -- wrong slot, stale
    buffer, unregistered payload silently truncated -- the outputs diverge. A
    timing-only test cannot see any of that.
    """
    if not torch.xpu.is_available():
        raise SystemExit("no XPU visible")
    device = torch.device("xpu:0")
    dtype = getattr(torch, args.dtype)
    properties = torch.xpu.get_device_properties(0)
    print(f"[client] {properties.name}, ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK', '<unset>')}")

    log = args.worker_log
    remote = connect_worker(args, device=device, dtype=dtype, log=log)
    local = LocalDraft(build_head(device, dtype, d_model=args.d_model, seed=args.draft_seed,
                                 checkpoint=args.draft_checkpoint), device)
    print(f"[client] connected, transport={args.transport}, worker log {log}")

    torch.manual_seed(args.draft_seed + 1)
    prefix_embs = torch.randn(1, PREFIX_LEN, VLM_HIDDEN, device=device, dtype=dtype)
    states = [torch.randn(1, STATE_DIM, device=device, dtype=dtype) for _ in range(args.iters + args.warmup)]

    try:
        refresh_samples = []
        for _ in range(max(1, args.refreshes)):
            torch.xpu.synchronize(device)
            start = time.perf_counter()
            remote_refresh = remote.refresh(prefix_embs, states[0])
            refresh_samples.append((time.perf_counter() - start) * 1000.0)
        local_refresh = local.refresh(prefix_embs, states[0])
        refresh_diff = float((remote_refresh - local_refresh).abs().max().item())

        samples, diffs, scale = [], [], 0.0
        for index, state in enumerate(states):
            torch.xpu.synchronize(device)
            start = time.perf_counter()
            remote_x0 = remote.draft(state)
            elapsed = (time.perf_counter() - start) * 1000.0
            local_x0 = local.draft(state)
            if index >= args.warmup:
                samples.append(elapsed)
                diffs.append(float((remote_x0 - local_x0).abs().max().item()))
                scale = max(scale, float(local_x0.abs().max().item()))
    finally:
        remote.shutdown()

    p90 = statistics.quantiles(samples, n=10)[-1] if len(samples) >= 10 else max(samples)
    print(f"\n[selftest] transport {args.transport}, {len(samples)} DRAFT calls")
    print(f"  draft round trip   median {statistics.median(samples):.3f} ms  "
          f"min {min(samples):.3f}  p90 {p90:.3f}")
    print(f"  refresh round trip median {statistics.median(refresh_samples):.3f} ms  "
          f"over {len(refresh_samples)} calls")
    print(f"  vs one 21.3 ms denoise step: {statistics.median(samples) / 21.3:.1%}")
    print(f"  max |igpu - dgpu| over x0_draft: {max(diffs):.2e}  (values up to {scale:.3f})")
    print(f"  same, on the refresh reply: {refresh_diff:.2e}")
    print(
        "\nThe two backends run identical weights on different devices, so a diff at the\n"
        "fp16 rounding scale is expected and a large one means the wire is wrong.\n"
        "Read the round trip against gate 4's 0.72 ms iGPU draft and §9's 0.14 ms hop."
    )
    if log:
        print(f"\n--- {log}")
        print(Path(log).read_text().rstrip())
    return 0


def add_draft_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared with `phase10_spec_runtime.py`, so both sides cannot drift apart."""
    parser.add_argument("--transport", choices=("oneccl", "shm"), default="oneccl")
    parser.add_argument("--ccl-lib", default=os.environ.get("VLLM_XPU_IGPU_LIB", DEFAULT_CCL_LIB),
                        help="oneCCL **dispatcher** (libccl.so); needs CCL_PLUGIN=ONECCL_IGPU")
    parser.add_argument("--uid-file", default="/tmp/phase10_draft_uid.bin")
    parser.add_argument("--shm-prefix", default="phase10draft")
    parser.add_argument("--d-model", type=int, default=512, help="narrow draft width; gate 4 priced 512")
    parser.add_argument("--draft-seed", type=int, default=0, help="both processes init from this")
    parser.add_argument("--draft-checkpoint", default=None, help="trained head (Phase 10.1); random if unset")
    parser.add_argument("--connect-timeout", type=float, default=600.0)
    parser.add_argument("--worker-log", default="/tmp/phase10_draft_worker.log")
    parser.add_argument("--worker-cpu", default=None,
                        help="CPUs to give the worker exclusively, e.g. 11 or 10-11; "
                             "required with --transport oneccl, whose recv hard-spins")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("selftest", "worker"), default="selftest")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--refreshes", type=int, default=20)
    add_draft_arguments(parser)
    args = parser.parse_args()

    if args.transport == "oneccl" and "oneccl" not in os.environ.get("LD_LIBRARY_PATH", ""):
        print(
            "[warn] LD_LIBRARY_PATH does not mention oneCCL. The dispatcher's own\n"
            "       dependencies are resolved by the loader, so exporting it after the\n"
            "       process starts is too late. If the next line is a load failure:\n"
            "         I=/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install\n"
            "         export LD_LIBRARY_PATH=$I/lib:$I/opt/mpi/lib:$LD_LIBRARY_PATH\n"
            "         export CCL_PLUGIN=ONECCL_IGPU",
            flush=True,
        )
    if args.role == "worker":
        return run_worker(args)
    return run_selftest(args)


if __name__ == "__main__":
    raise SystemExit(main())
