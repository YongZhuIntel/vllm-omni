# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The LingBot-VLA 2.0 draft head, and the iGPU process it runs in.

``config.spec_decode`` is one switch and it has no device option: the draft runs
on the integrated GPU, in a second process. That is not a preference —
``torch.xpu.device_count()`` is 1 per process on this host because the two cards
are different Level Zero platforms, so "the draft is on the iGPU" *is* a process
boundary. This module is that process, the protocol reaching it, and the head
itself.

    server process                        draft process
    ZE_AFFINITY_MASK=0 (dGPU)             ZE_AFFINITY_MASK=1 (iGPU)
    ------------------------------        -------------------------------
    full round:                           REFRESH: rebuild cached k/v from
      embed_prefix -> prefix_embs                  the 293 KiB projection,
      prefix_proj  -> [286,512] fp16 ---->         answer with the draft
      ten Euler steps                     DRAFT:   51 tokens through one
    speculative round:                             narrow layer over the
      state -------------------------->            cached prefix -> chunk
      K x predict_velocity  <---------- x0_draft [1,50,55] fp32

``prefix_proj`` (2560 -> 512) stays on the **dGPU** inside the full round, so the
payload is 293 KiB rather than the 1.4 MiB of ``prefix_embs``. The iGPU turns
that into k/v and keeps it resident until the next full round.

Measured on this host (``spikes/lingbot_vla_v2/PHASE10_SPECULATIVE.md`` §11):

    draft round trip      1.02 ms    (0.88 ms of it the iGPU's own compute)
    refresh round trip    1.54 ms    (full rounds only)
    one verify step      21.7 ms     -- what the above is read against
    idle worker CPU       1.00 core over oneCCL, 0.05 over shm

That last line is the operational trap and the reason ``spec_worker_cpu``
exists: ``onecclRecv`` hard-spins, and an unreserved spinning core collides with
this process's OpenMP pool badly enough to take ``preprocess`` from 3.9 ms to
218 ms. It slows the *non-speculative* path too, so it inflates the speculative
speedup instead of showing up as a regression. See §11.4.

Escape hatches, as environment variables rather than config fields because they
are host facts and not policy:

    VLLM_LINGBOT_SPEC_TRANSPORT   oneccl (default) | shm
    VLLM_LINGBOT_SPEC_CCL_LIB     oneCCL **dispatcher** (libccl.so), not libccl_igpu.so
    VLLM_LINGBOT_SPEC_DIR         rendezvous directory (default: a per-pid /tmp dir)

The oneCCL transport additionally needs, in the *parent's* environment (the
loader resolves the dispatcher's dependencies before this module can set them)::

    I=/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install
    export LD_LIBRARY_PATH=$I/lib:$I/opt/mpi/lib:$LD_LIBRARY_PATH CCL_PLUGIN=ONECCL_IGPU
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import logging
import os
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_CCL_LIB = "/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install/lib/libccl.so"
WORKER_MODULE = "vllm_omni.diffusion.models.lingbot_vla_v2.draft_igpu"

# Gate 4's priced shape: 2.31 M parameters, 4.4 MiB fp16, 0.72 ms on the iGPU.
# FLASH's full-width alternative measured 15.97 ms on the same device, which is
# 75% of a denoise step, which is why this is narrow.
DRAFT_WIDTH = 512
DRAFT_HEADS = 8
DRAFT_KV_HEADS = 2
DRAFT_HEAD_DIM = 64
DRAFT_INTERMEDIATE = 1024
# Both processes construct the head from this seed when no trained checkpoint is
# given, so the untrained latency mode is reproducible and the two sides agree.
DRAFT_SEED = 0

DGPU_RANK, IGPU_RANK = 0, 1
OP_SHUTDOWN, OP_DRAFT, OP_REFRESH = 0, 1, 2
# req = [opcode, seq, state...]
REQ_HEADER = 2


class LingbotDraftHead(nn.Module):
    """One narrow decoder layer over a **cached** prefix: state + chunk queries -> chunk.

    Regresses the whole action chunk in one shot, with no denoise loop -- that is
    the point of a draft. The prefix is not re-encoded per tick: ``project`` runs
    on the dGPU during a full round and ``kv_from_projection`` turns its output
    into resident k/v here, which is only sound because a speculative round
    reuses a stale prefix anyway. It is the same staleness assumption the
    verifier already makes, not a new one.

    Trained separately (Phase 10.1) and loaded from ``spec_draft_path``; it is
    not part of the served checkpoint and never enters its strict load.
    """

    def __init__(self, *, chunk_size: int, action_dim: int, state_dim: int, prefix_width: int,
                 d_model: int = DRAFT_WIDTH) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.heads, self.kv_heads, self.head_dim = DRAFT_HEADS, DRAFT_KV_HEADS, DRAFT_HEAD_DIM
        q_dim, kv_dim = self.heads * self.head_dim, self.kv_heads * self.head_dim

        self.prefix_proj = nn.Linear(prefix_width, d_model, bias=False)  # dGPU, full rounds only
        self.prefix_kv = nn.Linear(d_model, 2 * kv_dim, bias=False)  # iGPU, full rounds only

        self.state_proj = nn.Linear(state_dim, d_model, bias=False)
        self.queries = nn.Embedding(chunk_size, d_model)
        self.norm = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, q_dim + 2 * kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, d_model, bias=False)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.gate_up = nn.Linear(d_model, 2 * DRAFT_INTERMEDIATE, bias=False)
        self.down = nn.Linear(DRAFT_INTERMEDIATE, d_model, bias=False)
        self.action_out = nn.Linear(d_model, action_dim, bias=False)

    @torch.no_grad()
    def project(self, prefix_embs: torch.Tensor) -> torch.Tensor:
        """``[B,P,2560] -> [B,P,512]``: the only part that runs on the dGPU."""
        return self.prefix_proj(prefix_embs.to(self.prefix_proj.weight.dtype))

    @torch.no_grad()
    def kv_from_projection(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B,P,512]`` -> cached k/v at the draft's width."""
        key, value = self.prefix_kv(hidden).chunk(2, dim=-1)
        shape = (*key.shape[:2], self.kv_heads, self.head_dim)
        return key.view(shape).transpose(1, 2), value.view(shape).transpose(1, 2)

    def _attend(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if query.shape[1] != key.shape[1]:
            try:
                return F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
            except TypeError:  # older torch without enable_gqa
                repeat = query.shape[1] // key.shape[1]
                key = key.repeat_interleave(repeat, dim=1)
                value = value.repeat_interleave(repeat, dim=1)
        return F.scaled_dot_product_attention(query, key, value)

    def forward(self, state: torch.Tensor, prefix_kv: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch = state.shape[0]
        tokens = torch.cat(
            [self.state_proj(state)[:, None, :], self.queries.weight[None].expand(batch, self.chunk_size, -1)],
            dim=1,
        )
        residual = tokens
        hidden = self.norm(tokens)

        q_dim, kv_dim = self.heads * self.head_dim, self.kv_heads * self.head_dim
        query, key, value = self.qkv(hidden).split([q_dim, kv_dim, kv_dim], dim=-1)
        seq = hidden.shape[1]
        query = query.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)

        # The suffix sees the cached prefix and itself -- the same structure
        # `predict_velocity` gives the action expert.
        cached_k, cached_v = prefix_kv
        key = torch.cat([cached_k, key], dim=2)
        value = torch.cat([cached_v, value], dim=2)

        attended = self._attend(query, key, value).transpose(1, 2).reshape(batch, seq, q_dim)
        tokens = residual + self.o_proj(attended)
        gate, up = self.gate_up(self.mlp_norm(tokens)).chunk(2, dim=-1)
        tokens = tokens + self.down(F.silu(gate) * up)
        return self.action_out(tokens[:, 1:])


def build_draft_head(
    setup: dict, *, device: torch.device, dtype: torch.dtype, checkpoint: str | None
) -> LingbotDraftHead:
    """Both processes build the same head: CPU under a fixed seed, then moved.

    Identical weights on both devices is what lets a differential check tell a
    transport bug from initialisation noise, and it is what makes the untrained
    mode deterministic.
    """
    with torch.device("cpu"):
        torch.manual_seed(DRAFT_SEED)
        head = LingbotDraftHead(
            chunk_size=setup["chunk_size"],
            action_dim=setup["action_dim"],
            state_dim=setup["state_dim"],
            prefix_width=setup["prefix_width"],
            d_model=setup["d_model"],
        )
    if checkpoint:
        state = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)
        head.load_state_dict(state.get("model", state))
    return head.to(device=device, dtype=dtype).eval()


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Slot:
    name: str
    numel: int
    dtype: torch.dtype


def slot_table(setup: dict) -> tuple[Slot, ...]:
    """The three buffers, in the order both ranks must allocate and register them."""
    return (
        Slot("req", REQ_HEADER + setup["state_dim"], torch.float32),
        Slot("prefix", setup["prefix_len"] * setup["d_model"], torch.float16),
        Slot("reply", setup["chunk_size"] * setup["action_dim"], torch.float32),
    )


ONECCL_SUCCESS = 0
ONECCL_DTYPE = {torch.float16: 6, torch.float32: 7, torch.bfloat16: 9, torch.int8: 0, torch.uint8: 1}
UNIQUE_ID_BYTES = 4096


class _OneCCLUniqueId(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char * UNIQUE_ID_BYTES)]


class _OneCCL:
    """Minimal ctypes binding for the oneCCL v2 C API.

    Not vLLM's ``oneccl_igpu_communicator``: that wraps a ``torch.distributed``
    process group, and its pt2pt path is documented as accepting one dtype per
    packed transfer and hanging on repeated sequential transfers. What is needed
    here is three fixed-size registered buffers and a strict request/reply, so
    the raw API is both smaller and better behaved. ``onecclCommRegister`` is the
    reason it is fast: the plugin then runs its fd handshake once per buffer
    instead of once per message.
    """

    def __init__(self, lib_path: str) -> None:
        self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        self._comm = ctypes.c_void_p()
        lib = self.lib
        lib.onecclGetUniqueId.argtypes = [ctypes.POINTER(_OneCCLUniqueId)]
        lib.onecclSetDevice.argtypes = [ctypes.c_uint]
        lib.onecclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, _OneCCLUniqueId, ctypes.c_int,
        ]
        for name in ("onecclSend", "onecclRecv"):
            getattr(lib, name).argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ]
        lib.onecclMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.onecclCommRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
        ]

    @staticmethod
    def _check(name: str, code: int) -> None:
        if code != ONECCL_SUCCESS:
            raise RuntimeError(f"{name} failed with onecclResult_t={code}")

    def get_unique_id(self) -> bytes:
        uid = _OneCCLUniqueId()
        self._check("onecclGetUniqueId", self.lib.onecclGetUniqueId(ctypes.byref(uid)))
        return bytes(uid.data)

    def init(self, *, rank: int, uid_bytes: bytes, device: int) -> None:
        self._check("onecclSetDevice", self.lib.onecclSetDevice(device))
        uid = _OneCCLUniqueId()
        uid.data = uid_bytes.ljust(UNIQUE_ID_BYTES, b"\x00")[:UNIQUE_ID_BYTES]
        self._check("onecclCommInitRank", self.lib.onecclCommInitRank(ctypes.byref(self._comm), 2, uid, rank))

    def mem_alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self._check("onecclMemAlloc", self.lib.onecclMemAlloc(ctypes.byref(ptr), nbytes))
        return ptr.value

    def register(self, ptr: int, nbytes: int) -> None:
        handle = ctypes.c_void_p()
        self._check(
            "onecclCommRegister",
            self.lib.onecclCommRegister(self._comm, ctypes.c_void_p(ptr), nbytes, ctypes.byref(handle)),
        )

    def send(self, ptr: int, count: int, dtype: int, peer: int, stream: int) -> None:
        self._check("onecclSend", self.lib.onecclSend(
            ctypes.c_void_p(ptr), count, dtype, peer, self._comm, ctypes.c_void_p(stream)))

    def recv(self, ptr: int, count: int, dtype: int, peer: int, stream: int) -> None:
        self._check("onecclRecv", self.lib.onecclRecv(
            ctypes.c_void_p(ptr), count, dtype, peer, self._comm, ctypes.c_void_p(stream)))


class OneCCLTransport:
    """Registered fixed-size buffers over the oneCCL iGPU plugin."""

    def __init__(self, *, rank: int, device: torch.device, slots: tuple[Slot, ...], lib: str,
                 uid_file: Path, timeout: float) -> None:
        self.rank, self.peer, self.device = rank, 1 - rank, device
        self.ccl = _OneCCL(lib)
        if rank == DGPU_RANK:
            uid_file.write_bytes(self.ccl.get_unique_id())
        else:
            _wait_for(uid_file, timeout, what="the oneCCL unique id")
            time.sleep(0.2)  # the write is not atomic; let it land
        self.ccl.init(rank=rank, uid_bytes=uid_file.read_bytes(), device=rank)
        self.stream = torch.xpu.current_stream().sycl_queue
        # Only the iGPU rank stages through plugin-managed USM host memory; the
        # dGPU rank sends straight from ``data_ptr()``.
        self.staged = rank == IGPU_RANK
        self.buffers: dict[str, tuple[torch.Tensor, int, int, int]] = {}
        for slot in slots:
            tensor = torch.zeros(slot.numel, dtype=slot.dtype, device=device)
            nbytes = tensor.numel() * tensor.element_size()
            ptr = self.ccl.mem_alloc(nbytes) if self.staged else tensor.data_ptr()
            try:
                self.ccl.register(ptr, nbytes)
            except RuntimeError as exc:  # an optimisation, not a requirement
                logger.warning("onecclCommRegister unavailable (%s); using the unregistered path", exc)
            self.buffers[slot.name] = (tensor, ptr, tensor.numel(), ONECCL_DTYPE[slot.dtype])

    def send(self, slot: str, value: torch.Tensor) -> None:
        tensor, ptr, count, dtype = self.buffers[slot]
        tensor.copy_(value.reshape(-1))
        if self.staged:
            host = tensor.detach().contiguous().cpu()
            ctypes.memmove(ptr, host.data_ptr(), host.numel() * host.element_size())
        self.ccl.send(ptr, count, dtype, self.peer, self.stream)
        torch.xpu.synchronize(self.device)

    def recv(self, slot: str) -> torch.Tensor:
        tensor, ptr, count, dtype = self.buffers[slot]
        self.ccl.recv(ptr, count, dtype, self.peer, self.stream)
        torch.xpu.synchronize(self.device)
        if self.staged:
            host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
            ctypes.memmove(host.data_ptr(), ptr, host.numel() * host.element_size())
            tensor.copy_(host.to(tensor.device))
        return tensor

    def close(self) -> None:
        pass  # the comm dies with the process; destroying it mid-teardown hangs


class ShmTransport:
    """POSIX shared memory plus a sequence flag: 8-byte header, then the payload.

    Kept as the diagnostic arm. It needs no plugin, no ``LD_LIBRARY_PATH`` and no
    rendezvous file, and its wait sleeps instead of spinning, so it is what to
    reach for when a two-process result looks strange. It also found the protocol
    bug oneCCL's send queueing hid: this channel is a single-slot mailbox, which
    is why every request below is acknowledged.
    """

    SPIN_SECONDS = 0.05

    def __init__(self, *, rank: int, device: torch.device, slots: tuple[Slot, ...], prefix: str,
                 timeout: float) -> None:
        self.rank, self.device, self.timeout = rank, device, timeout
        create = rank == DGPU_RANK
        if not create:
            _wait_for(Path(f"/dev/shm/{prefix}_{slots[-1].name}"), timeout, what="the shared-memory segments")
        self.shm: dict[str, Any] = {}
        self.host: dict[str, torch.Tensor] = {}
        self.view: dict[str, np.ndarray] = {}
        self.resident: dict[str, torch.Tensor] = {}
        self.nbytes: dict[str, int] = {}
        self.seq: dict[str, int] = {}
        for slot in slots:
            nbytes = slot.numel * torch.empty(0, dtype=slot.dtype).element_size()
            name = f"{prefix}_{slot.name}"
            if create:
                segment = shared_memory.SharedMemory(name=name, create=True, size=nbytes + 8)
                segment.buf[:8] = struct.pack("<Q", 0)
            else:
                segment = shared_memory.SharedMemory(name=name)
            host = torch.empty(slot.numel, dtype=slot.dtype, device="cpu")
            self.shm[slot.name] = segment
            self.host[slot.name] = host
            # Staging buffers are allocated once and the device copies go
            # straight into them: writing ``value.cpu()`` instead costs an
            # allocation and an extra host copy per hop, which measured 6 ms on
            # the 293 KiB refresh.
            self.view[slot.name] = host.numpy().view("uint8")
            self.resident[slot.name] = torch.empty(slot.numel, dtype=slot.dtype, device=device)
            self.nbytes[slot.name] = nbytes
            self.seq[slot.name] = 0

    def send(self, slot: str, value: torch.Tensor) -> None:
        self.host[slot].copy_(value.reshape(-1))
        self.seq[slot] += 1
        segment, nbytes = self.shm[slot], self.nbytes[slot]
        segment.buf[8 : 8 + nbytes] = self.view[slot].data
        segment.buf[:8] = struct.pack("<Q", self.seq[slot])

    def recv(self, slot: str) -> torch.Tensor:
        self.seq[slot] += 1
        segment, nbytes, want = self.shm[slot], self.nbytes[slot], self.seq[slot]
        start = time.time()
        deadline = start + self.timeout
        while struct.unpack("<Q", bytes(segment.buf[:8]))[0] != want:
            now = time.time()
            if now > deadline:
                raise TimeoutError(f"shared-memory slot {slot!r} stalled at sequence {want}")
            # Spin briefly, then sleep: the worker's first wait spans the 6B
            # model load, and burning a core next to that load is not free.
            if now - start > self.SPIN_SECONDS:
                time.sleep(5e-5)
        self.view[slot][:] = np.frombuffer(segment.buf[8 : 8 + nbytes], dtype=np.uint8)
        self.resident[slot].copy_(self.host[slot])
        return self.resident[slot]

    def close(self) -> None:
        for segment in self.shm.values():
            segment.close()
            if self.rank == DGPU_RANK:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass


def _wait_for(path: Path, timeout: float, *, what: str) -> None:
    deadline = time.time() + timeout
    while not path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what} at {path}")
        time.sleep(0.01)


def make_transport(setup: dict, *, rank: int, device: torch.device, timeout: float):
    slots = slot_table(setup)
    directory = Path(setup["dir"])
    if setup["transport"] == "oneccl":
        return OneCCLTransport(
            rank=rank, device=device, slots=slots, lib=setup["ccl_lib"],
            uid_file=directory / "uid.bin", timeout=timeout,
        )
    return ShmTransport(
        rank=rank, device=device, slots=slots, prefix=setup["shm_prefix"], timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Server side
# ---------------------------------------------------------------------------
# Loaded here, in the parent, so the post-fork hook below only makes one libc
# call: `preexec_fn` runs between fork and exec, where allocating or taking a
# lock in a multithreaded parent can deadlock.
try:
    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:  # not glibc; the atexit path still covers a clean shutdown
    _LIBC = None
_PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """Ask the kernel to SIGTERM this child when the server process dies."""
    if _LIBC is not None:
        _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)


def reserve_cpus_for_worker(spec: str) -> list[int]:
    """Give the worker its cores and keep the rest, because its recv spins.

    ``onecclRecv`` hard-spins (1.00 core idle, against 0.05 under shm). On a
    12-core host that one spinner collides with this process's OpenMP pool, whose
    barriers active-wait, and ``processor.preprocess`` goes from 3.9 ms to
    218 ms -- fifty times what a speculative round saves. It slows the
    non-speculative path too, so it shows up as a *better* speedup rather than as
    a regression, which is why this is not left to chance.
    """
    reserved: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            low, high = (int(value) for value in part.split("-", 1))
            reserved.update(range(low, high + 1))
        else:
            reserved.add(int(part))
    available = sorted(set(os.sched_getaffinity(0)) - reserved)
    if not available:
        raise ValueError(f"spec_worker_cpu={spec!r} would leave the server process no CPUs")
    os.sched_setaffinity(0, available)
    torch.set_num_threads(len(available))
    logger.info(
        "LingBot draft worker pinned to CPUs %s; server process keeps %s (%d torch threads)",
        sorted(reserved), available, len(available),
    )
    return sorted(reserved)


class IGpuDraftClient:
    """The server's end of the protocol. Satisfies ``spec_decode.DraftBackend``.

    Spawn order matters: the worker is started first and only then does the
    server learn its prefix length and publish the rendezvous, so the worker's
    interpreter and XPU initialisation overlap the 6B weight load instead of
    following it.
    """

    def __init__(
        self,
        *,
        config: Any,
        prefix_len: int,
        prefix_width: int,
        device: torch.device,
        dtype: torch.dtype,
        timeout: float = 600.0,
    ) -> None:
        transport = os.environ.get("VLLM_LINGBOT_SPEC_TRANSPORT", "oneccl")
        if transport not in ("oneccl", "shm"):
            raise ValueError(f"VLLM_LINGBOT_SPEC_TRANSPORT must be 'oneccl' or 'shm'; got {transport!r}")
        directory = Path(os.environ.get("VLLM_LINGBOT_SPEC_DIR", f"/tmp/lingbot-spec-{os.getpid()}"))
        directory.mkdir(parents=True, exist_ok=True)
        self.setup = {
            "dir": str(directory),
            "transport": transport,
            "ccl_lib": os.environ.get("VLLM_LINGBOT_SPEC_CCL_LIB", DEFAULT_CCL_LIB),
            "shm_prefix": f"lingbotspec{os.getpid()}",
            "prefix_len": int(prefix_len),
            "prefix_width": int(prefix_width),
            "d_model": DRAFT_WIDTH,
            "chunk_size": int(config.chunk_size),
            "action_dim": int(config.max_action_dim),
            "state_dim": int(config.max_state_dim),
            "dtype": str(dtype).removeprefix("torch."),
            "draft_path": config.spec_draft_path,
            "worker_cpu": config.spec_worker_cpu,
        }
        self.device, self.dtype = device, dtype
        self.seq = 0
        self._closed = False
        self.request = torch.zeros(REQ_HEADER + self.setup["state_dim"], dtype=torch.float32, device=device)

        (directory / "uid.bin").unlink(missing_ok=True)
        for stale in Path("/dev/shm").glob(f"{self.setup['shm_prefix']}_*"):
            stale.unlink(missing_ok=True)
        if config.spec_worker_cpu:
            reserve_cpus_for_worker(config.spec_worker_cpu)
        elif transport == "oneccl" and os.environ.get("OMP_WAIT_POLICY", "").upper() != "PASSIVE":
            logger.warning(
                "spec_worker_cpu is unset and oneCCL's recv hard-spins a core. On this host that collides "
                "with the OpenMP pool and adds ~215 ms to every full round's preprocess -- and because it "
                "slows the non-speculative path too, it inflates the apparent speculative speedup. Set "
                "spec_worker_cpu (e.g. \"11\") or export OMP_WAIT_POLICY=PASSIVE."
            )

        self.process = self._spawn(directory)
        # An orphaned worker blocks in `recv` forever while holding a core (its
        # oneCCL wait spins), so it must not outlive this process. `atexit`
        # covers an ordinary shutdown; PR_SET_PDEATHSIG in `_spawn` covers a
        # crash, where no Python cleanup runs at all.
        atexit.register(self.close)
        # Published after the spawn so the worker boots in parallel, and it is
        # what tells the worker how to size its buffers -- the prefix length is a
        # property of the checkpoint's align layout, not a constant.
        (directory / "setup.json").write_text(json.dumps(self.setup, indent=2) + "\n")
        try:
            self.transport = make_transport(self.setup, rank=DGPU_RANK, device=device, timeout=timeout)
        except Exception:
            self.process.terminate()
            raise
        self.projector = build_draft_head(
            self.setup, device=device, dtype=dtype, checkpoint=config.spec_draft_path
        )
        logger.info(
            "LingBot draft worker ready: pid %d, %s transport, prefix %d x %d -> %d, rendezvous %s",
            self.process.pid, transport, prefix_len, prefix_width, DRAFT_WIDTH, directory,
        )

    def _spawn(self, directory: Path) -> subprocess.Popen:
        env = dict(os.environ)
        env["ZE_AFFINITY_MASK"] = str(IGPU_RANK)  # 1 is the iGPU on this host
        env.pop("ONEAPI_DEVICE_SELECTOR", None)  # would re-enumerate under the mask
        env.setdefault("OMP_NUM_THREADS", "2")
        command = [sys.executable, "-u", "-m", WORKER_MODULE, "--role", "worker", "--dir", str(directory)]
        if self.setup["worker_cpu"]:
            # The server already dropped these CPUs in `reserve_cpus_for_worker`;
            # this is the other half of the split.
            command = ["taskset", "-c", self.setup["worker_cpu"], *command]
        log = directory / "worker.log"
        return subprocess.Popen(
            command, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT, preexec_fn=_die_with_parent
        )

    # -- DraftBackend ------------------------------------------------------
    def _send_request(self, opcode: int, state: torch.Tensor | None) -> None:
        self.seq += 1
        self.request[0] = float(opcode)
        self.request[1] = float(self.seq)
        if state is not None:
            self.request[REQ_HEADER:] = state.reshape(-1).to(torch.float32)
        self.transport.send("req", self.request)

    def _reply(self) -> torch.Tensor:
        reply = self.transport.recv("reply")
        # ``copy=True``: oneCCL hands back the registered buffer itself, and the
        # caller keeps the chunk alive across the verify steps.
        return reply.view(1, self.setup["chunk_size"], self.setup["action_dim"]).to(self.dtype, copy=True)

    @torch.no_grad()
    def refresh(self, prefix_embs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        hidden = self.projector.project(prefix_embs)
        self._send_request(OP_REFRESH, state)
        self.transport.send("prefix", hidden.to(torch.float16))
        return self._reply()

    @torch.no_grad()
    def draft(self, state: torch.Tensor) -> torch.Tensor:
        self._send_request(OP_DRAFT, state)
        return self._reply()

    def close(self) -> None:
        # Idempotent: both `SpecDecoder.close()` and the atexit hook call this.
        if self._closed:
            return
        self._closed = True
        try:
            self._send_request(OP_SHUTDOWN, None)
        except Exception as exc:  # the worker may already be gone
            logger.warning("LingBot draft worker shutdown message failed: %s", exc)
        self.transport.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                logger.warning("LingBot draft worker did not exit; terminating")
                self.process.terminate()


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------
def run_worker(directory: Path, timeout: float) -> int:
    """The resident iGPU process. One outstanding message, every request answered.

        DRAFT     recv req(opcode=1, state)                  -> send reply
        REFRESH   recv req(opcode=2, state) + recv prefix     -> send reply
        SHUTDOWN  recv req(opcode=0)                           exit

    ``REFRESH`` is acknowledged, and its reply is the draft for the state it
    carried. The first version made it fire-and-forget so the iGPU could rebuild
    its cache inside the dGPU's own denoise loop; oneCCL tolerated that because
    sends queue, but the shm channel is a single-slot mailbox and the following
    ``DRAFT`` overwrote the unread ``REFRESH``, deadlocking both sides. The
    acknowledgement is what makes one-message-outstanding true on both
    transports, and it pays for itself: the server gets a draft for the very
    frame it is computing the teacher's answer for, which is the pair that
    predicts acceptance.
    """
    setup_file = directory / "setup.json"
    _wait_for(setup_file, timeout, what="the server's setup.json")
    setup = json.loads(setup_file.read_text())

    if not torch.xpu.is_available():
        raise SystemExit("the LingBot draft worker sees no XPU; it must be spawned with ZE_AFFINITY_MASK=1")
    device = torch.device("xpu:0")
    dtype = getattr(torch, setup["dtype"])
    properties = torch.xpu.get_device_properties(0)
    print(
        f"[draft] {properties.name}, {properties.total_memory / 2**30:.2f} GiB, "
        f"eu {getattr(properties, 'gpu_eu_count', '?')}, ZE_AFFINITY_MASK="
        f"{os.environ.get('ZE_AFFINITY_MASK')}, transport={setup['transport']}",
        flush=True,
    )

    head = build_draft_head(setup, device=device, dtype=dtype, checkpoint=setup.get("draft_path"))
    transport = make_transport(setup, rank=IGPU_RANK, device=device, timeout=timeout)
    parameters = sum(tensor.numel() for tensor in head.parameters())
    print(f"[draft] ready, {parameters / 1e6:.2f} M parameters", flush=True)

    cached: tuple[torch.Tensor, torch.Tensor] | None = None
    state_dim, chunk, action_dim = setup["state_dim"], setup["chunk_size"], setup["action_dim"]
    drafts = refreshes = 0
    expected_seq = 0
    while True:
        request = transport.recv("req").cpu()  # 228 B; the D2H is inside the numbers
        opcode, seq = int(request[0].item()), int(request[1].item())
        expected_seq += 1
        if seq != expected_seq:
            # Loud, because a dropped or duplicated message otherwise shows up as
            # a plausible latency computed from the wrong state.
            raise RuntimeError(f"draft protocol desync: got sequence {seq}, expected {expected_seq}")
        if opcode == OP_SHUTDOWN:
            break
        if opcode == OP_REFRESH:
            hidden = transport.recv("prefix").view(1, setup["prefix_len"], setup["d_model"])
            with torch.no_grad():
                cached = head.kv_from_projection(hidden.to(dtype))
            refreshes += 1
        elif opcode != OP_DRAFT:
            raise RuntimeError(f"unknown draft opcode {opcode}")
        if cached is None:
            raise RuntimeError("DRAFT arrived before any REFRESH: the session has no prefix cache")

        state = request[REQ_HEADER:].view(1, state_dim).to(device=device, dtype=dtype)
        with torch.no_grad():
            x0_draft = head(state, cached)
        transport.send("reply", x0_draft.reshape(1, chunk, action_dim).to(torch.float32))
        if opcode == OP_DRAFT:
            drafts += 1

    print(f"[draft] exiting after {drafts} drafts and {refreshes} refreshes", flush=True)
    transport.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LingBot-VLA 2.0 iGPU draft worker")
    parser.add_argument("--role", choices=("worker",), default="worker")
    parser.add_argument("--dir", required=True, help="rendezvous directory holding setup.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    return run_worker(Path(args.dir), args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["IGpuDraftClient", "LingbotDraftHead", "build_draft_head", "reserve_cpus_for_worker"]
