#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 10 gate 4 — can a draft head live on the iGPU?

The task is speculative inference with the **draft on the iGPU and the verifier
on the dGPU**, so this probe prices the draft on both devices and prices what
running it costs the dGPU. Three numbers decide the architecture:

1. **Draft latency on the iGPU.** The budget is set by the thing it runs in
   front of: one `predict_velocity` is 21.3 ms on the dGPU (PHASE9 P4), so a
   speculative round is `draft + K*21.3`. A draft that costs 20 ms has doubled a
   K=1 round before the verifier starts.
2. **The same on the dGPU**, which gives our own `k` for *this* shape rather
   than §K's borrowed 12.9x for the MoE GEMM.
3. **The contention tax.** §K's rule is that iGPU work is free only when it does
   not keep the EU array busy. `--role busy` drives the draft at a realistic duty
   cycle so a second process can measure what the dGPU pays.

Two candidate shapes, because the choice is not obvious and the wrong one kills
the design:

* **`wide`** — FLASH's actual architecture (`draft.py:49`): one full-width VLM
  decoder layer over `prefix + state + M queries`, re-encoding the prefix every
  tick. At LingBot's width (2560 hidden, 9728 intermediate, 32/8 heads) that is
  ~101 M parameters, 202 MiB in fp16. On a device with 29 GB/s of read bandwidth
  (§K) the weights alone are ~7 ms before a single FLOP is useful.
* **`narrow`** — the shape this plan proposes instead: project the prefix to 512
  **once per full round, on the dGPU**, ship 286x512 fp16 = 293 KiB, and keep it
  resident on the iGPU. Per tick the iGPU then runs 51 tokens through one 512-wide
  layer attending to the cached prefix -- ~2.3 M parameters. This is only possible
  because a speculative round reuses a cached prefix anyway; it is the same
  staleness assumption the verifier already makes, not a new one.

Both are built with random weights: this measures the *shape*, not the model, and
it must run before anything is trained. Usage, inside the container:

    # dGPU arm
    ONEAPI_DEVICE_SELECTOR=level_zero:0 PYTHONPATH=. \\
        python spikes/lingbot_vla_v2/phase10_igpu_draft_cost_probe.py

    # iGPU arm
    ONEAPI_DEVICE_SELECTOR=level_zero:1 PYTHONPATH=. \\
        python spikes/lingbot_vla_v2/phase10_igpu_draft_cost_probe.py

    # contention arm: leave this running, then time the dGPU request elsewhere
    ONEAPI_DEVICE_SELECTOR=level_zero:1 PYTHONPATH=. \\
        python spikes/lingbot_vla_v2/phase10_igpu_draft_cost_probe.py \\
            --role busy --shape narrow --duty 0.5

`torch.xpu.device_count()` is 1 whatever the selector (verified on this container,
torch 2.12.0+xpu, including `level_zero:*`), so the selected device is always
`xpu:0` and the two arms are two processes. That is §K's finding and it is why
this is a probe rather than a flag.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# The released LingBot-VLA 2.0 shapes, read from the prepared checkpoint's
# configs rather than assumed. VLM tower: Qwen3-VL, hidden 2560 / intermediate
# 9728 / 32 heads / 8 kv heads / head_dim 128. Prefix is 3*66 + 72 + 8 + 8 = 286.
VLM_HIDDEN = 2560
VLM_INTERMEDIATE = 9728
VLM_HEADS = 32
VLM_KV_HEADS = 8
VLM_HEAD_DIM = 128
PREFIX_LEN = 286
CHUNK = 50
ACTION_DIM = 55
STATE_DIM = 55

# One dGPU denoise step, PHASE9 P4's measured figure -- the budget this is read against.
DENOISE_STEP_MS = 21.3


def sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """GQA attention, without materialising the repeated kv heads when possible."""
    if query.shape[1] != key.shape[1]:
        try:
            return F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
        except TypeError:  # older torch without enable_gqa
            repeat = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(query, key, value)


class NarrowDraftHead(nn.Module):
    """51 tokens through one narrow layer, attending to a **cached** prefix.

    The prefix projection is deliberately not part of the per-tick path: it runs
    on the dGPU during a full round and its output is shipped once. `refresh()`
    is here only so the probe can price that hop separately.
    """

    def __init__(self, *, d_model: int = 512, heads: int = 8, kv_heads: int = 2, head_dim: int = 64,
                 intermediate: int = 1024) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        q_dim, kv_dim = heads * head_dim, kv_heads * head_dim

        self.prefix_proj = nn.Linear(VLM_HIDDEN, d_model, bias=False)  # full rounds only
        self.prefix_kv = nn.Linear(d_model, 2 * kv_dim, bias=False)  # full rounds only

        self.state_proj = nn.Linear(STATE_DIM, d_model, bias=False)
        self.queries = nn.Embedding(CHUNK, d_model)
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, q_dim + 2 * kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, d_model, bias=False)
        self.mlp_norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
        self.gate_up = nn.Linear(d_model, 2 * intermediate, bias=False)
        self.down = nn.Linear(intermediate, d_model, bias=False)
        self.action_out = nn.Linear(d_model, ACTION_DIM, bias=False)

    @torch.no_grad()
    def project(self, prefix_embs: torch.Tensor) -> torch.Tensor:
        """``[B,286,2560]`` -> ``[B,286,512]``, the payload that crosses devices.

        Split out of `refresh` because the two halves run on different devices in
        the two-process arm (`phase10_draft_worker.py`): this half runs on the
        dGPU during a full round, and only its 293 KiB output is shipped.
        """
        return self.prefix_proj(prefix_embs)

    @torch.no_grad()
    def kv_from_projection(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B,286,512]`` -> cached k/v at the draft's width. Runs where the draft runs."""
        key, value = self.prefix_kv(hidden).chunk(2, dim=-1)
        shape = (*key.shape[:2], self.kv_heads, self.head_dim)
        return key.view(shape).transpose(1, 2), value.view(shape).transpose(1, 2)

    @torch.no_grad()
    def refresh(self, prefix_embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-round only: ``[B,286,2560]`` -> cached k/v at the draft's width."""
        return self.kv_from_projection(self.project(prefix_embs))

    def forward(self, state: torch.Tensor, prefix_kv: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch = state.shape[0]
        tokens = torch.cat(
            [
                self.state_proj(state)[:, None, :],
                self.queries.weight[None].expand(batch, CHUNK, -1),
            ],
            dim=1,
        )
        residual = tokens
        hidden = self.norm(tokens)

        q_dim = self.heads * self.head_dim
        kv_dim = self.kv_heads * self.head_dim
        query, key, value = self.qkv(hidden).split([q_dim, kv_dim, kv_dim], dim=-1)
        seq = hidden.shape[1]
        query = query.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)

        # Suffix rows see the cached prefix and each other -- the same structure
        # `predict_velocity` uses, so the shape is comparable to a denoise step.
        cached_k, cached_v = prefix_kv
        key = torch.cat([cached_k, key], dim=2)
        value = torch.cat([cached_v, value], dim=2)

        attended = sdpa(query, key, value).transpose(1, 2).reshape(batch, seq, q_dim)
        tokens = residual + self.o_proj(attended)

        gate, up = self.gate_up(self.mlp_norm(tokens)).chunk(2, dim=-1)
        tokens = tokens + self.down(F.silu(gate) * up)
        return self.action_out(tokens[:, 1:])


class WideDraftHead(nn.Module):
    """FLASH's shape at LingBot's width: one full VLM-width layer, prefix re-encoded."""

    def __init__(self) -> None:
        super().__init__()
        q_dim, kv_dim = VLM_HEADS * VLM_HEAD_DIM, VLM_KV_HEADS * VLM_HEAD_DIM
        self.state_proj = nn.Linear(STATE_DIM, VLM_HIDDEN, bias=False)
        self.queries = nn.Embedding(CHUNK, VLM_HIDDEN)
        self.norm = nn.RMSNorm(VLM_HIDDEN) if hasattr(nn, "RMSNorm") else nn.LayerNorm(VLM_HIDDEN)
        self.qkv = nn.Linear(VLM_HIDDEN, q_dim + 2 * kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, VLM_HIDDEN, bias=False)
        self.mlp_norm = nn.RMSNorm(VLM_HIDDEN) if hasattr(nn, "RMSNorm") else nn.LayerNorm(VLM_HIDDEN)
        self.gate_up = nn.Linear(VLM_HIDDEN, 2 * VLM_INTERMEDIATE, bias=False)
        self.down = nn.Linear(VLM_INTERMEDIATE, VLM_HIDDEN, bias=False)
        self.action_out = nn.Linear(VLM_HIDDEN, ACTION_DIM, bias=False)

    def forward(self, state: torch.Tensor, prefix_embs: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        tokens = torch.cat(
            [
                prefix_embs,
                self.state_proj(state)[:, None, :],
                self.queries.weight[None].expand(batch, CHUNK, -1),
            ],
            dim=1,
        )
        residual = tokens
        hidden = self.norm(tokens)

        q_dim, kv_dim = VLM_HEADS * VLM_HEAD_DIM, VLM_KV_HEADS * VLM_HEAD_DIM
        query, key, value = self.qkv(hidden).split([q_dim, kv_dim, kv_dim], dim=-1)
        seq = hidden.shape[1]
        query = query.view(batch, seq, VLM_HEADS, VLM_HEAD_DIM).transpose(1, 2)
        key = key.view(batch, seq, VLM_KV_HEADS, VLM_HEAD_DIM).transpose(1, 2)
        value = value.view(batch, seq, VLM_KV_HEADS, VLM_HEAD_DIM).transpose(1, 2)

        attended = sdpa(query, key, value).transpose(1, 2).reshape(batch, seq, q_dim)
        tokens = residual + self.o_proj(attended)

        gate, up = self.gate_up(self.mlp_norm(tokens)).chunk(2, dim=-1)
        tokens = tokens + self.down(F.silu(gate) * up)
        return self.action_out(tokens[:, -CHUNK:])


def parameter_bytes(module: nn.Module, exclude: tuple[str, ...] = ()) -> tuple[int, int]:
    params, nbytes = 0, 0
    for name, tensor in module.named_parameters():
        if any(name.startswith(prefix) for prefix in exclude):
            continue
        params += tensor.numel()
        nbytes += tensor.numel() * tensor.element_size()
    return params, nbytes


def timed(fn, device: torch.device, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize(device)
    samples = []
    for _ in range(iters):
        torch.xpu.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.xpu.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def build(shape: str, device: torch.device, dtype: torch.dtype, d_model: int):
    """Return ``(head, per_tick_callable, exclude_from_per_tick_weights)``."""
    batch = 1
    state = torch.randn(batch, STATE_DIM, device=device, dtype=dtype)
    if shape == "narrow":
        head = NarrowDraftHead(d_model=d_model).to(device=device, dtype=dtype).eval()
        prefix_embs = torch.randn(batch, PREFIX_LEN, VLM_HIDDEN, device=device, dtype=dtype)
        with torch.no_grad():
            cached = head.refresh(prefix_embs)
        return head, (lambda: head(state, cached)), ("prefix_proj", "prefix_kv"), (
            lambda: head.refresh(prefix_embs)
        )
    head = WideDraftHead().to(device=device, dtype=dtype).eval()
    prefix_embs = torch.randn(batch, PREFIX_LEN, VLM_HIDDEN, device=device, dtype=dtype)
    return head, (lambda: head(state, prefix_embs)), (), None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("bench", "busy"), default="bench")
    parser.add_argument("--shape", choices=("narrow", "wide", "both"), default="both")
    parser.add_argument("--d-model", type=int, default=512, help="narrow candidate's width")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--duty", type=float, default=0.5, help="--role busy: fraction of wall time running")
    parser.add_argument("--seconds", type=float, default=120.0, help="--role busy: how long to run")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("no XPU visible; set ONEAPI_DEVICE_SELECTOR=level_zero:0 (dGPU) or :1 (iGPU)")
    device = torch.device("xpu:0")
    dtype = getattr(torch, args.dtype)
    properties = torch.xpu.get_device_properties(0)
    selector = os.environ.get("ONEAPI_DEVICE_SELECTOR", "<unset>")
    print(f"[device] ONEAPI_DEVICE_SELECTOR={selector} -> {properties.name}, "
          f"{properties.total_memory / 2**30:.2f} GiB, eu {getattr(properties, 'gpu_eu_count', '?')}, "
          f"device_count {torch.xpu.device_count()}")

    shapes = ("narrow", "wide") if args.shape == "both" else (args.shape,)

    if args.role == "busy":
        shape = shapes[0]
        _, call, _, _ = build(shape, device, dtype, args.d_model)
        period = max(1e-4, statistics.median(timed(call, device, warmup=5, iters=20)) / 1000.0)
        idle = period * (1.0 - args.duty) / max(args.duty, 1e-6)
        print(f"[busy] {shape} draft, {period * 1000:.2f} ms/call, duty {args.duty:.0%} "
              f"-> sleeping {idle * 1000:.2f} ms between calls, for {args.seconds:.0f}s", flush=True)
        print("[busy] now time the dGPU request in another process", flush=True)

        # Heartbeat, because a silently-dead load process and a genuinely free
        # load look **identical** from the victim's side. The effective duty it
        # reports is what the victim's number should be read against -- not the
        # requested one.
        started = time.time()
        deadline = started + args.seconds
        calls, busy_seconds, next_beat = 0, 0.0, started + 5.0
        while time.time() < deadline:
            mark = time.perf_counter()
            call()
            torch.xpu.synchronize(device)
            busy_seconds += time.perf_counter() - mark
            calls += 1
            if idle > 0:
                time.sleep(idle)
            now = time.time()
            if now >= next_beat:
                elapsed = now - started
                print(f"[busy] +{elapsed:6.1f}s  calls {calls:7d}  "
                      f"effective duty {busy_seconds / elapsed:5.1%}  "
                      f"{busy_seconds / calls * 1000:.2f} ms/call", flush=True)
                next_beat = now + 5.0
        elapsed = time.time() - started
        print(f"[busy] done: {calls} calls in {elapsed:.1f}s, effective duty "
              f"{busy_seconds / elapsed:.1%}", flush=True)
        return 0

    results = {}
    for shape in shapes:
        head, call, exclude, refresh = build(shape, device, dtype, args.d_model)
        params, nbytes = parameter_bytes(head, exclude)
        total_params, total_bytes = parameter_bytes(head)
        samples = timed(call, device, warmup=args.warmup, iters=args.iters)
        median = statistics.median(samples)
        entry = {
            "per_tick_ms_median": median,
            "per_tick_ms_p10": statistics.quantiles(samples, n=10)[0],
            "per_tick_ms_p90": statistics.quantiles(samples, n=10)[-1],
            "per_tick_params_M": params / 1e6,
            "per_tick_weight_MiB": nbytes / 2**20,
            "total_params_M": total_params / 1e6,
            "total_weight_MiB": total_bytes / 2**20,
        }
        if refresh is not None:
            entry["refresh_ms_median"] = statistics.median(timed(refresh, device, warmup=5, iters=30))
        results[shape] = entry
        del head
        torch.xpu.empty_cache()

    print(f"\n{'shape':>8s} {'per-tick ms':>12s} {'p10-p90':>16s} {'params M':>10s} "
          f"{'weights MiB':>12s} {'refresh ms':>11s} {'vs 1 denoise step':>18s}")
    print("-" * 92)
    for shape, entry in results.items():
        refresh = f"{entry['refresh_ms_median']:11.2f}" if "refresh_ms_median" in entry else f"{'-':>11s}"
        ratio = entry["per_tick_ms_median"] / DENOISE_STEP_MS
        print(f"{shape:>8s} {entry['per_tick_ms_median']:12.2f} "
              f"{entry['per_tick_ms_p10']:7.2f}-{entry['per_tick_ms_p90']:<8.2f} "
              f"{entry['per_tick_params_M']:10.2f} {entry['per_tick_weight_MiB']:12.1f} {refresh} "
              f"{ratio:17.2f}x")

    print(
        f"\nBudget: a dGPU denoise step is {DENOISE_STEP_MS} ms (PHASE9 P4), so a K=1 speculative\n"
        f"round costs `draft + {DENOISE_STEP_MS}` and a K=2 round `draft + {2 * DENOISE_STEP_MS:.1f}`.\n"
        "Gate 4 passes if the chosen shape's iGPU time leaves the round well under the\n"
        f"{10 * DENOISE_STEP_MS:.0f} ms full-round denoise it replaces. Run this on both selectors to get k."
    )

    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"args": vars(args), "device": properties.name, "results": results}, indent=2) + "\n")
        print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
