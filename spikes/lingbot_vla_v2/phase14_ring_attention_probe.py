#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 14 — ring attention across the iGPU and the dGPU, for the denoise loop.

The question: run the ten-step denoise loop on **both** GPUs at once, with ring
attention (sequence parallelism) joining the halves, and see whether it is
faster than the 213.5 ms the dGPU takes alone.

Four things already on file bound the answer before any code runs, and they are
why this file is a probe and not a feature:

* **§K1** — torch-xpu enumerates one Level-Zero platform per process, so "both
  GPUs" is two processes, one ``ZE_AFFINITY_MASK`` each. There is no
  ``.to("xpu:1")``.
* **§K2** — the iGPU runs the real MoE layer-step in 3.246 ms against the dGPU's
  0.251 ms (``k`` = 12.9x), because it reads its weights from host LPDDR5 at
  29 GB/s against the B60's 449.
* **§M.6** — one dGPU<->iGPU round trip is **0.155 ms** (oneCCL, registered),
  against **0.251 ms** for a whole 32-expert block. The loop is 36 layers x 10
  steps = **360** layer-steps, so any per-layer-step exchange starts 55.8 ms in
  the hole.
* **§G5** — splitting the 51 suffix rows is capped near 13% *even with a free
  iGPU and zero communication*, because the loop's cost is expert weight bytes
  and those do not scale with row count.

Ring attention parallelises **attention**. This loop is bound on streaming MoE
expert weights. So the expected result is a large regression -- but two numbers
that decide it have never been measured on this model, and this file measures
them rather than asserting them:

  1. **what fraction of a denoise layer-step attention actually is** (arm
     ``attn-share``). That is the ceiling for any attention-only split, and
     §F2's layer-step decomposition never isolated it.
  2. **what a ring exchange costs at the real ring payload shapes** (arms
     ``ring-2p``). §M.6 priced a 76.5 KiB MoE activation exchange; the ring's
     payloads are different (a 408 KiB query, a 408 KiB partial output).

Arms, and the device each wants::

    ZE_AFFINITY_MASK=0 python phase14_ring_attention_probe.py --arm ring-math   --model ...
    ZE_AFFINITY_MASK=0 python phase14_ring_attention_probe.py --arm attn-share  --model ...
    ZE_AFFINITY_MASK=0 python phase14_ring_attention_probe.py --arm ring-2p --split kv  --model ...
    ZE_AFFINITY_MASK=0 python phase14_ring_attention_probe.py --arm ring-2p --split seq --model ...

``ring-math`` is the gate: until the 2-shard ring reproduces ``eager_attention``
at the real shapes, nothing downstream is measuring ring attention. It is the
only arm that needs no second device.

``ring-2p`` spawns its own iGPU worker (``--role worker``), so it is the only
arm that touches both cards.

The two splits are the same kernel with different sharding:

``--split kv`` — **context parallel.** The 286-token prefix KV cache is split
    143/143 across the cards; the 51 queries are replicated. The iGPU computes a
    partial ``(out, lse)`` over its half and ships it back; the dGPU merges and
    runs everything else. The iGPU does attention only, which is the smallest
    burden that can be put on it, and the ceiling is half the attention time.

``--split seq`` — **sequence parallel**, the literal ask. Both ranks hold the
    6B model; the 51 suffix rows are split 26/25; each rank computes its own
    q/k/v, one ring step exchanges the peer's suffix K/V, each merges its
    partials and runs its own MoE over its own rows. The prefix KV (42.2 MB) is
    replicated once per request, not per step.

Host state matters (Rules #2): ``load < 2.0`` and no other container holding a
GPU. Every arm prints the load average it started at; §L had to discard a full
set of absolutes for ignoring this.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase5_latency import build, compile_denoise_step, observation  # noqa: E402
from phase10_draft_worker import (  # noqa: E402
    DGPU_RANK,
    IGPU_RANK,
    OneCCLTransport,
    ShmTransport,
    Slot,
    reserve_cpus_for_worker,
)
from phase10_ipc_probe import DEFAULT_CCL_LIB  # noqa: E402
from phase12_paradigms_probe import ground, real_dims, rms  # noqa: E402

# The recorded baselines every latency number here is scored against.
# PHASE8 §11 / §M: full round 294.3 ms, grounding 80.8, so the loop is 213.5.
BASELINE_LOOP_MS = 213.5
BASELINE_REQUEST_MS = 294.0
# PHASE10 §10: the fp16 noise floor of an action chunk in per-dim RMS units.
FP16_FLOOR = 3.9e-3
# config.spec_tau -- "close enough that the accept rule would not have noticed".
SPEC_TAU = 0.15
# §M.6's measured oneCCL round trip, for the projections the arms print.
MEASURED_HOP_MS = 0.155


def sync(device: torch.device) -> None:
    if device.type in ("xpu", "cuda"):
        torch.accelerator.synchronize()


def load_average() -> float:
    return os.getloadavg()[0]


def median_ms(fn, *, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# The ring kernel
# ---------------------------------------------------------------------------
# `vllm_omni/diffusion/attention/backends/ring_pytorch_attn.py` already ships a
# ring attention, and this is deliberately *not* it: that one is
# `torch.distributed`-based and masks only `causal` / `joint`, while the denoise
# attention carries an arbitrary `[B, Lq, Lk]` bool block mask built by
# `make_att_2d_masks` + `_block_query_columns`. What is shared is the algebra,
# and `arm ring-math` cross-checks the merge below against that file's
# `update_out_and_lse` so this is not a private variant of it.


def merge_out_lse(
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax merge of two *normalised* partial attentions.

    ``out`` is ``[B, H, Lq, D]`` and ``lse`` is ``[B, H, Lq, 1]``, both fp32.

    The ``-inf`` guard is the one thing a causal-only implementation never
    needs. Here a *shard* can be fully masked for a row even when the row is
    not globally masked -- the suffix's ``att_masks`` hide the action tokens
    from the state token, so a shard holding only action keys is dead for that
    row -- and then ``lse`` is ``-inf`` on both sides and the rescale is 0/0.
    """
    if out is None or lse is None:
        return block_out, block_lse
    merged_lse = torch.logaddexp(lse, block_lse)
    dead = torch.isneginf(merged_lse)
    safe = torch.where(dead, torch.zeros_like(merged_lse), merged_lse)
    weight_a = torch.where(dead, torch.zeros_like(safe), torch.exp(lse - safe))
    weight_b = torch.where(dead, torch.zeros_like(safe), torch.exp(block_lse - safe))
    return out * weight_a + block_out * weight_b, merged_lse


def block_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    accum: str = "fp32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One ring block: normalised partial attention plus its log-sum-exp.

    Same contract as ``eager_attention`` -- ``[B, L, H, D]`` q/k/v with GQA and a
    ``[B, Lq, Lk]`` bool mask -- but returns ``(out [B, H, Lq, D], lse
    [B, H, Lq, 1])`` instead of a finished softmax, so blocks can be merged.

    ``accum="fp32"`` runs the softmax and the ``probs @ v`` in fp32 regardless of
    the input dtype; ``accum="input"`` keeps the model dtype, which is what
    ``eager_attention`` does and therefore what a bit-comparison needs.
    """
    if accum not in ("fp32", "input"):
        raise ValueError(f"accum must be fp32 or input; got {accum!r}")
    bsize, q_len, num_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    groups = num_heads // num_kv_heads
    if groups > 1:
        key = key.repeat_interleave(groups, dim=2)
        value = value.repeat_interleave(groups, dim=2)

    q = query.transpose(1, 2)  # [B, H, Lq, D]
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim**-0.5)
    if accum == "fp32":
        scores = scores.float()
        v = v.float()
    scores = scores.masked_fill(~mask[:, None, :, :], float("-inf"))

    row_max = scores.amax(dim=-1, keepdim=True)  # -inf where the block is dead
    alive = torch.isfinite(row_max)
    shift = torch.where(alive, row_max, torch.zeros_like(row_max))
    probs = torch.exp(scores - shift)  # exactly 0 in the masked columns
    denom = probs.sum(dim=-1, keepdim=True)
    safe_denom = torch.where(alive, denom, torch.ones_like(denom))

    out = torch.matmul(probs.to(v.dtype), v) / safe_denom.to(v.dtype)
    out = torch.where(alive, out, torch.zeros_like(out))
    lse = torch.where(alive, shift + torch.log(safe_denom), torch.full_like(row_max, float("-inf")))
    return out.float(), lse.float()


def even_bounds(length: int, shards: int) -> list[tuple[int, int]]:
    """``shards`` contiguous, near-equal spans covering ``[0, length)``."""
    if shards < 1 or shards > length:
        raise ValueError(f"cannot split {length} keys into {shards} shards")
    edges = [round(i * length / shards) for i in range(shards + 1)]
    return [(edges[i], edges[i + 1]) for i in range(shards)]


def ring_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    shards: int = 2,
    accum: str = "fp32",
) -> torch.Tensor:
    """``eager_attention``'s output, computed one key shard at a time.

    Drop-in for ``eager_attention``: returns ``[B, Lq, H * D]`` in ``query``'s
    dtype. ``shards=1`` is the degenerate case and exists so ``ring-math`` can
    separate "the merge is wrong" from "the block kernel is wrong".
    """
    bsize, q_len, num_heads, head_dim = query.shape
    out = lse = None
    for start, end in even_bounds(key.shape[1], shards):
        block_out, block_lse = block_attention(
            query, key[:, start:end], value[:, start:end], mask[:, :, start:end], accum=accum
        )
        out, lse = merge_out_lse(out, lse, block_out, block_lse)
    return out.transpose(1, 2).reshape(bsize, q_len, num_heads * head_dim).to(query.dtype)


# ---------------------------------------------------------------------------
# Patching the model's attention
# ---------------------------------------------------------------------------
# `LingbotJointModel.forward` resolves `eager_attention` as a module global, so
# rebinding the module attribute is enough to swap the kernel -- no subclassing
# and no edit under `vllm_omni/`. §F1's lesson applies in the other direction
# too: `attention_backend` is read per call, so this only takes effect for the
# `eager` backend the deployed config selects.


class attention_patch:
    """Context manager rebinding ``modeling.eager_attention`` to ``fn``."""

    def __init__(self, fn) -> None:
        from vllm_omni.diffusion.models.lingbot_vla_v2 import modeling_lingbot_vla_v2 as mod

        self.mod = mod
        self.fn = fn
        self.previous = None

    def __enter__(self):
        self.previous = self.mod.eager_attention
        self.mod.eager_attention = self.fn
        return self.fn

    def __exit__(self, *exc) -> None:
        self.mod.eager_attention = self.previous


def capture_attention_inputs(model: Any, session: Any, state: torch.Tensor, x_t: torch.Tensor) -> dict[str, Any]:
    """Run one ``predict_velocity`` and keep layer 0's real q/k/v/mask."""
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import eager_attention

    captured: dict[str, Any] = {}

    def record(query, key, value, mask):
        if not captured:
            captured.update(query=query.clone(), key=key.clone(), value=value.clone(), mask=mask.clone())
        return eager_attention(query, key, value, mask)

    with attention_patch(record):
        model.predict_velocity(
            state=state,
            prefix_pad_masks=session.pad_masks,
            prefix_position_ids=session.position_ids,
            past_key_values=session.past_key_values,
            x_t=x_t,
            timestep=torch.full((state.shape[0],), 1.0, device=state.device, dtype=state.dtype),
        )
    if not captured:
        raise RuntimeError("attention was never called -- is attention_backend still 'eager'?")
    return captured


def deviation(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Divergence of ``a`` from reference ``b``.

    ``max_rel`` is **per element**, and on an attention output that makes it
    useless as a gate: the tensor contains entries near 1e-5, so one fp16 ulp of
    absolute error there reads as a relative error of 3. The gate metrics are
    ``max_abs_norm`` (largest error as a fraction of the reference's own largest
    magnitude) and ``rel_l2``; ``max_rel`` is kept only because a small value for
    it is still informative.
    """
    delta = (a.float() - b.float()).abs()
    ref = b.float().abs()
    scale = float(ref.max())
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "max_abs_norm": float(delta.max()) / max(scale, 1e-30),
        "rel_l2": float(delta.pow(2).sum().sqrt() / b.float().pow(2).sum().sqrt().clamp_min(1e-30)),
        "max_rel": float((delta / ref.clamp_min(1e-6)).max()),
        "ref_absmax": scale,
    }


def fp16_ulp(scale: float) -> float:
    """One fp16 ulp at magnitude ``scale`` -- the floor a fp16 output can hit."""
    return float(torch.tensor(scale, dtype=torch.float16).nextafter(
        torch.tensor(float("inf"), dtype=torch.float16)
    ) - torch.tensor(scale, dtype=torch.float16))


# ---------------------------------------------------------------------------
# Arm: ring-math  (one device -- the gate)
# ---------------------------------------------------------------------------
def arm_ring_math(args, processor, model, obs, device, dtype) -> dict:
    """Does the ring reproduce ``eager_attention`` at the real denoise shapes?

    Three checks, in the order that makes a failure readable:

    1. ``shards=1`` in the input dtype. The block kernel alone, no merge. Must
       be ~exact against ``eager_attention``; anything else is a kernel bug.
    2. ``shards=2,4,8`` in the input dtype and in fp32. The merge. fp32 should
       hold to ~1e-6 relative; fp16 is allowed to *beat* eager, since the ring
       accumulates the softmax in fp32 and eager does not.
    3. The merge itself against the shipped ``update_out_and_lse``.
    4. End to end: the 2-shard ring inside the real 10-step loop, graded on the
       Phase 7 metric (Rules #3).
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import eager_attention

    decoder, session, state = ground(model, processor, obs, device, dtype)
    dims = real_dims(processor, device)
    horizon = model.config.spec_max_exec_steps
    noise = torch.randn((1, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype)

    captured = capture_attention_inputs(model, session, state, noise)
    q, k, v, mask = captured["query"], captured["key"], captured["value"], captured["mask"]
    print(f"[shapes] q={tuple(q.shape)} k={tuple(k.shape)} mask={tuple(mask.shape)} dtype={q.dtype}")
    rows_all_masked = int((~mask.any(dim=-1)).sum())
    print(f"[mask] fully-masked query rows: {rows_all_masked} (must be 0, or eager's uniform-softmax "
          f"fallback and the ring's zero disagree by construction)")

    shard_counts = [int(s) for s in args.shards.split(",") if s.strip()]

    # Three sweeps, and the **fp32-input** one is the gate. With fp16 inputs the
    # output is an fp16 tensor and both sides land within one ulp of each other
    # whatever the algebra does, so a fp16 agreement cannot distinguish a correct
    # ring from a subtly wrong one. Casting q/k/v to fp32 lifts that floor and
    # the merge has to be right to ~1e-6.
    q32, k32, v32 = q.float(), k.float(), v.float()
    reference = eager_attention(q, k, v, mask)
    reference32 = eager_attention(q32, k32, v32, mask)
    rows = []
    for inputs, ref, accums in (("fp32", reference32, ("input",)), ("fp16", reference, ("input", "fp32"))):
        qq, kk, vv = (q32, k32, v32) if inputs == "fp32" else (q, k, v)
        for accum in accums:
            for shards in shard_counts:
                got = ring_masked_attention(qq, kk, vv, mask, shards=shards, accum=accum)
                row = {"inputs": inputs, "accum": accum, "shards": shards, **deviation(got, ref)}
                rows.append(row)
                print(f"[ring] inputs={inputs} accum={accum:<5} shards={shards:<2} "
                      f"max_abs={row['max_abs']:.3e} max_abs_norm={row['max_abs_norm']:.3e} "
                      f"rel_l2={row['rel_l2']:.3e}")
    ulp = fp16_ulp(rows[-1]["ref_absmax"])
    print(f"[floor] one fp16 ulp at the output's own magnitude ({rows[-1]['ref_absmax']:.3f}) "
          f"is {ulp:.3e} -- the fp16 rows above cannot do better than this")

    # A dead shard is the case the guard in `merge_out_lse` exists for, and the
    # real mask contains one: the state token (row 0) cannot see the action
    # tokens, so a shard holding only action keys is fully masked for it. Shard
    # the *suffix* keys alone to force that, rather than trusting it happens.
    prefix_len = session.pad_masks.shape[1]
    suffix_mask = mask[:, :, prefix_len:]
    suffix_only = ring_masked_attention(
        q32, k32[:, prefix_len:], v32[:, prefix_len:], suffix_mask, shards=2, accum="input"
    )
    suffix_ref = eager_attention(q32, k32[:, prefix_len:], v32[:, prefix_len:], suffix_mask)
    alive = suffix_mask.any(dim=-1)[0]
    dead_rows = int((~alive).sum())
    dead_shard = deviation(suffix_only[:, alive], suffix_ref[:, alive])
    print(f"[dead-shard] suffix keys only, 2 shards, {dead_rows} globally-dead rows excluded: "
          f"max_abs_norm={dead_shard['max_abs_norm']:.3e}  "
          f"(a shard holding only action keys is dead for the state-token row)")

    # -- what the ring costs *arithmetically*, before any device split -----
    # The context-parallel arm's transport control comes out slower than the
    # unsplit loop, so the ring's own cost has to be priced here, on one card,
    # where the wire cannot be blamed. Two effects are separated because they
    # are separately fixable: splitting one masked softmax into N, and running
    # the merge in fp32.
    cost_rows = []
    eager_us = median_ms(lambda: eager_attention(q, k, v, mask),
                         iters=50, warmup=10, device=q.device) * 1e3
    print(f"\n[cost] eager_attention, 337 keys           {eager_us:8.1f} us")
    # SDPA is the fused comparand, and it is what makes the "just fuse the ring
    # kernel" question answerable instead of hypothetical: it is one launch for
    # the whole masked softmax, so the gap below is what fusion is worth at this
    # shape before anyone writes a ring version of it.
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import sdpa_attention

    sdpa_us = median_ms(lambda: sdpa_attention(q, k, v, mask),
                        iters=50, warmup=10, device=q.device) * 1e3
    print(f"[cost] sdpa_attention (fused), 337 keys    {sdpa_us:8.1f} us"
          f"   {sdpa_us / eager_us:5.2f}x eager")
    for accum in ("input", "fp32"):
        for shards in shard_counts:
            got_us = median_ms(
                lambda a=accum, s=shards: ring_masked_attention(q, k, v, mask, shards=s, accum=a),
                iters=50, warmup=10, device=q.device,
            ) * 1e3
            cost_rows.append({"accum": accum, "shards": shards, "us": got_us,
                              "vs_eager": got_us / eager_us})
            print(f"[cost] ring accum={accum:<5} shards={shards:<2}        {got_us:8.1f} us"
                  f"   {got_us / eager_us:5.2f}x eager")

    # -- the merge, against the shipped one --------------------------------
    shipped = {"available": False}
    try:
        from vllm_omni.diffusion.attention.backends.ring.ring_utils import update_out_and_lse

        b1, l1 = block_attention(q, k[:, :prefix_len], v[:, :prefix_len], mask[:, :, :prefix_len])
        b2, l2 = block_attention(q, k[:, prefix_len:], v[:, prefix_len:], mask[:, :, prefix_len:])
        ours, _ = merge_out_lse(*merge_out_lse(None, None, b1, l1), b2, l2)
        # That file's convention is [B, S, H, D] out and [B, S, H, 1] lse.
        theirs_out, theirs_lse = update_out_and_lse(None, None, b1.transpose(1, 2), l1.transpose(1, 2))
        theirs_out, theirs_lse = update_out_and_lse(theirs_out, theirs_lse, b2.transpose(1, 2), l2.transpose(1, 2))
        shipped = {"available": True, **deviation(ours.transpose(1, 2), theirs_out)}
        print(f"[merge] vs ring_utils.update_out_and_lse: max_abs_norm={shipped['max_abs_norm']:.3e} "
              f"rel_l2={shipped['rel_l2']:.3e}")
    except Exception as exc:  # noqa: BLE001 - informational cross-check only
        shipped["error"] = repr(exc)
        print(f"[merge] shipped update_out_and_lse unavailable: {exc}")

    # -- end to end --------------------------------------------------------
    def loop() -> torch.Tensor:
        return model.denoise_actions(
            state=state,
            prefix_pad_masks=session.pad_masks,
            prefix_position_ids=session.position_ids,
            past_key_values=session.past_key_values,
            noise=noise.clone(),
            num_steps=args.num_steps or model.config.num_steps,
        )

    chunk_eager = loop()
    ring_kernel = lambda qq, kk, vv, mm: ring_masked_attention(  # noqa: E731
        qq, kk, vv, mm, shards=2, accum=args.accum
    )
    with attention_patch(ring_kernel):
        chunk_ring = loop()
    chunk = {**deviation(chunk_ring, chunk_eager), "rms": rms(chunk_ring, chunk_eager, dims, horizon)}
    print(f"[chunk] 2-shard ring vs eager over 10 steps: max_abs={chunk['max_abs']:.3e} "
          f"rms={chunk['rms']:.3e} (fp16 floor {FP16_FLOOR:.1e}, spec_tau {SPEC_TAU})")

    # Three conditions, each gating a different failure:
    #   kernel+merge  the fp32-input rows, where no storage floor hides a bug
    #   fp16 parity   the shipped dtype, allowed one ulp and no more
    #   task          the action chunk, on the Phase 7 / spec_tau metric (Rules #3)
    # The fp16 check is over `accum="fp32"` only, because that is the mode every
    # other arm runs: `block_attention` accumulates in fp32 by default. The
    # `accum="input"` fp16 rows are diagnostic, and they show why -- a fp16
    # softmax inside a *block* loses more the smaller the block gets (1 ulp at
    # shards=1, 3.5 at shards=2), which is the one way a ring can be worse than
    # the monolith it replaces. Accumulating the merge in fp32 removes it
    # entirely, and it costs nothing here: the block matmuls stay fp16.
    fp32_rows = [r for r in rows if r["inputs"] == "fp32"]
    fp16_rows = [r for r in rows if r["inputs"] == "fp16" and r["accum"] == "fp32"]
    gate_fp32 = max(r["max_abs_norm"] for r in fp32_rows)
    gate_fp16 = max(r["max_abs"] for r in fp16_rows)
    fp16_input_accum = max(r["max_abs"] for r in rows if r["inputs"] == "fp16" and r["accum"] == "input")
    print(f"[floor] fp16 blocks with a fp16 softmax reach {fp16_input_accum / ulp:.1f} ulp; "
          f"with the fp32 merge they stay at 1")
    checks = {
        "fp32_algebra": gate_fp32 < 1e-5,
        "fp16_within_one_ulp": gate_fp16 <= ulp * 1.01,
        "chunk_within_spec_tau": chunk["rms"] <= SPEC_TAU,
        "no_fully_masked_rows": rows_all_masked == 0,
    }
    passed = all(checks.values())
    print()
    print(f"[gate] fp32 algebra        max_abs_norm={gate_fp32:.3e}  (< 1e-5)  "
          f"{'ok' if checks['fp32_algebra'] else 'FAIL'}")
    print(f"[gate] fp16 vs eager       max_abs={gate_fp16:.3e}  (<= 1 ulp = {ulp:.3e}, fp32 merge)  "
          f"{'ok' if checks['fp16_within_one_ulp'] else 'FAIL'}")
    print(f"[gate] action chunk        rms={chunk['rms']:.3e}  (<= spec_tau {SPEC_TAU})  "
          f"{'ok' if checks['chunk_within_spec_tau'] else 'FAIL'}")
    print(f"[gate] {'PASS' if passed else 'FAIL'}")

    decoder.close()
    return {
        "arm": "ring-math",
        "shapes": {"q": list(q.shape), "k": list(k.shape), "mask": list(mask.shape)},
        "fully_masked_rows": rows_all_masked,
        "rows": rows,
        "dead_shard": dead_shard,
        "dead_shard_rows": dead_rows,
        "shipped_merge": shipped,
        "eager_us": eager_us,
        "sdpa_us": sdpa_us,
        "cost_rows": cost_rows,
        "chunk": chunk,
        "fp16_ulp": ulp,
        "fp16_input_accum_max_abs": fp16_input_accum,
        "checks": checks,
        "gate_passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# Arm: attn-share  (one device -- the ceiling)
# ---------------------------------------------------------------------------
def arm_attn_share(args, processor, model, obs, device, dtype) -> dict:
    """How much of the denoise loop is attention? The ceiling for any split.

    Two independent estimates, because neither is trustworthy alone on a
    dispatch-bound path:

    ``fenced``  the attention call with a device sync either side, summed over
                all 360 layer-steps. An **upper** bound: it charges attention
                for the sync it forces and for overlap it destroys.
    ``ablated`` the whole loop with attention replaced by a correctly-shaped
                zero tensor, differenced against the whole loop. A **lower**
                bound, and the one to quote: `moe_implementation="dense"` runs
                all 32 experts unconditionally (§F2), so changing the attention
                output does not change one byte of the weight traffic that
                dominates the loop.

    §F1's method note applies: the profiler reports zero device time for
    ``mm``/``einsum`` on this backend, which is what produced a retracted
    "51 ms device / 243 ms host" split. Neither estimate here uses it.
    """
    decoder, session, state = ground(model, processor, obs, device, dtype)
    steps = args.num_steps or model.config.num_steps
    noise = torch.randn((1, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype)

    def loop() -> None:
        model.denoise_actions(
            state=state,
            prefix_pad_masks=session.pad_masks,
            prefix_position_ids=session.position_ids,
            past_key_values=session.past_key_values,
            noise=noise.clone(),
            num_steps=steps,
        )

    layer_steps = model.config.expert_num_layers * steps
    full_ms = median_ms(loop, iters=args.iters, warmup=args.warmup, device=device)
    print(f"[loop] eager, unpatched         {full_ms:8.2f} ms  ({layer_steps} layer-steps)")

    # -- fenced ------------------------------------------------------------
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import eager_attention

    accrued: list[float] = []

    def fenced(query, key, value, mask):
        sync(device)
        start = time.perf_counter()
        out = eager_attention(query, key, value, mask)
        sync(device)
        accrued.append((time.perf_counter() - start) * 1e3)
        return out

    with attention_patch(fenced):
        loop()  # warm
        accrued.clear()
        fenced_loop_ms = median_ms(lambda: (accrued.clear(), loop()), iters=args.iters, warmup=0, device=device)
    fenced_attn_ms = sum(accrued)
    print(f"[fenced] attention, summed      {fenced_attn_ms:8.2f} ms over {len(accrued)} calls "
          f"({fenced_attn_ms / max(len(accrued), 1) * 1e3:7.1f} us each); loop {fenced_loop_ms:.2f} ms")

    # -- ablated -----------------------------------------------------------
    def stub(query, key, value, mask):
        bsize, q_len, num_heads, head_dim = query.shape
        return query.new_zeros((bsize, q_len, num_heads * head_dim))

    with attention_patch(stub):
        stub_ms = median_ms(loop, iters=args.iters, warmup=args.warmup, device=device)
    ablated_attn_ms = full_ms - stub_ms
    print(f"[ablated] loop without attention{stub_ms:8.2f} ms  -> attention {ablated_attn_ms:8.2f} ms "
          f"({ablated_attn_ms / full_ms:.1%} of the loop)")

    # -- the same two numbers, compiled ------------------------------------
    # Everything above is eager, and the recorded baseline (213.5 ms) is
    # compiled, so the share was being *projected* onto it. This measures both
    # instead: compile with the real attention, then reset dynamo and compile
    # again with the stub. Inductor sees the stub at trace time, so the second
    # graph really has no attention in it.
    compiled = {}
    if args.compile_denoise_step:
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        # 4e-2, not the 2e-2 `phase12_paradigms_probe.py` passes. On this
        # container (torch 2.12.0+xpu) inductor's single-velocity-step drift
        # reads max_rel 3.18e-2, over 2e-2 -- §B's "Inductor drift" item, and not
        # introduced here. The number that Rules #3 actually gates on is the
        # 10-step chunk, and that reads 8.86e-3; both are printed by
        # `compile_denoise_step` so a change in either is visible in the log.
        compile_denoise_step(processor, model, obs, device, dtype, "inductor", False, False, 4e-2)
        compiled["full_ms"] = median_ms(loop, iters=args.iters, warmup=args.warmup, device=device)
        print(f"\n[compiled] loop, unpatched       {compiled['full_ms']:8.2f} ms   "
              f"(recorded baseline {BASELINE_LOOP_MS:.1f} ms)")
        torch._dynamo.reset()
        with attention_patch(stub):
            compile_denoise_step(processor, model, obs, device, dtype, "inductor", False, False, 1e9)
            compiled["stub_ms"] = median_ms(loop, iters=args.iters, warmup=args.warmup, device=device)
        compiled["attn_ms"] = compiled["full_ms"] - compiled["stub_ms"]
        compiled["share"] = compiled["attn_ms"] / compiled["full_ms"]
        print(f"[compiled] loop without attention{compiled['stub_ms']:8.2f} ms   -> attention "
              f"{compiled['attn_ms']:7.2f} ms  ({compiled['share']:.1%} of the compiled loop)")
        print("[compiled]   ** UPPER BOUND on attention: the stub returns a constant-shaped")
        print("[compiled]      zero, so inductor can dead-code-eliminate the whole fused q/k/v")
        print("[compiled]      GEMM and apply_mrope behind it. The eager ablation cannot do that")
        print("[compiled]      (Python runs compute_qkv either way), which is why the two shares")
        print("[compiled]      disagree. For the KV-split question use `--sweep loop` instead:")
        print("[compiled]      it halves the prefix KV for real and needs no ablation. **")
        torch._dynamo.reset()

    share = ablated_attn_ms / full_ms
    # The two ceilings. A KV/context split halves attention and nothing else. A
    # sequence split over the rows is §G5's curve, not this share -- quoted from
    # that measurement rather than re-derived.
    #
    # `--compile-denoise-step` measures the compiled share directly; without it
    # the eager share is projected onto the compiled baseline, which is a
    # projection and is labelled as one in the output.
    if compiled:
        reference_loop, share_used, basis = compiled["full_ms"], compiled["share"], "measured, compiled"
    else:
        reference_loop, share_used, basis = BASELINE_LOOP_MS, share, "projected from eager"
    kv_ceiling = share_used * reference_loop / 2
    hop_cost = layer_steps * 2 * MEASURED_HOP_MS
    print()
    print(f"attention share of the eager loop                 {share:8.1%}")
    print(f"ceiling basis: {basis}, loop = {reference_loop:.1f} ms, share = {share_used:.1%}")
    print(f"ceiling, KV split (half of attention)             {kv_ceiling:8.2f} ms")
    print("  ** an UPPER BOUND, and not reachable on this hardware: it assumes")
    print("     attention's cost scales with the key count, and `--arm share-sweep`")
    print("     measures the dGPU flat from 1 key to 286. Achievable saving is 0. **")
    print(f"ceiling, row split (§G5, flat in M)               {0.13 * reference_loop:8.2f} ms")
    print(f"cost, 2 hops x {layer_steps} layer-steps at §M.6's 0.155 ms  {hop_cost:8.2f} ms")
    print(f"  -> transport / ceiling                          {hop_cost / max(kv_ceiling, 1e-9):8.1f}x")

    decoder.close()
    return {
        "arm": "attn-share",
        "layer_steps": layer_steps,
        "loop_eager_ms": full_ms,
        "loop_stub_ms": stub_ms,
        "loop_fenced_ms": fenced_loop_ms,
        "attn_fenced_ms": fenced_attn_ms,
        "attn_fenced_calls": len(accrued),
        "attn_ablated_ms": ablated_attn_ms,
        "attn_share": share,
        "compiled_loop": compiled,
        "ceiling_basis": basis,
        "ceiling_kv_split_ms": kv_ceiling,
        "ceiling_row_split_ms": 0.13 * reference_loop,
        "projected_transport_ms": hop_cost,
    }


# ---------------------------------------------------------------------------
# Two processes: the slot tables
# ---------------------------------------------------------------------------
# The oneCCL plugin hangs on a **fourth** live payload in one process
# (`phase10_ipc_probe.py:110` -- each payload registers two endpoints), so every
# table here stays at three slots. That constraint is why `reply` packs `out`
# and `lse` into one fp32 buffer and why `--split seq` returns its final rows
# through the `ring` slot instead of a fourth.

KV_HEADS, HEAD_DIM = 8, 128
PREFIX_LEN = 286


def kv_slots(config, prefix_shard: int) -> tuple[Slot, ...]:
    layers = config.expert_num_layers
    suffix = config.chunk_size + 1
    heads = config.expert_num_attention_heads
    return (
        # K and V for the iGPU's prefix shard, every layer, plus its pad mask.
        Slot("setup", layers * prefix_shard * KV_HEADS * HEAD_DIM * 2 + prefix_shard, torch.float16),
        # [layer_idx | query]. The header rides in the payload so the worker
        # needs no second channel to know where it is.
        Slot("query", 1 + suffix * heads * HEAD_DIM, torch.float16),
        # [out | lse], fp32 so the merge is not the thing that loses precision.
        Slot("reply", suffix * heads * HEAD_DIM + suffix * heads, torch.float32),
    )


def wire_slots(config, max_rows: int) -> tuple[Slot, ...]:
    """Exactly the two ring payloads, and nothing else.

    `--split wire` measures the transport on its own: the worker echoes a
    preallocated buffer without touching a weight or running a kernel, so what
    comes out is the wire and only the wire. This exists because PHASE14's first
    two attempts at a wire cost were both indirect -- one had the dGPU's own
    attention inside the timing span, the other interpolated
    `phase10_ipc_probe.py`'s round-trip curve to an asymmetric payload -- and
    they disagreed by 2.6x.
    """
    suffix = config.chunk_size + 1
    heads = config.expert_num_attention_heads
    return (
        Slot("query", 1 + suffix * heads * HEAD_DIM, torch.float16),  # 408 KiB, kv split up
        Slot("reply", suffix * heads * HEAD_DIM + suffix * heads, torch.float32),  # 822 KiB, kv split down
        Slot("ring", 1 + 2 * max_rows * KV_HEADS * HEAD_DIM, torch.float16),  # 104 KiB, seq split each way
    )


def wire_driver(args, processor, model, obs, device, dtype) -> dict:
    """Round-trip cost of the real ring payloads, with no compute on either end."""
    config = model.config
    _, theirs = suffix_row_split(config)
    max_rows = max(theirs.stop - theirs.start, (config.chunk_size + 1) - (theirs.stop - theirs.start))
    slots = wire_slots(config, max_rows)
    transport = make_transport(args, slots, rank=DGPU_RANK, device=device)
    by_name = {slot.name: slot for slot in slots}

    buffers = {
        name: torch.zeros(by_name[name].numel, dtype=by_name[name].dtype, device=device)
        for name in ("query", "reply", "ring")
    }

    def kv_round() -> None:
        transport.send("query", buffers["query"])
        transport.recv("reply")

    def seq_round() -> None:
        transport.send("ring", buffers["ring"])
        transport.recv("ring")

    rows = []
    for name, fn, up, down in (
        ("kv split: query up + reply down", kv_round,
         by_name["query"].numel * 2, by_name["reply"].numel * 4),
        ("seq split: ring both ways", seq_round,
         by_name["ring"].numel * 2, by_name["ring"].numel * 2),
    ):
        for _ in range(args.warmup * 10):
            fn()
        samples = []
        for _ in range(args.wire_iters):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1e3)
        median = statistics.median(samples)
        total = up + down
        rows.append({"name": name, "up_bytes": up, "down_bytes": down, "ms": median,
                     "p90_ms": statistics.quantiles(samples, n=10)[8],
                     "gb_per_s": total / median * 1e-6,
                     "ms_per_loop": median * config.expert_num_layers * (args.num_steps or config.num_steps)})
        print(f"[wire] {name:<34} {median:7.3f} ms  "
              f"(p90 {rows[-1]['p90_ms']:.3f}; {up / 2**10:.0f}+{down / 2**10:.0f} KiB, "
              f"{rows[-1]['gb_per_s']:.2f} GB/s)")
        print(f"[wire]   x360 layer-steps                {rows[-1]['ms_per_loop']:7.1f} ms")

    transport.send("query", buffers["query"].clone().fill_(OP_SHUTDOWN))
    transport.close()
    return {"arm": "wire", "transport": args.transport, "rows": rows}


def wire_worker(args, config, device, dtype) -> int:
    """Echo a preallocated buffer. No model, no kernels -- just the transport."""
    _, theirs = suffix_row_split(config)
    max_rows = max(theirs.stop - theirs.start, (config.chunk_size + 1) - (theirs.stop - theirs.start))
    slots = wire_slots(config, max_rows)
    transport = make_transport(args, slots, rank=IGPU_RANK, device=device)
    by_name = {slot.name: slot for slot in slots}
    reply = torch.zeros(by_name["reply"].numel, dtype=torch.float32, device=device)
    ring = torch.zeros(by_name["ring"].numel, dtype=torch.float16, device=device)
    print("[worker] wire echo up, no weights and no kernels", flush=True)

    served = 0
    for _ in range(args.warmup * 10 + args.wire_iters):
        transport.recv("query")
        transport.send("reply", reply)
        served += 1
    for _ in range(args.warmup * 10 + args.wire_iters):
        transport.recv("ring")
        transport.send("ring", ring)
    transport.recv("query")  # shutdown
    print(f"[worker] echoed {served} kv round trips and as many ring ones", flush=True)
    transport.close()
    return 0


def seq_slots(config, max_rows: int) -> tuple[Slot, ...]:
    layers = config.expert_num_layers
    return (
        # The full prefix KV, replicated: 36 x 286 x 1024 x 2 fp16 = 42.2 MB.
        Slot("setup", layers * PREFIX_LEN * KV_HEADS * HEAD_DIM * 2, torch.float16),
        # position ids, pad masks, state, and the worker's noise rows.
        Slot("meta", 3 * PREFIX_LEN + PREFIX_LEN + config.max_state_dim + max_rows * config.max_action_dim,
             torch.float32),
        # [rows | k | v] one way, and the worker's final x_t rows the other.
        Slot("ring", 1 + 2 * max_rows * KV_HEADS * HEAD_DIM, torch.float16),
    )


def make_transport(args, slots: tuple[Slot, ...], *, rank: int, device: torch.device):
    if args.transport == "oneccl":
        return OneCCLTransport(
            rank=rank,
            device=device,
            slots=slots,
            lib=args.ccl_lib,
            ccl_device=rank,  # aligned with ZE_AFFINITY_MASK; PHASE10 §9
            uid_file=args.uid_file,
            connect_timeout=args.connect_timeout,
        )
    return ShmTransport(
        rank=rank, device=device, slots=slots, prefix=args.shm_prefix, connect_timeout=args.connect_timeout
    )


def spawn_worker(args, *, log: str | None = None) -> subprocess.Popen:
    """Start the iGPU process. ``ZE_AFFINITY_MASK=1`` is the iGPU here (§K1)."""
    env = dict(os.environ)
    env["ZE_AFFINITY_MASK"] = str(IGPU_RANK)
    env.pop("ONEAPI_DEVICE_SELECTOR", None)  # would re-enumerate under the mask
    env["OMP_NUM_THREADS"] = "2"
    command = [
        sys.executable, "-u", __file__,
        "--role", "worker",
        "--arm", "ring-2p",
        "--split", args.split,
        "--model", str(args.model),
        "--dtype", args.dtype,
        "--transport", args.transport,
        "--ccl-lib", args.ccl_lib,
        "--uid-file", args.uid_file,
        "--shm-prefix", args.shm_prefix,
        "--connect-timeout", str(args.connect_timeout),
        "--iters", str(args.iters),
        "--wire-iters", str(args.wire_iters),
        "--warmup", str(args.warmup),
        "--seed", str(args.seed),
    ]
    if args.num_steps:
        command += ["--num-steps", str(args.num_steps)]
    if args.worker_cpu:
        command = ["taskset", "-c", args.worker_cpu, *command]
    handle = open(log, "w") if log else None  # noqa: SIM115 - lives as long as the child
    return subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)


def clear_rendezvous(args) -> None:
    Path(args.uid_file).unlink(missing_ok=True)
    for stale in Path("/dev/shm").glob(f"{args.shm_prefix}_*"):
        stale.unlink(missing_ok=True)  # a killed run leaves its segments behind


# ---------------------------------------------------------------------------
# --split kv : context parallel over the prefix KV cache
# ---------------------------------------------------------------------------
OP_SHUTDOWN = -1.0


def kv_driver(args, processor, model, obs, device, dtype) -> dict:
    """dGPU rank. Keeps prefix[:s], gives prefix[s:] to the iGPU."""
    decoder, session, state = ground(model, processor, obs, device, dtype)
    config = model.config
    steps = args.num_steps or config.num_steps
    prefix_len = session.pad_masks.shape[1]
    split_at = prefix_len // 2
    shard = prefix_len - split_at
    suffix = config.chunk_size + 1
    heads = config.expert_num_attention_heads

    # The worker rebuilds its mask columns from the pad mask alone, which is only
    # equal to the real thing while both blocking flags are off. Check it here
    # rather than let a deploy-config change silently corrupt the comparison.
    probe = capture_attention_inputs(model, session, state,
                                     torch.zeros((1, config.chunk_size, config.max_action_dim),
                                                 device=device, dtype=dtype))
    expected = session.pad_masks[:, None, :].expand(1, suffix, prefix_len)
    if not torch.equal(probe["mask"][:, :, :prefix_len], expected):
        raise RuntimeError(
            "the prefix half of the real mask is not the pad mask expanded -- "
            "block_future_depth_to_action / block_suffix_to_future_video must be on, "
            "and the worker's reconstructed mask would be wrong"
        )
    del probe

    slots = kv_slots(config, shard)
    transport = make_transport(args, slots, rank=DGPU_RANK, device=device)
    print(f"[kv] prefix {prefix_len} -> dGPU[:{split_at}] iGPU[{split_at}:] ({shard} keys), "
          f"setup {slots[0].numel * 2 / 2**20:.1f} MiB, query {slots[1].numel * 2 / 2**10:.0f} KiB, "
          f"reply {slots[2].numel * 4 / 2**10:.0f} KiB")

    # -- setup: ship the iGPU its shard of every layer's K/V ---------------
    parts = []
    for key, value in session.past_key_values:
        parts.append(key[:, split_at:].reshape(-1).to(torch.float16))
        parts.append(value[:, split_at:].reshape(-1).to(torch.float16))
    parts.append(session.pad_masks[0, split_at:].to(torch.float16))
    sync(device)
    setup_start = time.perf_counter()
    transport.send("setup", torch.cat(parts))
    setup_ms = (time.perf_counter() - setup_start) * 1e3
    print(f"[kv] setup shipped in {setup_ms:.1f} ms")

    # Three timers, not one. A single span around the whole thing would charge
    # the dGPU's *own* attention block to "the exchange", which is how the first
    # version of this arm reported 1.794 ms of "wire" that was mostly compute.
    #   send  loading the staging buffer and handing it off (returns without an ack)
    #   local the dGPU's own block, which runs while the iGPU has the query
    #   recv  the blocking wait -- the wire both ways plus whatever of the iGPU's
    #         own attention did not overlap `local`
    timers = {"send": [], "local": [], "recv": []}
    layer_counter = {"n": 0}

    def ring_kernel(query, key, value, mask):
        """dGPU's half locally, the iGPU's half over the wire, then merge."""
        layer = layer_counter["n"] % config.expert_num_layers
        layer_counter["n"] += 1

        header = torch.full((1,), float(layer), dtype=torch.float16, device=query.device)
        t0 = time.perf_counter()
        transport.send("query", torch.cat([header, query.reshape(-1).to(torch.float16)]))
        t1 = time.perf_counter()
        # Local block while the iGPU works on its own: prefix[:split] + the
        # whole suffix, which is the mask's remaining columns.
        local_out, local_lse = block_attention(
            query,
            torch.cat([key[:, :split_at], key[:, prefix_len:]], dim=1),
            torch.cat([value[:, :split_at], value[:, prefix_len:]], dim=1),
            torch.cat([mask[:, :, :split_at], mask[:, :, prefix_len:]], dim=2),
        )
        sync(device)  # so `local` is the block's cost, not its submission's
        t2 = time.perf_counter()
        reply = transport.recv("reply").clone()
        t3 = time.perf_counter()
        timers["send"].append((t1 - t0) * 1e3)
        timers["local"].append((t2 - t1) * 1e3)
        timers["recv"].append((t3 - t2) * 1e3)

        cut = suffix * heads * HEAD_DIM
        peer_out = reply[:cut].view(1, heads, suffix, HEAD_DIM)
        peer_lse = reply[cut:].view(1, heads, suffix, 1)
        out, _ = merge_out_lse(local_out, local_lse, peer_out, peer_lse)
        return out.transpose(1, 2).reshape(1, suffix, heads * HEAD_DIM).to(query.dtype)

    def solo_kernel(query, key, value, mask):
        """The same two blocks and the same merge, both computed on the dGPU.

        The transport control: identical arithmetic, identical Python, no wire
        and no iGPU. `split_ms - solo_ms` is therefore the cost of the device
        boundary and nothing else.
        """
        local_out, local_lse = block_attention(
            query,
            torch.cat([key[:, :split_at], key[:, prefix_len:]], dim=1),
            torch.cat([value[:, :split_at], value[:, prefix_len:]], dim=1),
            torch.cat([mask[:, :, :split_at], mask[:, :, prefix_len:]], dim=2),
        )
        peer_out, peer_lse = block_attention(
            query, key[:, split_at:prefix_len], value[:, split_at:prefix_len],
            mask[:, :, split_at:prefix_len],
        )
        out, _ = merge_out_lse(local_out, local_lse, peer_out, peer_lse)
        return out.transpose(1, 2).reshape(1, suffix, heads * HEAD_DIM).to(query.dtype)

    def loop() -> torch.Tensor:
        return model.denoise_actions(
            state=state,
            prefix_pad_masks=session.pad_masks,
            prefix_position_ids=session.position_ids,
            past_key_values=session.past_key_values,
            noise=noise.clone(),
            num_steps=steps,
        )

    noise = torch.randn((1, config.chunk_size, config.max_action_dim), device=device, dtype=dtype)
    reference = loop()
    # Same process, same host state, so this is the honest comparand -- the
    # recorded 213.5 ms is compiled and this arm is eager (Rules #2).
    single_ms = median_ms(loop, iters=args.iters, warmup=args.warmup, device=device)

    # The transport control: both ring blocks on the dGPU, same merge, no wire.
    with attention_patch(solo_kernel):
        solo_ms = median_ms(loop, iters=args.iters, warmup=args.warmup, device=device)

    with attention_patch(ring_kernel):
        for _ in range(args.warmup):
            loop()
        for samples_list in timers.values():
            samples_list.clear()
        samples = []
        for _ in range(args.iters):
            sync(device)
            start = time.perf_counter()
            chunk = loop()
            sync(device)
            samples.append((time.perf_counter() - start) * 1e3)
        transport.send("query", torch.cat([noise.new_full((1,), OP_SHUTDOWN).to(torch.float16),
                                           torch.zeros(slots[1].numel - 1, dtype=torch.float16, device=device)]))

    split_ms = statistics.median(samples)
    parts = {key: statistics.median(values) for key, values in timers.items()}
    per_exchange = parts["send"] + parts["recv"]
    layer_steps = config.expert_num_layers * steps
    dims = real_dims(processor, device)
    accuracy = {**deviation(chunk, reference), "rms": rms(chunk, reference, dims, config.spec_max_exec_steps)}
    print()
    print(f"loop, dGPU alone, full 337 keys     {single_ms:8.2f} ms   (denoise_actions)")
    print(f"loop, both ring blocks on the dGPU  {solo_ms:8.2f} ms   <- transport control, no wire")
    print(f"loop, dGPU + iGPU ring              {split_ms:8.2f} ms   -> {single_ms / split_ms:.2f}x")
    print(f"cost of the device boundary         {split_ms - solo_ms:8.2f} ms   "
          f"({(split_ms - solo_ms) / layer_steps:.3f} ms/layer-step, "
          f"{(split_ms - solo_ms) / split_ms:.0%} of the wall)")
    print()
    print("  per layer-step, split three ways:")
    print(f"    send   (hand off a 408 KiB query)   {parts['send']:7.3f} ms")
    print(f"    local  (the dGPU's own block)       {parts['local']:7.3f} ms   <- not the exchange")
    print(f"    recv   (wire back + the iGPU's block){parts['recv']:7.3f} ms")
    print(f"    -> exchange = send + recv           {per_exchange:7.3f} ms  x {layer_steps}"
          f" = {per_exchange * layer_steps:.1f} ms")
    print(f"chunk vs single-device             max_abs={accuracy['max_abs']:.3e} rms={accuracy['rms']:.3e}")

    transport.close()
    decoder.close()
    return {
        "arm": "ring-2p", "split": "kv",
        "prefix_len": prefix_len, "split_at": split_at, "shard": shard,
        "setup_ms": setup_ms,
        "loop_single_ms": single_ms,
        "loop_solo_ms": solo_ms,
        "loop_split_ms": split_ms,
        "boundary_ms": split_ms - solo_ms,
        "speedup": single_ms / split_ms,
        "exchange_ms": per_exchange,
        "exchange_parts_ms": parts,
        "exchanges_per_loop": layer_steps,
        "accuracy": accuracy,
        "payload_bytes": {"setup": slots[0].numel * 2, "query": slots[1].numel * 2, "reply": slots[2].numel * 4},
    }


def kv_worker(args, config, device, dtype) -> int:
    """iGPU rank. Attention over its prefix shard, and nothing else.

    Takes a config, not a model: this rank never touches a weight.
    """
    prefix_len = PREFIX_LEN
    split_at = prefix_len // 2
    shard = prefix_len - split_at
    suffix = config.chunk_size + 1
    heads = config.expert_num_attention_heads
    layers = config.expert_num_layers

    slots = kv_slots(config, shard)
    transport = make_transport(args, slots, rank=IGPU_RANK, device=device)
    print(f"[worker] up, shard={shard} keys x {layers} layers", flush=True)

    flat = transport.recv("setup").clone()
    per = shard * KV_HEADS * HEAD_DIM
    keys, values = [], []
    for layer in range(layers):
        base = layer * 2 * per
        keys.append(flat[base : base + per].view(1, shard, KV_HEADS, HEAD_DIM))
        values.append(flat[base + per : base + 2 * per].view(1, shard, KV_HEADS, HEAD_DIM))
    pad = flat[layers * 2 * per :].view(1, shard) > 0.5
    # Every suffix row sees every valid prefix column; the two blocking flags
    # that could carve holes in that are off in the released config
    # (`block_future_depth_to_action`, `block_suffix_to_future_video`).
    mask = pad[:, None, :].expand(1, suffix, shard).contiguous()
    print(f"[worker] setup received, {flat.numel() * 2 / 2**20:.1f} MiB", flush=True)

    # Time the iGPU's own block, so the driver's `recv` wait can be split into
    # wire and iGPU compute by measurement rather than by interpolating a
    # bandwidth curve. This is the number that answers "is it the wire or the
    # iGPU?" for the context-parallel split.
    served, block_ms = 0, []
    while True:
        payload = transport.recv("query")
        header = float(payload[0])
        if header == OP_SHUTDOWN:
            break
        layer = int(header)
        query = payload[1:].view(1, suffix, heads, HEAD_DIM).to(dtype)
        start = time.perf_counter()
        out, lse = block_attention(query, keys[layer], values[layer], mask)
        reply = torch.cat([out.reshape(-1).float(), lse.reshape(-1).float()])
        torch.xpu.synchronize(device)
        block_ms.append((time.perf_counter() - start) * 1e3)
        transport.send("reply", reply)
        served += 1

    if block_ms:
        tail = block_ms[len(block_ms) // 2 :]  # drop the warmup half
        print(f"[worker] iGPU's own attention block over {shard} keys: "
              f"{statistics.median(tail):.3f} ms/layer-step "
              f"(min {min(tail):.3f}, {len(tail)} samples)", flush=True)
    print(f"[worker] served {served} layer-steps, shutting down", flush=True)
    transport.close()
    return 0


# ---------------------------------------------------------------------------
# --split seq : sequence parallel over the 51 suffix rows
# ---------------------------------------------------------------------------
def suffix_row_split(config) -> tuple[slice, slice]:
    """Rank 0 takes the state token and the first half of the actions.

    The state token is row 0 and belongs to whoever holds it, so the halves are
    26/25 rather than 25.5 each.
    """
    suffix = config.chunk_size + 1
    cut = (suffix + 1) // 2
    return slice(0, cut), slice(cut, suffix)


def embed_suffix_rows(
    model, state: torch.Tensor, x_rows: torch.Tensor, timestep: torch.Tensor, *, with_state: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """``embed_suffix`` for one rank's rows. Row-wise, so this is exact.

    Reproduces ``LingbotVlaV2ForActionPrediction.embed_suffix``
    (``modeling_lingbot_vla_v2.py:1501``) rather than calling it, because that
    method always emits the state token and all ``chunk_size`` action rows.
    Returns ``(time_emb, embs)``.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
        TIME_MAX_PERIOD,
        TIME_MIN_PERIOD,
        create_sinusoidal_pos_embedding,
    )

    dtype, device = state.dtype, state.device
    time_emb = create_sinusoidal_pos_embedding(
        timestep, model.proj_width, min_period=TIME_MIN_PERIOD, max_period=TIME_MAX_PERIOD, device=device
    ).to(dtype=dtype)

    action_emb = model.action_in_proj(x_rows)
    action_time = torch.cat([action_emb, time_emb[:, None, :].expand(-1, action_emb.shape[1], -1)], dim=-1)
    action_time = model.action_time_mlp_out(F.silu(model.action_time_mlp_in(action_time)))
    if with_state:
        return time_emb, torch.cat([model.state_proj(state)[:, None], action_time], dim=1)
    return time_emb, action_time


def full_suffix_mask(model, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
    """The ``[1, 51, 337]`` mask ``predict_velocity`` builds, before sharding.

    Built here rather than inside the sharded forward because the block scheme
    is cumulative over the *whole* suffix -- a rank that ran
    ``make_att_2d_masks`` on its own 25 rows would restart the cumsum and let
    its rows see the state token.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import make_att_2d_masks

    config = model.config
    bsize, prefix_len = prefix_pad_masks.shape
    suffix = config.chunk_size + 1
    device = prefix_pad_masks.device
    pad = torch.ones((bsize, suffix), device=device, dtype=torch.bool)
    att = torch.zeros((bsize, suffix), device=device, dtype=torch.bool)
    att[:, :2] = True
    prefix_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix, prefix_len)
    full = torch.cat([prefix_2d, make_att_2d_masks(pad, att)], dim=2)
    return model._block_query_columns(full, prefix_len)


def sp_predict_velocity(
    model,
    *,
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_position_ids: torch.Tensor,
    past_key_values: list,
    x_rows: torch.Tensor,
    timestep: torch.Tensor,
    rows: slice,
    peer_rows: slice,
    exchange,
) -> torch.Tensor:
    """``predict_velocity`` for one rank's rows, with ring attention.

    Reproduces ``predict_velocity`` (``:1665``) and ``LingbotJointModel.forward``
    (``:1127``) for the expert tower only -- which is all the denoise loop runs,
    since it passes ``inputs_embeds=[None, suffix_embs]``.

    Every operation in the layer is row-wise except attention: ``compute_qkv``,
    ``o_proj``, both AdaRMSNorms and the MoE all act per row. So the shard is
    exact, and ``exchange`` -- one ring step, world size 2 -- is the only
    communication.
    """
    config = model.config
    joint = model.qwenvl_with_expert
    prefix_len = prefix_pad_masks.shape[1]
    with_state = rows.start == 0
    n_rows = rows.stop - rows.start

    time_emb, hidden = embed_suffix_rows(model, state, x_rows, timestep, with_state=with_state)
    ada_cond = time_emb if config.adanorm_time else None

    full_mask = full_suffix_mask(model, prefix_pad_masks)
    mask_local = torch.cat(
        [full_mask[:, rows, :prefix_len], full_mask[:, rows, prefix_len + rows.start : prefix_len + rows.stop]],
        dim=2,
    )
    mask_peer = full_mask[:, rows, prefix_len + peer_rows.start : prefix_len + peer_rows.stop]

    suffix_pad = torch.ones((state.shape[0], config.chunk_size + 1), device=state.device, dtype=torch.bool)
    full_position_ids = model._build_full_position_ids(prefix_position_ids, prefix_pad_masks, suffix_pad)
    position_ids = full_position_ids[:, :, prefix_len + rows.start : prefix_len + rows.stop]

    for layer_idx in range(joint.num_layers):
        layer = joint.qwen_expert.model.layers[layer_idx]
        query, key, value = layer.compute_qkv(hidden, ada_cond)
        if joint.attention_precision == "fp32":
            query, key, value = query.float(), key.float(), value.float()
        query, key = joint.apply_mrope(query, key, position_ids)

        peer_key, peer_value = exchange(layer_idx, key, value)

        cached_key, cached_value = past_key_values[layer_idx]
        out, lse = block_attention(
            query,
            torch.cat([cached_key, key], dim=1),
            torch.cat([cached_value, value], dim=1),
            mask_local,
        )
        if peer_key is not None:
            out, lse = merge_out_lse(out, lse, *block_attention(query, peer_key, peer_value, mask_peer))
        att_output = out.transpose(1, 2).reshape(
            state.shape[0], n_rows, config.expert_num_attention_heads * config.expert_head_dim
        ).to(query.dtype)
        hidden = layer.apply_attention(hidden, att_output, 0, n_rows, ada_cond)

    tower = joint.qwen_expert.model
    hidden = tower.norm(hidden, ada_cond) if tower.final_norm_adanorm else tower.norm(hidden)
    action_rows = hidden[:, 1:] if with_state else hidden
    return model.action_out_proj(action_rows.to(model.action_out_proj.weight.dtype))


def seq_exchange(transport, *, max_rows: int, dtype: torch.dtype, first: bool):
    """One ring step over the shared ``ring`` slot, ordered rather than duplex.

    The slot is one buffer per rank, so a simultaneous send/recv on it would
    clobber. Rank 0 sends first and rank 1 second, which makes the exchange
    **two** sequential hops. A duplex transport would halve it; the findings
    quote both the measured cost and that halved projection.
    """
    payload = 2 * max_rows * KV_HEADS * HEAD_DIM

    def pack(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        rows = key.shape[1]
        flat = torch.zeros(1 + payload, dtype=torch.float16, device=key.device)
        flat[0] = float(rows)
        span = rows * KV_HEADS * HEAD_DIM
        flat[1 : 1 + span] = key.reshape(-1).to(torch.float16)
        flat[1 + payload // 2 : 1 + payload // 2 + span] = value.reshape(-1).to(torch.float16)
        return flat

    def unpack(flat: torch.Tensor):
        rows = int(flat[0])
        span = rows * KV_HEADS * HEAD_DIM
        key = flat[1 : 1 + span].view(1, rows, KV_HEADS, HEAD_DIM).to(dtype)
        value = flat[1 + payload // 2 : 1 + payload // 2 + span].view(1, rows, KV_HEADS, HEAD_DIM).to(dtype)
        return key, value

    cost: list[float] = []

    def exchange(layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        start = time.perf_counter()
        if first:
            transport.send("ring", pack(key, value))
            peer = unpack(transport.recv("ring").clone())
        else:
            peer = unpack(transport.recv("ring").clone())
            transport.send("ring", pack(key, value))
        cost.append((time.perf_counter() - start) * 1e3)
        return peer

    return exchange, cost


def seq_driver(args, processor, model, obs, device, dtype) -> dict:
    """dGPU rank. Rows [0:26] here, [26:51] on the iGPU."""
    decoder, session, state = ground(model, processor, obs, device, dtype)
    config = model.config
    steps = args.num_steps or config.num_steps
    mine, theirs = suffix_row_split(config)
    max_rows = max(mine.stop - mine.start, theirs.stop - theirs.start)

    slots = seq_slots(config, max_rows)
    transport = make_transport(args, slots, rank=DGPU_RANK, device=device)
    print(f"[seq] suffix rows dGPU[{mine.start}:{mine.stop}] iGPU[{theirs.start}:{theirs.stop}], "
          f"setup {slots[0].numel * 2 / 2**20:.1f} MiB, ring {slots[2].numel * 2 / 2**10:.0f} KiB/hop")

    noise = torch.randn((1, config.chunk_size, config.max_action_dim), device=device, dtype=dtype)
    # `predict_velocity` drops the state token from its output, so the action
    # rows are the suffix rows shifted by one.
    mine_actions = slice(max(mine.start - 1, 0), mine.stop - 1)
    theirs_actions = slice(theirs.start - 1, theirs.stop - 1)

    def single_loop() -> torch.Tensor:
        return model.denoise_actions(
            state=state,
            prefix_pad_masks=session.pad_masks,
            prefix_position_ids=session.position_ids,
            past_key_values=session.past_key_values,
            noise=noise.clone(),
            num_steps=steps,
        )

    reference = single_loop()
    # Same process, same host state; the recorded 213.5 ms is compiled.
    single_ms = median_ms(single_loop, iters=args.iters, warmup=args.warmup, device=device)
    print(f"[seq] loop on the dGPU alone (eager): {single_ms:.2f} ms")

    # The decomposition that makes the result readable: the same sharded forward
    # on 26 of the 51 rows with the ring step stubbed out. No transport, no
    # peer -- so this is what the dGPU's own half costs, and everything above it
    # in the two-process run is either the wire or waiting for the iGPU.
    def solo_loop(rows: slice, peer: slice, actions: slice) -> torch.Tensor:
        dt = torch.tensor(-1.0 / steps, dtype=dtype, device=device)
        now = torch.tensor(1.0, dtype=dtype, device=device)
        x_rows = noise[:, actions].clone()
        for _ in range(steps):
            v_rows = sp_predict_velocity(
                model,
                state=state,
                prefix_pad_masks=session.pad_masks,
                prefix_position_ids=session.position_ids,
                past_key_values=session.past_key_values,
                x_rows=x_rows,
                timestep=now.expand(1),
                rows=rows,
                peer_rows=peer,
                exchange=lambda layer_idx, key, value: (None, None),
            )
            x_rows = x_rows + dt * v_rows
            now = now + dt
        return x_rows

    # The control. `sp_predict_velocity` is a reimplementation of
    # `LingbotJointModel.forward`, so before reading anything into the sharded
    # cost, run the same code path on **all 51 rows** with no peer: that is
    # `denoise_actions`' own work done by this file's forward, and the gap to
    # `single_ms` is the probe's overhead rather than the split's.
    whole = slice(0, config.chunk_size + 1)
    unsharded_ms = median_ms(
        lambda: solo_loop(whole, slice(whole.stop, whole.stop), slice(0, config.chunk_size)),
        iters=args.iters, warmup=args.warmup, device=device,
    )
    solo_ms = median_ms(
        lambda: solo_loop(mine, theirs, mine_actions),
        iters=args.iters, warmup=args.warmup, device=device,
    )
    layer_steps_ = config.expert_num_layers * steps
    print(f"[seq] this file's forward, all 51 rows, no ring: {unsharded_ms:.2f} ms "
          f"({unsharded_ms / layer_steps_:.3f} ms/layer-step)  <- control for the probe's own overhead")
    print(f"[seq] the dGPU's own 26-row half, ring stubbed: {solo_ms:.2f} ms "
          f"({solo_ms / layer_steps_:.3f} ms/layer-step)")
    print(f"[seq] halving the rows buys {1 - solo_ms / unsharded_ms:+.1%} of that forward's time "
          f"(§G5 predicted at most ~13%)")

    # -- setup -------------------------------------------------------------
    parts = []
    for key, value in session.past_key_values:
        parts.append(key.reshape(-1).to(torch.float16))
        parts.append(value.reshape(-1).to(torch.float16))
    sync(device)
    setup_start = time.perf_counter()
    transport.send("setup", torch.cat(parts))
    meta = torch.cat([
        session.position_ids.reshape(-1).float(),
        session.pad_masks.reshape(-1).float(),
        state.reshape(-1).float(),
        F.pad(noise[0, theirs_actions].float(),
              (0, 0, 0, max_rows - (theirs_actions.stop - theirs_actions.start))).reshape(-1),
    ])
    transport.send("meta", meta)
    setup_ms = (time.perf_counter() - setup_start) * 1e3
    print(f"[seq] setup shipped in {setup_ms:.1f} ms")

    exchange, cost = seq_exchange(transport, max_rows=max_rows, dtype=dtype, first=True)
    ring_numel = slots[2].numel

    def loop() -> torch.Tensor:
        dt = torch.tensor(-1.0 / steps, dtype=dtype, device=device)
        now = torch.tensor(1.0, dtype=dtype, device=device)
        x_rows = noise[:, mine_actions].clone()
        for _ in range(steps):
            v_rows = sp_predict_velocity(
                model,
                state=state,
                prefix_pad_masks=session.pad_masks,
                prefix_position_ids=session.position_ids,
                past_key_values=session.past_key_values,
                x_rows=x_rows,
                timestep=now.expand(1),
                rows=mine,
                peer_rows=theirs,
                exchange=exchange,
            )
            x_rows = x_rows + dt * v_rows
            now = now + dt
        return x_rows

    for _ in range(args.warmup):
        loop()
    cost.clear()
    samples = []
    for _ in range(args.iters):
        sync(device)
        start = time.perf_counter()
        mine_out = loop()
        sync(device)
        samples.append((time.perf_counter() - start) * 1e3)

    # The worker returns its rows through the ring slot; then shut it down.
    peer_out = transport.recv("ring").clone()
    rows_back = theirs_actions.stop - theirs_actions.start
    peer_rows = peer_out[1 : 1 + rows_back * config.max_action_dim].view(
        1, rows_back, config.max_action_dim
    ).to(dtype)
    stop = torch.zeros(ring_numel, dtype=torch.float16, device=device)
    stop[0] = OP_SHUTDOWN
    transport.send("ring", stop)

    chunk = torch.cat([mine_out, peer_rows], dim=1)
    split_ms = statistics.median(samples)
    per_exchange = statistics.median(cost)
    dims = real_dims(processor, device)
    accuracy = {**deviation(chunk, reference), "rms": rms(chunk, reference, dims, config.spec_max_exec_steps)}
    layer_steps = config.expert_num_layers * steps
    blocked_ms = split_ms - solo_ms
    print()
    print(f"loop, dGPU alone, all 51 rows       {single_ms:8.2f} ms   (denoise_actions)")
    print(f"loop, this file's forward, 51 rows  {unsharded_ms:8.2f} ms   (probe overhead control)")
    print(f"loop, dGPU's own 26 rows, no ring   {solo_ms:8.2f} ms   <- what the split leaves it")
    print(f"loop, dGPU + iGPU sequence parallel {split_ms:8.2f} ms   -> {single_ms / split_ms:.2f}x   "
          f"(compiled single-card baseline {BASELINE_LOOP_MS:.1f} ms)")
    print(f"per layer-step ring exchange        {per_exchange:8.3f} ms  ({len(cost) // max(args.iters, 1)} per loop)")
    print(f"blocked in the exchange             {blocked_ms:8.2f} ms   "
          f"({blocked_ms / layer_steps:.3f} ms/layer-step, {blocked_ms / split_ms:.0%} of the wall)")
    print(f"chunk vs single-device             max_abs={accuracy['max_abs']:.3e} rms={accuracy['rms']:.3e} "
          f"(fp16 floor {FP16_FLOOR:.1e})")
    # The best case this topology could ever reach: a duplex, non-blocking
    # transport that overlaps the dGPU's local block with the peer's, and a free
    # wire. Then the wall is whichever card is slower, and nothing else.
    print()
    print(f"best case, free wire and perfect overlap: max(dGPU {solo_ms:.0f}, iGPU >= "
          f"{blocked_ms:.0f}) = {max(solo_ms, blocked_ms):8.2f} ms   "
          f"-> {single_ms / max(solo_ms, blocked_ms):.2f}x")

    transport.close()
    decoder.close()
    return {
        "arm": "ring-2p", "split": "seq",
        "rows": [mine.start, mine.stop, theirs.start, theirs.stop],
        "setup_ms": setup_ms,
        "loop_single_ms": single_ms,
        "loop_unsharded_probe_ms": unsharded_ms,
        "loop_solo_shard_ms": solo_ms,
        "row_halving_gain": 1 - solo_ms / unsharded_ms,
        "loop_split_ms": split_ms,
        "blocked_ms": blocked_ms,
        "speedup": single_ms / split_ms,
        "best_case_ms": max(solo_ms, blocked_ms),
        "best_case_speedup": single_ms / max(solo_ms, blocked_ms),
        "exchange_ms": per_exchange,
        "exchanges_per_loop": len(cost) // max(args.iters, 1),
        "accuracy": accuracy,
        "payload_bytes": {"setup": slots[0].numel * 2, "meta": slots[1].numel * 4, "ring": slots[2].numel * 2},
    }


def seq_worker(args, model, device, dtype) -> int:
    """iGPU rank. The same sharded forward, on the other half of the rows."""
    config = model.config
    steps = args.num_steps or config.num_steps
    theirs, mine = suffix_row_split(config)  # rank 1's rows are the second half
    max_rows = max(mine.stop - mine.start, theirs.stop - theirs.start)
    layers = config.expert_num_layers

    slots = seq_slots(config, max_rows)
    transport = make_transport(args, slots, rank=IGPU_RANK, device=device)
    print(f"[worker] up, rows [{mine.start}:{mine.stop}]", flush=True)

    flat = transport.recv("setup").clone()
    per = PREFIX_LEN * KV_HEADS * HEAD_DIM
    past_key_values = []
    for layer in range(layers):
        base = layer * 2 * per
        past_key_values.append((
            flat[base : base + per].view(1, PREFIX_LEN, KV_HEADS, HEAD_DIM).to(dtype),
            flat[base + per : base + 2 * per].view(1, PREFIX_LEN, KV_HEADS, HEAD_DIM).to(dtype),
        ))
    meta = transport.recv("meta").clone()
    cut = 3 * PREFIX_LEN
    position_ids = meta[:cut].view(3, 1, PREFIX_LEN).long()
    pad_masks = (meta[cut : cut + PREFIX_LEN].view(1, PREFIX_LEN) > 0.5)
    state = meta[cut + PREFIX_LEN : cut + PREFIX_LEN + config.max_state_dim].view(1, -1).to(dtype)
    noise_rows = meta[cut + PREFIX_LEN + config.max_state_dim :].view(1, max_rows, config.max_action_dim)
    rows_here = mine.stop - mine.start
    noise_rows = noise_rows[:, :rows_here].to(dtype)
    print(f"[worker] setup received, {flat.numel() * 2 / 2**20:.1f} MiB prefix KV", flush=True)

    exchange, _ = seq_exchange(transport, max_rows=max_rows, dtype=dtype, first=False)

    def loop(exch=None) -> torch.Tensor:
        dt = torch.tensor(-1.0 / steps, dtype=dtype, device=device)
        now = torch.tensor(1.0, dtype=dtype, device=device)
        x_rows = noise_rows.clone()
        for _ in range(steps):
            v_rows = sp_predict_velocity(
                model,
                state=state,
                prefix_pad_masks=pad_masks,
                prefix_position_ids=position_ids,
                past_key_values=past_key_values,
                x_rows=x_rows,
                timestep=now.expand(1),
                rows=mine,
                peer_rows=theirs,
                exchange=exch if exch is not None else exchange,
            )
            x_rows = x_rows + dt * v_rows
            now = now + dt
        return x_rows

    # The iGPU's own half, ring stubbed -- the mirror of the driver's `solo_loop`
    # and the number that decides "is the split iGPU-bound or wire-bound?".
    # Measured here, before the lockstep starts, so it costs the driver only a
    # longer wait on its first warmup exchange and nothing in the timed loops.
    solo_ms = median_ms(
        lambda: loop(lambda layer_idx, key, value: (None, None)),
        iters=max(args.iters, 1), warmup=1, device=device,
    )
    print(f"[worker] iGPU's own {rows_here}-row half, ring stubbed: {solo_ms:.2f} ms "
          f"({solo_ms / (layers * steps):.3f} ms/layer-step)", flush=True)

    # Lockstep with the driver: the exchange is what synchronises the two, so
    # the worker must run exactly as many loops as the driver does.
    out = None
    for _ in range(args.warmup + args.iters):
        out = loop()

    payload = torch.zeros(slots[2].numel, dtype=torch.float16, device=device)
    payload[0] = float(out.shape[1])
    payload[1 : 1 + out.numel()] = out.reshape(-1).to(torch.float16)
    transport.send("ring", payload)
    final = transport.recv("ring")
    print(f"[worker] returned {out.shape[1]} action rows, shutdown={float(final[0])}", flush=True)
    transport.close()
    return 0


# ---------------------------------------------------------------------------
# Arm: ring-2p
# ---------------------------------------------------------------------------
def arm_ring_2p(args, processor, model, obs, device, dtype) -> dict:
    clear_rendezvous(args)
    if args.worker_cpu:
        reserve_cpus_for_worker(args.worker_cpu)
    log = args.worker_log or f"/tmp/lingbot-ring-worker-{args.split}.log"
    child = spawn_worker(args, log=log)
    print(f"[2p] worker pid {child.pid}, log {log}")
    try:
        driver = {"kv": kv_driver, "seq": seq_driver, "wire": wire_driver}[args.split]
        result = driver(args, processor, model, obs, device, dtype)
    finally:
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait(timeout=30)
        clear_rendezvous(args)
    result["worker_log"] = log
    result["worker_returncode"] = child.returncode
    return result


# ---------------------------------------------------------------------------
# Arm: share-sweep  (run once per device)
# ---------------------------------------------------------------------------
# Gate 2/3 found the iGPU's compute, not the wire, to be the dominant term. The
# obvious follow-up is load balancing: **give the iGPU a smaller share.** That
# only works if the iGPU's cost actually scales with its share, and this arm
# measures whether it does -- on whichever card `ZE_AFFINITY_MASK` selects, so
# the two curves can be put side by side the way §L's probe does it.
#
# Both splits get a curve, because "less work" means a different thing in each:
#   rows  the sequence split's share -- how many of the 51 suffix rows
#   keys  the context split's share -- how many of the 286 prefix KV columns


def expert_weight_bytes(config) -> int:
    """Routed expert weights read per denoise step, all layers, fp16.

    ``32 experts x 3 matrices x expert_hidden_size x token_moe_intermediate_size``
    per layer -- §F2's 75.5 MB, times 36 layers. This is what sets the iGPU's
    floor in the sequence split, because sequence parallelism **replicates the
    weights**: a rank with one row still streams all of them.
    """
    per_layer = (
        config.token_num_experts * 3 * config.expert_hidden_size * config.token_moe_intermediate_size * 2
    )
    return per_layer * config.expert_num_layers


def arm_share_sweep(args, processor, model, obs, device, dtype) -> dict:
    """Does the iGPU get cheaper if it is given less to do? Two curves."""
    decoder, session, state = ground(model, processor, obs, device, dtype)
    config = model.config
    steps = args.num_steps or config.num_steps
    suffix = config.chunk_size + 1
    prefix_len = session.pad_masks.shape[1]
    affinity = os.environ.get("ZE_AFFINITY_MASK", "unset")
    noise = torch.randn((1, config.chunk_size, config.max_action_dim), device=device, dtype=dtype)

    # -- the sequence split's share: rows -----------------------------------
    row_rows = []
    if "rows" in args.sweep:
        def sharded_loop(n_rows: int, step_fn=None) -> None:
            step_fn = step_fn or sp_predict_velocity
            rows = slice(0, n_rows)
            peer = slice(n_rows, suffix)
            actions = slice(0, n_rows - 1)
            dt = torch.tensor(-1.0 / steps, dtype=dtype, device=device)
            now = torch.tensor(1.0, dtype=dtype, device=device)
            x_rows = noise[:, actions].clone()
            for _ in range(steps):
                v_rows = step_fn(
                    model,
                    state=state,
                    prefix_pad_masks=session.pad_masks,
                    prefix_position_ids=session.position_ids,
                    past_key_values=session.past_key_values,
                    x_rows=x_rows,
                    timestep=now.expand(1),
                    rows=rows,
                    peer_rows=peer,
                    exchange=lambda layer_idx, key, value: (None, None),
                )
                x_rows = x_rows + dt * v_rows
                now = now + dt

        # Eager and, with --compile-denoise-step, compiled. The compiled curve is
        # the one that answers "how much of the loop is the 51 rows' own work?":
        # everything row-proportional -- the fused q/k/v projection that builds
        # their K/V, o_proj, the norms, attention's query side -- rides on this
        # slope, while the MoE's expert weight *bytes* do not (§F2: 75.5 MB per
        # layer-step whether there are 51 rows or 1). The eager curve is flat for
        # the reason §8 (5) records, so it cannot answer it.
        from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
            denoise_compile_options,
        )

        eager_pv_rows = model.predict_velocity
        for label, want_compile in (("eager", False), ("compiled", args.compile_denoise_step)):
            if label == "compiled" and not want_compile:
                continue
            if want_compile:
                torch._dynamo.reset()
                torch._dynamo.config.cache_size_limit = max(128, torch._dynamo.config.cache_size_limit)
            print(f"[rows] {label}: the sequence split's share -- {steps} steps, ring stubbed, no peer")
            full = None
            for n_rows in [int(v) for v in args.row_shares.split(",") if v.strip()]:
                if n_rows < 2 or n_rows > suffix:
                    continue
                run = sharded_loop
                if want_compile:
                    compiled_step = torch.compile(
                        sp_predict_velocity, backend="inductor", dynamic=False, fullgraph=False,
                        options=denoise_compile_options() or None,
                    )
                    run = lambda n, _c=compiled_step: sharded_loop(n, _c)  # noqa: E731
                ms = median_ms(lambda n=n_rows, r=run: r(n), iters=args.iters, warmup=2, device=device)
                full = full if full is not None else ms
                row_rows.append({"mode": label, "suffix_rows": n_rows, "action_rows": n_rows - 1,
                                 "ms": ms, "vs_full": ms / full})
                print(f"[rows] {label:<8} {n_rows - 1:>2} of {config.chunk_size} action rows  {ms:9.2f} ms  "
                      f"{ms / full:5.3f}x   "
                      f"({ms / (config.expert_num_layers * steps):.3f} ms/layer-step)")
            if full is not None:
                last = [r for r in row_rows if r["mode"] == label][-1]
                print(f"[rows] {label}: {suffix} -> {last['suffix_rows']} rows changes the loop by "
                      f"{last['vs_full'] - 1:+.1%}  "
                      f"=> row-proportional work is <= {full - last['ms']:.1f} ms of {full:.1f}")
        model.predict_velocity = eager_pv_rows
        torch._dynamo.reset()

        # The byte budget, exact from the config -- the denominator for "where
        # does the time go". fp16 throughout.
        h, hd = config.expert_hidden_size, config.expert_head_dim
        qkv_out = (config.expert_num_attention_heads + 2 * config.expert_num_key_value_heads) * hd
        budget = {
            "routed experts (MoE)": config.token_num_experts * 3 * h * config.token_moe_intermediate_size * 2,
            "fused q/k/v proj": h * qkv_out * 2,
            "o_proj": config.expert_num_attention_heads * hd * h * 2,
            "prefix KV read": prefix_len * KV_HEADS * HEAD_DIM * 2 * 2,
        }
        total = sum(budget.values())
        print("[rows] per-layer-step byte budget, fp16, exact from the config:")
        for name, nbytes in budget.items():
            print(f"[rows]   {name:<22} {nbytes / 2**20:7.2f} MiB   {nbytes / total:6.1%}")
        print(f"[rows]   {'(these four)':<22} {total / 2**20:7.2f} MiB")
        bytes_per_step = expert_weight_bytes(config)
        print(f"[rows] expert weights streamed per step: {bytes_per_step / 1e6:.0f} MB, "
              f"x{steps} steps = {bytes_per_step * steps / 1e9:.1f} GB")
        for name, bandwidth in (("dGPU 449 GB/s", 449e9), ("iGPU 29 GB/s", 29e9)):
            print(f"[rows]   zero-row floor at {name}: "
                  f"{bytes_per_step * steps / bandwidth * 1e3:8.1f} ms")

    # -- the context split's share: prefix KV columns -----------------------
    key_rows = []
    if "keys" in args.sweep:
        captured = capture_attention_inputs(model, session, state, noise)
        q, k, v, mask = captured["query"], captured["key"], captured["value"], captured["mask"]
        print(f"\n[keys] the context split's share -- one block_attention, "
              f"q={tuple(q.shape)}")
        full = None
        for n_keys in [int(v) for v in args.key_shares.split(",") if v.strip()]:
            if n_keys < 1 or n_keys > prefix_len:
                continue
            us = median_ms(
                lambda n=n_keys: block_attention(q, k[:, :n], v[:, :n], mask[:, :, :n]),
                iters=50, warmup=10, device=device,
            ) * 1e3
            full = full if full is not None else us
            key_rows.append({"keys": n_keys, "us": us, "vs_full": us / full})
            print(f"[keys] {n_keys:>4} of {prefix_len} prefix keys  {us:9.1f} us  "
                  f"{us / full:5.2f}x the full share")

    # -- the question without an ablation: shorten the prefix KV for real ---
    # "The loop is memory-bound, so why does halving the KV not help?" is
    # answerable directly: truncate the prefix KV cache and time the **whole**
    # loop. No stub, so nothing can be dead-code-eliminated, and no ceiling
    # arithmetic. The outputs are meaningless (the suffix sees fewer prefix
    # tokens) -- this measures cost, not behaviour.
    loop_rows = []
    if "loop" in args.sweep:
        from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
            denoise_compile_options,
        )

        kv_bytes_per_layer = prefix_len * KV_HEADS * HEAD_DIM * 2 * 2  # K and V, fp16
        print(f"\n[loop] prefix KV is {kv_bytes_per_layer / 2**20:.2f} MiB per layer-step "
              f"({kv_bytes_per_layer * config.expert_num_layers / 2**20:.1f} MiB for all "
              f"{config.expert_num_layers} layers), against "
              f"{expert_weight_bytes(config) / config.expert_num_layers / 2**20:.1f} MiB of "
              f"expert weights per layer-step -- i.e. "
              f"{kv_bytes_per_layer / (expert_weight_bytes(config) / config.expert_num_layers):.1%} "
              f"of the bytes")

        def truncated(n: int):
            return {
                "prefix_pad_masks": session.pad_masks[:, :n].contiguous(),
                "prefix_position_ids": session.position_ids[:, :, :n].contiguous(),
                "past_key_values": [
                    (key[:, :n].contiguous(), value[:, :n].contiguous())
                    for key, value in session.past_key_values
                ],
            }

        eager_pv = model.predict_velocity
        for label, compiled_mode in (("eager", False), ("compiled", args.compile_denoise_step)):
            if not compiled_mode and label == "compiled":
                continue
            if compiled_mode:
                torch._dynamo.reset()
                torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
                model.predict_velocity = torch.compile(
                    eager_pv, backend="inductor", dynamic=False, fullgraph=True,
                    options=denoise_compile_options() or None,
                )
            base = None
            for n_keys in [int(v) for v in args.loop_prefix.split(",") if v.strip()]:
                if n_keys < 1 or n_keys > prefix_len:
                    continue
                cond = truncated(n_keys)
                ms = median_ms(
                    lambda c=cond: model.denoise_actions(
                        state=state, noise=noise.clone(), num_steps=steps, **c
                    ),
                    iters=args.iters, warmup=2, device=device,
                )
                base = base if base is not None else ms
                loop_rows.append({"mode": label, "prefix_keys": n_keys, "ms": ms, "vs_full": ms / base})
                print(f"[loop] {label:<8} prefix {n_keys:>4} / {prefix_len} keys  {ms:8.2f} ms  "
                      f"{ms / base:5.3f}x")
            if base is not None:
                last = [r for r in loop_rows if r["mode"] == label][-1]
                print(f"[loop] {label}: prefix {prefix_len} -> {last['prefix_keys']} keys changes the "
                      f"loop by {last['vs_full'] - 1:+.1%}")
        model.predict_velocity = eager_pv
        torch._dynamo.reset()

    decoder.close()
    return {
        "arm": "share-sweep",
        "affinity": affinity,
        "rows": row_rows,
        "keys": key_rows,
        "loop": loop_rows,
        "prefix_kv_bytes_per_layer_step": prefix_len * KV_HEADS * HEAD_DIM * 2 * 2,
        "expert_weight_bytes_per_step": expert_weight_bytes(config),
    }


ARMS = {
    "ring-math": arm_ring_math,
    "attn-share": arm_attn_share,
    "ring-2p": arm_ring_2p,
    "share-sweep": arm_share_sweep,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", default="ring-math", choices=sorted(ARMS))
    parser.add_argument("--role", default="driver", choices=["driver", "worker"])
    parser.add_argument("--split", default="kv", choices=["kv", "seq", "wire"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--shards", default="1,2,4,8", help="ring-math: key-shard counts to grade")
    parser.add_argument("--sweep", default="rows,keys,loop", help="share-sweep: which curves to measure")
    parser.add_argument("--loop-prefix", default="286,143,72,8",
                        help="share-sweep loop: prefix KV lengths to time the whole loop at")
    parser.add_argument("--row-shares", default="51,26,13,7,3,2",
                        help="share-sweep: suffix rows to give one rank (state token included)")
    parser.add_argument("--key-shares", default="286,143,72,36,8,1",
                        help="share-sweep: prefix KV columns to give one rank")
    parser.add_argument("--accum", default="fp32", choices=["fp32", "input"],
                        help="ring-math: softmax accumulation dtype for the end-to-end run")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--wire-iters", type=int, default=200,
                        help="ring-2p --split wire: round trips to time (no compute on either end)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-denoise-step", action="store_true")
    parser.add_argument("--transport", default="oneccl", choices=["oneccl", "shm"])
    parser.add_argument("--ccl-lib", default=DEFAULT_CCL_LIB)
    parser.add_argument("--uid-file", default="/tmp/lingbot-ring-uid")
    parser.add_argument("--shm-prefix", default="lingbot_ring")
    parser.add_argument("--connect-timeout", type=float, default=900.0)
    parser.add_argument("--worker-cpu", default=None,
                        help="cores handed to the worker, e.g. 10-11. oneCCL recv hard-spins (PHASE10 §11.4)")
    parser.add_argument("--worker-log", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    role = args.role
    print(f"[setup] arm={args.arm} role={role} split={args.split} "
          f"affinity={os.environ.get('ZE_AFFINITY_MASK', 'unset')} device={device} "
          f"dtype={args.dtype} transport={args.transport} load1m={load_average():.2f}")
    if load_average() > 2.0:
        print("WARNING: load average above 2.0. Rules #2 -- absolute numbers from this run are suspect.")

    torch.manual_seed(args.seed)
    raw = json.loads((args.model / "transformer" / "config.json").read_text())

    # The `kv` worker does attention over a shipped KV shard and nothing else,
    # so it needs the config and no weights at all. Skipping the 6B load there
    # is not an optimisation of the probe -- it is the arm's whole point, that
    # the iGPU carries attention only.
    if role == "worker" and args.split in ("kv", "wire"):
        from vllm_omni.diffusion.models.lingbot_vla_v2.config import LingbotVlaV2Config

        config = LingbotVlaV2Config.from_model_config(raw)
        print(f"[setup] {args.split} worker: no weights loaded")
        with torch.inference_mode():
            runner = kv_worker if args.split == "kv" else wire_worker
            return runner(args, config, device, dtype)

    processor, model = build(args.model, device, dtype, num_steps=args.num_steps)
    # §F1: the module hardcodes attention_precision="fp32" in __init__ and the
    # pipeline overrides it from the config afterwards. A harness that builds
    # the module directly gets the wrong one silently, so set it explicitly and
    # print what is actually in force.
    joint = model.qwenvl_with_expert
    joint.attention_precision = raw.get("attention_precision", joint.attention_precision)
    joint.attention_backend = raw.get("attention_backend", joint.attention_backend)
    print(f"[setup] attention_backend={joint.attention_backend} "
          f"attention_precision={joint.attention_precision} moe={model.config.moe_implementation} "
          f"inference_mode=True")

    if role == "worker":
        with torch.inference_mode():
            return seq_worker(args, model, device, dtype)

    obs = observation(processor.spec, args.seed)
    # Compiled outside `inference_mode`, the way `phase12_paradigms_probe.py`
    # does it: dynamo's guard capture and inference tensors do not mix well.
    # `share-sweep` compiles per prefix length inside the arm; everything else
    # that wants a compiled loop gets it here. 4e-2 rather than
    # `phase12_paradigms_probe.py`'s 2e-2: on this container inductor's
    # single-velocity-step drift is 3.18e-2 (§B), while the 10-step chunk --
    # what Rules #3 gates on -- is 8.86e-3. Both are printed by the checker.
    if args.compile_denoise_step and args.arm not in ("attn-share", "share-sweep"):
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        compile_denoise_step(processor, model, obs, device, dtype, "inductor", False, False, 4e-2)

    with torch.inference_mode():
        result = ARMS[args.arm](args, processor, model, obs, device, dtype)

    result["load1m"] = load_average()
    result["dtype"] = args.dtype
    result["compiled"] = args.compile_denoise_step
    result["baseline_loop_ms"] = BASELINE_LOOP_MS
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"\n[json] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
