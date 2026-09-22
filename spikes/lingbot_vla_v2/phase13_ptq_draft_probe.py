#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 13 — a PTQ-compressed copy of the real model as the iGPU draft.

The scheme under test is ``spec_reground`` (PHASE10_SPECULATIVE.md §12.9) with
one substitution: the draft stops being a 4 M-parameter ``LingbotDraftHead``
that has to be trained, and becomes **the real action-expert tower, quantized**.
That substitution is worth measuring because it removes Phase 10's two largest
open items at once -- the draft head's training (accept rate is 0 today) and KV
staleness (probe 2 never ran; ``reground`` re-grounds every tick so there is
none).

**The budget is 58 ms, not 90.** ``_ground`` is 80.8 ms, but the draft only
needs ``embs``, which ``embed_prefix`` produces after 22.6 ms; the dGPU then
spends 58.1 ms in ``prefix_forward`` building the verifier's KV
(``spec_decode.py:440``). A draft that fits in that window is free -- §M §4
measured a memory-bound iGPU load costing the dGPU 1.01x, not §K's 1.75x.
Every millisecond past it lands on the tick:

    no speculation, baseline                                   292.6 ms
    cached K=1 @ accept 0.9  (ships today, prefix up to 5 ticks stale)
                                                                85.9 ms
    reground + draft <= 58 ms   (this scheme, fresh prefix)    103.6 ms
    reground + draft 90 ms                                      135   ms
    reground K=1 @ accept 1.0  (measured, 4 M head not hidden) 114.5 ms

So the scheme needs a point where **both** curves are good at once: accept rate
high enough to be worth speculating, and iGPU 10-step time inside 58 ms. This
probe measures the two curves and reports where they fail to meet.

Arms::

    ZE_AFFINITY_MASK=1 python phase13_ptq_draft_probe.py --arm kernel
    ZE_AFFINITY_MASK=0 python phase13_ptq_draft_probe.py --arm accept --model ... --dataset ...
    ZE_AFFINITY_MASK=1 python phase13_ptq_draft_probe.py --arm step   --model ... --launch-floor
    ZE_AFFINITY_MASK=0 python phase13_ptq_draft_probe.py --arm depth  --model ... --dataset ...

``kernel`` needs no checkpoint and settles which quantized GEMM to use at all.
``accept`` is dGPU-only and can close the scheme on accuracy without touching
the iGPU. ``step`` is the iGPU speed curve. ``depth`` is the layer-count curve,
which is what is left once ``step`` shows the floor is per-layer overhead rather
than bytes.

Why the MoE needed solving before any of this could run
-------------------------------------------------------
The routed experts are 69.5% of a denoise step's bytes and ``GroupedExperts``
contracts them with ``einsum``/``bmm``, while every weight-only-quant kernel on
XPU is a 2D ``mm``. The two are reconcilable without changing the arithmetic:

    gate/up   [E,I,H] -> [E*I, H]          x @ W.T   -> [T, E*I]
    down      [E,H,I] -> [H, E*I]          H @ D.T   -> [T, H]

for ``down`` the routing weights fold into the activation first, so
``H[t,(e,i)] = w[t,e] * h[e,t,i]`` and ``H @ D.T = sum_e w_e * h_e @ down_e``,
which is exactly what ``forward_dense``'s two einsums compute. Launch count
stays at 3 per layer. ``--check-folding`` asserts this against the shipped
``forward_dense`` in fp16, so the folding is validated separately from the
quantization error -- otherwise a folding bug and a quantization loss are the
same number.

Host state matters (Rules #2): ``load < 2.0`` and no other container holding a
GPU. §L's absolute numbers swung 2.4x under a busy host.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase5_latency import build, observation  # noqa: E402

# PHASE10_SPECULATIVE.md probe 3: the draft's per-dim RMS against the teacher
# has to be at or under this for the accept rule to have anything to accept.
DRAFT_RMS_TARGET = 0.02
# The window the draft has to fit in to be free -- `prefix_forward`, §F1.
HIDDEN_BUDGET_MS = 58.1
# §12.9: the tick this scheme costs when the draft is fully hidden and the
# round is accepted, and what a rejection adds on top (the Euler loop).
REGROUND_TICK_MS = 103.6
REJECT_COST_MS = 213.0


def sync(device: torch.device) -> None:
    if device.type in ("xpu", "cuda"):
        torch.accelerator.synchronize()


def load_average() -> float:
    return os.getloadavg()[0]


def auto_batch(fn, device: torch.device, cap: int, target_ms: float = 120.0) -> int:
    """Calls per timed region: enough to amortise submit/drain, no more.

    Back-to-back timing is not a convenience. §L had to retract a "the cache is
    useless" result that came from syncing once per call: at this size the
    submit/drain round trip is 0.37 ms on the iGPU, which is 30% of an int4
    layer-step and 0% of an fp16 one, so a per-call sync silently flatters
    whichever arm is slower. But a fixed batch is wrong in the other direction --
    ``aten``'s int8 path takes 468 ms per call, and 40 of those is a 19-second
    region. Size the region by time instead.
    """
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    once = (time.perf_counter() - start) * 1e3
    return max(1, min(cap, int(target_ms / max(once, 1e-3))))


def median_ms(fn, iters: int, warmup: int, device: torch.device, *, batch: int = 1) -> float:
    """Median ms per call, ``batch`` calls back to back inside each timed region."""
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(iters):
        sync(device)
        start = time.perf_counter()
        for _ in range(batch):
            fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1e3 / batch)
    return statistics.median(samples)


def issue_ms(fn, iters: int, warmup: int, device: torch.device) -> float:
    """Median wall time of one call with **no** sync -- Python plus dispatch.

    The queue absorbs the device work, so this is the part of a step that a
    graph capture or a compiled region could in principle remove. Reported next
    to the synced time because "the iGPU is too slow" and "we cannot submit
    work fast enough" call for completely different responses, and at 36 layers
    of small kernels the second is a live possibility (F3b measured 4.6x off
    host issue on a dispatch-bound chain).
    """
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
        sync(device)
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# Q4_0 via Intel's ESIMD kernels
# ---------------------------------------------------------------------------
# Layout is dictated by `vllm/model_executor/layers/quantization/sym_int4.py`:
# the quantizer emits int32 `[N, K/8]` and fp16 scales `[N, K/128]`, and the
# GEMM consumes the same storage viewed as uint8. `_register_linear_int4_layouts`
# does exactly this `.view(torch.uint8)`, so this is the shipped aliasing, not a
# reinterpretation of our own.
PACK_FACTOR = 8
GROUP_SIZE = 128
_ESIMD: Any = None


def esimd() -> Any:
    """The ESIMD op namespace, or ``None`` when the kernels are not installed."""
    global _ESIMD
    if _ESIMD is None:
        try:
            importlib.import_module("custom_esimd_kernels_vllm.q4_0_quant_ops")
            _ESIMD = torch.ops.custom_esimd_kernels_vllm
        except Exception:
            _ESIMD = False
    return _ESIMD or None


class Q40:
    """One Q4_0-packed weight plus the ESIMD GEMM/GEMV that consumes it.

    ``supported`` is a hard gate rather than a best effort: the kernel asserts
    ``N % 16 == 0`` and ``K % 128 == 0`` internally, and silently falling back
    to fp16 for some layers while claiming an int4 measurement is exactly the
    kind of "the fast path never engaged" error probe 4 had to retract once.
    Unsupported layers are counted and printed.
    """

    MIN_ROWS, MAX_ROWS = 2, 64

    @staticmethod
    def supported(weight: torch.Tensor) -> bool:
        if esimd() is None or weight.ndim != 2:
            return False
        out_features, in_features = weight.shape
        return out_features % 16 == 0 and in_features % GROUP_SIZE == 0

    def __init__(self, weight: torch.Tensor) -> None:
        out_features, in_features = weight.shape
        device = weight.device
        packed = torch.empty(out_features, in_features // PACK_FACTOR, dtype=torch.int32, device=device)
        scales = torch.empty(out_features, in_features // GROUP_SIZE, dtype=torch.float16, device=device)
        esimd().q4_0_quantize(weight.detach().to(torch.float16).contiguous(), packed, scales)
        self.qweight = packed.view(torch.uint8)
        self.scales = scales
        self.out_features = out_features
        # int4 nibble + one fp16 scale per group, which is what the device reads.
        self.bytes = out_features * in_features * 0.5 + scales.numel() * 2
        self._out: dict[int, torch.Tensor] = {}

    def mm(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.shape[0]
        # The kernel writes into a caller-supplied buffer, so allocating one per
        # call is measurable: 3 allocations per layer cost 0.35 ms on the iGPU,
        # 29% of the int4 layer-step. Shapes are fixed here (M=51 every step),
        # so a real implementation caches, and so does this.
        out = self._out.get(rows)
        if out is None:
            out = torch.empty(rows, self.out_features, dtype=torch.float16, device=x.device)
            self._out[rows] = out
        op = esimd().esimd_gemv_int4 if rows == 1 else esimd().esimd_gemm_int4_pgrp
        op(x.contiguous(), self.qweight, self.scales, out)
        return out

    def fits(self, rows: int) -> bool:
        return rows == 1 or self.MIN_ROWS <= rows <= self.MAX_ROWS


class Q4GroupedExperts(nn.Module):
    """``GroupedExperts`` with the three einsums folded into three Q4_0 GEMMs.

    Exposes ``forward_dense`` with the same signature, so ``TokenMoeBlock``
    needs no change: the swap happens at the ``experts`` attribute.
    """

    def __init__(self, experts: nn.Module) -> None:
        super().__init__()
        num_experts, intermediate, hidden = experts.gate_proj.shape
        self.num_experts, self.intermediate, self.hidden = num_experts, intermediate, hidden
        self.q_gate = Q40(experts.gate_proj.reshape(num_experts * intermediate, hidden))
        self.q_up = Q40(experts.up_proj.reshape(num_experts * intermediate, hidden))
        # down is [E, H, I]; the fold needs D[h, (e,i)] = down[e, h, i].
        self.q_down = Q40(experts.down_proj.permute(1, 0, 2).reshape(hidden, num_experts * intermediate))
        self.bytes = self.q_gate.bytes + self.q_up.bytes + self.q_down.bytes

    @staticmethod
    def supported(experts: nn.Module) -> bool:
        num_experts, intermediate, hidden = experts.gate_proj.shape
        return Q40.supported(experts.gate_proj.reshape(num_experts * intermediate, hidden)) and Q40.supported(
            experts.down_proj.permute(1, 0, 2).reshape(hidden, num_experts * intermediate)
        )

    def forward_dense(
        self, hidden_flat: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        tokens = hidden_flat.shape[0]
        x = hidden_flat.to(torch.float16)
        inter = F.silu(self.q_gate.mm(x)) * self.q_up.mm(x)
        # [T, E] dense routing weights, zero for unselected pairs -- the same
        # contraction `forward_dense` builds, kept in the identical order so the
        # only difference between the two paths is the weight precision.
        mask = F.one_hot(selected_experts, num_classes=self.num_experts).to(routing_weights.dtype)
        weights = (mask * routing_weights.unsqueeze(-1)).sum(dim=1)
        inter = inter.view(tokens, self.num_experts, self.intermediate) * weights.to(torch.float16).unsqueeze(-1)
        return self.q_down.mm(inter.reshape(tokens, self.num_experts * self.intermediate))

    def forward_gather(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("the folded Q4_0 path implements `dense` only")


class FoldedGroupedExperts(nn.Module):
    """The same fold at fp16 -- the control arm for ``--check-folding``.

    Without this, a folding bug and a quantization loss produce the same
    number and neither can be attributed.
    """

    def __init__(self, experts: nn.Module) -> None:
        super().__init__()
        num_experts, intermediate, hidden = experts.gate_proj.shape
        self.num_experts, self.intermediate = num_experts, intermediate
        self.gate = experts.gate_proj.reshape(num_experts * intermediate, hidden)
        self.up = experts.up_proj.reshape(num_experts * intermediate, hidden)
        self.down = experts.down_proj.permute(1, 0, 2).reshape(hidden, num_experts * intermediate).contiguous()

    def forward_dense(
        self, hidden_flat: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        tokens = hidden_flat.shape[0]
        inter = F.silu(hidden_flat @ self.gate.t()) * (hidden_flat @ self.up.t())
        mask = F.one_hot(selected_experts, num_classes=self.num_experts).to(routing_weights.dtype)
        weights = (mask * routing_weights.unsqueeze(-1)).sum(dim=1)
        inter = inter.view(tokens, self.num_experts, self.intermediate) * weights.unsqueeze(-1).to(inter.dtype)
        return inter.reshape(tokens, -1) @ self.down.t()


class Q4Linear(nn.Module):
    """``nn.Linear`` with a Q4_0 weight, falling back to fp16 outside the
    kernel's row range (it accepts 1, or 2..64; the expert tower runs at 51)."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.q = Q40(linear.weight.data)
        self.bias = linear.bias
        self.fallback = linear
        self.bytes = self.q.bytes

    @property
    def weight(self) -> torch.Tensor:
        """``compute_qkv`` reads ``qkv_proj.weight.dtype`` to pick the cast it
        applies to the hidden states. Keep the fp16 original visible for that:
        the activations really are fp16 here, the *weight storage* is what
        changed."""
        return self.fallback.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if not self.q.fits(flat.shape[0]):
            return self.fallback(x)
        out = self.q.mm(flat.to(torch.float16))
        # ``mm`` hands back a buffer it will overwrite on its next call. Every
        # other consumer here immediately folds it into an add, but a bias-free
        # Linear whose input is already fp16 would return the raw buffer to
        # arbitrary downstream code -- which is how F3b's XPU-graph probe
        # produced references that looked bit-exact and were aliased. Bias adds
        # already copy; the bias-free case is made to copy too.
        out = out + self.bias.to(torch.float16) if self.bias is not None else out.clone()
        return out.view(*x.shape[:-1], self.q.out_features).to(x.dtype)


# ---------------------------------------------------------------------------
# Installing / restoring a quantization mode
# ---------------------------------------------------------------------------
QUANT_MODES = ("fp16", "fold16", "int4-moe", "int4-all", "floor")


def expert_layers(model: nn.Module) -> list[nn.Module]:
    """The 36 ``ExpertDecoderLayer``s -- the only tower the denoise loop runs.

    Found by type rather than by path: the VLM tower must not be touched (it
    runs at 286 rows, outside the kernel's range, and it is not on the denoise
    loop's critical path at all -- its KV is already cached by then).
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import ExpertDecoderLayer

    return [m for m in model.modules() if isinstance(m, ExpertDecoderLayer)]


def _linear_children(layer: nn.Module, include_moe_experts: bool) -> list[tuple[nn.Module, str, nn.Linear]]:
    """Every ``nn.Linear`` in one expert layer that is safe to quantize.

    Two deliberate exclusions, both load-bearing:

    * ``TokenMoeBlock.gate`` -- the router. MODEL_ARCH.md §4 note 1: it must run
      in true fp32 with autocast off, because bf16 logits flip the top-4 choice
      on near-ties and the output jumps discontinuously. That is a change of
      *selection*, not a loss of precision, and no verifier tolerance covers it.
    * ``shared_expert.down_proj`` -- ``K = 704`` is not a multiple of 128, so
      Q4_0's group layout does not fit. 1.1 MB of a ~100 MB layer.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import TokenMoeBlock

    found: list[tuple[nn.Module, str, nn.Linear]] = []
    for module in layer.modules():
        router = module.gate if isinstance(module, TokenMoeBlock) else None
        for name, child in module.named_children():
            if not isinstance(child, nn.Linear) or child is router:
                continue
            if not include_moe_experts and isinstance(module, TokenMoeBlock):
                continue
            found.append((module, name, child))
    return found


def install(model: nn.Module, mode: str, *, experts_kept: int | None = None) -> dict:
    """Swap in ``mode`` and return what was swapped, skipped and why.

    Returns a handle that :func:`restore` undoes, so one loaded checkpoint
    serves every mode and the teacher stays available for scoring.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import TokenMoeBlock

    saved: list[tuple[nn.Module, str, Any]] = []
    report = {"mode": mode, "moe_swapped": 0, "linear_swapped": 0, "linear_skipped": 0, "bytes": 0.0}
    if mode == "fp16":
        return {"saved": saved, "report": report}

    for layer in expert_layers(model):
        for module in layer.modules():
            if not isinstance(module, TokenMoeBlock):
                continue
            experts = module.experts
            if experts_kept is not None:
                # The launch-floor arm: fewer experts is fewer *bytes* at an
                # unchanged kernel count, because the folded path issues three
                # GEMMs whatever E is. So this isolates the floor from the
                # stream. It is not a pruning proposal -- routing still selects
                # over the original E and the output is wrong on purpose.
                experts = _truncate_experts(experts, experts_kept)
            if mode == "fold16":
                replacement: nn.Module = FoldedGroupedExperts(experts)
            elif Q4GroupedExperts.supported(experts):
                replacement = Q4GroupedExperts(experts)
            else:
                report["linear_skipped"] += 1
                continue
            saved.append((module, "experts", module.experts))
            module.experts = replacement
            report["moe_swapped"] += 1
            report["bytes"] += getattr(replacement, "bytes", 0.0)

        if mode in ("int4-all", "floor"):
            for parent, name, child in _linear_children(layer, include_moe_experts=True):
                if not Q40.supported(child.weight.data):
                    report["linear_skipped"] += 1
                    continue
                replacement = Q4Linear(child)
                saved.append((parent, name, child))
                setattr(parent, name, replacement)
                report["linear_swapped"] += 1
                report["bytes"] += replacement.bytes

    return {"saved": saved, "report": report}


def _truncate_experts(experts: nn.Module, keep: int) -> Any:
    """A shallow view of ``experts`` holding only the first ``keep`` experts."""
    from types import SimpleNamespace

    return SimpleNamespace(
        gate_proj=experts.gate_proj[:keep],
        up_proj=experts.up_proj[:keep],
        down_proj=experts.down_proj[:keep],
    )


def restore(handle: dict) -> None:
    for parent, name, original in reversed(handle["saved"]):
        setattr(parent, name, original)


# ---------------------------------------------------------------------------
# Depth pruning
# ---------------------------------------------------------------------------
class depth_pruned:
    """Run the denoise pass over ``keep`` expert layers instead of all 36.

    No copy of the 60-line joint-tower loop, because during denoise
    ``inputs_embeds = [None, suffix_embs]`` and the VLM half is skipped by the
    loop's own ``if hidden_states is None: continue``. That leaves exactly two
    things indexed by ``layer_idx`` -- ``towers[1].layers[i]`` and
    ``past_key_values[i]`` -- so subsetting the expert ``ModuleList`` and the
    cache, and shortening ``num_layers``, is the whole change.

    ``prefix_forward`` must still run all 36 VLM layers, and does: grounding
    happens before this context is entered. What a dropped layer removes is the
    *use* of its cached K/V, which is also what removes it from the transfer.

    Note the coupling this exposes: expert layer ``i`` attends to VLM layer
    ``i``'s K/V, so the kept set decides **which levels of the VLM's hierarchy
    the action expert ever sees**. Keeping a contiguous prefix shows it only the
    shallow ones. That is the opposite of the usual LLM depth-pruning result
    (drop contiguous deep blocks) and is why ``--patterns`` exists.
    """

    def __init__(self, model: Any, keep: list[int]) -> None:
        self.joint = model.qwenvl_with_expert
        self.tower = self.joint.qwen_expert.model
        self.keep = list(keep)

    def __enter__(self) -> depth_pruned:
        self.saved_layers = self.tower.layers
        self.saved_count = self.joint.num_layers
        self.tower.layers = nn.ModuleList([self.saved_layers[i] for i in self.keep])
        self.joint.num_layers = len(self.keep)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.tower.layers = self.saved_layers
        self.joint.num_layers = self.saved_count

    def cache(self, past_key_values: list) -> list:
        """The kept layers' slice of a full-depth prefix KV cache."""
        return [past_key_values[i] for i in self.keep]


def keep_pattern(name: str, total: int, count: int) -> list[int]:
    """Three ways to choose which ``count`` of ``total`` layers survive.

    ``stride`` samples the whole hierarchy, ``head`` keeps the shallowest,
    ``tail`` keeps the deepest. The LLM literature's answer is that deep
    contiguous blocks are the redundant ones (so ``head`` wins); this model's
    per-layer VLM-KV coupling argues for ``stride``. Measured, not assumed.
    """
    if name == "stride":
        # Always keep layer 0 and the last layer: the first sees the raw
        # embedding and the last feeds `action_out_proj`.
        if count == 1:
            return [0]
        step = (total - 1) / (count - 1)
        return sorted({min(total - 1, round(i * step)) for i in range(count)})
    if name == "head":
        return list(range(count))
    if name == "tail":
        return list(range(total - count, total))
    raise ValueError(f"unknown keep pattern {name!r}")


# ---------------------------------------------------------------------------
# Shared model setup
# ---------------------------------------------------------------------------
def ground(model: Any, processor: Any, obs: dict, device: torch.device, dtype: torch.dtype):
    """One real full round, so the session holds the prefix KV the product builds."""
    import dataclasses

    from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import SpecDecoder

    class _ZeroDraft:
        def __init__(self, chunk: torch.Tensor) -> None:
            self.chunk = chunk

        def refresh(self, prefix_embs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            return self.chunk

        def draft(self, state: torch.Tensor) -> torch.Tensor:
            return self.chunk

        def close(self) -> None:
            pass

    config = dataclasses.replace(model.config, spec_decode=True)
    chunk = torch.zeros((1, config.chunk_size, config.max_action_dim), device=device, dtype=dtype)
    decoder = SpecDecoder(
        transformer=model, processor=processor, config=config, device=device, dtype=dtype, draft=_ZeroDraft(chunk)
    )
    decoder.decode(obs, session_id="p13", reset=True, noise=None, num_steps=config.num_steps)
    state = processor.preprocess_state(obs).to(device=device, dtype=dtype)
    return decoder, decoder.sessions["p13"], state


def euler(model: Any, session: Any, state: torch.Tensor, noise: torch.Tensor, num_steps: int,
          past_key_values: list | None = None) -> torch.Tensor:
    """``denoise_actions``' arithmetic, opened up so ``num_steps`` is a variable.

    Reproduced rather than called because the draft's step count is one of the
    two axes here, and it has to stay in the model dtype with ``time``
    accumulated by repeated addition, exactly as the shipped loop does.
    """
    dtype, device, bsize = state.dtype, state.device, int(state.shape[0])
    dt = torch.tensor(-1.0 / num_steps, dtype=dtype, device=device)
    now = torch.tensor(1.0, dtype=dtype, device=device)
    cache = session.past_key_values if past_key_values is None else past_key_values
    x_t = noise
    for _ in range(num_steps):
        v_t = model.predict_velocity(
            state=state,
            prefix_pad_masks=session.pad_masks,
            prefix_position_ids=session.position_ids,
            past_key_values=cache,
            x_t=x_t,
            timestep=now.expand(bsize),
        )
        x_t = x_t + dt * v_t
        now = now + dt
    return x_t


def observations(args: argparse.Namespace, processor: Any) -> list[dict]:
    """``--observations`` observations, from the open-loop bundle if given.

    ``phase5_latency.observation`` fills the cameras with uniform noise, which
    is fine for timing and misleading for accuracy: a draft's error depends on
    how structured the trajectory is, and noise frames do not produce one. Any
    accept-rate number has to come from real frames.
    """
    if args.dataset is None:
        return [observation(processor.spec, args.seed + i) for i in range(args.observations)]

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.offline_inference.lingbot_vla_v2.open_loop_eval import CAMERA_KEYS, load_bundle

    bundle = load_bundle(args.dataset)
    count = min(args.observations, bundle.num_samples)
    if count < args.observations:
        print(f"[data] bundle holds {bundle.num_samples} samples; using all of them")
    return [
        {
            "images": {key: bundle.images[i, camera] for camera, key in enumerate(CAMERA_KEYS)},
            "state": bundle.states[i].astype("float32", copy=False),
            "prompt": str(bundle.prompts[i]),
        }
        for i in range(count)
    ]


def euler_trajectory(
    model: Any, session: Any, state: torch.Tensor, noise: torch.Tensor, num_steps: int,
    past_key_values: list | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """``euler`` but keeping every ``x_t`` and the ``t`` it was evaluated at.

    Returns ``(x_0..x_S, t_0..t_{S-1})``. Same arithmetic as :func:`euler`;
    only the bookkeeping differs, so the endpoint is bit-identical.
    """
    dtype, device, bsize = state.dtype, state.device, int(state.shape[0])
    dt = torch.tensor(-1.0 / num_steps, dtype=dtype, device=device)
    now = torch.tensor(1.0, dtype=dtype, device=device)
    cache = session.past_key_values if past_key_values is None else past_key_values
    x_t = noise
    traj, times = [x_t], []
    for _ in range(num_steps):
        times.append(now.clone())
        v_t = model.predict_velocity(
            state=state, prefix_pad_masks=session.pad_masks,
            prefix_position_ids=session.position_ids, past_key_values=cache,
            x_t=x_t, timestep=now.expand(bsize),
        )
        x_t = x_t + dt * v_t
        traj.append(x_t)
        now = now + dt
    return traj, times


def trajectory_accept(
    model: Any, session: Any, state: torch.Tensor, traj: list[torch.Tensor],
    times: list[torch.Tensor], tau: float, dims: torch.Tensor, horizon: int,
) -> tuple[int, list[float]]:
    """How many of the draft's **denoise** steps survive a full-model check.

    The literal reading of "verify the ten steps": for each ``j``, take the
    draft's ``x_{j-1}``, ask the *unpruned* model for one Euler step from it,
    and compare against the draft's own ``x_j``. Longest accepted prefix.

    All ``S`` checks are one batched ``B=S`` forward (§M §3 priced it at 96.9 ms
    compiled for S=10), so this is ParaDiGMS-shaped verification, not the
    shipped rule. The shipped rule checks the **endpoint** at K near-terminal
    ``t`` for 22.8 ms and ignores the path, because in flow matching the
    intermediate ``x_t`` are scaffolding rather than outputs (§M §7). This
    function exists to answer the question, not to propose the scheme.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import _expand_rows

    steps = len(times)
    cond = {
        "state": _expand_rows(state, steps),
        "prefix_pad_masks": _expand_rows(session.pad_masks, steps),
        "prefix_position_ids": _expand_rows(session.position_ids, steps, dim=1),
        "past_key_values": [
            (_expand_rows(k, steps), _expand_rows(v, steps)) for k, v in session.past_key_values
        ],
    }
    x_prev = torch.cat(traj[:-1], dim=0)
    v_full = model.predict_velocity(
        **cond, x_t=x_prev, timestep=torch.cat([t.reshape(1) for t in times])
    )
    dt = -1.0 / steps
    x_hat = x_prev + dt * v_full                      # teacher's x_j from the draft's x_{j-1}
    x_draft = torch.cat(traj[1:], dim=0)

    delta = (x_hat - x_draft)[:, :horizon].index_select(-1, dims).to(torch.float32)
    per_step = delta.pow(2).mean(dim=(1, 2)).sqrt()   # per-dim RMS for each denoise step
    ok = (per_step <= tau).to(torch.int64)
    accepted = int(ok.cumprod(dim=0).sum().item())
    return accepted, [float(v) for v in per_step]


def real_dims(processor: Any, device: torch.device) -> torch.Tensor:
    from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import gripper_dims, pose_dims

    return torch.as_tensor(sorted(pose_dims(processor) + gripper_dims(processor)), device=device, dtype=torch.long)


def rms(a: torch.Tensor, b: torch.Tensor, dims: torch.Tensor, horizon: int) -> float:
    delta = (a - b)[:, :horizon].index_select(-1, dims).to(torch.float32)
    return float(delta.pow(2).mean().sqrt().item())


# ---------------------------------------------------------------------------
# Arm: kernel
# ---------------------------------------------------------------------------
def arm_kernel(args: argparse.Namespace, device: torch.device, **_: Any) -> dict:
    """Price every weight-only-quant GEMM this stack offers, at the MoE shape.

    No checkpoint: this is a shape question. Run it on both cards -- the answer
    differs, and that difference is what decides where a compressed draft lives.
    """
    tokens, num_experts, hidden, intermediate = 51, 32, 768, 512
    wide = num_experts * intermediate
    x = torch.randn(tokens, hidden, dtype=torch.float16, device=device)

    gate = torch.randn(wide, hidden, dtype=torch.float16, device=device) * 0.02
    up = torch.randn(wide, hidden, dtype=torch.float16, device=device) * 0.02
    down = torch.randn(hidden, wide, dtype=torch.float16, device=device) * 0.02
    fp16_bytes = (gate.numel() + up.numel() + down.numel()) * 2 / 1e6

    def run_fp16() -> torch.Tensor:
        return (F.silu(x @ gate.t()) * (x @ up.t())) @ down.t()

    rows = [("fp16 mm (baseline)", run_fp16, fp16_bytes)]

    if esimd() is not None:
        qg, qu, qd = Q40(gate), Q40(up), Q40(down)

        def run_esimd() -> torch.Tensor:
            return qd.mm((F.silu(qg.mm(x)) * qu.mm(x)).contiguous())

        rows.append(("esimd int4 (Q4_0)", run_esimd, (qg.bytes + qu.bytes + qd.bytes) / 1e6))
    else:
        print("[kernel] custom_esimd_kernels_vllm not installed -- skipping the only usable int4 path")

    if args.include_stock:
        # Both are in every torch-xpu build and both are traps: int8 dispatches
        # to a scalar reference and int4 is slower than fp16 on both cards. They
        # are measured so the write-up can say so with a number.
        g8 = torch.randint(-127, 127, (wide, hidden), dtype=torch.int8, device=device)
        d8 = torch.randint(-127, 127, (hidden, wide), dtype=torch.int8, device=device)
        s_w = torch.rand(wide, dtype=torch.float16, device=device) + 0.01
        s_h = torch.rand(hidden, dtype=torch.float16, device=device) + 0.01

        def run_int8() -> torch.Tensor:
            a = torch.ops.aten._weight_int8pack_mm(x, g8, s_w)
            b = torch.ops.aten._weight_int8pack_mm(x, g8, s_w)
            return torch.ops.aten._weight_int8pack_mm((F.silu(a) * b).contiguous(), d8, s_h)

        rows.append(("aten int8pack_mm", run_int8, (g8.numel() * 2 + d8.numel()) / 1e6))

    results = []
    reference = None
    for name, fn, megabytes in rows:
        out = fn()
        if reference is None:
            reference = out.float()
            rel = 0.0
        else:
            rel = float((out.float() - reference).abs().mean() / reference.abs().mean().clamp_min(1e-9))
        batch = auto_batch(fn, device, args.kernel_batch)
        millis = median_ms(fn, args.iters, args.warmup, device, batch=batch)
        results.append(
            {"kernel": name, "ms": millis, "mb": megabytes, "gbs": megabytes / millis, "rel_err": rel,
             "batch": batch, "ten_step_36l_ms": millis * 360}
        )
        print(f"[kernel] {name:22s} {millis:8.3f} ms/layer {megabytes:7.1f} MB "
              f"{megabytes / millis:7.1f} GB/s  rel={rel:.3e}  x{batch:<3d} "
              f"10-step/36L: {millis * 360:9.1f} ms")

    return {"arm": "kernel", "affinity": os.environ.get("ZE_AFFINITY_MASK", "unset"), "rows": results}


# ---------------------------------------------------------------------------
# Arm: accept
# ---------------------------------------------------------------------------
def arm_accept(args: argparse.Namespace, processor: Any, model: Any, device: torch.device,
               dtype: torch.dtype, **_: Any) -> dict:
    """Accept rate of a PTQ draft against the shipped verifier. dGPU only.

    The teacher is the fp16 model at ``config.num_steps``; the draft is the same
    weights quantized, at whichever step count is being swept. Scoring goes
    through the production ``_verify`` and ``radius_prefix_acceptance``, not a
    reimplementation, so the number is the one the decoder would produce.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import (
        gripper_dims,
        pose_dims,
        radius_prefix_acceptance,
        truncate_on_gripper_switch,
    )

    # Exactly what ``SpecDecoder.__init__`` binds: the accept radius is over the
    # pose dims only, the guard over the gripper dims only. Passing the union to
    # either one would be a different rule than the product's.
    pose, grippers = pose_dims(processor), gripper_dims(processor)
    dims = real_dims(processor, device)  # pose + gripper, `_draft_rms`'s units
    horizon = model.config.spec_max_exec_steps
    modes = [m for m in args.quant.split(",") if m.strip()]
    steps = [int(s) for s in args.num_steps.split(",") if s.strip()]

    # Three passes rather than one nested loop, because installing a mode
    # re-quantizes 2.7 GB and the naive ordering pays that once per cell (72
    # times) instead of once per mode (3).
    #
    # Pass 1: ground and score the teacher, unquantized. The dGPU does the
    # grounding in this scheme either way, so it stays fp16 throughout.
    sessions = []
    for obs in observations(args, processor):
        decoder, session, state = ground(model, processor, obs, device, dtype)
        noise = torch.randn(
            (1, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype
        )
        sessions.append({
            "decoder": decoder, "session": session, "state": state, "noise": noise,
            "teacher": euler(model, session, state, noise, model.config.num_steps),
        })
    print(f"[accept] {len(sessions)} groundings + teachers done")

    cells: dict[tuple[str, int], list[dict]] = {(m, s): [] for m in modes for s in steps}
    for mode in modes:
        # Pass 2: every draft this mode produces, with the mode installed once.
        handle = install(model, mode)
        print(f"[accept] install {mode}: {handle['report']}")
        try:
            drafts = {
                (seed, num_steps): euler(model, item["session"], item["state"], item["noise"], num_steps)
                for seed, item in enumerate(sessions)
                for num_steps in steps
            }
        finally:
            # Pass 3: verify on the *unquantized* model. The verifier is the
            # dGPU's teacher; quantizing it too would check the draft against
            # itself and accept everything by construction.
            restore(handle)

        for (seed, num_steps), draft in drafts.items():
            item = sessions[seed]
            x0_hat = item["decoder"]._verify(item["session"], item["state"], item["noise"], draft)
            accepted, dist = radius_prefix_acceptance(
                draft, x0_hat, tau=model.config.spec_tau, dims=pose, eval_h=horizon
            )
            accepted, cut = truncate_on_gripper_switch(draft, accepted, gripper_prev=None, dims=grippers)
            cells[(mode, num_steps)].append({
                "draft_rms": rms(draft, item["teacher"], dims, horizon),
                "accepted": int(accepted.item()),
                "dist": float(dist.mean().item()),
                "gripper_cut": bool(cut.any().item()),
            })

    for item in sessions:
        item["decoder"].close()

    rows = []
    for (mode, num_steps), samples in cells.items():
        accept_rate = sum(1 for s in samples if s["accepted"] > 0) / len(samples)
        mean_rms = statistics.mean(s["draft_rms"] for s in samples)
        tick = REGROUND_TICK_MS + (1.0 - accept_rate) * REJECT_COST_MS
        rows.append({
            "mode": mode, "num_steps": num_steps, "accept_rate": accept_rate,
            "draft_rms": mean_rms, "mean_accepted": statistics.mean(s["accepted"] for s in samples),
            "mean_dist": statistics.mean(s["dist"] for s in samples), "tick_ms": tick,
        })
        print(f"[accept] {mode:9s} steps={num_steps:>2}  rms={mean_rms:.4f} "
              f"({'PASS' if mean_rms <= DRAFT_RMS_TARGET else 'over'} {DRAFT_RMS_TARGET}) "
              f"accept={accept_rate:5.1%}  tick={tick:6.1f} ms")

    return {"arm": "accept", "rows": rows, "observations": len(sessions),
            "dataset": str(args.dataset) if args.dataset else "synthetic-noise-frames",
            "spec_tau": model.config.spec_tau, "teacher_steps": model.config.num_steps}


# ---------------------------------------------------------------------------
# Arm: step
# ---------------------------------------------------------------------------
def arm_step(args: argparse.Namespace, processor: Any, model: Any, device: torch.device,
             dtype: torch.dtype, **_: Any) -> dict:
    """iGPU speed curve: the full Euler loop per (quant mode, step count).

    ``--launch-floor`` adds a cell that keeps every kernel and drops almost all
    the bytes (one expert instead of 32, on top of ``int4-all``). Its *output*
    is wrong by construction; its *time* is the floor below which no amount of
    further compression can go. If that floor is already over 58 ms the scheme
    is closed regardless of how good the accept curve looks.
    """
    obs = observation(processor.spec, args.seed)
    decoder, session, state = ground(model, processor, obs, device, dtype)
    noise = torch.randn((1, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype)

    # `--keep-layers N` prices the depth the `depth` arm says is affordable,
    # instead of trusting that per-layer cost is constant. Which layers survive
    # does not change the time, only the accuracy, so the pattern is irrelevant
    # here and `stride` is used for consistency with that arm.
    total = model.qwenvl_with_expert.num_layers
    pruner = None
    if args.keep_layers is not None and args.keep_layers < total:
        pruner = depth_pruned(model, keep_pattern("stride", total, args.keep_layers))
        pruner.__enter__()
        session.past_key_values = pruner.cache(session.past_key_values)
        print(f"[step] depth {total} -> {args.keep_layers} layers")

    modes = [(m, None) for m in args.quant.split(",") if m.strip()]
    if args.launch_floor:
        modes.append(("floor", 1))
    steps = [int(s) for s in args.num_steps.split(",") if s.strip()]

    rows = []
    for mode, kept in modes:
        handle = install(model, mode, experts_kept=kept)
        print(f"[step] install {mode}"
              f"{'' if kept is None else f' (experts={kept})'}: {handle['report']}")
        try:
            timestep = torch.full((1,), 0.5, device=device, dtype=dtype)

            def one_step() -> None:
                model.predict_velocity(
                    state=state, prefix_pad_masks=session.pad_masks,
                    prefix_position_ids=session.position_ids,
                    past_key_values=session.past_key_values, x_t=noise,
                    timestep=timestep,
                )

            one = median_ms(one_step, args.iters, args.warmup, device)
            # Host issue time: the same call with no sync, so the queue absorbs
            # the device work and what is left is Python + dispatch. The gap
            # between this and `one` is what a graph capture could not remove,
            # and the split decides whether the floor is fixable at all
            # (F3b got 4.6x off host issue on a dispatch-bound chain).
            issue = issue_ms(one_step, args.iters, args.warmup, device)
            for num_steps in steps:
                total = median_ms(
                    lambda n=num_steps: euler(model, session, state, noise, n),
                    max(3, args.iters // 2), args.warmup, device,
                )
                verdict = "FITS" if total <= HIDDEN_BUDGET_MS else f"{total / HIDDEN_BUDGET_MS:.1f}x over"
                rows.append({
                    "mode": mode, "experts_kept": kept, "num_steps": num_steps,
                    "step_ms": one, "issue_ms": issue, "total_ms": total,
                    "budget_ratio": total / HIDDEN_BUDGET_MS,
                    "weight_mb": handle["report"]["bytes"] / 1e6,
                })
                print(f"[step] {mode:9s} steps={num_steps:>2}  1 step {one:8.2f} ms "
                      f"(issue {issue:6.2f}) {num_steps} steps {total:9.2f} ms  "
                      f"vs {HIDDEN_BUDGET_MS:.0f} ms budget: {verdict}")
        finally:
            restore(handle)

    if pruner is not None:
        pruner.__exit__()
    decoder.close()
    return {"arm": "step", "affinity": os.environ.get("ZE_AFFINITY_MASK", "unset"),
            "budget_ms": HIDDEN_BUDGET_MS, "layers": args.keep_layers or total, "rows": rows}


# ---------------------------------------------------------------------------
# Arm: depth
# ---------------------------------------------------------------------------
def arm_depth(args: argparse.Namespace, processor: Any, model: Any, device: torch.device,
              dtype: torch.dtype, **_: Any) -> dict:
    """Training-free depth pruning: which layers can go, and how many.

    Gate 2 left ``L <= 8`` of 36 as the only way a PTQ draft reaches the 58 ms
    window, because the iGPU's per-step floor is per-*layer* overhead and that
    is the one thing quantization cannot touch. This measures how far depth can
    actually be cut with no weight update at all.

    The selection criterion is the unusual part, and it is better than the
    cosine-similarity proxy the LLM depth-pruning literature uses: this model
    ships a **verifier with a known tolerance**, so a layer's importance can be
    scored in exactly the units that decide acceptance. Phase 13 §2 measured a
    20-50x margin against ``spec_tau`` at int4-all, and that margin is the
    pruning budget -- it is spendable, and this arm spends it.

    Three passes:

    1. ``sensitivity`` -- drop each layer alone, score the whole denoise.
    2. ``greedy`` -- drop in ascending sensitivity until the production accept
       rule refuses, which is the training-free depth.
    3. ``patterns`` -- at each depth, compare greedy against ``stride`` / ``head``
       / ``tail``, because this model's per-layer VLM-KV coupling predicts a
       different winner than the LLM result does.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import (
        gripper_dims,
        pose_dims,
        radius_prefix_acceptance,
        truncate_on_gripper_switch,
    )

    pose, grippers = pose_dims(processor), gripper_dims(processor)
    dims = real_dims(processor, device)
    horizon = model.config.spec_max_exec_steps
    total = model.qwenvl_with_expert.num_layers
    steps = int(args.num_steps.split(",")[0])

    sessions = []
    for obs in observations(args, processor):
        decoder, session, state = ground(model, processor, obs, device, dtype)
        noise = torch.randn(
            (1, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype
        )
        sessions.append({
            "decoder": decoder, "session": session, "state": state, "noise": noise,
            "teacher": euler(model, session, state, noise, model.config.num_steps),
        })
    print(f"[depth] {len(sessions)} groundings, teacher = full {total} layers x "
          f"{model.config.num_steps} steps, draft = {steps} steps, quant={args.quant}")

    handle = install(model, args.quant.split(",")[0])
    print(f"[depth] install {handle['report']}")

    def score(keep: list[int]) -> dict:
        """Mean per-dim RMS and production accept over every grounding.

        ``profile`` is the per-horizon-step distance, worst over the K verify
        timesteps and averaged over groundings. It is the useful column when a
        cell fails: ``prefix`` says *how many* action steps survived, ``profile``
        says **where** the draft parts company with the teacher. Note this is a
        prefix of the 50-step action chunk (12 of them evaluated), not of the
        denoise trajectory -- the verifier checks the endpoint, not the path.
        """
        rms_values, accepted_any, prefix_len, dists = [], 0, [], []
        profiles = []
        with depth_pruned(model, keep) as pruned:
            drafts = [
                euler(model, s["session"], s["state"], s["noise"], steps,
                      past_key_values=pruned.cache(s["session"].past_key_values))
                for s in sessions
            ]
        for item, draft in zip(sessions, drafts):
            rms_values.append(rms(draft, item["teacher"], dims, horizon))
            x0_hat = item["decoder"]._verify(item["session"], item["state"], item["noise"], draft)
            accepted, dist = radius_prefix_acceptance(
                draft, x0_hat, tau=model.config.spec_tau, dims=pose, eval_h=horizon
            )
            accepted, _ = truncate_on_gripper_switch(draft, accepted, gripper_prev=None, dims=grippers)
            accepted_any += int(int(accepted.item()) > 0)
            prefix_len.append(int(accepted.item()))
            dists.append(float(dist.mean().item()))
            # The accept rule is a conjunction over K, so the worst verify
            # timestep is the one that decides each horizon step.
            profiles.append(dist[0].amax(dim=0).tolist())
        return {
            "rms": statistics.mean(rms_values), "accept_rate": accepted_any / len(sessions),
            "prefix": statistics.mean(prefix_len), "dist": statistics.mean(dists),
            "prefix_all": prefix_len,
            "profile": [statistics.mean(col) for col in zip(*profiles)],
        }

    try:
        full = score(list(range(total)))
        print(f"[depth] full {total} layers @ {steps} steps: rms={full['rms']:.4f} "
              f"accept={full['accept_rate']:.0%} dist={full['dist']:.4f}")

        # -- 1. sensitivity ------------------------------------------------
        sensitivity = []
        for victim in range(total):
            keep = [i for i in range(total) if i != victim]
            row = score(keep)
            sensitivity.append({"layer": victim, **row})
            print(f"[depth] drop L{victim:<2d} alone: rms={row['rms']:.4f} "
                  f"dist={row['dist']:.4f} accept={row['accept_rate']:.0%}")
        order = [r["layer"] for r in sorted(sensitivity, key=lambda r: r["rms"])]
        print(f"[depth] least -> most sensitive: {order}")

        # -- 2. greedy + 3. patterns ---------------------------------------
        depths = [int(d) for d in args.depths.split(",") if d.strip()]
        rows = []
        for count in depths:
            for pattern in ["greedy", *args.patterns.split(",")]:
                # `order` is ascending sensitivity, so the survivors are its
                # *tail*: drop the least sensitive first, keep the most.
                keep = sorted(order[total - count:]) if pattern == "greedy" else keep_pattern(pattern, total, count)
                row = score(keep)
                # Both iGPU terms are per-layer: 3.29 ms/layer/step (int4-all,
                # Gate 2) and 0.44 ms/layer of prefix-KV transfer (Gate 3).
                projected = count * (3.29 * steps + 0.44)
                rows.append({"pattern": pattern, "layers": count, "keep": keep,
                             "igpu_ms": projected, **row})
                print(f"[depth] {pattern:6s} L={count:<2d} rms={row['rms']:.4f} "
                       f"accept={row['accept_rate']:5.1%} prefix={row['prefix']:4.1f} "
                       f"{row['prefix_all']} dist={row['dist']:.4f}  iGPU~{projected:6.1f} ms "
                       f"({'FITS' if projected <= HIDDEN_BUDGET_MS else 'over'} {HIDDEN_BUDGET_MS:.0f})")
                print(f"[depth]   profile (tau={model.config.spec_tau}): "
                      + " ".join(f"{d:.3f}" for d in row["profile"]))
                if args.trajectory_verify:
                    traj_rows = []
                    for item in sessions:
                        with depth_pruned(model, keep) as pruned:
                            traj, times = euler_trajectory(
                                model, item["session"], item["state"], item["noise"], steps,
                                past_key_values=pruned.cache(item["session"].past_key_values),
                            )
                        traj_rows.append(trajectory_accept(
                            model, item["session"], item["state"], traj, times,
                            model.config.spec_tau, dims, horizon,
                        ))
                    lens = [n for n, _ in traj_rows]
                    per_step = [statistics.mean(col) for col in zip(*[d for _, d in traj_rows])]
                    rows[-1]["traj_accepted"] = statistics.mean(lens)
                    rows[-1]["traj_per_step"] = per_step
                    print(f"[depth]   denoise-step accept {statistics.mean(lens):.1f}/{steps} "
                          f"{lens}  per-step rms: " + " ".join(f"{d:.3f}" for d in per_step))
    finally:
        restore(handle)
        for item in sessions:
            item["decoder"].close()

    return {"arm": "depth", "total_layers": total, "draft_steps": steps, "quant": args.quant,
            "dataset": str(args.dataset) if args.dataset else "synthetic-noise-frames",
            "full": full, "sensitivity": sensitivity, "order": order, "rows": rows,
            "spec_tau": model.config.spec_tau}


# ---------------------------------------------------------------------------
# Folding check
# ---------------------------------------------------------------------------
def check_folding(model: Any, processor: Any, device: torch.device, dtype: torch.dtype) -> dict:
    """``fold16`` against the shipped ``forward_dense``, on a real denoise.

    Validates the algebra before any quantization number is quoted. Tolerance is
    the fp16 reconstruction floor Phase 10 measured (3.9e-3), not zero: the fold
    reduces over ``E*I`` in one GEMM where the einsum reduces over ``I`` then
    ``E``, so the two round differently by construction.
    """
    obs = observation(processor.spec, 0)
    decoder, session, state = ground(model, processor, obs, device, dtype)
    noise = torch.randn((1, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype)
    dims = real_dims(processor, device)
    horizon = model.config.spec_max_exec_steps

    reference = euler(model, session, state, noise, model.config.num_steps)
    handle = install(model, "fold16")
    try:
        folded = euler(model, session, state, noise, model.config.num_steps)
    finally:
        restore(handle)
    decoder.close()

    error = rms(folded, reference, dims, horizon)
    ok = error <= 3.9e-3
    print(f"[folding] fold16 vs forward_dense: per-dim RMS {error:.3e} -- {'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit("folding check failed; every int4 number downstream would be uninterpretable")
    return {"folding_rms": error}


ARMS = {"kernel": arm_kernel, "accept": arm_accept, "step": arm_step, "depth": arm_depth}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--model", type=Path, default=None, help="required by accept/step")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--quant", default="fp16,int4-moe,int4-all",
                        help=f"comma-separated from {QUANT_MODES}")
    parser.add_argument("--num-steps", default="10,4,2", help="draft step counts to sweep")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="accept: open-loop bundle to draw real frames from; noise frames otherwise")
    parser.add_argument("--observations", type=int, default=8, help="accept: distinct groundings to score")
    parser.add_argument("--launch-floor", action="store_true", help="step: add the kernel-count floor cell")
    parser.add_argument("--keep-layers", type=int, default=None,
                        help="step: price the tower at this depth instead of all 36")
    parser.add_argument("--depths", default="30,24,18,12,8,4", help="depth: layer counts to evaluate")
    parser.add_argument("--trajectory-verify", action="store_true",
                        help="depth: also check each denoise step against the full model "
                             "(one batched B=S forward); answers 'how many of the S steps pass'")
    parser.add_argument("--patterns", default="stride,head,tail",
                        help="depth: fixed keep patterns to compare against the greedy order")
    parser.add_argument("--check-folding", action="store_true", help="validate the fold before quantizing")
    parser.add_argument("--include-stock", action="store_true",
                        help="kernel: also price aten's int8 path (it is a trap; measured for the record)")
    parser.add_argument("--kernel-batch", type=int, default=40,
                        help="kernel: cap on calls per timed region; the region is sized by time "
                             "(~120 ms) up to this, so slow kernels do not blow the run up")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    print(f"[setup] arm={args.arm} affinity={os.environ.get('ZE_AFFINITY_MASK', 'unset')} "
          f"device={device} dtype={args.dtype} esimd={'yes' if esimd() else 'NO'} "
          f"load1m={load_average():.2f}")
    if load_average() > 2.0:
        print("WARNING: load average above 2.0. Rules #2 -- absolute numbers from this run are suspect.")

    torch.manual_seed(args.seed)
    result: dict[str, Any]
    if args.arm == "kernel":
        with torch.no_grad():
            result = arm_kernel(args, device=device)
    else:
        if args.model is None:
            parser.error(f"--arm {args.arm} needs --model")
        processor, model = build(args.model, device, dtype, num_steps=None)
        with torch.no_grad():
            folding = check_folding(model, processor, device, dtype) if args.check_folding else {}
            result = ARMS[args.arm](args, processor=processor, model=model, device=device, dtype=dtype)
            result.update(folding)

    result["load1m"] = load_average()
    result["dtype"] = args.dtype
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"\n[json] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
