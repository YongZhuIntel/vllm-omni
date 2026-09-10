#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 8 step I — what in the denoise loop does not depend on the loop variable?

F3 closed graph capture: no XPU graph API exists, so the ~40 ms host dispatch tax
looked unreachable from Python. That conclusion was about *one mechanism*. This
probe asks the cheaper question — instead of submitting the same work faster, do
not submit it at all — and finds ~9 ms.

`predict_velocity` is called ten times per request and rebuilds, every time:

* the attention mask (`make_att_2d_masks` + `cat` + `_block_query_columns`) and
  the mrope position ids, both functions of the pad masks only;
* `state_proj(state)`, a function of the observation only;
* `AdaRMSNorm.gamma(cond)` / `.beta(cond)` — 72 modules x 2 Linears =
  **144 [1,768]x[768,768] matmuls per step, 1440 per request** — whose only
  input is the timestep, and all ten timesteps are known before the loop starts.

`torch.compile` cannot hoist any of it: each step is a separate call into the
same graph, so Inductor CSEs within a step and re-runs everything across steps.

Two measurements, because the cheap one is misleading:

1. `--mode micro` times the 144 matmuls in isolation, per-step versus batched
   over 10 timesteps. It reports ~21.8 ms of savings and is **wrong by 3x**: in
   the real graph Inductor fuses most of those Linears into surrounding kernels,
   so their marginal launch cost is already partly paid.
2. `--mode real` (default) patches `AdaRMSNorm.forward` to read a cached γ/β
   instead of computing it, and times the real compiled loop. This is a *timing
   proxy* — reusing one timestep's FiLM makes the output numerically wrong on
   purpose (`max|Δ| ~ 2.8`) — but the runtime of a lookup versus two matmuls is
   exactly what a correct precompute-then-index would pay. It reports 7.3-7.5 ms.

It also times the strictly step-invariant setup (masks + position ids), 9/10 of
which is redundant, for a further ~1.6 ms.

    PYTHONPATH=.:spikes/lingbot_vla_v2 \\
        python spikes/lingbot_vla_v2/phase8_loop_invariant_probe.py --model /tmp/lingbot-vla-v2-perf

Findings are recorded in `PHASE8_LATENCY_PARITY.md` under F4.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from phase5_latency import DEFAULT_MODEL, build, observation

HIDDEN = 768
LAYERS = 36
NORMS_PER_LAYER = 2  # input_layernorm + post_attention_layernorm
STEPS = 10


def _bench(fn, iters: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run_micro(args: argparse.Namespace) -> None:
    """The isolated version. Kept because its 3x error is the lesson."""
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    n = LAYERS * NORMS_PER_LAYER * 2  # gamma and beta
    weights = [torch.empty(HIDDEN, HIDDEN, device=device, dtype=dtype).normal_(0, 0.02) for _ in range(n)]
    biases = [torch.empty(HIDDEN, device=device, dtype=dtype).normal_(0, 0.02) for _ in range(n)]
    one = torch.empty(1, HIDDEN, device=device, dtype=dtype).normal_()
    ten = torch.empty(STEPS, HIDDEN, device=device, dtype=dtype).normal_()

    def apply(cond):
        return [torch.nn.functional.linear(cond, w, b) for w, b in zip(weights, biases)]

    print(f"\nisolated microbenchmark: {n} Linears of [.,{HIDDEN}]x[{HIDDEN},{HIDDEN}]")
    per_step = _bench(lambda: [apply(one) for _ in range(STEPS)], 50)
    batched = _bench(lambda: apply(ten), 50)
    print(f"  today:   {n} ops x {STEPS} steps          {per_step:6.2f} ms/request")
    print(f"  hoisted: {n} ops x [{STEPS},{HIDDEN}], once   {batched:6.2f} ms/request")
    print(f"  => {per_step - batched:.1f} ms -- and this OVERSTATES the real gain by ~3x")


@torch.inference_mode()
def run_real(args: argparse.Namespace) -> None:
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    from vllm_omni.diffusion.models.lingbot_vla_v2 import modeling_lingbot_vla_v2 as mod

    processor, model = build(Path(args.model), device, dtype, None, None)
    joint = model.qwenvl_with_expert
    joint.attention_backend, joint.attention_precision = "eager", args.attention_precision
    inputs = processor.preprocess(observation(processor.spec, 0)).to(device=device, dtype=dtype).model_inputs()

    def compile_step():
        model.predict_velocity = torch.compile(
            mod.LingbotVlaV2ForActionPrediction.predict_velocity.__get__(model),
            backend="inductor",
            dynamic=False,
            fullgraph=True,
        )

    compile_step()

    embs, pad_masks, att_masks, position_ids, visual_masks, deepstack = model.embed_prefix(
        inputs["images"], inputs["img_masks"], inputs["lang_tokens"], inputs["lang_masks"], inputs["image_grid_thw"]
    )
    _, past_key_values = joint.forward(
        attention_mask=mod.make_att_2d_masks(pad_masks, att_masks),
        position_ids=position_ids,
        inputs_embeds=[embs, None],
        past_key_values=None,
        fill_kv_cache=True,
        visual_pos_masks=visual_masks,
        deepstack_visual_embeds=deepstack,
    )
    bsize = inputs["state"].shape[0]
    noise = torch.randn(
        (bsize, model.config.chunk_size, model.config.max_action_dim), device=device, dtype=dtype
    )
    kwargs = dict(
        state=inputs["state"],
        prefix_pad_masks=pad_masks,
        prefix_position_ids=position_ids,
        past_key_values=past_key_values,
        noise=noise,
    )

    # --- part 1: the step-invariant setup, measured on its own ---------------
    suffix_len = model.config.chunk_size + 1
    ones = torch.ones((bsize, suffix_len), device=device, dtype=torch.bool)
    opens = torch.zeros((bsize, suffix_len), device=device, dtype=torch.bool)
    opens[:, :2] = True

    def setup_invariant():
        prefix_len = pad_masks.shape[1]
        prefix_2d = pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        full = torch.cat([prefix_2d, mod.make_att_2d_masks(ones, opens)], dim=2)
        full = model._block_query_columns(full, prefix_len)
        return full, model._build_full_position_ids(position_ids, pad_masks, ones)[:, :, -suffix_len:]

    per_step = _bench(setup_invariant, 200, warmup=20)
    print(
        f"\nstep-invariant setup (masks + position ids): {per_step:.3f} ms/step"
        f" -> {per_step * STEPS:.2f} ms/request, {per_step * (STEPS - 1):.2f} ms of it redundant"
    )

    # --- part 2: the FiLM projections, in the real compiled loop -------------
    n_ada = sum(1 for m in model.modules() if isinstance(m, mod.AdaRMSNorm))
    print(
        f"\n{n_ada} AdaRMSNorm modules -> {n_ada * 2} gamma/beta matmuls per step, "
        f"{n_ada * 2 * STEPS} per request"
    )

    def timed(label: str) -> tuple[float, torch.Tensor]:
        for _ in range(args.warmup):
            model.denoise_actions(num_steps=STEPS, **kwargs)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            out = model.denoise_actions(num_steps=STEPS, **kwargs)
        torch.xpu.synchronize()
        ms = (time.perf_counter() - t0) / args.repeats * 1e3
        print(f"  {label:44s} {ms:6.1f} ms")
        return ms, out.float().clone()

    base_ms, base_out = timed("today (gamma/beta recomputed every step)")

    original = mod.AdaRMSNorm.forward
    for module in model.modules():
        if isinstance(module, mod.AdaRMSNorm):
            module._cached = None

    def patched(self, hidden_states, cond):
        """Timing proxy: γ/β from cache. Numerically wrong on purpose."""
        if self._cached is None:
            self._cached = (self.gamma(cond).unsqueeze(1).float(), self.beta(cond).unsqueeze(1).float())
        gamma, beta = self._cached
        hidden = hidden_states.float()
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = self.weight * (hidden * torch.rsqrt(variance + self.variance_epsilon))
        return ((1 + gamma) * hidden + beta).to(hidden_states.dtype)

    try:
        mod.AdaRMSNorm.forward = patched
        torch._dynamo.reset()
        compile_step()
        hoist_ms, hoist_out = timed("gamma/beta as a cached lookup (proxy)")
    finally:
        mod.AdaRMSNorm.forward = original

    print(
        f"\n  saving: {base_ms - hoist_ms:.1f} ms ({base_ms / hoist_ms:.3f}x) on the {STEPS}-step loop"
        f"\n  plus {per_step * (STEPS - 1):.2f} ms of redundant setup = "
        f"{base_ms - hoist_ms + per_step * (STEPS - 1):.1f} ms total (step I)"
        f"\n  proxy reuses one timestep's FiLM, so max|delta|="
        f"{(hoist_out - base_out).abs().max().item():.3e} is expected, not a result"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attention-precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--mode", default="real", choices=["real", "micro", "both"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.mode in ("micro", "both"):
        run_micro(args)
    if args.mode in ("real", "both"):
        run_real(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
