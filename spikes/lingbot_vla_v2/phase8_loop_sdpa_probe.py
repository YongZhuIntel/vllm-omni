#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 8 — isolate the denoise-loop SDPA experiment and grade its precision.

`PHASE8_LATENCY_PARITY.md` step D concluded "no default change" for SDPA, on two
findings that this probe revises:

* SDPA saved 6.3 ms of `prefix_fill` (64.5 → 58.2). That comparison predates the
  fp16 attention change, which took eager to 57.6 on its own — so SDPA's prefix
  advantage was really the accumulation precision, and is already banked.
* "its cached prefix produces NaNs", attributed to an SDPA KV-cache contract.

The NaN is real but the attribution was too broad. `attention_backend` is read
per call, so the prefix and the loop can be measured independently, and they
behave completely differently:

* **prefix (`fill_kv_cache=True`) under SDPA poisons the cache.** The `[286,286]`
  mask has 63 fully-masked rows (`pad_2d_masks` zeroes both the row and the
  column of every padding token). Layer 0's K comes out bit-identical to eager,
  layer 35's is NaN — it starts at those rows and spreads until essentially every
  valid slot is non-finite. Eager is immune because it masks with
  `torch.where`, a select, and never evaluates a fully-masked softmax.
* **the loop (`fill_kv_cache=False`) under SDPA is finite**, because suffix
    queries are all real tokens: no row is fully masked. It does *not*, however,
    agree closely enough with eager or the fp32 reference to ship.

The usable subset is therefore only a numerical diagnostic. Five-seed fp32
reference grading of compiled `suffix_sdpa` measured MAE `1.538e-01`, above the
`2.882e-02` vendor ceiling. This probe remains useful for separating Prefix NaNs
from loop behavior, but does not define a deployment path.

    PYTHONPATH=.:spikes/lingbot_vla_v2 \\
        python spikes/lingbot_vla_v2/phase8_loop_sdpa_probe.py --model /tmp/lingbot-vla-v2-perf
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from phase5_latency import DEFAULT_MODEL, build, observation


def _sync() -> None:
    torch.xpu.synchronize()


@torch.inference_mode()
def probe(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = build(Path(args.model), device, dtype, None, None)
    joint = model.qwenvl_with_expert
    joint.attention_precision = args.attention_precision

    inputs = processor.preprocess(observation(processor.spec, 0)).to(device=device, dtype=dtype).model_inputs()
    if args.compile:
        model.predict_velocity = torch.compile(
            model.predict_velocity, backend="inductor", dynamic=False, fullgraph=True
        )

    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import make_att_2d_masks

    prefix_args = (
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["image_grid_thw"],
    )
    embs, pad_masks, att_masks, position_ids, visual_masks, deepstack = model.embed_prefix(*prefix_args)
    mask_2d = make_att_2d_masks(pad_masks, att_masks)
    valid = pad_masks[0]
    print(
        f"prefix: {mask_2d.shape[-1]} slots, {int(valid.sum())} valid, "
        f"{int((~valid).sum())} padding, {int((~mask_2d[0].any(-1)).sum())} fully-masked rows"
    )

    # The prefix always runs eager: SDPA here is the half that poisons the cache.
    joint.attention_backend = "eager"
    _, past_key_values = joint.forward(
        attention_mask=mask_2d,
        position_ids=position_ids,
        inputs_embeds=[embs, None],
        past_key_values=None,
        fill_kv_cache=True,
        visual_pos_masks=visual_masks,
        deepstack_visual_embeds=deepstack,
    )
    _sync()

    # One fixed noise draw, so eager and SDPA are compared on the same trajectory.
    noise = torch.randn(
        (inputs["state"].shape[0], model.config.chunk_size, model.config.max_action_dim),
        device=device,
        dtype=dtype,
    )
    kwargs = dict(
        state=inputs["state"],
        prefix_pad_masks=pad_masks,
        prefix_position_ids=position_ids,
        past_key_values=past_key_values,
        noise=noise,
    )

    print(f"\nloop backend timings ({args.repeats} iterations, compile={args.compile})")
    results: dict[str, tuple[float, torch.Tensor]] = {}
    for backend in ("eager", "sdpa"):
        joint.attention_backend = backend
        for _ in range(args.warmup):
            out = model.denoise_actions(num_steps=model.config.num_steps, **kwargs)
        _sync()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            out = model.denoise_actions(num_steps=model.config.num_steps, **kwargs)
        _sync()
        per = (time.perf_counter() - t0) / args.repeats * 1e3
        results[backend] = (per, out.float().clone())
        print(f"  loop={backend:5s} {per:7.1f} ms   finite={bool(torch.isfinite(out).all())}")

    eager_ms, eager_out = results["eager"]
    sdpa_ms, sdpa_out = results["sdpa"]
    delta = (sdpa_out - eager_out).abs()
    denom = eager_out.abs().max().clamp_min(1e-12)
    print(
        f"\n  saving: {eager_ms - sdpa_ms:.1f} ms ({eager_ms / sdpa_ms:.3f}x) on the loop"
        f"\n  agreement vs eager: max_abs={delta.max().item():.3e} "
        f"max_rel={(delta.max() / denom).item():.3e} MAE={delta.mean().item():.3e}"
        f"\n  (the fp32-referenced ceiling used by step A's gate is 2.882e-02 MAE)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attention-precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    return probe(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
