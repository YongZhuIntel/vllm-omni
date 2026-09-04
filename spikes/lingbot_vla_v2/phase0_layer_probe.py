#!/usr/bin/env python3
"""Localize the CPU-vs-XPU divergence seen in ``phase0_torch_spike.py``.

The end-to-end action chunk showed bf16 alone costs ~4e-2 mean abs error against
the fp32 CPU golden, but XPU bf16 costs ~4e-1 — an order of magnitude more, so
something device-specific is happening rather than plain rounding. This walks the
``sample_actions`` pipeline stage by stage and reports where the two devices part
company, using ONE model load (build on CPU, probe, move to XPU, probe again).

Stages, in execution order:

  1. ``embed_image``   — Qwen3-VL vision tower only (SDPA attention)
  2. ``embed_prefix``  — vision + token embedding + mrope position ids
  3. prefix forward    — the 36-layer text backbone that fills the KV cache
  4. ``predict_velocity`` — one action-expert (MoE) denoise step off that cache

    python phase0_layer_probe.py --dtype bfloat16
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import bootstrap
from phase0_torch_spike import IMAGE_SEED, NOISE_SEED, PROMPT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--structural", action="store_true")
    return p.parse_args()


def stats(name: str, cpu: torch.Tensor, xpu: torch.Tensor) -> None:
    a = cpu.detach().float().cpu().numpy().astype(np.float64)
    b = xpu.detach().float().cpu().numpy().astype(np.float64)
    diff = np.abs(a - b)
    scale = np.abs(a).mean()
    print(
        f"  {name:24s} shape={str(tuple(a.shape)):22s} "
        f"mean|d|={diff.mean():.3e}  max|d|={diff.max():.3e}  "
        f"mean|ref|={scale:.3e}  rel={diff.mean() / max(scale, 1e-12):.3e}"
    )


@torch.inference_mode()
def probe(model, inputs, device):
    """Run the four stages on ``device``; return the intermediate tensors (on CPU)."""
    from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks

    to = lambda t: t.to(device) if isinstance(t, torch.Tensor) else t  # noqa: E731
    images, img_masks = to(inputs["images"]), to(inputs["img_masks"])
    lang_tokens, lang_masks = to(inputs["lang_tokens"]), to(inputs["lang_masks"])
    state, noise = to(inputs["state"]), to(inputs["noise"])
    grid = to(inputs["image_grid_thw"])

    out = {}

    # 1. vision tower only. embed_image takes the flattened (b*n) layout that
    #    embed_prefix builds for it (modeling_lingbot_vla_v2.py:512-528).
    import einops

    flat_images = einops.rearrange(images, "b n l d -> (b n) l d") if images.ndim == 4 else images
    flat_grid = einops.rearrange(grid, "b n d -> (b n) d") if grid.ndim == 3 else grid
    img_emb, _deepstack = model.qwenvl_with_expert.embed_image(flat_images, flat_grid)
    out["img_emb"] = img_emb

    # 2. full prefix embedding (vision + text tokens + mrope ids)
    (
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
    ) = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, image_grid_thw=grid)
    out["prefix_embs"] = prefix_embs
    out["prefix_position_ids"] = prefix_position_ids.to(torch.float32)

    # 3. the 36-layer text backbone -> KV cache
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    _, past_key_values, _ = model.qwenvl_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        vlm_position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    layers = sorted(past_key_values.keys())
    for idx in (layers[0], layers[len(layers) // 2], layers[-1]):
        out[f"kv_key_L{idx}"] = past_key_values[idx]["key_states"]

    # 4. one action-expert denoise step off that cache
    timestep = torch.tensor([1.0], dtype=state.dtype, device=device)
    out["v_t_step0"] = model.predict_velocity(
        state, prefix_pad_masks, past_key_values, noise, timestep, prefix_position_ids=prefix_position_ids
    )

    return {k: v.detach().to("cpu") for k, v in out.items()}


def main():
    args = parse_args()
    torch_dtype = getattr(torch, args.dtype)

    bootstrap.setup(verbose=False)
    bootstrap.import_modeling(verbose=False)

    from phase0_torch_spike import build_inputs, build_model

    spike_args = argparse.Namespace(structural=args.structural)
    print(f"[probe] building model (dtype={args.dtype}) ...")
    model, config, processor = build_model(spike_args, torch_dtype)
    inputs = build_inputs(model, config, processor, torch.device("cpu"), torch_dtype)

    print("[probe] running CPU stages ...")
    cpu_out = probe(model, inputs, torch.device("cpu"))

    print("[probe] moving model to XPU ...")
    model.to("xpu")
    torch.xpu.synchronize()
    print("[probe] running XPU stages ...")
    xpu_out = probe(model, inputs, torch.device("xpu"))

    print(f"\n[probe] CPU vs XPU, both {args.dtype}, in execution order:")
    for key in cpu_out:
        stats(key, cpu_out[key], xpu_out[key])


if __name__ == "__main__":
    main()
