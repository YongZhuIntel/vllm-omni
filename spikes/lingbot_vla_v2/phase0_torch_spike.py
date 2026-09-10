#!/usr/bin/env python3
"""Phase 0 go/no-go: run upstream LingBot-VLA-2.0 ``sample_actions`` unmodified,
inside the vLLM-Omni environment (transformers 5.x, torch-XPU).

Two things this answers before any porting work starts:

  1. Does the upstream model run at all against vLLM-Omni's pinned transformers?
     (upstream pins 4.57.3; vLLM-Omni wants 5.10+). ``bootstrap.py`` carries the
     compatibility shims that make it work, each one documented.
  2. What is the reference action chunk? Everything later — the vendored
     ``vllm_omni/diffusion/models/lingbot_vla_v2`` kernel, the XPU/bf16 path — is
     graded against the fp32 CPU golden this produces.

Usage
-----
    # golden reference (CPU, fp32, ~25 GB RAM, slow)
    python phase0_torch_spike.py --device cpu --dtype float32 --out golden_cpu_fp32.npz

    # XPU run, graded against the golden
    python phase0_torch_spike.py --device xpu --dtype bfloat16 --ref golden_cpu_fp32.npz

    # no checkpoint needed — validates the code paths only
    python phase0_torch_spike.py --structural
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import bootstrap

# Fixed seeds so every run is comparable. The observation is synthetic (random
# camera frames + a fixed prompt): Phase 0 grades numerics, not robot behaviour.
IMAGE_SEED = 0
NOISE_SEED = 1234
PROMPT = "pick up the object"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", choices=["cpu", "xpu"])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--structural", action="store_true", help="tiny random-init model; no checkpoint needed")
    p.add_argument("--num-steps", type=int, default=None, help="override the flow-matching step count")
    p.add_argument("--repeat", type=int, default=1, help="timed iterations after the first")
    p.add_argument("--out", default=None, help="write the action chunk + metadata to this .npz")
    p.add_argument("--ref", default=None, help="compare against a previously written .npz")
    return p.parse_args()


def build_model(args, torch_dtype):
    """Return ``(FlowMatchingV2, config, processor)``."""
    import build_model as bm
    from lingbotvla.models import build_processor

    qwen3vl = bootstrap.QWEN3VL_PATH.as_posix()
    processor = build_processor(qwen3vl)

    if args.structural:
        model, config = bm.build_structural(qwen3vl, num_layers=2, num_experts=4, dtype=torch_dtype)
        return model, config, processor

    # The released 6B checkpoint ships no ``lingbotvla_cli.yaml`` and only a stub
    # ``config.json`` ({"vlm_family": "qwen3_vl"}), so the architecture is inferred
    # from the safetensors headers.
    model, config = bm.build_from_release(bootstrap.CKPT_DIR.as_posix(), qwen3vl, dtype=torch_dtype)
    return model, config, processor


def build_inputs(model, config, processor, device, torch_dtype):
    import wrappers

    images, img_masks, lang_tokens, lang_masks, image_grid_thw = wrappers.make_prefix_example(
        model, processor, num_cams=3, size=224, prompt=PROMPT, dtype=torch_dtype, seed=IMAGE_SEED
    )
    state = torch.zeros(1, config.max_state_dim, dtype=torch_dtype)
    # Noise is always generated on CPU in float32 and cast, so the same seed gives
    # the same starting point on every device/dtype combination.
    generator = torch.Generator().manual_seed(NOISE_SEED)
    noise = torch.randn(1, config.chunk_size, config.max_action_dim, generator=generator).to(torch_dtype)

    to_dev = lambda t: t.to(device)  # noqa: E731
    return {
        "images": to_dev(images),
        "img_masks": to_dev(img_masks),
        "lang_tokens": to_dev(lang_tokens),
        "lang_masks": to_dev(lang_masks),
        "state": to_dev(state),
        "noise": to_dev(noise),
        "image_grid_thw": to_dev(image_grid_thw),
    }


def synchronize(device):
    if device.type == "xpu":
        torch.xpu.synchronize()


def run_once(model, inputs, device):
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        actions = model.sample_actions(
            inputs["images"],
            inputs["img_masks"],
            inputs["lang_tokens"],
            inputs["lang_masks"],
            inputs["state"],
            # .clone() is REQUIRED: sample_actions runs ``x_t = noise`` then
            # ``x_t += dt * v_t``, i.e. it denoises the caller's tensor in place
            # and returns that same storage. Without the clone, --repeat runs on
            # CPU would each start from the previous run's output.
            noise=inputs["noise"].clone(),
            image_grid_thw=inputs["image_grid_thw"],
        )
    synchronize(device)
    return actions, time.perf_counter() - start


def compare(actions: np.ndarray, ref_path: str) -> dict:
    ref = np.load(ref_path)
    ref_actions = ref["actions"]
    if ref_actions.shape != actions.shape:
        raise SystemExit(f"shape mismatch: ref {ref_actions.shape} vs run {actions.shape}")
    diff = np.abs(actions.astype(np.float64) - ref_actions.astype(np.float64))
    denom = np.maximum(np.abs(ref_actions.astype(np.float64)), 1e-6)
    return {
        "ref": ref_path,
        "ref_dtype": str(ref["dtype"]) if "dtype" in ref else "?",
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float((diff / denom).max()),
    }


def main():
    args = parse_args()
    torch_dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    bootstrap.setup()
    print(f"[spike] env: {json.dumps(bootstrap.env_report())}")
    bootstrap.import_modeling()

    print(f"[spike] building model (structural={args.structural}, dtype={args.dtype}) ...")
    t0 = time.perf_counter()
    model, config, processor = build_model(args, torch_dtype)
    print(f"[spike] model built in {time.perf_counter() - t0:.1f}s")

    if args.num_steps is not None:
        config.num_steps = args.num_steps
        model.config.num_steps = args.num_steps

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[spike] params={n_params / 1e9:.2f}B  chunk={config.chunk_size} "
        f"action_dim={config.max_action_dim} state_dim={config.max_state_dim} "
        f"num_steps={config.num_steps}"
    )

    inputs = build_inputs(model, config, processor, device, torch_dtype)
    print(f"[spike] moving model to {device} ...")
    t0 = time.perf_counter()
    model.to(device)
    synchronize(device)
    print(f"[spike] model on {device} in {time.perf_counter() - t0:.1f}s")

    actions, elapsed = run_once(model, inputs, device)
    print(f"[spike] first sample_actions: {elapsed:.2f}s -> {tuple(actions.shape)} {actions.dtype}")
    for i in range(args.repeat):
        _, elapsed = run_once(model, inputs, device)
        print(f"[spike] repeat {i + 1}: {elapsed:.2f}s")

    actions_np = actions.float().cpu().numpy()
    print(
        f"[spike] actions: mean={actions_np.mean():.6f} std={actions_np.std():.6f} "
        f"min={actions_np.min():.6f} max={actions_np.max():.6f}"
    )
    print(f"[spike] actions[0, 0, :8] = {np.array2string(actions_np[0, 0, :8], precision=6)}")

    if args.out:
        np.savez(
            args.out,
            actions=actions_np,
            dtype=args.dtype,
            device=args.device,
            num_steps=config.num_steps,
            image_seed=IMAGE_SEED,
            noise_seed=NOISE_SEED,
            prompt=PROMPT,
            structural=args.structural,
        )
        print(f"[spike] wrote {args.out}")

    if args.ref:
        result = compare(actions_np, args.ref)
        print(f"[spike] vs {result['ref']}: mean|d|={result['mean_abs_diff']:.3e} "
              f"max|d|={result['max_abs_diff']:.3e} max_rel={result['max_rel_diff']:.3e}")


if __name__ == "__main__":
    main()
