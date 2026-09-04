#!/usr/bin/env python3
"""How fast does the flow-matching loop amplify a 1-ulp bf16 difference?

``phase0_layer_probe.py`` showed CPU and XPU agree to ~1e-2 relative at *every*
stage of one forward pass (that is one bf16 ulp — different kernel accumulation
order, nothing more), yet the final action chunk differs by ~85% relative. The
only place that gap can come from is the Euler denoise loop, which feeds each
step's output back in as the next step's input.

This measures the growth directly: one model load, run ``sample_actions`` on CPU
and on XPU for an increasing number of denoise steps, and report the divergence
as a function of step count.

    python phase0_step_growth.py --dtype bfloat16
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import bootstrap


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 10])
    p.add_argument("--structural", action="store_true")
    return p.parse_args()


@torch.inference_mode()
def sample(model, inputs, device, num_steps):
    model.config.num_steps = num_steps
    to = lambda t: t.to(device)  # noqa: E731
    actions = model.sample_actions(
        to(inputs["images"]),
        to(inputs["img_masks"]),
        to(inputs["lang_tokens"]),
        to(inputs["lang_masks"]),
        to(inputs["state"]),
        # .clone() is REQUIRED: sample_actions does ``x_t = noise`` then ``x_t +=
        # dt * v_t``, so it denoises the caller's tensor in place and returns that
        # same storage. Reusing ``inputs["noise"]`` across calls silently starts
        # run N+1 from run N's output. (On XPU the ``.to(device)`` copy hid this.)
        noise=to(inputs["noise"]).clone(),
        image_grid_thw=to(inputs["image_grid_thw"]),
    )
    return actions.detach().float().cpu().numpy().astype(np.float64)


def main():
    args = parse_args()
    torch_dtype = getattr(torch, args.dtype)

    bootstrap.setup(verbose=False)
    bootstrap.import_modeling(verbose=False)
    from phase0_torch_spike import build_inputs, build_model

    model, config, processor = build_model(argparse.Namespace(structural=args.structural), torch_dtype)
    inputs = build_inputs(model, config, processor, torch.device("cpu"), torch_dtype)

    cpu_runs = {n: sample(model, inputs, torch.device("cpu"), n) for n in args.steps}
    model.to("xpu")
    torch.xpu.synchronize()
    xpu_runs = {n: sample(model, inputs, torch.device("xpu"), n) for n in args.steps}

    print(f"\n[growth] CPU vs XPU, both {args.dtype}, by denoise step count:")
    print(f"  {'steps':>6}  {'mean|d|':>10}  {'max|d|':>10}  {'mean|ref|':>10}  {'rel':>10}")
    for n in args.steps:
        diff = np.abs(cpu_runs[n] - xpu_runs[n])
        scale = np.abs(cpu_runs[n]).mean()
        print(
            f"  {n:>6}  {diff.mean():>10.3e}  {diff.max():>10.3e}  "
            f"{scale:>10.3e}  {diff.mean() / max(scale, 1e-12):>10.3e}"
        )


if __name__ == "__main__":
    main()
