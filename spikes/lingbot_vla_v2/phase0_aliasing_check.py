#!/usr/bin/env python3
"""Regression check for the ``sample_actions`` noise-aliasing bug.

``FlowMatchingV2.sample_actions`` (modeling_lingbot_vla_v2.py:967-996) does::

    x_t = noise
    while time >= -dt / 2:
        v_t = predict_velocity(...)
        x_t += dt * v_t          # in-place, on the CALLER's tensor
        time += dt
    return x_t                   # same storage as ``noise``

so it denoises the caller's ``noise`` argument in place and returns that same
storage. Upstream's own deploy path never notices: ``select_action`` passes
``noise=None``, so ``sample_actions`` allocates a fresh tensor each call.

A vLLM-Omni pipeline WILL notice. π0's pipeline already accepts a caller-supplied
``extra_args["noise"]`` (pipeline_pi0.py:219-224), and any pipeline that keeps a
preallocated noise buffer for a warm path hits it too: request 2 silently starts
from request 1's action chunk. It also means the returned action tensor aliases an
input, so writing to the output corrupts the buffer.

This check demonstrates both halves cheaply, on the tiny structural model.

    python phase0_aliasing_check.py
"""

from __future__ import annotations

import argparse

import torch

import bootstrap


@torch.inference_mode()
def sample(model, inputs, noise):
    return model.sample_actions(
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["state"],
        noise=noise,
        image_grid_thw=inputs["image_grid_thw"],
    )


def main():
    bootstrap.setup(verbose=False)
    bootstrap.import_modeling(verbose=False)
    from phase0_torch_spike import build_inputs, build_model

    model, config, processor = build_model(argparse.Namespace(structural=True), torch.float32)
    inputs = build_inputs(model, config, processor, torch.device("cpu"), torch.float32)

    noise = inputs["noise"]
    pristine = noise.clone()

    first_view = sample(model, inputs, noise)
    print(f"caller's noise mutated in place : {not torch.equal(pristine, noise)}")
    print(f"output aliases the noise storage: {first_view.data_ptr() == noise.data_ptr()}")
    # Must copy: `first_view` IS `noise`, so the next call would overwrite it too.
    first = first_view.clone()

    # Reusing the same buffer: run 2 starts from run 1's output.
    second = sample(model, inputs, noise).clone()
    reused_delta = (second - first).abs().max()

    # With a clone per call, the model is bit-exactly reproducible.
    a = sample(model, inputs, pristine.clone()).clone()
    b = sample(model, inputs, pristine.clone()).clone()
    print(f"reused buffer, run1 vs run2     : max|d|={reused_delta:.3e}  <- silently wrong")
    print(f"cloned noise,  run1 vs run2     : max|d|={(a - b).abs().max():.3e}  <- bit-exact")


if __name__ == "__main__":
    main()
