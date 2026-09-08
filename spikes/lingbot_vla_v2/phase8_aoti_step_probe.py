#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 8 step C — does AOTInductor remove the denoise step's host dispatch tax?

Step C wanted "one graph for all 10 steps, as the reference does" and failed
because Inductor's JIT stayed in fullgraph generation for over six minutes. That
was a *cold-compile* failure, not a verdict on precompilation, and AOTInductor
answers exactly that objection: `torch.export` plus `aoti_compile_and_package`
move lowering offline and ship a `.pt2` that loads in seconds. `phase8_graph_probe.py`
confirms both entry points exist on this stack and that AOTI is worth 1.2x on a
synthetic dispatch-bound chain.

So this script applies it to the real `predict_velocity`, which is the only
measurement that decides step C. It exports the step with the prefix KV cache
captured as constants, compiles it, and times it against the JIT-compiled
baseline of ~20.1 ms/step from `phase8_stage_device_profile.py`.

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase8_aoti_step_probe.py \
        --model /tmp/lingbot-vla-v2-perf

Result is recorded in `PHASE8_LATENCY_PARITY.md` under F3: it does not help. The
residual host cost is per-kernel submission into the Level-Zero queue, which
AOTInductor still performs one kernel at a time -- from C++ instead of Python,
which is why the tiny chain gains and the real step does not.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from phase5_latency import DEFAULT_MODEL, build, observation

# The JIT-compiled single-step figure this is trying to beat, from F1.
BASELINE_MS = 20.1


@torch.inference_mode()
def probe(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = build(Path(args.model), device, dtype, None, None)
    # The override trap from F1: `__init__` hardcodes fp32/eager and the deployed
    # values arrive from the config in `pipeline_lingbot_vla_v2.py:61`.
    model.qwenvl_with_expert.attention_backend = args.attention_backend
    model.qwenvl_with_expert.attention_precision = args.attention_precision

    inputs = processor.preprocess(observation(processor.spec, 0)).to(device=device, dtype=dtype).model_inputs()
    model.sample_actions(**inputs)
    torch.xpu.synchronize()

    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import make_att_2d_masks

    prefix_args = (
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["image_grid_thw"],
    )
    embs, pad_masks, att_masks, position_ids, visual_masks, deepstack = model.embed_prefix(*prefix_args)
    _, past_key_values = model.qwenvl_with_expert.forward(
        attention_mask=make_att_2d_masks(pad_masks, att_masks),
        position_ids=position_ids,
        inputs_embeds=[embs, None],
        past_key_values=None,
        fill_kv_cache=True,
        visual_pos_masks=visual_masks,
        deepstack_visual_embeds=deepstack,
    )
    torch.xpu.synchronize()
    noise = torch.randn(
        (inputs["state"].shape[0], model.config.chunk_size, model.config.max_action_dim),
        device=device,
        dtype=dtype,
    )
    timestep = torch.ones(noise.shape[0], device=device, dtype=dtype)

    class Step(torch.nn.Module):
        """One Euler step with the prefix cache and masks closed over.

        Closing over them is what makes the export succeed -- the KV cache is a
        36-element list of tuples, not a flat tensor pytree -- and also what
        makes the package enormous, since every captured tensor is frozen in as
        a constant. Both facts are part of the finding.
        """

        def forward(self, state: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return model.predict_velocity(
                state=state,
                prefix_pad_masks=pad_masks,
                prefix_position_ids=position_ids,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=t,
            )

    example = (inputs["state"], noise, timestep)

    t0 = time.perf_counter()
    exported = torch.export.export(Step(), example, strict=False)
    print(f"torch.export: {time.perf_counter() - t0:.1f}s, {len(list(exported.graph.nodes))} nodes")

    t0 = time.perf_counter()
    package = torch._inductor.aoti_compile_and_package(exported, package_path=args.package)
    print(
        f"aoti_compile_and_package: {time.perf_counter() - t0:.1f}s, "
        f"{os.path.getsize(package) / 1e6:.0f} MB"
    )

    fn = torch._inductor.aoti_load_package(package)
    for _ in range(args.warmup):
        fn(*example)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        fn(*example)
    torch.xpu.synchronize()
    per_step = (time.perf_counter() - t0) / args.repeats * 1e3

    print(
        f"\nAOTInductor predict_velocity: {per_step:.1f} ms/step "
        f"vs JIT-compiled baseline {BASELINE_MS:.1f} ms/step "
        f"= {BASELINE_MS / per_step:.2f}x"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--attention-precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--package", default="/tmp/phase8_aoti_step.pt2")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    return probe(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
