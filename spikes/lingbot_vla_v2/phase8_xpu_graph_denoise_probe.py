#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture LingBot's complete ten-step denoise loop in one XPU Graph.

The production path compiles one ``predict_velocity`` step and invokes it ten
times from Python. This probe captures those ten invocations as one replay and
measures both pure replay and the realistic cost of copying per-request Prefix
KV/state/noise into graph-stable buffers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase5_latency import DEFAULT_MODEL, build, observation  # noqa: E402

from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import (  # noqa: E402
    denoise_compile_options,
    make_att_2d_masks,
)


def sync() -> None:
    torch.accelerator.synchronize()


def clone_kv(values: list[tuple[torch.Tensor, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(key.clone(), value.clone()) for key, value in values]


def copy_kv(
    destination: list[tuple[torch.Tensor, torch.Tensor]],
    source: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for (dst_key, dst_value), (src_key, src_value) in zip(destination, source, strict=True):
        dst_key.copy_(src_key)
        dst_value.copy_(src_value)


def elapsed_ms(callable, repeats: int) -> float:
    sync()
    started = time.perf_counter()
    for _ in range(repeats):
        callable()
    sync()
    return (time.perf_counter() - started) * 1e3 / repeats


@torch.inference_mode()
def probe(args: argparse.Namespace) -> int:
    if not hasattr(torch.xpu, "XPUGraph"):
        raise RuntimeError("this PyTorch XPU build does not expose XPUGraph")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = build(Path(args.model), device, dtype, None, None)
    model.qwenvl_with_expert.attention_backend = "eager"
    model.qwenvl_with_expert.attention_precision = "fp16"
    if args.compile_step:
        model.predict_velocity = torch.compile(
            model.predict_velocity,
            backend="inductor",
            dynamic=False,
            fullgraph=True,
            options=denoise_compile_options(),
        )

    inputs = processor.preprocess(observation(processor.spec, 0)).to(device=device, dtype=dtype).model_inputs()
    embs, pad_masks, att_masks, position_ids, visual_masks, deepstack = model.embed_prefix(
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["image_grid_thw"],
    )
    _, past_key_values = model.prefix_forward(
        attention_mask=make_att_2d_masks(pad_masks, att_masks),
        position_ids=position_ids,
        inputs_embeds=[embs, None],
        past_key_values=None,
        fill_kv_cache=True,
        visual_pos_masks=visual_masks,
        deepstack_visual_embeds=deepstack,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        (inputs["state"].shape[0], model.config.chunk_size, model.config.max_action_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    static_state = inputs["state"].clone()
    static_pad_masks = pad_masks.clone()
    static_position_ids = position_ids.clone()
    static_kv = clone_kv(past_key_values)
    static_noise = noise.clone()
    dt = torch.tensor(-1.0 / model.config.num_steps, device=device, dtype=dtype)
    timestep = torch.tensor(1.0, device=device, dtype=dtype)
    timesteps = []
    for _ in range(model.config.num_steps):
        timesteps.append(timestep.expand(inputs["state"].shape[0]))
        timestep = timestep + dt

    def denoise(
        state: torch.Tensor,
        masks: torch.Tensor,
        positions: torch.Tensor,
        kv: list[tuple[torch.Tensor, torch.Tensor]],
        noise_tensor: torch.Tensor,
    ) -> torch.Tensor:
        output = noise_tensor
        for current_time in timesteps:
            velocity = model.predict_velocity(
                state=state,
                prefix_pad_masks=masks,
                prefix_position_ids=positions,
                past_key_values=kv,
                x_t=output,
                timestep=current_time,
            )
            output = output + dt * velocity
        return output

    for _ in range(args.warmup):
        expected = denoise(inputs["state"], pad_masks, position_ids, past_key_values, noise)
    sync()
    expected = expected.clone()

    graph = torch.xpu.XPUGraph()
    capture_stream = torch.xpu.Stream()
    with torch.xpu.graph(graph, stream=capture_stream):
        graph_output = denoise(
            static_state,
            static_pad_masks,
            static_position_ids,
            static_kv,
            static_noise,
        )
    for _ in range(3):
        graph.replay()
    sync()

    diff = (graph_output.float() - expected.float()).abs()
    scale = expected.float().abs().max().clamp_min(1e-12)

    def copy_inputs() -> None:
        static_state.copy_(inputs["state"])
        static_pad_masks.copy_(pad_masks)
        static_position_ids.copy_(position_ids)
        copy_kv(static_kv, past_key_values)
        static_noise.copy_(noise)

    def copy_and_replay() -> None:
        copy_inputs()
        graph.replay()

    def synchronized_copy_and_replay() -> None:
        copy_inputs()
        sync()
        graph.replay()
        sync()

    direct_ms = elapsed_ms(
        lambda: denoise(inputs["state"], pad_masks, position_ids, past_key_values, noise),
        args.repeats,
    )
    replay_ms = elapsed_ms(graph.replay, args.repeats)
    copy_ms = elapsed_ms(copy_inputs, args.repeats)
    total_ms = elapsed_ms(copy_and_replay, args.repeats)
    synchronized_total_ms = elapsed_ms(synchronized_copy_and_replay, args.repeats)

    print(f"torch={torch.__version__} device={torch.xpu.get_device_name(0)} compile_step={args.compile_step}")
    print(f"steps={model.config.num_steps} repeats={args.repeats}")
    print(f"direct compiled loop       {direct_ms:8.2f} ms")
    print(f"XPU Graph replay          {replay_ms:8.2f} ms")
    print(f"static input copies       {copy_ms:8.2f} ms")
    print(f"copies + graph replay     {total_ms:8.2f} ms")
    print(f"synchronized copies/replay {synchronized_total_ms:7.2f} ms")
    print(f"speedup direct/replay     {direct_ms / replay_ms:8.3f}x")
    print(f"speedup direct/realistic  {direct_ms / total_ms:8.3f}x")
    print(
        f"parity finite={bool(torch.isfinite(graph_output).all())} "
        f"max_abs={float(diff.max()):.3e} max_rel={float(diff.max() / scale):.3e} "
        f"mae={float(diff.mean()):.3e}"
    )

    next_inputs = (
        processor.preprocess(observation(processor.spec, args.seed + 1)).to(device=device, dtype=dtype).model_inputs()
    )
    next_embs, next_masks, next_att_masks, next_positions, next_visual_masks, next_deepstack = model.embed_prefix(
        next_inputs["images"],
        next_inputs["img_masks"],
        next_inputs["lang_tokens"],
        next_inputs["lang_masks"],
        next_inputs["image_grid_thw"],
    )
    _, next_kv = model.prefix_forward(
        attention_mask=make_att_2d_masks(next_masks, next_att_masks),
        position_ids=next_positions,
        inputs_embeds=[next_embs, None],
        past_key_values=None,
        fill_kv_cache=True,
        visual_pos_masks=next_visual_masks,
        deepstack_visual_embeds=next_deepstack,
    )
    next_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    next_noise = torch.randn(static_noise.shape, generator=next_generator, device=device, dtype=dtype)
    next_expected = denoise(next_inputs["state"], next_masks, next_positions, next_kv, next_noise)
    sync()
    next_expected = next_expected.clone()

    static_state.copy_(next_inputs["state"])
    static_pad_masks.copy_(next_masks)
    static_position_ids.copy_(next_positions)
    copy_kv(static_kv, next_kv)
    static_noise.copy_(next_noise)
    graph.replay()
    sync()
    next_diff = (graph_output.float() - next_expected.float()).abs()
    next_scale = next_expected.float().abs().max().clamp_min(1e-12)
    print(
        f"new-request parity finite={bool(torch.isfinite(graph_output).all())} "
        f"max_abs={float(next_diff.max()):.3e} "
        f"max_rel={float(next_diff.max() / next_scale):.3e} "
        f"mae={float(next_diff.mean()):.3e}"
    )

    worst_repeat_diff = 0.0
    for _ in range(5):
        static_state.copy_(inputs["state"])
        static_pad_masks.copy_(pad_masks)
        static_position_ids.copy_(position_ids)
        copy_kv(static_kv, past_key_values)
        static_noise.copy_(noise)
        sync()
        graph.replay()
        sync()
        worst_repeat_diff = max(worst_repeat_diff, float((graph_output.float() - expected.float()).abs().max()))
        static_state.copy_(next_inputs["state"])
        static_pad_masks.copy_(next_masks)
        static_position_ids.copy_(next_positions)
        copy_kv(static_kv, next_kv)
        static_noise.copy_(next_noise)
        sync()
        graph.replay()
        sync()
        worst_repeat_diff = max(
            worst_repeat_diff,
            float((graph_output.float() - next_expected.float()).abs().max()),
        )
    print(f"alternating-request max_abs over 10 replays={worst_repeat_diff:.3e}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compile-step", action=argparse.BooleanOptionalAction, default=True)
    return probe(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
