# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M5 step 1 — where the 2.43 s goes.

Phase 0 measured the bare kernel at 0.74 s/chunk on the B60. The M3 pipeline
through ``OmniDiffusion`` measures 2.43 s and the M4 WebSocket 2.74 s. Nothing so
far says whether the extra 1.7 s is the observation processor, the host-to-device
copy, the model itself, or the engine's worker IPC — and every M5 optimisation
idea (fixed-shape compile, an XPU grouped-MoE kernel) is a guess until it is.

This script measures everything *below* the engine: it builds exactly what
``LingbotVlaV2Pipeline.__init__`` builds and runs exactly what its ``forward``
runs, in-process, with a device sync at every stage boundary. Subtracting the
total it reports from the engine's 2.43 s leaves the engine/IPC share.

Syncing at stage boundaries perturbs the total slightly (it serialises what would
otherwise overlap), so the run also reports an unsynced total for comparison. If
the two disagree by much, the attribution below is hiding overlap.

What it found, for the record: the denoise loop is ~90% of the request, and
``moe_implementation="gather"`` was 3.7x slower than ``"dense"`` on the B60, which
was the whole of the 2.43 s. With ``dense`` the total is 0.703 s. **Check the host
is idle before believing any absolute number here** — the first run of this script
shared the machine with leaked worker processes and inflated a 1.9 ms CPU stage to
178 ms. ``pre.images_again`` exists as the control for exactly that.

    python phase5_latency.py --iters 5
    python phase5_latency.py --device cpu --iters 1   # no XPU needed
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/tmp/lingbot-vla-v2-prepared"
DEFAULT_PROMPT = "pick up the object"


def _sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class Stopwatch:
    """Accumulates per-stage wall time, syncing the device at each boundary."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.laps: dict[str, float] = {}
        self._mark = 0.0

    def start(self) -> None:
        _sync(self.device)
        self._mark = time.perf_counter()

    def lap(self, name: str) -> None:
        _sync(self.device)
        now = time.perf_counter()
        self.laps[name] = now - self._mark
        self._mark = now


def build(
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int | None,
    moe: str | None = None,
) -> tuple[Any, Any]:
    """Reproduce ``LingbotVlaV2Pipeline.__init__`` without the engine."""
    sys.path.insert(0, str(REPO_ROOT))
    from safetensors.torch import safe_open

    from vllm_omni.diffusion.models.lingbot_vla_v2.config import LingbotVlaV2Config
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
        LingbotVlaV2ForActionPrediction,
    )
    from vllm_omni.diffusion.models.lingbot_vla_v2.processor import (
        LingbotVlaV2Processor,
        RobotSpec,
        load_hf_processor,
    )

    raw = json.loads((model_dir / "transformer" / "config.json").read_text())
    config = LingbotVlaV2Config.from_model_config(raw)
    if num_steps is not None:
        config.num_steps = num_steps
    if moe is not None:
        config.moe_implementation = moe

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else model_dir / path

    spec = RobotSpec.from_files(
        resolve(raw["robot_config"]),
        resolve(raw["data_config"]),
        resolve(raw["norm_stats"]) if raw.get("norm_stats") else None,
    )
    tokenizer, image_processor = load_hf_processor(config.qwen3vl_path)
    processor = LingbotVlaV2Processor(spec, config, tokenizer=tokenizer, image_processor=image_processor)

    t0 = time.perf_counter()
    # Build straight onto the target device in the target dtype, the way the
    # engine's loader does. Constructing fp32-on-CPU first needs 25.5 GB of host
    # RAM on top of the mmapped checkpoint and gets the process OOM-killed on a
    # 60 GB host; `.data.copy_` casts each shard tensor as it lands.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = LingbotVlaV2ForActionPrediction(config).eval()
    finally:
        torch.set_default_dtype(previous_dtype)

    def shards():
        for shard in sorted(model_dir.glob("model-*.safetensors")):
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    yield key, f.get_tensor(key)

    model.load_weights(shards())
    _sync(device)
    print(f"[build] weights loaded and placed in {time.perf_counter() - t0:.1f}s")
    return processor, model


def observation(spec: Any, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sources = sorted(set(spec.camera_sources.values()))
    return {
        "images": {s: rng.integers(0, 256, (spec.image_size, spec.image_size, 3), dtype=np.uint8) for s in sources},
        "state": np.zeros(sum(sl.width for slices in spec.state_slices.values() for sl in slices), dtype=np.float32),
        "prompt": DEFAULT_PROMPT,
    }


def one_request(processor: Any, model: Any, obs: dict, device: torch.device, dtype: torch.dtype) -> dict[str, float]:
    """``LingbotVlaV2Pipeline.forward``, opened up and timed stage by stage."""
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import make_att_2d_masks

    # Private helpers on purpose: this reproduces ``preprocess`` stage by stage,
    # so it has to call the same pieces it calls, not an approximation.
    from vllm_omni.diffusion.models.lingbot_vla_v2.processor import (
        RobotFeatures,
        _action_key,
        _as_float_tensor,
        _state_key,
    )

    watch = Stopwatch(device)
    config = model.config

    # -- observation -> tensors (CPU) ---------------------------------------
    watch.start()
    with torch.device("cpu"):
        raw = {"observation.state": _as_float_tensor(obs["state"])}
        state, state_mask = processor._build_vector(raw, processor.spec.state_slices, _state_key, config.max_state_dim)
        _, action_mask = processor._build_vector(
            raw, processor.spec.action_slices, _action_key, config.max_action_dim, values=False
        )
        watch.lap("pre.state")
        images, img_masks, grid = processor._build_images(obs["images"])
        watch.lap("pre.images")
        # The same call again, on the same frames. Kept as a control: this stage
        # once measured 178 ms and looked like the second-biggest cost in the
        # request, but three 256x256 cameras are ~1.9 ms of work and the repeat
        # confirms it. The 178 ms was host CPU contention from leaked worker
        # processes, not the processor. If the two laps ever diverge again,
        # something is competing for the CPU.
        processor._build_images(obs["images"])
        watch.lap("pre.images_again")
        lang_tokens, lang_masks = processor._build_language(obs["prompt"])
        watch.lap("pre.language")

    features = RobotFeatures(
        images=images,
        img_masks=img_masks,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=state.unsqueeze(0),
        image_grid_thw=grid,
        state_mask=state_mask,
        action_mask=action_mask,
    )

    # -- host -> device ------------------------------------------------------
    inputs = features.to(device=device, dtype=dtype).model_inputs()
    watch.lap("h2d")

    # -- prefix fill ---------------------------------------------------------
    embs, pad_masks, att_masks, position_ids, visual_masks, deepstack = model.embed_prefix(
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["image_grid_thw"],
    )
    watch.lap("model.embed_prefix")

    _, past_key_values = model.qwenvl_with_expert.forward(
        attention_mask=make_att_2d_masks(pad_masks, att_masks),
        position_ids=position_ids,
        inputs_embeds=[embs, None],
        past_key_values=None,
        fill_kv_cache=True,
        visual_pos_masks=visual_masks,
        deepstack_visual_embeds=deepstack,
    )
    watch.lap("model.prefix_fill")

    # -- denoise -------------------------------------------------------------
    bsize = inputs["state"].shape[0]
    num_steps = config.num_steps
    dt = torch.tensor(-1.0 / num_steps, dtype=dtype, device=device)
    t = torch.tensor(1.0, dtype=dtype, device=device)
    x_t = torch.randn((bsize, config.chunk_size, config.max_action_dim), device=device, dtype=dtype)
    step_times: list[float] = []
    for _ in range(num_steps):
        _sync(device)
        step_start = time.perf_counter()
        v_t = model.predict_velocity(
            state=inputs["state"],
            prefix_pad_masks=pad_masks,
            prefix_position_ids=position_ids,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=t.expand(bsize),
        )
        x_t = x_t + dt * v_t
        t = t + dt
        _sync(device)
        step_times.append(time.perf_counter() - step_start)
    watch._mark = time.perf_counter()
    watch.laps["model.denoise"] = sum(step_times)

    # -- device -> host, unnormalize ----------------------------------------
    watch.start()
    processor.postprocess(x_t, features)
    watch.lap("post.d2h_unnormalize")

    laps = dict(watch.laps)
    laps["_step_first"] = step_times[0]
    laps["_step_rest"] = statistics.mean(step_times[1:]) if len(step_times) > 1 else step_times[0]
    return laps


def unsynced_total(processor: Any, model: Any, obs: dict, device: torch.device, dtype: torch.dtype) -> float:
    """The same work with no intermediate syncs — the honest end-to-end number."""
    _sync(device)
    start = time.perf_counter()
    features = processor.preprocess(obs)
    actions = model.sample_actions(**features.to(device=device, dtype=dtype).model_inputs())
    processor.postprocess(actions, features)
    _sync(device)
    return time.perf_counter() - start


def compile_denoise_step(
    processor: Any,
    model: Any,
    obs: dict,
    device: torch.device,
    dtype: torch.dtype,
    backend: str,
    force_same_precision: bool,
    emulate_precision_casts: bool,
    max_relative_error: float,
) -> None:
    """Compile one fixed-shape denoise step and check it against eager output."""
    features = processor.preprocess(obs)
    inputs = features.to(device=device, dtype=dtype).model_inputs()
    generator = torch.Generator(device=device).manual_seed(1234)
    noise = torch.randn(
        (1, model.config.chunk_size, model.config.max_action_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    eager_predict_velocity = model.predict_velocity
    captured: dict[str, Any] = {}

    def capture_predict_velocity(**kwargs: Any) -> torch.Tensor:
        captured.update(kwargs)
        return eager_predict_velocity(**kwargs)

    model.predict_velocity = capture_predict_velocity
    model.sample_actions(**inputs, noise=noise, num_steps=1)
    model.predict_velocity = eager_predict_velocity
    _sync(device)

    eager = eager_predict_velocity(**captured)
    eager_repeat = eager_predict_velocity(**captured)
    eager_chunk = model.sample_actions(
        **inputs,
        noise=noise,
        num_steps=model.config.num_steps,
    )
    _sync(device)

    options = {}
    if force_same_precision:
        options["force_same_precision"] = True
    if emulate_precision_casts:
        options["emulate_precision_casts"] = True
    print(
        f"[compile] torch.compile(predict_velocity, backend={backend!r}, "
        f"dynamic=False, fullgraph=True, options={options})"
    )
    compiled_predict_velocity = torch.compile(
        eager_predict_velocity,
        backend=backend,
        dynamic=False,
        fullgraph=True,
        options=options or None,
    )
    compiled = compiled_predict_velocity(**captured)
    compiled_repeat = compiled_predict_velocity(**captured)
    model.predict_velocity = compiled_predict_velocity
    compiled_chunk = model.sample_actions(
        **inputs,
        noise=noise,
        num_steps=model.config.num_steps,
    )
    _sync(device)

    eager_repeat_abs = float((eager_repeat.float() - eager.float()).abs().max())
    compiled_repeat_abs = float((compiled_repeat.float() - compiled.float()).abs().max())
    diff = (compiled.float() - eager.float()).abs()
    max_abs = float(diff.max())
    scale = max(float(eager.float().abs().max()), 1e-12)
    max_rel = max_abs / scale
    finite = bool(torch.isfinite(compiled).all())
    chunk_diff = (compiled_chunk.float() - eager_chunk.float()).abs()
    chunk_max_abs = float(chunk_diff.max())
    chunk_scale = max(float(eager_chunk.float().abs().max()), 1e-12)
    chunk_max_rel = chunk_max_abs / chunk_scale
    print(
        f"[compile] repeat eager={eager_repeat_abs:.3e} compiled={compiled_repeat_abs:.3e}; "
        f"parity finite={finite} max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )
    print(f"[compile] {model.config.num_steps}-step chunk max_abs={chunk_max_abs:.3e} max_rel={chunk_max_rel:.3e}")
    if not finite or max(max_rel, chunk_max_rel) > max_relative_error:
        raise RuntimeError("compiled denoise step failed numerical parity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=None, help="override the flow-matching step count")
    parser.add_argument("--moe", choices=("gather", "dense"), default=None, help="override the MoE kernel")
    parser.add_argument(
        "--compile-denoise-step",
        action="store_true",
        help="compile predict_velocity with a fixed-shape full Inductor graph",
    )
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="torch.compile backend used by --compile-denoise-step",
    )
    parser.add_argument(
        "--compile-force-same-precision",
        action="store_true",
        help="enable Inductor force_same_precision for the compiled denoise step",
    )
    parser.add_argument(
        "--compile-emulate-precision-casts",
        action="store_true",
        help="enable Inductor emulate_precision_casts for the compiled denoise step",
    )
    parser.add_argument(
        "--compile-max-relative-error",
        type=float,
        default=1e-3,
        help="maximum velocity/chunk relative error accepted by the compile probe",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = build(Path(args.model), device, dtype, args.num_steps, args.moe)
    print(f"[build] moe_implementation={model.config.moe_implementation} num_steps={model.config.num_steps}")
    obs = observation(processor.spec, seed=0)

    with torch.inference_mode():
        if args.compile_denoise_step:
            compile_denoise_step(
                processor,
                model,
                obs,
                device,
                dtype,
                args.compile_backend,
                args.compile_force_same_precision,
                args.compile_emulate_precision_casts,
                args.compile_max_relative_error,
            )
        for _ in range(args.warmup):
            unsynced_total(processor, model, obs, device, dtype)

        runs = [one_request(processor, model, obs, device, dtype) for _ in range(args.iters)]
        plain = [unsynced_total(processor, model, obs, device, dtype) for _ in range(args.iters)]

    stages = [k for k in runs[0] if not k.startswith("_")]
    total = sum(statistics.median([r[s] for r in runs]) for s in stages)

    print(f"\n{'stage':26s} {'median ms':>10s} {'share':>7s}")
    print("-" * 46)
    for stage in stages:
        ms = statistics.median([r[stage] for r in runs]) * 1e3
        print(f"{stage:26s} {ms:10.1f} {ms / (total * 1e3) * 100:6.1f}%")
    print("-" * 46)
    print(f"{'total (synced)':26s} {total * 1e3:10.1f} {100.0:6.1f}%")
    print(f"{'total (unsynced)':26s} {statistics.median(plain) * 1e3:10.1f}")

    first = statistics.median([r["_step_first"] for r in runs]) * 1e3
    rest = statistics.median([r["_step_rest"] for r in runs]) * 1e3
    print(f"\ndenoise step 0 {first:.1f} ms, steps 1+ {rest:.1f} ms each ({model.config.num_steps} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
