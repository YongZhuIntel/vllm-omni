#!/usr/bin/env python3
"""Phase 1 gate: the vendored kernel must match upstream **per stage, in fp32**.

Phase 0 established that a bf16 action chunk carries ~10 % relative error, so
comparing final chunks in bf16 cannot resolve a real regression. And comparing
only the final chunk — even in fp32 — tells you *that* something diverged, not
where: the 10-step Euler loop mixes every stage together. So this walks the
pipeline stage by stage and reports each one separately, in fp32 on CPU.

``golden_cpu_fp32.npz`` from Phase 0 stores only the final chunk, so the upstream
side is re-run here to capture intermediates. The final chunk is still checked
against the stored golden as a cross-check that the upstream side reproduces.

Two sides, two subprocesses
---------------------------
The upstream model needs ``bootstrap.py``'s nine transformers-5 monkeypatches to
import at all. Running the vendored kernel in that same interpreter would grade
it in a patched environment — exactly the thing the port exists to eliminate. So
each side runs in its own subprocess, and the vendored side never imports
``bootstrap``: it reads the *inputs* the upstream side captured, so both see
byte-identical tensors without sharing any setup.

The upstream side is also *repaired* before it is used as a reference: its action
expert is built in bf16 and only widened to fp32 after the weights are loaded, so
the attention projections of all 36 expert layers arrive bf16-rounded. See
``repair_upstream_precision``. ``golden_cpu_fp32.npz`` predates that discovery and
was produced without the repair, so the cross-check against it now shows the size
of the defect rather than zero.

Usage
-----
    python phase1_parity.py                     # both sides + report (~2 x 26 GB, slow)
    python phase1_parity.py --num-steps 2       # faster; still exercises every stage
    python phase1_parity.py --keep /tmp/parity  # keep the captures for a re-diff
    python phase1_parity.py --report /tmp/parity/{upstream,vendored}.npz  # re-diff only
    python phase1_parity.py --raw-upstream      # grade against the unrepaired reference
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Same fixed observation Phase 0 used, so the final chunk is comparable to
# golden_cpu_fp32.npz.
IMAGE_SEED = 0
NOISE_SEED = 1234
PROMPT = "pick up the object"
NUM_CAMS = 3
IMAGE_SIZE = 224

# Layers to snapshot the KV cache at: first, middle, last. A divergence that
# starts mid-stack shows up here and nowhere else.
PROBE_LAYERS = (0, 18, 35)

# The action expert is walked only during a denoise step, so the KV-cache probes
# above say nothing about it. These four layers are probed *inside* the first
# velocity prediction, at the Linear boundaries both implementations share.
EXPERT_PROBE_LAYERS = (0, 1, 18, 35)

# Per-stage fp32 tolerance. Stages are graded on max|Δ| relative to the
# magnitude of the stage itself, so a 2560-wide embedding and a 55-wide action
# are held to the same *relative* standard.
TOLERANCE = 1e-4
# The action chunk is the output of 10 chained velocity predictions, so it
# accumulates the per-step error; π0's kernel is graded at 1e-4 on the chunk too,
# which is the bar to match rather than relax.
CHUNK_TOLERANCE = 1e-4

INPUT_KEYS = ("images", "img_masks", "lang_tokens", "lang_masks", "state", "noise", "image_grid_thw")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    diff = np.abs(a64 - b64)
    scale = max(float(np.abs(b64).max()), 1e-12)
    return {
        "shape": tuple(a.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel": float(diff.max() / scale),
    }


def report(upstream_path: str, vendored_path: str, golden: str | None) -> int:
    up = np.load(upstream_path)
    ven = np.load(vendored_path)

    stages = [key for key in up.files if key.startswith("stage.")]
    shared = [key for key in stages if key in ven.files]

    print(f"\n{'stage':<28} {'shape':<22} {'max|d|':>11} {'mean|d|':>11} {'rel':>10}  verdict")
    print("-" * 98)
    failures: list[str] = []
    for key in shared:
        name = key[len("stage.") :]
        a, b = up[key], ven[key]
        if a.shape != b.shape:
            print(f"{name:<28} {str(tuple(a.shape)) + ' vs ' + str(tuple(b.shape)):<22} {'SHAPE MISMATCH':>46}")
            failures.append(name)
            continue
        if a.dtype == bool or np.issubdtype(a.dtype, np.integer):
            exact = bool((a == b).all())
            verdict = "PASS (exact)" if exact else "FAIL (exact)"
            print(f"{name:<28} {str(tuple(a.shape)):<22} {'-':>11} {'-':>11} {'-':>10}  {verdict}")
            if not exact:
                failures.append(name)
            continue
        s = _stats(a, b)
        limit = CHUNK_TOLERANCE if name == "actions" else TOLERANCE
        ok = s["rel"] <= limit
        print(
            f"{name:<28} {str(s['shape']):<22} {s['max_abs']:>11.3e} "
            f"{s['mean_abs']:>11.3e} {s['rel']:>10.2e}  {'PASS' if ok else f'FAIL (>{limit:g})'}"
        )
        if not ok:
            failures.append(name)

    missing = sorted(set(stages) - set(ven.files))
    if missing:
        print(f"\nmissing from the vendored capture: {[k[6:] for k in missing]}")
        failures.extend(k[6:] for k in missing)

    if golden and Path(golden).exists() and "stage.actions" in up.files:
        ref = np.load(golden)["actions"]
        if ref.shape == up["stage.actions"].shape:
            s = _stats(up["stage.actions"], ref)
            print(
                f"\ncross-check, upstream re-run vs {Path(golden).name}: "
                f"max|d|={s['max_abs']:.3e} rel={s['rel']:.2e}"
            )
        else:
            print(f"\ncross-check skipped: golden is {ref.shape}, this run is {up['stage.actions'].shape}")

    if failures:
        print(f"\nFAIL - {len(failures)} stage(s) outside tolerance: {failures}")
        return 1
    print(f"\nPASS - {len(shared)} stage(s) agree within {TOLERANCE:g} relative.")
    return 0


# ---------------------------------------------------------------------------
# Shared input construction (upstream side only; vendored side reloads it)
# ---------------------------------------------------------------------------
def make_inputs(processor, tokenizer_max_length: int, state_dim: int, chunk: int, action_dim: int) -> dict:
    """Reproduce ``wrappers.make_prefix_example`` plus state and noise.

    Inlined rather than imported so the capture is self-describing: this is the
    exact input contract the vendored processor (M2) has to reproduce.
    """
    generator = torch.Generator().manual_seed(IMAGE_SEED)
    image_processor = processor.image_processor
    pixel_values, grids = [], []
    for _ in range(NUM_CAMS):
        frame = torch.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8, generator=generator)
        out = image_processor(frame)
        pixel_values.append(torch.as_tensor(np.array(out["pixel_values"])))
        grids.append(torch.as_tensor(np.array(out["image_grid_thw"])).reshape(-1)[:3])

    tokens = processor.tokenizer(
        PROMPT, return_tensors="pt", padding="max_length", max_length=tokenizer_max_length, truncation=True
    )
    noise_generator = torch.Generator().manual_seed(NOISE_SEED)
    return {
        "images": torch.stack(pixel_values, 0).unsqueeze(0).float(),
        "img_masks": torch.ones(1, NUM_CAMS, dtype=torch.bool),
        "lang_tokens": tokens.input_ids,
        "lang_masks": tokens.attention_mask.bool(),
        "state": torch.zeros(1, state_dim, dtype=torch.float32),
        "noise": torch.randn(1, chunk, action_dim, generator=noise_generator),
        "image_grid_thw": torch.stack(grids, 0).unsqueeze(0).long(),
    }


# ---------------------------------------------------------------------------
# Upstream precision repair
# ---------------------------------------------------------------------------
def repair_upstream_precision(model, checkpoint: str, safe_open) -> int:
    """Restore full fp32 precision to upstream's weights before grading.

    Upstream's action-expert config hardcodes ``torch_dtype="bfloat16"``, so
    ``Qwen2ForCausalLM._from_config`` builds the expert in bf16. The layernorms
    and the MoE blocks are then *replaced* by freshly constructed fp32 modules
    (``replace_lnorm_with_adanorm``, ``_install_moe_blocks``), which is why only
    ``self_attn.{q,k,v,o}_proj`` survive as bf16 parameters. ``load_state_dict``
    then rounds the fp32 checkpoint into them, and ``build_model._finalize``'s
    ``.to(torch.float32)`` widens the *already rounded* values back out — so the
    "fp32" reference silently carries bf16 attention weights in all 36 expert
    layers. Measured: every one of those tensors is bit-identical to its own
    ``.to(bfloat16).to(float32)`` (``phase1_weight_diff.py``).

    That is a real upstream defect, not a porting question, and grading against
    it would mean reproducing it. So the reference is repaired here instead: any
    parameter that does not match the checkpoint is re-copied at full precision.
    Pass ``--raw-upstream`` to skip this and measure the defect's size.
    """
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    repaired = 0
    for shard in sorted(Path(checkpoint).glob("*.safetensors")):
        with safe_open(shard.as_posix(), framework="pt") as handle:
            for name in handle.keys():
                key = name[len("model.") :] if name.startswith("model.") else name
                param = params.get(key)
                if param is None:
                    continue
                tensor = handle.get_tensor(name).to(param.dtype)
                if not torch.equal(param.data, tensor):
                    param.data.copy_(tensor)
                    repaired += 1
    print(f"[upstream] repaired {repaired} parameter(s) that had lost precision at build time")
    return repaired


# ---------------------------------------------------------------------------
# Expert-layer probes
# ---------------------------------------------------------------------------
# Both implementations build the expert out of the same named submodules
# (``qwen_expert.model.layers.<i>.{self_attn.q_proj,self_attn.o_proj,mlp}``), so
# forward hooks on those Linears are the one instrumentation that needs no
# per-side code. The ladder they give, per layer:
#
#   q_proj in   the post-``input_layernorm`` hidden state -> AdaRMSNorm + residual
#   q_proj out  the projection itself
#   o_proj in   this layer's slice of the *joint* attention output -> mask, rope,
#               KV cache and the attention kernel
#   o_proj out  the projection itself
#   mlp   in    the post-``post_attention_layernorm`` hidden state
#   mlp   out   the MoE (routing + experts + shared expert)
#
# The first rung that diverges names the culprit.
_PROBE_SITES = ("self_attn.q_proj", "self_attn.o_proj", "mlp")


def install_expert_probes(joint, stages: dict) -> list:
    """Hook the expert's shared Linear boundaries; first call per site wins."""

    def hook_for(name: str):
        def hook(_module, args, output):
            for suffix, tensor in (("in", args[0]), ("out", output)):
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                key = f"{name}_{suffix}"
                if key not in stages and torch.is_tensor(tensor):
                    stages[key] = tensor.detach().clone()

        return hook

    handles = []
    layers = joint.qwen_expert.model.layers
    for idx in EXPERT_PROBE_LAYERS:
        for site in _PROBE_SITES:
            module = layers[idx].get_submodule(site)
            short = site.rsplit(".", 1)[-1]
            handles.append(module.register_forward_hook(hook_for(f"expL{idx}_{short}")))
    return handles


# ---------------------------------------------------------------------------
# Upstream side
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_upstream(num_steps: int | None, out_path: str, repair: bool = True) -> None:
    sys.path.insert(0, str(HERE))
    import bootstrap

    bootstrap.setup()
    modeling, _ = bootstrap.import_modeling()

    import build_model
    from lingbotvla.models import build_processor

    qwen3vl = bootstrap.QWEN3VL_PATH.as_posix()
    processor = build_processor(qwen3vl)
    fm, config = build_model.build_from_release(
        bootstrap.CKPT_DIR.as_posix(), qwen3vl, dtype=torch.float32
    )
    fm.eval()
    if repair:
        from safetensors.torch import safe_open

        repair_upstream_precision(fm, bootstrap.CKPT_DIR.as_posix(), safe_open)
    # The golden path. flex_attention is CUDA-only and the vendored kernel has no
    # flex path at all, so pin both sides to the pure-torch reference kernel.
    fm.qwenvl_with_expert.config.attention_implementation = "eager"
    if num_steps is not None:
        config.num_steps = num_steps
        fm.config.num_steps = num_steps

    inputs = make_inputs(
        processor,
        getattr(fm.config, "tokenizer_max_length", 72),
        config.max_state_dim,
        config.chunk_size,
        config.max_action_dim,
    )
    stages = {}

    flat_images = inputs["images"].reshape(-1, *inputs["images"].shape[2:])
    flat_grid = inputs["image_grid_thw"].reshape(-1, 3)
    image_embeds, deepstack = fm.qwenvl_with_expert.embed_image(flat_images, flat_grid)
    stages["image_embeds"] = image_embeds
    for i, feature in enumerate(deepstack):
        stages[f"deepstack_{i}"] = feature

    embs, pad_masks, att_masks, position_ids, visual_pos_masks, deepstack_visual = fm.embed_prefix(
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        image_grid_thw=inputs["image_grid_thw"],
    )
    stages["prefix_embs"] = embs
    stages["prefix_pad_masks"] = pad_masks
    stages["prefix_att_masks"] = att_masks
    stages["prefix_position_ids"] = position_ids
    stages["prefix_visual_masks"] = visual_pos_masks

    att_2d = modeling.make_att_2d_masks(pad_masks, att_masks)
    stages["prefix_att_2d"] = att_2d
    _, past_key_values, _ = fm.qwenvl_with_expert.forward(
        attention_mask=att_2d,
        position_ids=position_ids,
        vlm_position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[embs, None],
        use_cache=True,
        fill_kv_cache=True,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual,
    )
    for layer in PROBE_LAYERS:
        entry = past_key_values[layer]
        stages[f"kv_key_L{layer}"] = entry["key_states"]
        stages[f"kv_value_L{layer}"] = entry["value_states"]

    t1 = torch.ones(1, dtype=torch.float32)
    time_emb, suffix_embs, suffix_pad_masks, _ = fm.embed_suffix(inputs["state"], inputs["noise"], t1)
    stages["time_emb_t1"] = time_emb
    stages["suffix_embs_t1"] = suffix_embs
    full_position_ids = fm._build_full_position_ids(position_ids, pad_masks, suffix_pad_masks)
    stages["suffix_position_ids"] = full_position_ids[:, :, -suffix_pad_masks.shape[1] :]

    handles = install_expert_probes(fm.qwenvl_with_expert, stages)
    stages["v_t_step0"] = fm.predict_velocity(
        inputs["state"],
        pad_masks,
        past_key_values,
        inputs["noise"],
        t1,
        prefix_position_ids=position_ids,
    )
    for handle in handles:
        handle.remove()

    stages["actions"] = fm.sample_actions(
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["state"],
        # Upstream aliases and overwrites its noise argument; without the clone
        # every stage captured above would be read back against a mutated input.
        noise=inputs["noise"].clone(),
        image_grid_thw=inputs["image_grid_thw"],
    )
    _write(out_path, inputs, stages, num_steps=fm.config.num_steps)


# ---------------------------------------------------------------------------
# Vendored side
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_vendored(inputs_path: str, num_steps: int | None, out_path: str) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from safetensors.torch import safe_open

    from vllm_omni.diffusion.models.lingbot_vla_v2 import (
        LingbotVlaV2Config,
        LingbotVlaV2ForActionPrediction,
    )
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import make_att_2d_masks

    checkpoint = os.environ.get("LINGBOT_CKPT", "/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b")
    qwen3vl = os.environ["QWEN3VL_PATH"]

    config = LingbotVlaV2Config.from_release_checkpoint(checkpoint, qwen3vl_path=qwen3vl)
    if num_steps is not None:
        config.num_steps = num_steps
    print(f"[vendored] inferred config: {config}")

    model = LingbotVlaV2ForActionPrediction(config).to(torch.float32).eval()
    model.load_weights(_iter_checkpoint(checkpoint, safe_open))

    loaded = np.load(inputs_path)
    inputs = {key: torch.as_tensor(loaded[f"input.{key}"]) for key in INPUT_KEYS}
    stages = {}

    joint = model.qwenvl_with_expert
    flat_images = inputs["images"].reshape(-1, *inputs["images"].shape[2:])
    flat_grid = inputs["image_grid_thw"].reshape(-1, 3)
    image_embeds, deepstack = joint.embed_image(flat_images, flat_grid)
    stages["image_embeds"] = image_embeds
    for i, feature in enumerate(deepstack):
        stages[f"deepstack_{i}"] = feature

    embs, pad_masks, att_masks, position_ids, visual_pos_masks, deepstack_visual = model.embed_prefix(
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["image_grid_thw"],
    )
    stages["prefix_embs"] = embs
    stages["prefix_pad_masks"] = pad_masks
    stages["prefix_att_masks"] = att_masks
    stages["prefix_position_ids"] = position_ids
    stages["prefix_visual_masks"] = visual_pos_masks

    att_2d = make_att_2d_masks(pad_masks, att_masks)
    stages["prefix_att_2d"] = att_2d
    _, past_key_values = joint.forward(
        attention_mask=att_2d,
        position_ids=position_ids,
        inputs_embeds=[embs, None],
        past_key_values=None,
        fill_kv_cache=True,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual,
    )
    for layer in PROBE_LAYERS:
        key, value = past_key_values[layer]
        stages[f"kv_key_L{layer}"] = key
        stages[f"kv_value_L{layer}"] = value

    t1 = torch.ones(1, dtype=torch.float32)
    time_emb, suffix_embs, suffix_pad_masks, _ = model.embed_suffix(inputs["state"], inputs["noise"], t1)
    stages["time_emb_t1"] = time_emb
    stages["suffix_embs_t1"] = suffix_embs
    full_position_ids = model._build_full_position_ids(position_ids, pad_masks, suffix_pad_masks)
    stages["suffix_position_ids"] = full_position_ids[:, :, -suffix_pad_masks.shape[1] :]

    handles = install_expert_probes(joint, stages)
    stages["v_t_step0"] = model.predict_velocity(
        state=inputs["state"],
        prefix_pad_masks=pad_masks,
        prefix_position_ids=position_ids,
        past_key_values=past_key_values,
        x_t=inputs["noise"],
        timestep=t1,
    )
    for handle in handles:
        handle.remove()

    stages["actions"] = model.sample_actions(
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["state"],
        inputs["image_grid_thw"],
        noise=inputs["noise"],
    )
    _write(out_path, inputs, stages, num_steps=config.num_steps)


def _iter_checkpoint(checkpoint_dir: str, safe_open):
    """Stream ``(name, tensor)`` from every shard, one tensor at a time."""
    shards = sorted(Path(checkpoint_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {checkpoint_dir}")
    for shard in shards:
        with safe_open(shard.as_posix(), framework="pt") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def _write(path: str, inputs: dict, stages: dict, num_steps: int) -> None:
    payload = {f"input.{key}": value.cpu().numpy() for key, value in inputs.items()}
    for key, value in stages.items():
        payload[f"stage.{key}"] = value.detach().cpu().numpy()
    payload["num_steps"] = np.asarray(num_steps)
    np.savez(path, **payload)
    print(f"[capture] wrote {path}: {len(stages)} stages")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--side",
        choices=["upstream", "vendored"],
        help="internal: capture one side in this process (the driver spawns these)",
    )
    parser.add_argument("--inputs", help="internal: the upstream capture the vendored side reads")
    parser.add_argument("--out", help="internal: where this side writes its capture")
    parser.add_argument("--num-steps", type=int, default=None, help="shorten the denoise loop")
    parser.add_argument(
        "--raw-upstream",
        action="store_true",
        help="do not repair upstream's bf16-rounded expert attention weights (see "
        "repair_upstream_precision); the report then measures that defect",
    )
    parser.add_argument("--keep", help="directory to keep the two captures in")
    parser.add_argument("--report", nargs=2, metavar=("UPSTREAM", "VENDORED"), help="re-diff two captures")
    parser.add_argument(
        "--golden",
        default=str(HERE / "golden_cpu_fp32.npz"),
        help="Phase 0 golden, cross-checked against the upstream re-run",
    )
    args = parser.parse_args()

    if args.report:
        return report(args.report[0], args.report[1], args.golden)

    if args.side == "upstream":
        capture_upstream(args.num_steps, args.out, repair=not args.raw_upstream)
        return 0
    if args.side == "vendored":
        capture_vendored(args.inputs, args.num_steps, args.out)
        return 0

    workdir = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="lingbot_parity_"))
    workdir.mkdir(parents=True, exist_ok=True)
    upstream_path = workdir / "upstream.npz"
    vendored_path = workdir / "vendored.npz"

    common = [sys.executable, __file__]
    if args.num_steps is not None:
        common += ["--num-steps", str(args.num_steps)]
    if args.raw_upstream:
        common += ["--raw-upstream"]

    # The vendored side never imports bootstrap, so it cannot pick these up from
    # bootstrap's ``setdefault``; pass them through explicitly, same defaults.
    sys.path.insert(0, str(HERE))
    import bootstrap

    env = dict(os.environ)
    env.setdefault("QWEN3VL_PATH", bootstrap.QWEN3VL_PATH.as_posix())
    env.setdefault("LINGBOT_CKPT", bootstrap.CKPT_DIR.as_posix())

    # Sequential, not parallel: each fp32 side is ~26 GB resident.
    for side, extra in (
        ("upstream", ["--out", upstream_path.as_posix()]),
        ("vendored", ["--out", vendored_path.as_posix(), "--inputs", upstream_path.as_posix()]),
    ):
        print(f"\n=== capturing {side} ===", flush=True)
        result = subprocess.run(common + ["--side", side] + extra, cwd=HERE, env=env)
        if result.returncode != 0:
            print(f"{side} capture failed with exit code {result.returncode}")
            return result.returncode

    code = report(upstream_path.as_posix(), vendored_path.as_posix(), args.golden)
    if not args.keep:
        print(f"\ncaptures left in {workdir} (pass --keep to choose the location)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
