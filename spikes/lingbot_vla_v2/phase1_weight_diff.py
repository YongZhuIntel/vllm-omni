#!/usr/bin/env python3
"""Which side's action-expert weights disagree with the checkpoint on disk?

``phase1_parity.py``'s expert probes localised the divergence to a place that
cannot be a maths bug: at expert layer 0 the *input* to ``q_proj`` is
bit-identical between the two implementations while its *output* is not. A
``nn.Linear`` is deterministic, so the only free variable left is the parameter.

So each side is built, and every expert parameter is compared against the raw
tensor in the safetensors shards. Whichever side is not a bit-for-bit copy of
the checkpoint is the one loading weights wrong.

    python phase1_weight_diff.py            # both sides, sequentially
    python phase1_weight_diff.py --side vendored
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Enough of the layer to tell "one tensor is wrong" from "the whole layer is".
SITES = (
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "input_layernorm.weight",
    "input_layernorm.gamma.weight",
    "input_layernorm.gamma.bias",
    "post_attention_layernorm.weight",
    "mlp.gate.weight",
    "mlp.e_score_correction_bias",
    "mlp.experts.gate_proj",
    "mlp.experts.down_proj",
    "mlp.shared_expert.gate_proj.weight",
)
LAYERS = (0, 18, 35)


def load_checkpoint_tensors(checkpoint: str, wanted: set[str]) -> dict[str, torch.Tensor]:
    from safetensors.torch import safe_open

    found = {}
    for shard in sorted(Path(checkpoint).glob("*.safetensors")):
        with safe_open(shard.as_posix(), framework="pt") as handle:
            for name in handle.keys():
                key = name[len("model.") :] if name.startswith("model.") else name
                if key in wanted:
                    found[key] = handle.get_tensor(name)
    return found


def compare(side: str, tensors: dict[str, torch.Tensor], checkpoint: str) -> int:
    """``tensors`` is {module-path: live parameter} for one side."""
    reference = load_checkpoint_tensors(checkpoint, set(tensors))
    print(f"\n{'parameter':<66} {'dtype':<10} {'max|d| vs ckpt':>16}")
    print("-" * 96)
    bad = 0
    for key, live in tensors.items():
        if key not in reference:
            print(f"{key:<66} {'-':<10} {'ABSENT FROM CHECKPOINT':>16}")
            bad += 1
            continue
        ref = reference[key].to(torch.float32)
        value = live.detach().float().cpu()
        delta = (value - ref).abs().max().item()
        flag = ""
        if delta != 0.0:
            # A weight that is exactly its own bf16 rounding was round-tripped
            # through a half-precision parameter at some point, not corrupted.
            rounded = ref.to(torch.bfloat16).to(torch.float32)
            flag = "   <-- bf16-rounded" if torch.equal(value, rounded) else "   <-- differs"
        print(f"{key:<66} {str(reference[key].dtype).replace('torch.', ''):<10} {delta:>16.3e}{flag}")
        bad += delta != 0.0
    print(f"\n[{side}] {bad} of {len(tensors)} expert parameters differ from the checkpoint.")
    return bad


def names() -> list[str]:
    return [f"qwenvl_with_expert.qwen_expert.model.layers.{i}.{s}" for i in LAYERS for s in SITES]


def run_upstream(checkpoint: str, survey_all: bool = False) -> int:
    sys.path.insert(0, str(HERE))
    import bootstrap

    bootstrap.setup()
    bootstrap.import_modeling()

    import build_model

    model, _ = build_model.build_from_release(
        checkpoint, os.environ["QWEN3VL_PATH"], dtype=torch.float32
    )
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    if survey_all:
        return survey("upstream", params, checkpoint)
    live = {key: params[key] for key in names() if key in params}
    absent = [key for key in names() if key not in params]
    if absent:
        print(f"[upstream] not a parameter of the model: {absent}")
    return compare("upstream", live, checkpoint)


def run_vendored(checkpoint: str, survey_all: bool = False) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from safetensors.torch import safe_open

    from vllm_omni.diffusion.models.lingbot_vla_v2 import (
        LingbotVlaV2Config,
        LingbotVlaV2ForActionPrediction,
    )

    config = LingbotVlaV2Config.from_release_checkpoint(
        checkpoint, qwen3vl_path=os.environ["QWEN3VL_PATH"]
    )
    model = LingbotVlaV2ForActionPrediction(config).to(torch.float32).eval()

    def stream():
        for shard in sorted(Path(checkpoint).glob("*.safetensors")):
            with safe_open(shard.as_posix(), framework="pt") as handle:
                for name in handle.keys():
                    yield name, handle.get_tensor(name)

    model.load_weights(stream())
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    if survey_all:
        return survey("vendored", params, checkpoint)
    live = {key: params[key] for key in names() if key in params}
    absent = [key for key in names() if key not in params]
    if absent:
        print(f"[vendored] not a parameter of the model: {absent}")
    return compare("vendored", live, checkpoint)


def survey(side: str, params: dict, checkpoint: str) -> int:
    """Every parameter, grouped by module path with the layer index folded out."""
    import re
    from collections import Counter

    from safetensors.torch import safe_open

    exact = Counter()
    rounded = Counter()
    other = Counter()
    for shard in sorted(Path(checkpoint).glob("*.safetensors")):
        with safe_open(shard.as_posix(), framework="pt") as handle:
            for name in handle.keys():
                key = name[len("model.") :] if name.startswith("model.") else name
                live = params.get(key)
                if live is None:
                    continue
                ref = handle.get_tensor(name).to(torch.float32)
                group = re.sub(r"\.\d+\.", ".*.", key)
                value = live.detach().float().cpu()
                if torch.equal(value, ref):
                    exact[group] += 1
                elif torch.equal(value, ref.to(torch.bfloat16).to(torch.float32)):
                    rounded[group] += 1
                else:
                    other[group] += 1

    print(f"\n[{side}] parameters that are NOT a bit-for-bit copy of the checkpoint:")
    if not rounded and not other:
        print("  (none)")
    for label, counter in (("bf16-rounded", rounded), ("differs otherwise", other)):
        for group, count in sorted(counter.items()):
            print(f"  {count:>4} x {group:<64} {label}")
    print(f"[{side}] exact={sum(exact.values())} rounded={sum(rounded.values())} other={sum(other.values())}")
    return sum(rounded.values()) + sum(other.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["upstream", "vendored"])
    parser.add_argument(
        "--all", action="store_true", help="sweep every parameter instead of the probe list"
    )
    args = parser.parse_args()

    sys.path.insert(0, str(HERE))
    import bootstrap

    checkpoint = os.environ.get("LINGBOT_CKPT", bootstrap.CKPT_DIR.as_posix())
    os.environ.setdefault("QWEN3VL_PATH", bootstrap.QWEN3VL_PATH.as_posix())

    if args.side == "upstream":
        return 1 if run_upstream(checkpoint, args.all) else 0
    if args.side == "vendored":
        return 1 if run_vendored(checkpoint, args.all) else 0

    status = 0
    for side in ("upstream", "vendored"):
        print(f"\n=== {side} ===", flush=True)
        result = subprocess.run(
            [sys.executable, __file__, "--side", side] + (["--all"] if args.all else []),
            cwd=HERE,
            env=dict(os.environ),
        )
        status |= result.returncode
    return status


if __name__ == "__main__":
    raise SystemExit(main())
