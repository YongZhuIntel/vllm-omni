# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Is folding the expert's q/k/v into one GEMM bit-exact on the 6B checkpoint?

Phase 9 P1 concatenates ``q_proj``/``k_proj``/``v_proj`` along dim 0 and splits
the product back apart. That is the same arithmetic per output element, so the
action chunk should not move by one bit -- but "should" is how a misaligned
split ships silently, because the wrong slice still has the right shape whenever
the kv-head count divides the query-head count.

The check has to happen inside one process on one set of weights: two 6B models
do not fit in 23.9 GiB together, and comparing across processes would confound
the fusion with load order. So: load unfused, sample, fuse in place, sample
again, compare.

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase9_p1_qkv_fusion.py \
        --model /tmp/lingbot-vla-v2-perf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from spikes.lingbot_vla_v2.phase5_latency import build, observation  # noqa: E402


def chunk(processor, model, obs, device, dtype, noise) -> torch.Tensor:
    features = processor.preprocess(obs).to(device=device, dtype=dtype)
    with torch.inference_mode():
        return model.sample_actions(**features.model_inputs(), noise=noise).float().cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="/tmp/lingbot-vla-v2-perf")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--compile",
        action="store_true",
        help="run both sides through Inductor, which is the deployment default. The "
        "graph is rebuilt after the fusion rather than reused: dynamo's guards on a "
        "module attribute that changes from None to a Module are not what is under "
        "test here, and a stale graph would silently pass.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = build(Path(args.model), device, dtype, None, None, fuse_expert_qkv=False)

    attn = model.qwenvl_with_expert.qwen_expert.model.layers[0].self_attn
    assert attn.qkv_proj is None, "build() was asked for the split path and did not deliver it"

    obs = [observation(processor.spec, seed) for seed in args.seeds]
    noise = [
        torch.randn(
            (1, model.config.chunk_size, model.config.max_action_dim),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        for seed in args.seeds
    ]
    eager_predict_velocity = model.predict_velocity
    if args.compile:
        model.predict_velocity = torch.compile(
            eager_predict_velocity, backend="inductor", dynamic=False, fullgraph=True
        )
    split = [chunk(processor, model, o, device, dtype, n) for o, n in zip(obs, noise)]

    fused_layers = model.fuse_expert_qkv()
    print(f"[p1] fused {fused_layers} expert layers in place")
    assert attn.qkv_proj is not None
    if args.compile:
        torch._dynamo.reset()
        model.predict_velocity = torch.compile(
            eager_predict_velocity, backend="inductor", dynamic=False, fullgraph=True
        )

    worst = 0.0
    for seed, o, n, before in zip(args.seeds, obs, noise, split):
        after = chunk(processor, model, o, device, dtype, n)
        delta = (after - before).abs().max().item()
        worst = max(worst, delta)
        print(f"[p1] seed {seed}: max|delta| = {delta:.3e}  {'EXACT' if delta == 0.0 else 'MOVED'}")

    print(f"[p1] worst over {len(args.seeds)} seeds: {worst:.3e}")
    return 0 if worst == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
