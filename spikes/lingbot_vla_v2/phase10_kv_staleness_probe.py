#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 10 gate 2 — how stale is the prefix KV cache allowed to get?

FLASH's speculative rounds (`spec_pi0_pytorch.py:868`, `_get_cached_past_key_values`)
skip the VLM prefill entirely and verify against the KV cache built by the last
*full* round. That is where most of the speedup comes from — on our model
`embed_prefix` + `prefix_fill` is 80.8 ms of the 301.2 ms request — and it is
also the single assumption the whole design rests on: **that the visual context
can be several frames old without changing the chunk.**

Nothing in Phases 0-9 measured that. This does, before any draft head is trained.

Method, per anchor frame `t` and staleness `n`:

    stale arm : prefix KV from frame t   + state from frame t+n  -> 10-step chunk
    fresh arm : prefix KV from frame t+n + state from frame t+n  -> 10-step chunk

Both arms share the same noise (keyed on the target frame), so the only
difference is which observation filled the cache. Reported against the fresh
arm, and against ground truth so the numbers have a floor to be read next to:
`mae_fresh_gt` is what the model scores when nothing is stale, and a
`mae_stale_fresh` well under it means staleness is not the binding error.

Requires a **dense** bundle (consecutive frames). The committed
`adjust_bottle_3ep_2chunks.npz` strides by 50, which is exactly the thing this
probe cannot use. Build one with the `--stride` flag added to
`export_open_loop_bundle.py`:

    python examples/offline_inference/lingbot_vla_v2/export_open_loop_bundle.py \
        --lingbot-root <upstream lingbot-vla-v2 checkout> \
        --data-path ~/zhuyong/lingbovla/datasets/lerobot/adjust_bottle_demo_clean \
        --output ~/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_dense.npz \
        --episodes 0 1 2 --stride 1 --max-chunks-per-episode 200

Then, inside the container:

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_kv_staleness_probe.py \
        --model /tmp/lingbot-open-loop \
        --dataset ~/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_dense.npz

Read the result as: the largest `n` whose `mae_stale_fresh` stays under the
`mae_fresh_gt` floor is the ceiling on `periodic_full_every_n_draft_rounds`.
If that `n` is 0, speculative rounds must re-run `prefix_fill` (+80.8 ms) and
the expected speedup drops from ~6x to ~2.5x — the design survives, the
arithmetic changes. See `PHASE10_SPECULATIVE.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from open_loop_conditioning_probe import CAMERA_KEYS, make_noise  # noqa: E402
from phase5_latency import build  # noqa: E402


def jerk(chunk: np.ndarray) -> float:
    """Mean |second difference| along time -- 0 for a straight line, high for noise."""
    return float(np.abs(np.diff(chunk, n=2, axis=-2)).mean())


def load_dense_bundle(path: Path) -> dict[str, np.ndarray]:
    """Load the NPZ and refuse the strided one with an actionable message."""
    with np.load(path, allow_pickle=False) as data:
        bundle = {key: data[key] for key in data.files}
    required = {"images", "states", "actions", "prompts", "episode_ids", "frame_indices"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"bundle {path} lacks keys: {sorted(missing)}")

    # Consecutive frames within an episode are the whole point; a strided bundle
    # would silently measure staleness in units of 50 frames instead of 1.
    for episode in np.unique(bundle["episode_ids"]):
        frames = bundle["frame_indices"][bundle["episode_ids"] == episode]
        if len(frames) < 2:
            continue
        step = int(np.diff(np.sort(frames)).min())
        if step != 1:
            raise ValueError(
                f"bundle {path} has frame stride {step} in episode {episode}; this probe needs "
                "consecutive frames. Re-export with --stride 1 (see this file's docstring)."
            )
    return bundle


def episode_index(bundle: dict[str, np.ndarray]) -> dict[int, dict[int, int]]:
    """``episode -> {frame_index: row}`` so a staleness offset can be resolved."""
    rows: dict[int, dict[int, int]] = defaultdict(dict)
    for row, (episode, frame) in enumerate(zip(bundle["episode_ids"], bundle["frame_indices"], strict=True)):
        rows[int(episode)][int(frame)] = row
    return rows


def observation(bundle: dict[str, np.ndarray], row: int) -> dict[str, Any]:
    return {
        "images": {key: bundle["images"][row, camera] for camera, key in enumerate(CAMERA_KEYS)},
        "state": bundle["states"][row].astype(np.float32),
        "prompt": str(bundle["prompts"][row]),
    }


def embed_prefix_inputs(model: Any, inputs: dict[str, torch.Tensor]) -> tuple:
    """Just the encoder half: ``embed_prefix``'s six-tuple, unchanged.

    Separate from `fill_prefix` because the speculative runtime times the two
    halves apart (they are 22.6 ms and 58.1 ms) and needs `prefix_embs` itself to
    build the draft's projected prefix.
    """
    return model.embed_prefix(
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["image_grid_thw"],
    )


def fill_prefix(
    model: Any, inputs: dict[str, torch.Tensor], embedded: tuple | None = None
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """``embed_prefix`` + the Prefix walk -> everything ``denoise_actions`` needs.

    Returns the three things a speculative session would have to carry across
    ticks: the pad masks, the mrope position ids, and the KV cache itself.
    ``embedded`` lets a caller that already ran `embed_prefix_inputs` skip it.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import make_att_2d_masks

    embs, pad_masks, att_masks, position_ids, visual_masks, deepstack = (
        embed_prefix_inputs(model, inputs) if embedded is None else embedded
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
    return pad_masks, position_ids, past_key_values


def denoise_with_prefix(
    model: Any,
    prefix: tuple[torch.Tensor, torch.Tensor, list],
    state: torch.Tensor,
    noise: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    pad_masks, position_ids, past_key_values = prefix
    return model.denoise_actions(
        state=state,
        prefix_pad_masks=pad_masks,
        prefix_position_ids=position_ids,
        past_key_values=past_key_values,
        noise=noise,
        num_steps=num_steps,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="prepared model dir, e.g. /tmp/lingbot-open-loop")
    parser.add_argument("--dataset", required=True, help="dense (--stride 1) open-loop NPZ bundle")
    parser.add_argument("--device", default="xpu")
    # fp16, matching run_open_loop_eval.sh; see PHASE7_NUMERICS.md for why not bf16.
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--staleness", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--anchors", type=int, default=8, help="anchor frames per episode")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    bundle = load_dense_bundle(Path(args.dataset).expanduser())
    rows_by_episode = episode_index(bundle)
    processor, model = build(Path(args.model), device, dtype, args.num_steps)

    # Anchors are spread across each episode so the result is not dominated by
    # one phase of the motion; the largest staleness has to stay in-episode.
    max_n = max(args.staleness)
    anchors: list[tuple[int, int]] = []
    for episode, frames in sorted(rows_by_episode.items()):
        available = sorted(frame for frame in frames if frame + max_n in frames)
        if not available:
            continue
        picks = np.linspace(0, len(available) - 1, num=min(args.anchors, len(available)))
        anchors.extend((episode, available[int(round(index))]) for index in picks)
    if not anchors:
        raise ValueError(f"no anchor frame survives a staleness of {max_n}; use smaller --staleness")
    print(f"[setup] {len(anchors)} anchors across {len(rows_by_episode)} episodes, staleness {args.staleness}")

    def chunk_for(row: int, prefix: tuple | None) -> tuple[np.ndarray, tuple]:
        """Sample one chunk for bundle row ``row``; build its prefix if not given."""
        # Move first, then use the same moved features for postprocess -- the
        # pattern the other open-loop probes use; the masks have to be on-device.
        features = processor.preprocess(observation(bundle, row)).to(device=device, dtype=dtype)
        inputs = features.model_inputs()
        own_prefix = fill_prefix(model, inputs) if prefix is None else prefix
        noise = torch.as_tensor(make_noise(args.seed, row), device=device, dtype=dtype)
        with torch.no_grad():
            actions = denoise_with_prefix(model, own_prefix, inputs["state"], noise, args.num_steps)
        raw = next(iter(processor.postprocess(actions, features).values()))
        return np.asarray(raw, dtype=np.float32), own_prefix

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for episode, anchor_frame in anchors:
            anchor_row = rows_by_episode[episode][anchor_frame]
            # One prefix fill per anchor, reused for every staleness -- exactly
            # what a speculative session does with its cached KV.
            _, stale_prefix = chunk_for(anchor_row, None)
            for n in args.staleness:
                target_frame = anchor_frame + n
                if target_frame not in rows_by_episode[episode]:
                    continue
                target_row = rows_by_episode[episode][target_frame]
                stale, _ = chunk_for(target_row, stale_prefix)
                fresh, _ = chunk_for(target_row, None)
                gt = bundle["actions"][target_row].astype(np.float32)
                results.append(
                    {
                        "episode": episode,
                        "anchor_frame": anchor_frame,
                        "staleness": n,
                        "mae_stale_fresh": float(np.abs(stale - fresh).mean()),
                        "mae_stale_gt": float(np.abs(stale - gt).mean()),
                        "mae_fresh_gt": float(np.abs(fresh - gt).mean()),
                        "jerk_stale": jerk(stale),
                        "jerk_fresh": jerk(fresh),
                    }
                )

    print(f"\n{'n':>4s} {'frames':>7s} {'mae(stale,fresh)':>17s} {'mae(stale,gt)':>14s} "
          f"{'mae(fresh,gt)':>14s} {'jerk stale':>11s} {'jerk fresh':>11s}")
    print("-" * 84)
    for n in args.staleness:
        rows = [row for row in results if row["staleness"] == n]
        if not rows:
            continue
        mean = lambda key: float(np.mean([row[key] for row in rows]))  # noqa: E731
        print(f"{n:4d} {len(rows):7d} {mean('mae_stale_fresh'):17.5f} {mean('mae_stale_gt'):14.5f} "
              f"{mean('mae_fresh_gt'):14.5f} {mean('jerk_stale'):11.5f} {mean('jerk_fresh'):11.5f}")

    print(
        "\nRead: `mae(stale,fresh)` is the cost of reusing the cache; `mae(fresh,gt)` is the\n"
        "floor the model already carries. Staleness is affordable while the first stays\n"
        "below the second -- that n is the ceiling on periodic_full_every_n_draft_rounds."
    )

    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2) + "\n")
        print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
