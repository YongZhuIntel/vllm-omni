#!/usr/bin/env python3
"""Scope the vendoring job: which upstream code does inference actually execute?

The upstream tree is ~5000 lines across the ``lingbot_vla`` package alone, most of
it training (losses, MoE metrics, align/depth/DINO heads, distributed plans). Phase
1 vendors the inference subset into ``vllm_omni/diffusion/models/lingbot_vla_v2``,
and guessing that subset by reading is how you end up dragging the training stack
along — exactly what Phase 0 had to stub out.

So measure it instead. This profiles ``sample_actions`` and records every function
that actually runs, grouped by source file, separating construction from inference:

  * ``__init__`` surface  — needed to build the module tree and load weights
  * inference surface     — needed for a forward pass

Run on the structural model (seconds, no checkpoint); the code paths are identical
to the 6B model, only the layer counts differ.

    python phase1_trace_surface.py [--structural] [--out surface.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import torch

import bootstrap


class SurfaceTracer:
    """Record ``(file, qualname)`` of every call under the watched source roots."""

    def __init__(self, roots: tuple[str, ...]):
        self.roots = roots
        self.calls: set[tuple[str, str]] = set()
        self._active = False

    def _profile(self, frame, event, arg):
        if event != "call":
            return
        filename = frame.f_code.co_filename
        for root in self.roots:
            if filename.startswith(root):
                self.calls.add((filename, frame.f_code.co_qualname))
                break

    def __enter__(self):
        self._active = True
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)
        self._active = False
        return False


def relativize(path: str, roots: tuple[str, ...]) -> str:
    for root in roots:
        if path.startswith(root):
            return os.path.relpath(path, root)
    return path


def summarize(title: str, calls: set[tuple[str, str]], roots: tuple[str, ...]) -> dict:
    by_file: dict[str, set[str]] = collections.defaultdict(set)
    for filename, qualname in calls:
        by_file[relativize(filename, roots)].add(qualname)

    print(f"\n=== {title}: {len(calls)} functions across {len(by_file)} files ===")
    for filename in sorted(by_file, key=lambda f: -len(by_file[f])):
        names = sorted(by_file[filename])
        print(f"\n  {filename}  ({len(names)})")
        # Group by owning class so the vendoring unit is obvious.
        by_owner: dict[str, list[str]] = collections.defaultdict(list)
        for name in names:
            owner, _, method = name.rpartition(".")
            by_owner[owner or "<module-level>"].append(method or name)
        for owner in sorted(by_owner):
            methods = ", ".join(sorted(by_owner[owner]))
            print(f"      {owner}: {methods}")
    return {f: sorted(v) for f, v in by_file.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structural", action="store_true", default=True)
    parser.add_argument("--full", dest="structural", action="store_false", help="use the real 6B checkpoint")
    parser.add_argument("--out", default=None, help="write the surface to this JSON file")
    args = parser.parse_args()

    bootstrap.setup(verbose=False)
    bootstrap.import_modeling(verbose=False)
    from phase0_torch_spike import build_inputs, build_model

    # Only trace the upstream policy source, not transformers/torch.
    roots = (str(bootstrap.LINGBOT_SRC / "lingbotvla") + os.sep,)

    with SurfaceTracer(roots) as tracer:
        model, config, processor = build_model(argparse.Namespace(structural=args.structural), torch.float32)
    construction = set(tracer.calls)

    inputs = build_inputs(model, config, processor, torch.device("cpu"), torch.float32)

    with SurfaceTracer(roots) as tracer:
        with torch.inference_mode():
            model.sample_actions(
                inputs["images"],
                inputs["img_masks"],
                inputs["lang_tokens"],
                inputs["lang_masks"],
                inputs["state"],
                noise=inputs["noise"].clone(),
                image_grid_thw=inputs["image_grid_thw"],
            )
    inference = set(tracer.calls)

    surface = {
        "construction_only": summarize(
            "CONSTRUCTION ONLY (build + load weights)", construction - inference, roots
        ),
        "inference": summarize("INFERENCE (sample_actions)", inference, roots),
    }

    all_files = set(surface["construction_only"]) | set(surface["inference"])
    print(f"\n=== total upstream files touched: {len(all_files)} ===")
    for filename in sorted(all_files):
        print(f"  {filename}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(surface, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
