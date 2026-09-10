# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two LingBot open-loop result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def compare(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline_metrics = json.loads((baseline_dir / "metrics.json").read_text())
    candidate_metrics = json.loads((candidate_dir / "metrics.json").read_text())
    baseline_outputs = np.load(baseline_dir / "predictions.npz")
    candidate_outputs = np.load(candidate_dir / "predictions.npz")

    for key in ("selected_indices", "seed", "num_valid_steps"):
        if baseline_metrics[key] != candidate_metrics[key]:
            raise ValueError(f"result sets differ in {key!r}")
    for key in ("ground_truth", "episode_ids", "frame_indices", "noise_seeds", "noise_hashes"):
        if not np.array_equal(baseline_outputs[key], candidate_outputs[key]):
            raise ValueError(f"result sets differ in predictions.npz {key!r}")

    metric_deltas = {}
    for key, baseline_value in baseline_metrics["micro"].items():
        candidate_value = candidate_metrics["micro"][key]
        metric_deltas[key] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
            "relative_percent": (
                (candidate_value - baseline_value) / baseline_value * 100.0 if baseline_value else 0.0
            ),
        }

    baseline_prediction = baseline_outputs["predictions"].astype(np.float64)
    candidate_prediction = candidate_outputs["predictions"].astype(np.float64)
    difference = np.abs(candidate_prediction - baseline_prediction)
    summary = {
        "baseline": baseline_dir.resolve().as_posix(),
        "candidate": candidate_dir.resolve().as_posix(),
        "micro_metric_deltas": metric_deltas,
        "prediction_drift": {
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "max_relative_to_baseline_max": float(difference.max() / max(np.abs(baseline_prediction).max(), 1e-12)),
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    summary = compare(Path(args.baseline), Path(args.candidate))
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
