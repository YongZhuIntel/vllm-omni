# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[4] / "examples/offline_inference/lingbot_vla_v2/compare_open_loop.py"
SPEC = importlib.util.spec_from_file_location("lingbot_compare_open_loop", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _result(path: Path, prediction: np.ndarray, mse: float) -> None:
    path.mkdir()
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "selected_indices": [0],
                "seed": 1234,
                "num_valid_steps": 1,
                "micro": {"mse": mse, "mae": mse},
            }
        )
    )
    np.savez_compressed(
        path / "predictions.npz",
        predictions=prediction,
        ground_truth=np.zeros_like(prediction),
        episode_ids=np.array([0]),
        frame_indices=np.array([0]),
        noise_seeds=np.array([1234]),
        noise_hashes=np.array(["hash"]),
    )


def test_compare_reports_metric_and_prediction_drift(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _result(baseline, np.ones((1, 1, 14)), 2.0)
    _result(candidate, np.full((1, 1, 14), 1.5), 2.5)

    summary = MODULE.compare(baseline, candidate)

    assert summary["micro_metric_deltas"]["mse"]["relative_percent"] == pytest.approx(25.0)
    assert summary["prediction_drift"]["max_abs"] == pytest.approx(0.5)
