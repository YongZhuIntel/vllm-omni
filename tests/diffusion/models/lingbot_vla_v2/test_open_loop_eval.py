# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[4] / "examples/offline_inference/lingbot_vla_v2/open_loop_eval.py"
SPEC = importlib.util.spec_from_file_location("lingbot_open_loop_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_prediction_metrics_use_robotwin_joint_and_gripper_indices():
    ground_truth = np.zeros((1, 2, 14), dtype=np.float32)
    prediction = ground_truth.copy()
    prediction[..., 0] = 2.0
    prediction[..., 6] = 4.0

    metrics = MODULE.prediction_metrics(ground_truth, prediction)

    assert metrics["mse_joint"] == pytest.approx(4.0 / 12.0)
    assert metrics["mae_joint"] == pytest.approx(2.0 / 12.0)
    assert metrics["mse_gripper"] == pytest.approx(16.0 / 2.0)
    assert metrics["mae_gripper"] == pytest.approx(4.0 / 2.0)


def test_aggregate_metrics_reports_macro_and_micro():
    ground_truth = np.zeros((3, 1, 14), dtype=np.float32)
    prediction = ground_truth.copy()
    prediction[0] = 1.0
    prediction[1:] = 3.0
    summary = MODULE.aggregate_metrics(
        ground_truth,
        prediction,
        np.array([0, 1, 1]),
    )

    assert summary["num_episodes"] == 2
    assert summary["micro"]["mse"] == pytest.approx(19.0 / 3.0)
    assert summary["macro"]["mse"] == pytest.approx(5.0)


def test_prediction_metrics_report_the_openvino_metric_set():
    ground_truth = np.zeros((1, 4, 14), dtype=np.float32)
    prediction = ground_truth.copy()
    prediction[0, 0, 0] = 5.0  # one outlier, so max and p99 can diverge from the mean
    prediction[0, 1:, 0] = 1.0

    metrics = MODULE.prediction_metrics(ground_truth, prediction)

    assert metrics["max_abs"] == pytest.approx(5.0)
    # Metrics are computed in float64, so the expectation has to be too.
    assert metrics["p99_abs"] == pytest.approx(np.percentile(np.abs(prediction.astype(np.float64)), 99))
    assert metrics["max_abs_joint"] == pytest.approx(5.0)
    assert metrics["max_abs_gripper"] == 0.0
    # Ground truth is all zeros, so its norm is clipped and cosine collapses to
    # zero rather than dividing by it.
    assert metrics["cosine_mean"] == 0.0


def test_cosine_uses_whole_chunks_and_ignores_padding():
    ground_truth = np.zeros((2, 3, 14), dtype=np.float32)
    ground_truth[:, :, 0] = 1.0
    prediction = ground_truth.copy()
    # Sample 0 is exact within its valid step; sample 1 points the other way.
    prediction[1, :2, 0] = -1.0
    # Garbage in both padded tails must not reach the metric.
    prediction[0, 1:] = 100.0
    prediction[1, 2:] = 100.0

    cosine = MODULE.cosine_per_sample(ground_truth, prediction, valid_steps=np.array([1, 2]))

    assert cosine.tolist() == pytest.approx([1.0, -1.0])


def test_metrics_ignore_padded_episode_tail():
    ground_truth = np.zeros((2, 3, 14), dtype=np.float32)
    prediction = ground_truth.copy()
    prediction[0, 1:] = 100.0
    prediction[1, 2:] = 100.0

    summary = MODULE.aggregate_metrics(
        ground_truth,
        prediction,
        np.array([0, 1]),
        valid_steps=np.array([1, 2]),
    )

    assert summary["num_valid_steps"] == 3
    assert summary["micro"]["mse"] == 0.0
    assert summary["macro"]["mae"] == 0.0


def test_bundle_validation_and_deterministic_noise(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez_compressed(
        path,
        images=np.zeros((2, 3, 8, 8, 3), dtype=np.uint8),
        states=np.zeros((2, 14), dtype=np.float32),
        actions=np.zeros((2, 50, 14), dtype=np.float32),
        prompts=np.array(["a", "b"]),
        episode_ids=np.array([0, 1]),
        frame_indices=np.array([0, 50]),
        valid_steps=np.array([50, 7]),
    )

    bundle = MODULE.load_bundle(path)
    first, first_hash = MODULE.make_noise(1234, 1)
    second, second_hash = MODULE.make_noise(1234, 1)

    assert bundle.num_samples == 2
    assert bundle.valid_steps.tolist() == [50, 7]
    np.testing.assert_array_equal(first, second)
    assert first_hash == second_hash


def test_old_bundle_defaults_to_full_horizon(tmp_path):
    path = tmp_path / "old_bundle.npz"
    np.savez_compressed(
        path,
        images=np.zeros((1, 3, 8, 8, 3), dtype=np.uint8),
        states=np.zeros((1, 14), dtype=np.float32),
        actions=np.zeros((1, 50, 14), dtype=np.float32),
        prompts=np.array(["move"]),
        episode_ids=np.array([0]),
        frame_indices=np.array([0]),
    )

    bundle = MODULE.load_bundle(path)

    assert bundle.valid_steps.tolist() == [50]


def test_episode_plot_uses_only_valid_steps(tmp_path):
    # Plotting is behind the opt-in --plots flag, so matplotlib is not a
    # declared dependency (write_episode_plots raises its own error without it).
    pytest.importorskip("matplotlib", reason="matplotlib required for --plots output")

    ground_truth = np.zeros((1, 3, 14), dtype=np.float32)
    prediction = ground_truth.copy()
    prediction[:, 2] = 100.0

    MODULE.write_episode_plots(
        tmp_path,
        ground_truth,
        prediction,
        episode_ids=np.array([7]),
        valid_steps=np.array([2]),
    )

    assert (tmp_path / "plots/episode_7.png").stat().st_size > 0
