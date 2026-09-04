# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).parents[4] / "examples/offline_inference/lingbot_vla_v2/export_open_loop_bundle.py"
SPEC = importlib.util.spec_from_file_location("lingbot_export_open_loop", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_image_conversion_accepts_lerobot_float_chw():
    image = torch.tensor([[[0.0, 1.0]], [[0.5, 0.25]], [[1.0, 0.0]]])

    converted = MODULE.image_to_uint8_hwc(image)

    assert converted.shape == (1, 2, 3)
    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted[0, 0], [0, 128, 255])


def test_valid_steps_prefers_action_padding_mask():
    sample = {"action_is_pad": torch.tensor([False, False, True, True])}

    assert MODULE.valid_action_steps(sample, horizon=4, episode_remaining=4) == 2
    assert MODULE.valid_action_steps({}, horizon=4, episode_remaining=3) == 3


def test_transform_ground_truth_uses_upstream_round_trip():
    class FakeTransform:
        def apply(self, sample):
            assert sample["token"] == "sample"
            return {"normalized": True}

        def unapply(self, transformed):
            assert transformed == {"normalized": True}
            return {
                "observation.state": torch.arange(14, dtype=torch.float32),
                "action": torch.arange(50 * 14, dtype=torch.float32).reshape(50, 14),
            }

    state, actions = MODULE.transform_ground_truth({"token": "sample"}, FakeTransform(), horizon=50)

    assert state.shape == (14,)
    assert actions.shape == (50, 14)
    assert actions.dtype == np.float32
