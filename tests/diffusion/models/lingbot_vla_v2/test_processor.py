# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the LingBot-VLA 2.0 robot observation processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.lingbot_vla_v2.config import LingbotVlaV2Config
from vllm_omni.diffusion.models.lingbot_vla_v2.processor import (
    LingbotVlaV2Processor,
    RobotSpec,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return f"chat:{messages[0]['content']}"

    def __call__(self, texts, **kwargs):
        assert texts == ["chat:move"]
        length = kwargs["max_length"]
        return {
            "input_ids": torch.arange(length).unsqueeze(0),
            "attention_mask": torch.ones(1, length),
        }


class FakeImageProcessor:
    def __call__(self, image):
        assert image.shape == (3, 8, 8)
        return {
            "pixel_values": torch.full((4, 12), float(image.float().mean())),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }


def _spec() -> RobotSpec:
    robot_config = {
        "states": [
            {
                "observation.state.arm.position": {
                    "origin_keys": [
                        {"observation.state": {"start": 0, "end": 2}},
                        {"observation.state": {"start": 3, "end": 5}},
                    ]
                }
            },
            {
                "observation.state.effector.position": {
                    "origin_keys": [
                        {"observation.state": {"start": 2, "end": 3}},
                        {"observation.state": {"start": 5, "end": 6}},
                    ]
                }
            },
        ],
        "actions": [
            {
                "action.arm.position": {
                    "origin_keys": [
                        {"action": {"start": 0, "end": 2}},
                        {"action": {"start": 3, "end": 5}},
                    ]
                }
            },
            {
                "action.effector.position": {
                    "origin_keys": [
                        {"action": {"start": 2, "end": 3}},
                        {"action": {"start": 5, "end": 6}},
                    ]
                }
            },
        ],
        "images": [
            {"observation.images.camera_top": {"origin_keys": "observation.images.cam_high"}},
            {"observation.images.camera_wrist": {"origin_keys": "observation.images.cam_wrist"}},
        ],
    }
    data_config = {
        "joints": [
            {"arm.position": 4},
            {"effector.position": 2},
        ],
        "cameras": ["camera_top", "camera_wrist"],
        "norm_type": [
            {"arm.position": "identity"},
            {"effector.position": "identity"},
        ],
        "img_size": 8,
    }
    return RobotSpec.from_dicts(robot_config, data_config)


def _processor() -> tuple[LingbotVlaV2Processor, FakeTokenizer]:
    tokenizer = FakeTokenizer()
    config = LingbotVlaV2Config(
        chunk_size=3,
        max_action_dim=8,
        max_state_dim=8,
        tokenizer_max_length=5,
        max_cameras=2,
    )
    return (
        LingbotVlaV2Processor(
            _spec(),
            config,
            tokenizer=tokenizer,
            image_processor=FakeImageProcessor(),
        ),
        tokenizer,
    )


def test_preprocess_builds_fixed_model_inputs_and_masks_missing_camera():
    processor, tokenizer = _processor()
    with torch.device("meta"):
        features = processor.preprocess(
            {
                "images": {"observation.images.cam_high": np.zeros((8, 8, 3), dtype=np.uint8)},
                "state": np.arange(6, dtype=np.float32),
                "prompt": "move",
            }
        )

    assert {key: tuple(value.shape) for key, value in features.model_inputs().items()} == {
        "images": (1, 2, 4, 12),
        "img_masks": (1, 2),
        "lang_tokens": (1, 5),
        "lang_masks": (1, 5),
        "state": (1, 8),
        "image_grid_thw": (1, 2, 3),
    }
    assert features.img_masks.tolist() == [[True, False]]
    assert torch.equal(features.images[0, 1], torch.full((4, 12), -1.0))
    assert features.state.tolist() == [[0.0, 1.0, 3.0, 4.0, 2.0, 5.0, 0.0, 0.0]]
    assert features.state_mask.tolist() == [True, True, True, True, True, True, False, False]
    assert {tensor.device.type for tensor in features.model_inputs().values()} == {"cpu"}
    assert tokenizer.messages == [{"role": "user", "content": "move"}]


def test_postprocess_reassembles_the_robot_action_vector():
    processor, _ = _processor()
    features = processor.preprocess(
        {
            "images": {"observation.images.cam_high": np.zeros((8, 8, 3), dtype=np.uint8)},
            "state": np.arange(6, dtype=np.float32),
            "prompt": "move",
        }
    )
    row = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0]
    actions = torch.tensor([[row] * 3])

    output = processor.postprocess(actions, features)

    expected = np.array([[10.0, 11.0, 14.0, 12.0, 13.0, 15.0]] * 3, dtype=np.float32)
    np.testing.assert_array_equal(output["action"], expected)


def test_preprocess_rejects_observations_without_any_configured_camera():
    processor, _ = _processor()

    with pytest.raises(ValueError, match="no slot maps from"):
        processor.preprocess(
            {
                "images": {"observation.images.other": np.zeros((8, 8, 3))},
                "state": np.arange(6, dtype=np.float32),
                "prompt": "move",
            }
        )


def test_bounds_normalization_round_trips_through_public_contract():
    robot_config = {
        "states": [{"observation.state.arm.position": {"origin_keys": "observation.state"}}],
        "actions": [{"action.arm.position": {"origin_keys": "action", "subtract_state": False}}],
        "images": ["observation.images.camera_top"],
    }
    data_config = {
        "joints": [{"arm.position": 2}],
        "cameras": ["camera_top"],
        "norm_type": [{"arm.position": "bounds_99_woclip"}],
        "img_size": 8,
    }
    stats = {
        key: {"mean": [5.0, 5.0], "q01": [0.0, 0.0], "q99": [10.0, 10.0]}
        for key in ("observation.state.arm.position", "action.arm.position")
    }
    previous_device = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        spec = RobotSpec.from_dicts(robot_config, data_config, stats)
    finally:
        torch.set_default_device(previous_device)
    assert {stat.device.type for entry in spec.norm_stats.values() for stat in entry.values()} == {"cpu"}
    config = LingbotVlaV2Config(
        chunk_size=2,
        max_action_dim=2,
        max_state_dim=2,
        tokenizer_max_length=5,
        max_cameras=1,
    )
    processor = LingbotVlaV2Processor(
        spec,
        config,
        tokenizer=FakeTokenizer(),
        image_processor=FakeImageProcessor(),
    )
    features = processor.preprocess(
        {
            "images": {"observation.images.camera_top": np.zeros((8, 8, 3))},
            "state": np.array([2.5, 7.5], dtype=np.float32),
            "prompt": "move",
        }
    )

    np.testing.assert_allclose(features.state.numpy(), [[-0.5, 0.5]], atol=1e-6)
    output = processor.postprocess(torch.tensor([[[-0.5, 0.5], [-0.5, 0.5]]]), features)
    np.testing.assert_allclose(output["action"], [[2.5, 7.5], [2.5, 7.5]], atol=1e-6)
