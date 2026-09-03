# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Action-output contract tests for the diffusion engine."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, supports_action_output
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def test_lingbot_pipeline_is_registered_as_action_output():
    assert supports_action_output("LingbotVlaV2Pipeline") is True


def test_step_returns_actions_as_multimodal_output():
    actions = np.arange(12, dtype=np.float32).reshape(3, 4)
    request = OmniDiffusionRequest(
        prompts=["move"],
        sampling_params=OmniDiffusionSamplingParams(),
        request_ids=["request-1"],
    )
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(model_class_name="ActionPipeline")
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.add_req_and_wait_for_response = Mock(
        return_value=SimpleNamespace(
            output=actions,
            error=None,
            trajectory_timesteps=None,
            trajectory_latents=None,
        )
    )

    with patch(
        "vllm_omni.diffusion.diffusion_engine.supports_action_output",
        return_value=True,
    ):
        outputs = engine.step(request)

    assert len(outputs) == 1
    assert outputs[0].request_id == "request-1"
    assert outputs[0].images == []
    assert outputs[0].final_output_type == "actions"
    np.testing.assert_array_equal(outputs[0].multimodal_output["actions"], actions)
