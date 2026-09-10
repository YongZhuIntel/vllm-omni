# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Action-output contract tests for LingBot-VLA 2.0.

The generic ``actions`` payload contract is covered by
``tests/diffusion/test_diffusion_output_formatter.py``. These tests pin the
LingBot-specific half: the pipeline is registered, and an actions-only payload
from it reaches the client as ``multimodal_output["actions"]`` with no images.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.diffusion import output_formatter
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.output_formatter import (
    format_diffusion_outputs,
    normalize_diffusion_postprocess_output,
)
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def test_lingbot_pipeline_is_registered():
    assert _DIFFUSION_MODELS["LingbotVlaV2Pipeline"] == (
        "lingbot_vla_v2",
        "pipeline_lingbot_vla_v2",
        "LingbotVlaV2Pipeline",
    )


def test_actions_payload_becomes_multimodal_output(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    actions = np.arange(12, dtype=np.float32).reshape(3, 4)

    request = OmniDiffusionRequest(
        prompt="move",
        request_id="request-1",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )
    # What LingbotVlaV2Pipeline.forward() returns.
    postprocess_output = normalize_diffusion_postprocess_output({"actions": actions})

    outputs = format_diffusion_outputs(
        request=request,
        od_config=SimpleNamespace(model_class_name="LingbotVlaV2Pipeline"),
        diffusion_output=DiffusionOutput(output=None),
        output_data={"actions": actions},
        postprocess_output=postprocess_output,
    )

    assert len(outputs) == 1
    assert outputs[0].request_id == "request-1"
    # An actions-only payload has no primary key, so nothing lands in images.
    assert outputs[0].images == []
    np.testing.assert_array_equal(outputs[0].multimodal_output["actions"], actions)
