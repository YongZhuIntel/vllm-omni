# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the LingBot-VLA 2.0 diffusion pipeline."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2 import (
    LingbotVlaV2Pipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def _request_batch(prompt: str, sampling_params: OmniDiffusionSamplingParams) -> DiffusionRequestBatch:
    """Wrap one request the way the diffusion runner invokes ``forward()``."""
    return DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id="test-lingbot-vla-v2",
            )
        ]
    )


class FakeFeatures:
    def __init__(self) -> None:
        self.moved_to = None

    def to(self, *, device=None, dtype=None):
        self.moved_to = (device, dtype)
        return self

    def model_inputs(self):
        return {"state": torch.zeros(1, 4)}


class FakeProcessor:
    def __init__(self, spec, config, *, tokenizer, image_processor) -> None:
        self.spec = spec
        self.observation = None
        self.features = FakeFeatures()

    def preprocess(self, observation):
        self.observation = observation
        return self.features

    def postprocess(self, actions, features):
        assert features is self.features
        return {"action": actions.detach().cpu().numpy()[0]}


class FakePolicy:
    def __init__(self, config) -> None:
        self.sample_kwargs = None

    def sample_actions(self, **kwargs):
        self.sample_kwargs = kwargs
        return torch.ones(1, 3, 4)

    def predict_velocity(self, **kwargs):
        return kwargs

    def load_weights(self, weights):
        return {name for name, _ in weights}


def _pipeline(tmp_path, *, compile_denoise_step=False) -> LingbotVlaV2Pipeline:
    config = OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="LingbotVlaV2Pipeline",
        dtype=torch.float32,
        tf_model_config=TransformerConfig.from_dict(
            {
                "robot_config": "robot.yaml",
                "data_config": "data.yaml",
                "qwen3vl_path": "qwen",
                "chunk_size": 3,
                "max_action_dim": 4,
                "max_state_dim": 4,
                "num_steps": 10,
                "compile_denoise_step": compile_denoise_step,
            }
        ),
    )
    spec = SimpleNamespace(
        image_size=8,
        camera_sources={"camera": "source_camera"},
    )
    with (
        patch(
            "vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2.get_local_device",
            return_value=torch.device("cpu"),
        ),
        patch(
            "vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2.RobotSpec.from_files",
            return_value=spec,
        ) as from_files,
        patch(
            "vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2.load_hf_processor",
            return_value=(Mock(), Mock()),
        ),
        patch(
            "vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2.LingbotVlaV2Processor",
            FakeProcessor,
        ),
        patch(
            "vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2.LingbotVlaV2ForActionPrediction",
            FakePolicy,
        ),
    ):
        pipeline = LingbotVlaV2Pipeline(od_config=config)
    from_files.assert_called_once_with(tmp_path / "robot.yaml", tmp_path / "data.yaml", None)
    return pipeline


def test_forward_uses_robot_observation_noise_and_num_steps(tmp_path):
    pipeline = _pipeline(tmp_path)
    noise = torch.zeros(1, 3, 4)
    observation = {"images": {}, "state": np.zeros(4)}
    request = _request_batch(
        "move",
        OmniDiffusionSamplingParams(extra_args={"robot_obs": observation, "noise": noise, "num_steps": 2}),
    )

    output = pipeline(request)

    assert pipeline.processor.observation["prompt"] == "move"
    assert pipeline.processor.features.moved_to == (torch.device("cpu"), torch.float32)
    assert pipeline.transformer.sample_kwargs["noise"] is noise
    assert pipeline.transformer.sample_kwargs["num_steps"] == 2
    np.testing.assert_array_equal(output.output["actions"], np.ones((3, 4), dtype=np.float32))


def test_forward_synthesizes_warmup_observation(tmp_path):
    pipeline = _pipeline(tmp_path)
    request = _request_batch("dummy run", OmniDiffusionSamplingParams())

    pipeline(request)

    observation = pipeline.processor.observation
    assert observation["prompt"] == "dummy run"
    assert observation["state"].shape == (4,)
    assert observation["images"]["source_camera"].shape == (8, 8, 3)
    assert pipeline.transformer.sample_kwargs["num_steps"] == 10


def test_load_weights_delegates_to_policy(tmp_path):
    pipeline = _pipeline(tmp_path)
    loaded = pipeline.load_weights([("weight", torch.ones(1))])
    assert loaded == {"weight"}


def test_pipeline_compiles_denoise_step_only_when_enabled(tmp_path):
    compiled = Mock()
    with patch("torch.compile", return_value=compiled) as compile_mock:
        pipeline = _pipeline(tmp_path, compile_denoise_step=True)

    compile_mock.assert_called_once()
    original = compile_mock.call_args.args[0]
    assert original.__self__ is pipeline.transformer
    assert original.__name__ == "predict_velocity"
    assert compile_mock.call_args.kwargs == {
        "backend": "inductor",
        "dynamic": False,
        "fullgraph": True,
    }
    assert pipeline.transformer.predict_velocity is compiled
