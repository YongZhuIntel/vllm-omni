# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-Omni diffusion-stage pipeline for LingBot-VLA 2.0."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.lingbot_vla_v2.config import LingbotVlaV2Config
from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
    LingbotVlaV2ForActionPrediction,
)
from vllm_omni.diffusion.models.lingbot_vla_v2.processor import (
    LingbotVlaV2Processor,
    RobotSpec,
    load_hf_processor,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch


class LingbotVlaV2Pipeline(nn.Module):
    """Single-request robot policy pipeline using the generic diffusion worker.

    Like π0 and GR00T, this returns ``DiffusionOutput(output={"actions": ...})``;
    the engine's output formatter promotes an ``actions`` payload key to
    ``multimodal_output["actions"]`` and leaves ``images`` empty.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        if od_config.model is None:
            raise ValueError("LingbotVlaV2Pipeline requires a prepared model directory")

        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = od_config.dtype
        raw_config = od_config.tf_model_config.to_dict()
        self.config = LingbotVlaV2Config.from_model_config(raw_config)

        model_root = Path(od_config.model)
        robot_config = self._required_path(raw_config, "robot_config", model_root)
        data_config = self._required_path(raw_config, "data_config", model_root)
        norm_stats = self._optional_path(raw_config.get("norm_stats"), model_root)
        spec = RobotSpec.from_files(robot_config, data_config, norm_stats)
        tokenizer, image_processor = load_hf_processor(self.config.qwen3vl_path)
        self.processor = LingbotVlaV2Processor(
            spec,
            self.config,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        self.transformer = LingbotVlaV2ForActionPrediction(self.config)
        joint_model = getattr(self.transformer, "qwenvl_with_expert", None)
        if joint_model is not None:
            joint_model.attention_precision = self.config.attention_precision
            joint_model.attention_backend = self.config.attention_backend
        if self.config.compile_prefix:
            self.transformer.prefix_forward = torch.compile(
                self.transformer.prefix_forward,
                backend="inductor",
                dynamic=False,
                fullgraph=True,
            )
        if self.config.compile_denoise_step:
            self.transformer.predict_velocity = torch.compile(
                self.transformer.predict_velocity,
                backend="inductor",
                dynamic=False,
                fullgraph=True,
            )
        self.vae = None
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=False,
            )
        ]

    @staticmethod
    def _optional_path(value: Any, model_root: Path) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else model_root / path

    @classmethod
    def _required_path(cls, config: Mapping[str, Any], key: str, model_root: Path) -> Path:
        if not config.get(key):
            raise ValueError(f"transformer/config.json must define {key!r}")
        path = cls._optional_path(config[key], model_root)
        assert path is not None
        return path

    @staticmethod
    def _request_prompt(request: DiffusionRequestBatch) -> str:
        prompt = request.prompts[0]
        if isinstance(prompt, str):
            return prompt
        return str(prompt.get("prompt") or "")

    def _dummy_observation(self, prompt: str) -> dict[str, Any]:
        spec = self.processor.spec
        images = {
            source: np.zeros((spec.image_size, spec.image_size, 3), dtype=np.uint8)
            for source in set(spec.camera_sources.values())
        }
        return {
            "images": images,
            "state": np.zeros(self.config.max_state_dim, dtype=np.float32),
            "prompt": prompt,
        }

    def _noise(self, request: DiffusionRequestBatch) -> torch.Tensor | None:
        params = request.sampling_params
        supplied = params.extra_args.get("noise")
        if supplied is not None:
            return torch.as_tensor(supplied, device=self.device, dtype=self.dtype)

        generator = params.generator
        if isinstance(generator, list):
            raise ValueError("LingbotVlaV2Pipeline supports one generator per request")
        if generator is None and params.seed is None:
            return None
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)
        return torch.randn(
            (1, self.config.chunk_size, self.config.max_action_dim),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.inference_mode()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.prompts) != 1:
            raise ValueError("LingbotVlaV2Pipeline supports exactly one request at a time")

        prompt = self._request_prompt(request)
        observation = request.sampling_params.extra_args.get("robot_obs")
        if observation is None:
            observation = self._dummy_observation(prompt)
        elif not isinstance(observation, Mapping):
            raise TypeError("extra_args['robot_obs'] must be a mapping")
        else:
            observation = dict(observation)
            observation.setdefault("prompt", prompt)

        features = self.processor.preprocess(observation)
        model_features = features.to(device=self.device, dtype=self.dtype)
        actions = self.transformer.sample_actions(
            **model_features.model_inputs(),
            noise=self._noise(request),
            num_steps=request.sampling_params.extra_args.get("num_steps", self.config.num_steps),
        )
        robot_actions = self.processor.postprocess(actions, features)
        if len(robot_actions) != 1:
            raise ValueError(f"action output must resolve to one robot source key; got {sorted(robot_actions)}")
        return DiffusionOutput(output={"actions": next(iter(robot_actions.values()))})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.transformer.load_weights(weights)


__all__ = ["LingbotVlaV2Pipeline"]
