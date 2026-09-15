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
    denoise_compile_options,
)
from vllm_omni.diffusion.models.lingbot_vla_v2.processor import (
    LingbotVlaV2Processor,
    RobotSpec,
    load_hf_processor,
)
from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import SpecDecoder, prefix_shape
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
            compile_kwargs = {"backend": "inductor", "dynamic": False, "fullgraph": True}
            options = denoise_compile_options()
            if options is not None:
                compile_kwargs["options"] = options
            self.transformer.predict_velocity = torch.compile(self.transformer.predict_velocity, **compile_kwargs)
        # Speculative decoding is one switch, and it owns a second process. It is
        # built here so the iGPU worker boots while this process loads the 6B
        # weights, rather than after. `None` means the served path below is
        # byte-for-byte today's stateless one.
        self.spec: SpecDecoder | None = None
        if self.config.spec_decode:
            self.spec = self._build_spec_decoder()
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

    def _build_spec_decoder(self) -> SpecDecoder:
        """Start the iGPU draft process and wrap it in the decoder.

        Import is local so that a build without the speculative path never pays
        for the ctypes/oneCCL module at all.
        """
        from vllm_omni.diffusion.models.lingbot_vla_v2.draft_igpu import IGpuDraftClient

        prefix_len, prefix_width = self._prefix_shape()
        draft = IGpuDraftClient(
            config=self.config,
            prefix_len=prefix_len,
            prefix_width=prefix_width,
            device=self.device,
            dtype=self.dtype,
        )
        return SpecDecoder(
            transformer=self.transformer,
            processor=self.processor,
            config=self.config,
            device=self.device,
            dtype=self.dtype,
            draft=draft,
        )

    def _prefix_shape(self) -> tuple[int, int]:
        return prefix_shape(
            self.transformer,
            self.processor,
            self._dummy_observation(""),
            device=self.device,
            dtype=self.dtype,
        )

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

        extra_args = request.sampling_params.extra_args
        num_steps = extra_args.get("num_steps", self.config.num_steps)
        session_id = extra_args.get("session_id")
        if self.spec is not None and session_id is not None:
            # Stateful path: the session's prefix KV cache decides whether this
            # tick is a full round or a speculative one. `session_id`/`reset` are
            # already carried by the OpenPI serving entrypoint, so this needed no
            # new hook.
            result = self.spec.decode(
                observation,
                session_id=str(session_id),
                reset=bool(extra_args.get("reset", False)),
                noise=self._noise(request),
                num_steps=num_steps,
            )
            return DiffusionOutput(output={"actions": self._single_action(result.actions)})

        features = self.processor.preprocess(observation)
        model_features = features.to(device=self.device, dtype=self.dtype)
        actions = self.transformer.sample_actions(
            **model_features.model_inputs(),
            noise=self._noise(request),
            num_steps=num_steps,
        )
        return DiffusionOutput(output={"actions": self._single_action(self.processor.postprocess(actions, features))})

    @staticmethod
    def _single_action(robot_actions: Mapping[str, Any]) -> Any:
        if len(robot_actions) != 1:
            raise ValueError(f"action output must resolve to one robot source key; got {sorted(robot_actions)}")
        return next(iter(robot_actions.values()))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The loader compares the returned names against this pipeline's own
        # named_parameters(), so re-qualify the transformer-relative names it
        # gets back. Without the prefix every parameter looks unloaded and the
        # strict check in DiffusersPipelineLoader fails the whole load.
        return {f"transformer.{name}" for name in self.transformer.load_weights(weights)}


__all__ = ["LingbotVlaV2Pipeline"]
