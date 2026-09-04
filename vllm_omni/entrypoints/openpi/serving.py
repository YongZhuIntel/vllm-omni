# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving adapter for robot policy inference through AsyncOmni."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)

ActionOutput = np.ndarray | dict[str, np.ndarray]


def _to_builtin_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: _to_builtin_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin_container(item) for item in value]
    return value


@dataclass(frozen=True)
class PolicyServerConfig:
    """Model-specific metadata sent to an OpenPI client at connection time."""

    values: dict[str, Any]

    @classmethod
    def from_model_config(cls, model_config: Any) -> PolicyServerConfig:
        if isinstance(model_config, Mapping):
            raw_config = model_config.get("policy_server_config")
        else:
            raw_config = getattr(model_config, "policy_server_config", None)
        if raw_config is None or not isinstance(raw_config, Mapping):
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        return cls(_to_builtin_container(raw_config))

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin_container(self.values)


class ServingRealtimeRobotOpenPI:
    """Translate OpenPI observations to the v0.14.0 AsyncOmni API."""

    def __init__(self, engine_client: Any, model_name: str | None = None) -> None:
        self.engine_client = engine_client
        self.model_name = model_name
        self.policy_server_config = self._get_policy_server_config(engine_client, model_name)
        self._request_counter = count()

    @classmethod
    def create_policy_server(
        cls, engine_client: Any, model_name: str | None = None
    ) -> ServingRealtimeRobotOpenPI | None:
        try:
            return cls(engine_client=engine_client, model_name=model_name)
        except ValueError as exc:
            if "policy_server_config" not in str(exc):
                raise
            logger.info("Robot OpenPI serving disabled for model %s", model_name)
            return None

    @classmethod
    def _get_policy_server_config(cls, engine_client: Any, model_name: str | None = None) -> PolicyServerConfig:
        for stage_config in getattr(engine_client, "stage_configs", []) or []:
            if getattr(stage_config, "stage_type", None) != "diffusion":
                continue
            engine_args = getattr(stage_config, "engine_args", None)
            model_config = getattr(engine_args, "model_config", None)
            if model_config is not None:
                return PolicyServerConfig.from_model_config(model_config)
            model = getattr(engine_args, "model", None)
            if model:
                config = get_hf_file_to_dict("transformer/config.json", model)
                return PolicyServerConfig.from_model_config(config)

        model = getattr(engine_client, "model", None)
        if model:
            config = get_hf_file_to_dict("transformer/config.json", model)
            return PolicyServerConfig.from_model_config(config)
        if model_name:
            config = get_hf_file_to_dict("transformer/config.json", model_name)
            return PolicyServerConfig.from_model_config(config)
        raise ValueError("Robot OpenPI serving requires policy_server_config.")

    def reset(self, obs: dict[str, Any]) -> None:
        """Compatibility hook; connection-local state is reset by the transport."""

    def _next_request_id(self, session_id: str) -> str:
        return f"robot-{session_id}-{next(self._request_counter)}"

    async def infer(self, obs: dict[str, Any], *, session_id: str, reset: bool) -> ActionOutput:
        extra_args = {
            "reset": reset,
            "session_id": session_id,
            "robot_obs": obs,
        }
        sampling_params = OmniDiffusionSamplingParams(extra_args=extra_args)
        result = None
        async for output in self.engine_client.generate(
            prompt=obs.get("prompt", ""),
            request_id=self._next_request_id(session_id),
            sampling_params_list=[sampling_params],
        ):
            result = output
        if result is None:
            raise RuntimeError("Robot OpenPI request produced no output.")
        return self._extract_actions(result)

    @staticmethod
    def _extract_actions(result: Any) -> ActionOutput:
        current = result
        while getattr(current, "request_output", None) is not None:
            current = current.request_output
        multimodal_output = getattr(current, "multimodal_output", None)
        if not isinstance(multimodal_output, Mapping):
            raise RuntimeError("Missing multimodal_output in robot policy result")
        actions = multimodal_output.get("actions")
        if actions is None:
            raise RuntimeError("Missing multimodal_output['actions'] in robot policy result")
        if isinstance(actions, Mapping):
            return {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
        return np.asarray(actions, dtype=np.float32)


__all__ = ["PolicyServerConfig", "ServingRealtimeRobotOpenPI"]
