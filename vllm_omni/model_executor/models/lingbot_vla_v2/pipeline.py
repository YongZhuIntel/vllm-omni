# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LingBot-VLA 2.0 single-stage policy topology.

A prepared LingBot directory ships a ``model_index.json`` and no root
``config.json``, so it is auto-detected through ``diffusers_class_name`` rather
than an inferred ``model_type``. Registering this topology is also what lets a
``--deploy-config`` file apply: without a resolved ``PipelineConfig`` the deploy
overrides (including ``model_config.policy_server_config``, which turns on the
OpenPI WebSocket endpoint) are silently dropped.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

LINGBOT_VLA_V2_PIPELINE = PipelineConfig(
    model_type="lingbot_vla_v2",
    model_arch="LingbotVlaV2Pipeline",
    diffusers_class_name="LingbotVlaV2Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="actions",
            model_arch="LingbotVlaV2Pipeline",
        ),
    ),
)
