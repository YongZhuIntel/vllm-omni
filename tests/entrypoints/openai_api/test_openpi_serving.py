# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace

import numpy as np

from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.outputs import OmniRequestOutput


class FakeEngine:
    def __init__(self, model: str) -> None:
        self.stage_configs = [
            SimpleNamespace(
                stage_type="diffusion",
                engine_args=SimpleNamespace(model=model),
            )
        ]
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        yield OmniRequestOutput.from_diffusion(
            request_id=kwargs["request_id"],
            images=[],
            multimodal_output={"actions": np.ones((50, 14), dtype=np.float32)},
            final_output_type="actions",
        )


def _prepared_model(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    config = {
        "policy_server_config": {
            "action_dim": 14,
            "action_horizon": 50,
            "action_space": "joint_position",
        }
    }
    (transformer / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_infer_reads_prepared_config_and_forwards_observation(tmp_path):
    engine = FakeEngine(str(_prepared_model(tmp_path)))
    serving = ServingRealtimeRobotOpenPI(engine, model_name="lingbot")
    observation = {"state": np.zeros(14), "prompt": "move"}

    actions = asyncio.run(serving.infer(observation, session_id="session-a", reset=True))

    assert serving.policy_server_config.to_dict()["action_dim"] == 14
    assert actions.shape == (50, 14)
    call = engine.calls[0]
    assert call["prompt"] == "move"
    assert call["request_id"] == "robot-session-a-0"
    extra_args = call["sampling_params_list"][0].extra_args
    assert extra_args["robot_obs"] is observation
    assert extra_args["session_id"] == "session-a"
    assert extra_args["reset"] is True


def test_policy_server_is_disabled_without_metadata(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text("{}")

    assert ServingRealtimeRobotOpenPI.create_policy_server(FakeEngine(str(tmp_path))) is None


def test_policy_config_falls_back_to_served_model_path(tmp_path):
    engine = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(
                stage_type="diffusion",
                engine_args=SimpleNamespace(),
            )
        ]
    )

    serving = ServingRealtimeRobotOpenPI(engine, model_name=str(_prepared_model(tmp_path)))

    assert serving.policy_server_config.to_dict()["action_horizon"] == 50


def test_extract_actions_unwraps_async_omni_output():
    inner = OmniRequestOutput.from_diffusion(
        request_id="request-1",
        images=[],
        multimodal_output={"actions": np.ones((50, 14))},
        final_output_type="actions",
    )
    outer = OmniRequestOutput.from_pipeline(
        stage_id=0,
        final_output_type="image",
        request_output=inner,
    )

    actions = ServingRealtimeRobotOpenPI._extract_actions(outer)

    assert actions.shape == (50, 14)
    assert actions.dtype == np.float32


def test_openpi_websocket_route_is_registered():
    from vllm_omni.entrypoints.openai.api_server import router

    assert any(getattr(route, "path", None) == "/v1/realtime/robot/openpi" for route in router.routes)
