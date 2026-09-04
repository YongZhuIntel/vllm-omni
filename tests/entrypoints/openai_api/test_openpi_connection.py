# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import numpy as np

from vllm_omni.entrypoints.openpi.connection import (
    RobotRealtimeConnection,
    _pack,
    _unpack,
)
from vllm_omni.entrypoints.openpi.serving import PolicyServerConfig


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        return self.messages.pop(0)

    async def send_bytes(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True


class FakeServing:
    policy_server_config = PolicyServerConfig({"action_dim": 14, "action_horizon": 50})

    def __init__(self):
        self.calls = []
        self.resets = []

    async def infer(self, observation, *, session_id, reset):
        self.calls.append((observation, session_id, reset))
        return np.ones((50, 14), dtype=np.float32)

    def reset(self, observation):
        self.resets.append(observation)


def test_numpy_round_trip_uses_openpi_markers():
    value = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "state": np.float32(1.5),
    }
    decoded = _unpack(_pack(value))
    np.testing.assert_array_equal(decoded["image"], value["image"])
    assert decoded["state"] == value["state"]


def test_connection_sends_metadata_and_tracks_session_reset():
    messages = [
        {
            "type": "websocket.receive",
            "bytes": _pack({"state": np.zeros(14), "prompt": "move", "session_id": "a"}),
        },
        {
            "type": "websocket.receive",
            "bytes": _pack({"state": np.zeros(14), "prompt": "move", "session_id": "a"}),
        },
        {"type": "websocket.receive", "bytes": _pack({"endpoint": "reset"})},
        {"type": "websocket.disconnect"},
    ]
    websocket = FakeWebSocket(messages)
    serving = FakeServing()

    asyncio.run(RobotRealtimeConnection(websocket, serving).handle_connection())

    assert websocket.accepted is True
    assert _unpack(websocket.sent[0]) == {"action_dim": 14, "action_horizon": 50}
    assert _unpack(websocket.sent[3]) == {"status": "reset successful"}
    assert serving.calls[0][1:] == ("a", True)
    assert serving.calls[1][1:] == ("a", False)
    assert serving.resets == [{}]
    np.testing.assert_array_equal(_unpack(websocket.sent[1]), np.ones((50, 14)))
