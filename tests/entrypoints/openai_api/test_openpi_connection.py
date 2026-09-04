# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import msgspec
import numpy as np
import pytest

from vllm_omni.entrypoints.openpi import connection as connection_module
from vllm_omni.entrypoints.openpi.connection import (
    MAX_OPENPI_PAYLOAD_BYTES,
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


class SilentWebSocket(FakeWebSocket):
    """A client that connects and then never says anything."""

    def __init__(self):
        super().__init__([])

    async def receive(self):
        await asyncio.Event().wait()


class FakeServing:
    policy_server_config = PolicyServerConfig({"action_dim": 14, "action_horizon": 50})

    def __init__(self, error=None):
        self.calls = []
        self.resets = []
        self._error = error

    async def infer(self, observation, *, session_id, reset):
        self.calls.append((observation, session_id, reset))
        if self._error is not None:
            raise self._error
        return np.ones((50, 14), dtype=np.float32)

    def reset(self, observation):
        self.resets.append(observation)


def _observation(session_id="a"):
    return {"state": np.zeros(14), "prompt": "move", "session_id": session_id}


def _run(websocket, serving, **kwargs):
    asyncio.run(RobotRealtimeConnection(websocket, serving, **kwargs).handle_connection())
    return [_unpack(frame) for frame in websocket.sent[1:]]  # frame 0 is the handshake


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


def test_idle_connection_is_closed_without_an_error_frame():
    """A robot that stops sending must not hold the single-GPU policy forever.

    The close is the point: an abandoned session pins the worker, and there is
    only one of it.
    """
    websocket = SilentWebSocket()
    serving = FakeServing()

    replies = _run(websocket, serving, idle_timeout=0.01)

    assert websocket.closed is True
    assert replies == []  # a timeout is not an error to report to a gone client
    assert serving.calls == []


def test_oversized_payload_is_rejected_before_it_is_decoded(monkeypatch):
    """The size check has to precede the decode, or the limit buys nothing.

    A valid-but-too-large observation is what discriminates: garbage would be
    refused either way, so it would not prove the order.
    """
    accepted = _pack(_observation())
    oversized = _pack({**_observation(), "images": {"cam": np.zeros((64, 64, 3), dtype=np.uint8)}})
    limit = (len(accepted) + len(oversized)) // 2
    monkeypatch.setattr(connection_module, "MAX_OPENPI_PAYLOAD_BYTES", limit)
    assert len(accepted) <= limit < len(oversized)
    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": oversized},
            {"type": "websocket.receive", "bytes": accepted},
            {"type": "websocket.disconnect"},
        ]
    )
    serving = FakeServing()

    replies = _run(websocket, serving)

    assert replies[0] == {"type": "error", "message": "Invalid request payload"}
    # Rejected, but the connection survived and the next request still ran.
    assert len(serving.calls) == 1
    np.testing.assert_array_equal(replies[1], np.ones((50, 14)))


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"\xc1not-msgpack-at-all", id="undecodable"),
        pytest.param(msgspec.msgpack.encode([1, 2, 3]), id="not-a-mapping"),
        pytest.param(msgspec.msgpack.encode("just a string"), id="scalar"),
    ],
)
def test_malformed_payloads_are_refused_and_the_session_continues(payload):
    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": payload},
            {"type": "websocket.receive", "bytes": _pack(_observation())},
            {"type": "websocket.disconnect"},
        ]
    )
    serving = FakeServing()

    replies = _run(websocket, serving)

    assert replies[0] == {"type": "error", "message": "Invalid request payload"}
    assert len(serving.calls) == 1
    np.testing.assert_array_equal(replies[1], np.ones((50, 14)))


def test_text_frames_are_ignored():
    """OpenPI is a binary protocol; a stray text frame is not an error."""
    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "text": "hello?"},
            {"type": "websocket.receive", "bytes": _pack(_observation())},
            {"type": "websocket.disconnect"},
        ]
    )
    serving = FakeServing()

    replies = _run(websocket, serving)

    assert len(replies) == 1
    np.testing.assert_array_equal(replies[0], np.ones((50, 14)))


def test_reset_needs_no_observation():
    """A reset carries the endpoint marker and nothing else, and must stay valid.

    Anything that inspects an infer request's contents has to skip this path —
    a reset has no observation to inspect.
    """
    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": _pack({"endpoint": "reset"})},
            {"type": "websocket.disconnect"},
        ]
    )
    serving = FakeServing()

    replies = _run(websocket, serving)

    assert replies == [{"status": "reset successful"}]
    assert serving.resets == [{}]


def test_inference_errors_are_sanitized():
    """The client is told inference failed, not where the checkpoint lives."""
    secret = "/srv/private/checkpoints/lingbot-vla-v2-6b/model-00001.safetensors"
    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": _pack(_observation())},
            {"type": "websocket.disconnect"},
        ]
    )
    serving = FakeServing(error=RuntimeError(f"failed to mmap {secret}"))

    replies = _run(websocket, serving)

    assert replies == [{"type": "error", "message": "Internal inference error"}]
    assert secret not in str(replies)


def test_inference_failure_does_not_end_the_session():
    """One bad chunk must not cost the robot its connection and its warm worker."""
    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": _pack(_observation())},
            {"type": "websocket.receive", "bytes": _pack(_observation())},
            {"type": "websocket.disconnect"},
        ]
    )
    serving = FakeServing(error=ValueError("transient"))

    replies = _run(websocket, serving)

    assert replies == [{"type": "error", "message": "Internal inference error"}] * 2
    assert len(serving.calls) == 2


def test_decodes_msgpack_numpy_wire_format():
    """Interop with the official ``openpi-client``, which packs via ``msgpack_numpy``.

    That encoder emits ``nd``/``type``/``kind``/``data`` keys as *bytes*, not the
    ``__ndarray__`` keys this server's own ``_pack`` produces, so the two markers
    are separate decode paths and only one of them is exercised by a round trip.
    """
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    wire = msgspec.msgpack.encode(
        {
            "state": {
                b"nd": True,
                b"type": array.dtype.str,
                b"kind": b"",
                b"shape": array.shape,
                b"data": array.tobytes(),
            },
            "scalar": {b"nd": False, b"type": "<f4", b"kind": b"", b"data": np.float32(2.5).tobytes()},
        }
    )

    decoded = _unpack(wire)

    np.testing.assert_array_equal(decoded["state"], array)
    assert decoded["state"].dtype == np.float32
    assert decoded["scalar"] == np.float32(2.5)


def test_structured_msgpack_numpy_arrays_are_refused():
    """``kind="V"`` means ``type`` is a descr list, not a dtype string."""
    wire = msgspec.msgpack.encode(
        {b"nd": True, b"type": [["a", "<f4"]], b"kind": b"V", b"shape": (1,), b"data": b"\x00\x00\x00\x00"}
    )
    with pytest.raises(ValueError, match="structured arrays"):
        _unpack(wire)


@pytest.mark.parametrize("dtype", ["O", "V8", "c8"])
def test_object_and_void_dtypes_are_refused(dtype):
    """Decoding these is arbitrary-code and ill-defined territory; refuse both ways."""
    dtype_obj = np.dtype(dtype)
    wire = msgspec.msgpack.encode(
        {b"__ndarray__": True, b"data": b"\x00" * 16, b"dtype": dtype_obj.str, b"shape": (2,)}
    )
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _unpack(wire)
    if dtype_obj.kind == "c":  # only complex is constructible enough to pack
        with pytest.raises((ValueError, TypeError)):
            _pack({"x": np.zeros(2, dtype=dtype_obj)})


def test_payload_limit_is_not_larger_than_the_transport_allows():
    """The client passes this limit to ``websockets`` as ``max_size``.

    Uvicorn's own default ``ws_max_size`` is 16 MiB, so a payload between the two
    limits is dropped by the transport before this module ever sees it. Keeping
    the constant at or below 16 MiB is what makes the application-level check the
    binding one, and the error the client gets a protocol error rather than a
    silent disconnect.
    """
    assert MAX_OPENPI_PAYLOAD_BYTES <= 16 * 1024 * 1024
