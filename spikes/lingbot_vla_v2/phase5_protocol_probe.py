# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M5 step 6 — exercise the OpenPI endpoint's failure paths against a live server.

``tests/entrypoints/openai_api/test_openpi_connection.py`` covers these with a fake
websocket, which is where they belong. This script is the other half: it checks the
same behaviours survive uvicorn, the real ASGI receive loop and a real robot client,
which the unit tests cannot see.

Three things it can catch that the unit tests structurally cannot:

* a payload the *transport* drops before the application limit applies (uvicorn's
  ``ws_max_size``), which looks like a disconnect rather than an error frame;
* the msgpack-numpy ``nd``/``type``/``kind`` wire format, which no test fixture in
  this repo produces because ``_pack`` emits openpi's ``__ndarray__`` keys instead;
* whether the server keeps serving after a refused request, on the same connection.

    examples/online_serving/lingbot_vla_v2/run_openpi_server.sh   # in another shell
    PYTHONPATH=. python spikes/lingbot_vla_v2/phase5_protocol_probe.py
"""

from __future__ import annotations

import argparse
import time

import msgspec
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from vllm_omni.entrypoints.openpi.connection import MAX_OPENPI_PAYLOAD_BYTES, _pack, _unpack

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def observation(metadata: dict, rng: np.random.Generator) -> dict:
    height, width = metadata["image_resolution"]
    return {
        "images": {key: rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for key in CAMERA_KEYS},
        "state": np.zeros(metadata["action_dim"], dtype=np.float32),
        "prompt": "pick up the object",
    }


def _as_msgpack_numpy(value):
    """Re-encode an observation the way the ``msgpack-numpy`` library would.

    ``kind`` is "" for every ordinary dtype — it is *not* ``dtype.kind``. Getting
    that wrong is what made this decode path dead until step 6.
    """
    if isinstance(value, np.ndarray):
        return {b"nd": True, b"type": value.dtype.str, b"kind": b"", b"shape": value.shape, b"data": value.tobytes()}
    if isinstance(value, dict):
        return {key: _as_msgpack_numpy(item) for key, item in value.items()}
    return value


def _request(websocket, payload: bytes, label: str) -> object:
    start = time.perf_counter()
    websocket.send(payload)
    reply = _unpack(websocket.recv())
    elapsed = time.perf_counter() - start
    shape = getattr(np.asarray(reply, dtype=object), "shape", None) if not isinstance(reply, dict) else None
    print(f"  {label:38s} -> {reply if isinstance(reply, dict) else f'actions {shape}'}  [{elapsed:.3f}s]")
    return reply


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--idle-timeout", type=float, default=None, help="seconds to hold the connection open silently")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    uri = f"ws://{args.host}:{args.port}/v1/realtime/robot/openpi"
    failures: list[str] = []

    with connect(uri, max_size=MAX_OPENPI_PAYLOAD_BYTES) as websocket:
        metadata = _unpack(websocket.recv())
        print(f"handshake: {metadata}")
        obs = observation(metadata, rng)

        print("\nmalformed payloads (each must be refused, connection must survive):")
        for label, payload in (
            ("undecodable bytes", b"\xc1not-msgpack"),
            ("a list, not a mapping", msgspec.msgpack.encode([1, 2, 3])),
            ("an empty mapping", _pack({})),
            # Decodes fine, but the model's processor rejects it in the worker, so
            # this one comes back as a sanitized internal error rather than a
            # named missing key. See the note in connection.py.
            ("an observation with no state", _pack({"images": obs["images"], "prompt": "move"})),
        ):
            reply = _request(websocket, payload, label)
            if not (isinstance(reply, dict) and reply.get("type") == "error"):
                failures.append(f"{label}: expected an error frame, got {type(reply).__name__}")

        print("\nthe server still serves on the same connection:")
        reply = _request(websocket, _pack(obs), "valid observation after refusals")
        actions = np.asarray(reply, dtype=np.float32)
        if actions.shape != (metadata["action_horizon"], metadata["action_dim"]) or not np.isfinite(actions).all():
            failures.append(f"recovery request returned {actions.shape}, finite={np.isfinite(actions).all()}")

        print("\nmsgpack-numpy wire format (what a non-openpi-client robot sends):")
        reply = _request(websocket, msgspec.msgpack.encode(_as_msgpack_numpy(obs)), "nd/type/kind markers")
        if isinstance(reply, dict):
            failures.append(f"msgpack-numpy observation was refused: {reply}")
        else:
            other = np.asarray(reply, dtype=np.float32)
            if other.shape != actions.shape or not np.isfinite(other).all():
                failures.append(f"msgpack-numpy observation returned {other.shape}")

        print("\noversized payload (application limit is %.0f MiB):" % (MAX_OPENPI_PAYLOAD_BYTES / 1024**2))
        big = dict(obs)
        big["images"] = dict(obs["images"])
        big["images"]["observation.images.cam_high"] = np.zeros((4096, 4096, 3), dtype=np.uint8)  # 48 MiB
        try:
            reply = _request(websocket, _pack(big), "48 MiB observation")
            if not (isinstance(reply, dict) and reply.get("type") == "error"):
                failures.append("oversized payload was accepted")
        except ConnectionClosed as exc:
            # Expected when the frame exceeds what the transport will carry: the
            # robot sees a 1009, not the application's error frame.
            print(f"  {'48 MiB observation':38s} -> transport closed the connection: {exc.code}")

    if args.idle_timeout is not None:
        print(f"\nidle timeout (holding a fresh connection open for {args.idle_timeout:.0f}s):")
        with connect(uri, max_size=MAX_OPENPI_PAYLOAD_BYTES) as websocket:
            _unpack(websocket.recv())
            start = time.perf_counter()
            try:
                websocket.recv(timeout=args.idle_timeout)
                failures.append("idle connection sent something instead of closing")
            except ConnectionClosed as exc:
                print(f"  server closed after {time.perf_counter() - start:.1f}s (code {exc.code})")
            except TimeoutError:
                failures.append(f"idle connection still open after {args.idle_timeout:.0f}s")

    print("\n" + ("FAIL\n  " + "\n  ".join(failures) if failures else "all protocol probes passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
