#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send a synthetic RobotWin observation to the LingBot OpenPI endpoint."""

from __future__ import annotations

import argparse
import json
import uuid

import numpy as np
from websockets.sync.client import connect

from vllm_omni.entrypoints.openpi.connection import (
    MAX_OPENPI_PAYLOAD_BYTES,
    _pack,
    _unpack,
)

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    observation = {
        "images": {key: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for key in CAMERA_KEYS},
        "state": np.zeros(14, dtype=np.float32),
        "prompt": args.prompt,
        "session_id": str(uuid.uuid4()),
    }
    uri = f"ws://{args.host}:{args.port}/v1/realtime/robot/openpi"
    with connect(uri, max_size=MAX_OPENPI_PAYLOAD_BYTES) as websocket:
        handshake = websocket.recv()
        if isinstance(handshake, str):
            try:
                error = json.loads(handshake)
            except json.JSONDecodeError:
                error = handshake
            raise RuntimeError(f"OpenPI endpoint did not return MessagePack metadata: {error}")
        metadata = _unpack(handshake)
        print(f"metadata={metadata}")
        for step in range(args.num_steps):
            websocket.send(_pack(observation))
            response = _unpack(websocket.recv())
            if isinstance(response, dict) and response.get("type") == "error":
                raise RuntimeError(response.get("message", "OpenPI inference failed"))
            actions = np.asarray(response, dtype=np.float32)
            if not np.isfinite(actions).all():
                raise RuntimeError("server returned non-finite actions")
            print(f"step={step} shape={actions.shape} mean={actions.mean():.6f} std={actions.std():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
