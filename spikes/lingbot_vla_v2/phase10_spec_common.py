# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 10 — FLASH's verify algebra. **Now a shim over the shipped code.**

This file held the port while it was being validated. It has since moved into
``vllm_omni/diffusion/models/lingbot_vla_v2/spec_decode.py`` (Phase 10.4), and
what is left here is a re-export, on purpose:

* ``phase10_port_exactness.py`` compares these functions against the FLASH
  originals with ``torch.equal`` on 200 random inputs. Pointing that probe at the
  production module means the **shipped** accept rule is the one under test, not
  a copy of it that can drift.
* ``phase10_verify_oracle_probe.py`` and ``phase10_spec_runtime.py`` keep
  importing from here, so the spike log's numbers still refer to the same code.

The two deliberate divergences from FLASH (gripper dims derived from the
``RobotSpec`` instead of hardcoded index 6, and the accept radius measured over
the ``arm.position`` group's real slots) are documented at the top of
``spec_decode.py``.
"""

from __future__ import annotations

from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import (
    GRIPPER_GROUP_MARKERS,
    action_group_slots,
    build_x_t,
    gripper_dims,
    pose_dims,
    radius_prefix_acceptance,
    stitch_prefix,
    truncate_on_gripper_switch,
    x0_from_velocity,
)

__all__ = [
    "GRIPPER_GROUP_MARKERS",
    "action_group_slots",
    "build_x_t",
    "gripper_dims",
    "pose_dims",
    "radius_prefix_acceptance",
    "stitch_prefix",
    "truncate_on_gripper_switch",
    "x0_from_velocity",
]
