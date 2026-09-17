# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative action decoding for LingBot-VLA 2.0.

Ten Euler steps over a freshly filled prefix cost 294 ms on this host. This
replaces most of those ticks with a **speculative round**: keep the prefix KV
cache from the last full round, take a whole action chunk from a small draft head
running on the iGPU, and check it with ``K = len(spec_t_list)`` near-terminal
``predict_velocity`` calls instead of ten sequential ones. Measured 45.6 ms at
K=2 and 24.8 ms at K=1, i.e. 95.8 ms per tick amortised at ``spec_full_every=4``
(3.1x); see ``spikes/lingbot_vla_v2/PHASE10_SPECULATIVE.md`` §11. Those two
figures are the **sequential** verify; ``spec_verify_batched`` (default) takes
K=2's verify from 42.6 to 30.6 ms measured (§12.8), which carried through §11's
own arithmetic puts the amortised tick near 86 ms -- derived, not re-measured.

That is the **cached** scheme, and its cost is low because its verifier is
stale: the teacher reads the prefix KV of whichever tick last ran a full round.
``spec_reground`` switches to the alternative -- ground every speculative round
on its own observation, draft off the fresh embeddings, and absorb a rejection
with an Euler loop inside the rejecting tick. Nothing is stale and no tick
executes a one-step guess, but grounding is no longer amortised and a rejection
is no longer free. Which one wins is entirely a function of the accept rate;
``spikes/lingbot_vla_v2/test_spec_oneccl.sh`` sweeps both over K and acceptance.

The accept rule is a port of Dexmal RealtimeVLA-FLASH's radius-prefix
acceptance, and the flow-matching conventions line up exactly, which is why it is
a port and not a re-derivation. ``sample_actions`` states ``t=1`` is noise and
``t=0`` is the action, integrated with ``dt = -1/num_steps``; FLASH builds
``x_t = t*noise + (1-t)*x0`` and recovers ``x0_hat = x_t - t*v_t``. With ``x_t``
linear in ``t``, ``v = dx/dt = noise - x0``, so ``x_t - t*v = x0`` identically.

``spikes/lingbot_vla_v2/phase10_port_exactness.py`` checks the four functions
below against the FLASH originals on 200 random inputs with ``torch.equal`` —
not a tolerance — so that probe guards this module.

Two things here are deliberately **not** ports, because FLASH's LIBERO
assumptions do not survive the move to a 14-DoF dual-arm robot:

1. **Gripper dimensions.** FLASH hardcodes action index 6, LIBERO being 7-DoF
   with the gripper last. RoboTwin's grippers are a separate ``effector.position``
   joint group, and the action space is 55 slots with every group padded to its
   own ``max_dim``, so they are at **28 and 29** — index 6 is a left-arm joint
   here, and watching it would fail silently rather than loudly.
   ``gripper_dims()`` derives them from the ``RobotSpec``.
2. **Distance dimensions.** FLASH measures the radius over the first
   ``min(dist_dims, 6)`` dims, i.e. pose-without-gripper. The equivalent here is
   the ``arm.position`` group's real slots, which ``pose_dims()`` returns.

One FLASH guard is **not implemented**: ``enable_gripper_verify``, the any-K
pre-verify stop that zeroes the accepted prefix when any verify timestep shows a
gripper transition. Only the post-verify truncation below is ported. Behaviour
differs only on gripper transitions.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)

# Joint groups whose slots are grippers. RoboTwin names the group
# ``effector.position``; the substring match keeps this working for a robot that
# calls it ``gripper.position`` without needing a per-robot table.
GRIPPER_GROUP_MARKERS = ("effector", "gripper")

# FLASH's gripper switch threshold, and the only value the port was checked at.
GRIPPER_SWITCH_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Action-space geometry
# ---------------------------------------------------------------------------
def action_group_slots(processor: Any) -> dict[str, tuple[int, int]]:
    """``joint group name -> [start, end)`` of its **real** slots in model space.

    Mirrors ``LingbotVlaV2Processor._build_vector(values=False)``: every joint in
    ``spec.joints`` occupies ``joint.max_dim`` consecutive slots, of which the
    first ``real`` are the robot's own dims and the rest are padding.
    """
    from vllm_omni.diffusion.models.lingbot_vla_v2.processor import _action_key

    spec = processor.spec
    slots: dict[str, tuple[int, int]] = {}
    offset = 0
    for joint in spec.joints:
        parts = spec.action_slices.get(joint.name)
        if parts is not None:
            key = _action_key(joint.name)
            real = spec.norm_width(key) if key in spec.norm_stats else sum(part.width for part in parts)
            slots[joint.name] = (offset, offset + int(real))
        offset += joint.max_dim
    return slots


def _dims_for(processor: Any, wanted_gripper: bool) -> list[int]:
    dims: list[int] = []
    for name, (start, end) in action_group_slots(processor).items():
        is_gripper = any(marker in name for marker in GRIPPER_GROUP_MARKERS)
        if is_gripper == wanted_gripper:
            dims.extend(range(start, end))
    return dims


def gripper_dims(processor: Any) -> list[int]:
    """Model-space slots holding gripper commands (RoboTwin: two, one per arm)."""
    return _dims_for(processor, wanted_gripper=True)


def pose_dims(processor: Any) -> list[int]:
    """Model-space slots holding arm pose -- what the accept radius is measured on."""
    return _dims_for(processor, wanted_gripper=False)


def prefix_shape(
    transformer: Any, processor: Any, observation: dict, *, device: torch.device, dtype: torch.dtype
) -> tuple[int, int]:
    """``(prefix_len, prefix_width)`` from one ``embed_prefix`` on ``observation``.

    The prefix length follows from the checkpoint's align-query layout and camera
    count -- 286 for the release, as ``3*66 + 72 + 8 + 8`` -- so it is not a
    constant the draft worker can assume, and it has to size its buffers to it
    before the first request. Running the real code once is cheaper than
    re-deriving that layout and cannot drift from it. Only the shape is read, so
    this is valid before the weights are loaded.
    """
    features = processor.preprocess(observation)
    inputs = features.to(device=device, dtype=dtype).model_inputs()
    with torch.no_grad():
        embs = transformer.embed_prefix(
            inputs["images"], inputs["img_masks"], inputs["lang_tokens"], inputs["lang_masks"],
            inputs["image_grid_thw"],
        )[0]
    return int(embs.shape[1]), int(embs.shape[2])


# ---------------------------------------------------------------------------
# Verify algebra
# ---------------------------------------------------------------------------
def build_x_t(noise: torch.Tensor, x0_draft: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
    """``x_t = t*noise + (1-t)*x0`` -- the forward interpolation the model trained on.

    ``t`` may also be a tensor that broadcasts against the operands, which is how
    the batched verify builds all K timesteps at once. Scalar ``t`` is unchanged,
    so ``phase10_port_exactness.py`` still guards this against FLASH.
    """
    return t * noise + (1.0 - t) * x0_draft


def x0_from_velocity(x_t: torch.Tensor, t: float | torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    """``x0_hat = x_t - t*v_t`` -- one near-terminal step's estimate of the answer."""
    return x_t - t * v_t


def radius_prefix_acceptance(
    x0_draft: torch.Tensor,
    x0_hat: torch.Tensor,
    *,
    tau: float,
    dims: list[int],
    eval_h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Longest per-step prefix the draft and **every** verify timestep agree on.

    ``x0_draft`` is ``[B,H,D]``, ``x0_hat`` is ``[B,K,H,D]``. Distance is the L2
    norm over ``dims`` divided by ``sqrt(len(dims))``, so ``tau`` is a per-dimension
    RMS and does not move when the robot's DoF count does.

    Returns ``(accepted_prefix_len [B], dist [B,K,eval_h])``.
    """
    if x0_draft.ndim != 3:
        raise ValueError(f"expected x0_draft to be (B,H,D), got {tuple(x0_draft.shape)}")
    if x0_hat.ndim != 4:
        raise ValueError(f"expected x0_hat to be (B,K,H,D), got {tuple(x0_hat.shape)}")
    if not dims:
        raise ValueError("dims must be non-empty; check the RobotSpec's joint groups")

    horizon = int(x0_draft.shape[1])
    eval_h = int(min(horizon, max(1, int(eval_h))))
    index = torch.as_tensor(dims, device=x0_draft.device, dtype=torch.long)

    draft = x0_draft[:, :eval_h].index_select(-1, index).to(torch.float32)
    hat = x0_hat[:, :, :eval_h].index_select(-1, index).to(torch.float32)
    dist = torch.linalg.vector_norm(hat - draft[:, None], ord=2, dim=3) / float(len(dims)) ** 0.5

    ok = (dist <= float(tau)).to(torch.int64).cumprod(dim=2)
    # A step counts only if every verify timestep accepted it and every earlier
    # step did too -- the min over K of each arm's own prefix length.
    return ok.sum(dim=2).min(dim=1).values, dist


def _expand_rows(x: torch.Tensor, k: int, *, dim: int = 0) -> torch.Tensor:
    """Repeat each row of ``dim`` ``k`` times, b-major, folding it into ``dim``.

    ``[B, ...] -> [B*K, ...]`` with ``row = b*K + k``. The result is a copy: an
    expanded stride cannot survive the reshape, and the attention path wants a
    real tensor. That copy is the prefix KV's ``42.2 MB * K``.
    """
    if k == 1:
        return x
    shape = list(x.shape)
    return x.unsqueeze(dim + 1).expand(*shape[: dim + 1], k, *shape[dim + 1 :]).reshape(
        *shape[:dim], shape[dim] * k, *shape[dim + 1 :]
    )


def stitch_prefix(x0_draft: torch.Tensor, x0_tail: torch.Tensor, accepted_prefix_len: torch.Tensor) -> torch.Tensor:
    """Accepted steps from the draft, the rest from the verifier."""
    index = torch.arange(int(x0_draft.shape[1]), device=x0_draft.device, dtype=torch.int64)
    keep = (index[None, :] < accepted_prefix_len.to(x0_draft.device)[:, None])[:, :, None]
    return torch.where(keep, x0_draft, x0_tail)


def truncate_on_gripper_switch(
    x0_out: torch.Tensor,
    accepted_prefix_len: torch.Tensor,
    *,
    gripper_prev: torch.Tensor | None,
    dims: list[int],
    threshold: float = GRIPPER_SWITCH_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut the accepted prefix at the first gripper open/close inside it.

    A mis-timed grasp is not a small error, so FLASH refuses to execute a
    speculated gripper transition and replans instead. Generalised here to
    several gripper dims: a switch on *any* of them cuts.

    ``gripper_prev`` is ``[B, len(dims)]`` -- the last executed gripper command.
    Returns ``(truncated_len [B], cut_mask [B])``.
    """
    batch, horizon = int(x0_out.shape[0]), int(x0_out.shape[1])
    device = x0_out.device
    accepted_prefix_len = accepted_prefix_len.to(device=device, dtype=torch.int64)
    if not dims or gripper_prev is None:
        return accepted_prefix_len, torch.zeros((batch,), device=device, dtype=torch.bool)

    index = torch.as_tensor(dims, device=device, dtype=torch.long)
    curr = x0_out.index_select(-1, index).to(torch.float32)  # [B,H,G]
    prev = torch.cat([gripper_prev.to(device=device, dtype=torch.float32)[:, None, :], curr[:, :-1]], dim=1)

    crossed = ((prev < threshold) & (curr >= threshold)) | ((prev >= threshold) & (curr < threshold))
    switch = crossed.any(dim=2)  # [B,H]
    step = torch.arange(horizon, device=device, dtype=torch.int64)[None, :]
    switch = switch & (step < accepted_prefix_len[:, None])

    cut = switch.any(dim=1)
    first = switch.to(torch.int64).argmax(dim=1)
    return torch.where(cut, first, accepted_prefix_len), cut


# ---------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------
class DraftBackend(Protocol):
    """The iGPU draft, behind the process boundary.

    ``refresh`` is called once per full round with the fresh ``prefix_embs`` and
    returns the draft for that same frame -- the full round is computing the
    teacher's answer for it anyway, so the pair is the draft's accuracy, measured
    for free. ``draft`` is called once per speculative round. Both return
    ``[1, chunk_size, max_action_dim]`` on the caller's device.
    """

    def refresh(self, prefix_embs: torch.Tensor, state: torch.Tensor) -> torch.Tensor: ...

    def draft(self, state: torch.Tensor) -> torch.Tensor: ...

    def close(self) -> None: ...


@dataclass
class SpecSession:
    """What survives between ticks of one robot session.

    The serving seam already exists: ``extra_args`` carries ``session_id`` and
    ``reset`` (``entrypoints/openpi/serving.py``), so this is indexed per session
    rather than hooked in anywhere new.
    """

    pad_masks: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: list
    # The full round's features. ``postprocess`` needs their masks, which are
    # static for a robot, so a speculative round reuses them instead of running
    # the image processor again.
    features: Any
    rounds_since_full: int = 0
    pending_full: bool = False
    gripper_prev: torch.Tensor | None = None


@dataclass
class _Grounding:
    """One tick's fresh prefix, as ``embed_prefix`` + ``prefix_forward`` leave it.

    Exists so the two schemes can share the expensive half of a full round: a
    ``spec_reground`` speculative round builds one of these too, and its
    rejection fallback runs the Euler loop over the same object rather than
    paying for a second ``_ground``.
    """

    features: Any
    inputs: dict
    embs: torch.Tensor
    pad_masks: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: list


@dataclass
class SpecStats:
    """Per-round record. The pipeline ignores it; the latency sweep reads it."""

    kind: str  # "full" | "spec"
    accepted: int = 0
    eval_h: int = 0
    gripper_cut: bool = False
    rounds_since_full: int = 0
    forced: bool = False
    dist_mean: float | None = None
    draft_rms: float | None = None
    # ``spec_reground`` only: this round rejected and ran the Euler loop in its
    # own tick. Kept off ``kind`` so existing consumers keep seeing "full"/"spec",
    # and reported separately by the sweep because it is the expensive tail.
    fell_back: bool = False
    # Whether this round re-grounded on its own observation. Distinguishes the
    # two schemes in a report without the reader having to know the config.
    regrounded: bool = False


@dataclass
class DecodeResult:
    actions: Any
    stats: SpecStats = field(default_factory=lambda: SpecStats(kind="full"))


class SpecDecoder:
    """Owns the session state and decides full vs speculative, per tick.

    Construction does not touch the draft: the pipeline passes an already
    connected backend, which keeps this class testable without an iGPU.
    """

    def __init__(
        self,
        *,
        transformer: Any,
        processor: Any,
        config: Any,
        device: torch.device,
        dtype: torch.dtype,
        draft: DraftBackend,
    ) -> None:
        self.transformer = transformer
        self.processor = processor
        self.config = config
        self.device = device
        self.dtype = dtype
        self.draft = draft

        self.pose = pose_dims(processor)
        self.grippers = gripper_dims(processor)
        if not self.pose:
            raise ValueError("no pose dims found in the RobotSpec; the accept radius has nothing to measure")
        self.sessions: dict[str, SpecSession] = {}
        # The draft is optional at runtime, not at configuration time: if the
        # worker dies mid-session the policy must keep answering, so it degrades
        # to full rounds rather than failing the request.
        self.draft_failed = False
        self.rounds = {"full": 0, "spec": 0, "forced_accept": 0, "reject": 0}

        self._forced_rate = config.spec_force_accept_rate
        self._forced_rng = random.Random(0)
        self._forced_queue: deque[bool] = deque()
        if self._forced_rate is not None:
            logger.warning(
                "LingBot speculative decoding is running with spec_force_accept_rate=%.3f: the verify passes "
                "still run so the latency is real, but an accepted chunk is the draft's own and the actions "
                "are NOT valid. This is a latency-measurement mode.",
                self._forced_rate,
            )
        logger.info(
            "LingBot speculative decoding on: K=%d at %s, tau=%.3f, full round every %d speculative rounds, "
            "eval horizon %d, accept radius over %d pose dims, gripper dims %s",
            len(config.spec_t_list), config.spec_t_list, config.spec_tau, config.spec_full_every,
            config.spec_max_exec_steps, len(self.pose), self.grippers,
        )

    # -- lifecycle ---------------------------------------------------------
    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def close(self) -> None:
        self.sessions.clear()
        self.draft.close()

    def _wants_full(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if self.draft_failed or session is None:
            return True
        if self.config.spec_reground:
            # Both remaining conditions are staleness guards, and under
            # ``spec_reground`` nothing is stale: the speculative round builds its
            # own prefix from this tick's observation and absorbs its own
            # rejection. So one full round per session, to create it, and then
            # never again.
            return False
        return session.pending_full or session.rounds_since_full >= self.config.spec_full_every

    # -- rounds ------------------------------------------------------------
    def decode(
        self,
        observation: dict,
        *,
        session_id: str,
        reset: bool,
        noise: torch.Tensor | None,
        num_steps: int,
    ) -> DecodeResult:
        """One control tick. Returns robot-space actions, as ``postprocess`` does."""
        if reset:
            self.reset(session_id)
        noise = self._noise(noise)
        if self._wants_full(session_id):
            return self._full_round(observation, session_id=session_id, noise=noise, num_steps=num_steps)
        return self._spec_round(observation, session_id=session_id, noise=noise, num_steps=num_steps)

    def _noise(self, noise: torch.Tensor | None) -> torch.Tensor:
        if noise is not None:
            return noise
        return torch.randn(
            (1, self.config.chunk_size, self.config.max_action_dim), device=self.device, dtype=self.dtype
        )

    def _ground(self, observation: dict) -> _Grounding:
        """Preprocess, embed the prefix, fill its KV cache. 80.8 ms (§11).

        The expensive half of a full round, and exactly what a speculative round
        skips when ``spec_reground`` is off.
        """
        from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import make_att_2d_masks

        features = self.processor.preprocess(observation)
        inputs = features.to(device=self.device, dtype=self.dtype).model_inputs()

        embs, pad_masks, att_masks, position_ids, visual_masks, deepstack = self.transformer.embed_prefix(
            inputs["images"], inputs["img_masks"], inputs["lang_tokens"], inputs["lang_masks"],
            inputs["image_grid_thw"],
        )
        _, past_key_values = self.transformer.prefix_forward(
            attention_mask=make_att_2d_masks(pad_masks, att_masks),
            position_ids=position_ids,
            inputs_embeds=[embs, None],
            past_key_values=None,
            fill_kv_cache=True,
            visual_pos_masks=visual_masks,
            deepstack_visual_embeds=deepstack,
        )
        return _Grounding(
            features=features,
            inputs=inputs,
            embs=embs,
            pad_masks=pad_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def _denoise(self, ground: _Grounding, *, noise: torch.Tensor, num_steps: int) -> torch.Tensor:
        """The Euler loop over a grounding. 213 ms of a full round's 294 (§11)."""
        return self.transformer.denoise_actions(
            state=ground.inputs["state"],
            prefix_pad_masks=ground.pad_masks,
            prefix_position_ids=ground.position_ids,
            past_key_values=ground.past_key_values,
            noise=noise,
            num_steps=num_steps,
        )

    def _adopt(self, session_id: str, ground: _Grounding) -> SpecSession:
        """Make ``ground`` the session's prefix, creating the session if needed.

        A full round starts a session over; a re-grounding speculative round only
        replaces the prefix, because ``gripper_prev`` and the round counters are
        continuity that has to survive the swap.
        """
        session = self.sessions.get(session_id)
        if session is None:
            session = SpecSession(
                pad_masks=ground.pad_masks,
                position_ids=ground.position_ids,
                past_key_values=ground.past_key_values,
                features=ground.features,
            )
            self.sessions[session_id] = session
            return session
        session.pad_masks = ground.pad_masks
        session.position_ids = ground.position_ids
        session.past_key_values = ground.past_key_values
        session.features = ground.features
        return session

    def _full_round(
        self,
        observation: dict,
        *,
        session_id: str,
        noise: torch.Tensor,
        num_steps: int,
        ground: _Grounding | None = None,
    ) -> DecodeResult:
        """``_ground`` plus ``num_steps`` Euler steps -- today's path exactly.

        ``ground`` lets a caller that already built this tick's prefix hand it
        over rather than pay the 80.8 ms twice; the only caller that does is a
        re-grounding speculative round whose draft worker just died.
        """
        if ground is None:
            ground = self._ground(observation)

        stats = SpecStats(kind="full", eval_h=self.config.spec_max_exec_steps)
        x0_draft = self._refresh_draft(ground.embs, ground.inputs["state"])

        x0 = self._denoise(ground, noise=noise, num_steps=num_steps)
        if x0_draft is not None:
            stats.draft_rms = self._draft_rms(x0_draft, x0)

        self.reset(session_id)
        session = self._adopt(session_id, ground)
        session.gripper_prev = self._gripper_of(x0)
        self.rounds["full"] += 1
        return DecodeResult(self.processor.postprocess(x0, ground.features), stats)

    def _spec_round(
        self, observation: dict, *, session_id: str, noise: torch.Tensor, num_steps: int
    ) -> DecodeResult:
        session = self.sessions[session_id]
        ground = self._ground(observation) if self.config.spec_reground else None

        if ground is not None:
            # This tick owns its whole prefix, so the draft gets the fresh
            # embeddings too: a draft conditioned on last tick's frame is the
            # other half of the staleness this scheme exists to remove.
            session = self._adopt(session_id, ground)
            state = ground.inputs["state"]
            x0_draft = self._refresh_draft(ground.embs, state)
            if x0_draft is None:
                # The worker just died, and ``draft_failed`` makes every later
                # tick a full round. Finish this one as a full round over the
                # prefix already in hand rather than grounding twice.
                return self._full_round(
                    observation, session_id=session_id, noise=noise, num_steps=num_steps, ground=ground
                )
        else:
            # Only the state is needed: the draft holds its own projected prefix
            # and the verifier reads the cached KV, so the image processor is
            # skipped.
            state = self.processor.preprocess_state(observation).to(device=self.device, dtype=self.dtype)
            try:
                x0_draft = self.draft.draft(state)
            except Exception:
                logger.exception("LingBot draft worker failed; falling back to full rounds for every tick")
                self.draft_failed = True
                session.pending_full = True
                return self._full_round(
                    observation, session_id=session_id, noise=noise, num_steps=num_steps
                )

        x0_hat = self._verify(session, state, noise, x0_draft)
        x0_tail = x0_hat.mean(dim=1)
        accepted, dist = radius_prefix_acceptance(
            x0_draft, x0_hat, tau=self.config.spec_tau, dims=self.pose, eval_h=self.config.spec_max_exec_steps
        )
        x0_out = stitch_prefix(x0_draft, x0_tail, accepted)
        accepted, cut = truncate_on_gripper_switch(
            x0_out, accepted, gripper_prev=session.gripper_prev, dims=self.grippers
        )
        forced = False
        if self._forced_rate is not None:
            # After the gripper guard, not before: the guard is part of the accept
            # rule, and an untrained draft's gripper dims swing across the
            # threshold on almost every step, so forcing before it gets truncated
            # straight back to 0 and the simulated rate never takes effect. The
            # whole real path still ran, so its cost is in the measurement; only
            # its verdict is replaced.
            accepted, cut, forced = self._force_accept(accepted)
            x0_out = stitch_prefix(x0_draft, x0_tail, accepted)

        accepted_len = int(accepted.item())
        session.rounds_since_full += 1
        rejected = accepted_len == 0 or bool(cut.any().item())
        if rejected:
            self.rounds["reject"] += 1

        fell_back = False
        if ground is None:
            # FLASH's rule: nothing accepted, or a speculated gripper transition,
            # means the next tick re-grounds on a fresh observation.
            if rejected:
                session.pending_full = True
        elif accepted_len == 0:
            # Nothing to defer to -- this scheme has no next-tick full round --
            # and ``x0_tail`` off a draft that was just rejected is not an action
            # worth executing. So pay the Euler loop here, over the prefix this
            # tick already built. A gripper cut with a non-empty accept does
            # *not* land here: what survives truncation is real accepted draft
            # steps, and there is no staleness left for a full round to repair.
            x0_out = self._denoise(ground, noise=noise, num_steps=num_steps)
            fell_back = True

        session.gripper_prev = self._gripper_of(x0_out)
        self.rounds["spec"] += 1
        stats = SpecStats(
            kind="spec",
            accepted=accepted_len,
            eval_h=self.config.spec_max_exec_steps,
            gripper_cut=bool(cut.any().item()),
            rounds_since_full=session.rounds_since_full,
            forced=forced,
            dist_mean=float(dist.mean().item()),
            fell_back=fell_back,
            regrounded=ground is not None,
        )
        return DecodeResult(self.processor.postprocess(x0_out, session.features), stats)

    # -- pieces ------------------------------------------------------------
    def _verify(
        self, session: SpecSession, state: torch.Tensor, noise: torch.Tensor, x0_draft: torch.Tensor
    ) -> torch.Tensor:
        """K near-terminal steps -> ``x0_hat`` stacked as ``[B,K,H,D]``.

        Batched or sequential per ``spec_verify_batched``; the two agree to fp
        tolerance, not bit-exactly, because a ``B*K`` GEMM reduces in a different
        order than K separate ones. See that field for why batching is cheap and
        where it stops being cheap.
        """
        if self.config.spec_verify_batched and len(self.config.spec_t_list) > 1:
            return self._verify_batched(session, state, noise, x0_draft)
        return self._verify_sequential(session, state, noise, x0_draft)

    def _verify_sequential(
        self, session: SpecSession, state: torch.Tensor, noise: torch.Tensor, x0_draft: torch.Tensor
    ) -> torch.Tensor:
        """One ``predict_velocity`` per timestep. The fallback, and the reference."""
        batch = int(state.shape[0])
        hats = []
        for t in self.config.spec_t_list:
            x_t = build_x_t(noise, x0_draft, t)
            v_t = self.transformer.predict_velocity(
                state=state,
                prefix_pad_masks=session.pad_masks,
                prefix_position_ids=session.position_ids,
                past_key_values=session.past_key_values,
                x_t=x_t,
                timestep=torch.full((batch,), float(t), device=x_t.device, dtype=x_t.dtype),
            )
            hats.append(x0_from_velocity(x_t, t, v_t))
        return torch.stack(hats, dim=1)

    def _verify_batched(
        self, session: SpecSession, state: torch.Tensor, noise: torch.Tensor, x0_draft: torch.Tensor
    ) -> torch.Tensor:
        """All K timesteps in one ``predict_velocity`` at batch ``B*K``.

        Row ordering is b-major, ``row = b*K + k``, and every conditioning tensor
        below is expanded the same way so they stay aligned. Note the two layouts:
        ``prefix_pad_masks`` is ``[B,S]`` but ``prefix_position_ids`` is
        ``[3,B,S]`` -- mrope's three axes come first -- so the batch axis to
        expand is 0 for one and 1 for the other.
        """
        batch, k = int(state.shape[0]), len(self.config.spec_t_list)
        t = torch.as_tensor(self.config.spec_t_list, device=x0_draft.device, dtype=x0_draft.dtype)

        # [B,K,H,D]: every timestep's x_t, built off the one draft.
        x_t = build_x_t(noise[:, None], x0_draft[:, None], t[None, :, None, None])
        v_t = self.transformer.predict_velocity(
            state=_expand_rows(state, k),
            prefix_pad_masks=_expand_rows(session.pad_masks, k),
            prefix_position_ids=_expand_rows(session.position_ids, k, dim=1),
            past_key_values=[
                (_expand_rows(key, k), _expand_rows(value, k)) for key, value in session.past_key_values
            ],
            x_t=x_t.reshape(batch * k, *x_t.shape[2:]),
            timestep=t[None, :].expand(batch, k).reshape(batch * k),
        ).reshape(batch, k, *x_t.shape[2:])
        return x0_from_velocity(x_t, t[None, :, None, None], v_t)

    def _refresh_draft(self, prefix_embs: torch.Tensor, state: torch.Tensor) -> torch.Tensor | None:
        if self.draft_failed:
            return None
        try:
            return self.draft.refresh(prefix_embs, state)
        except Exception:
            logger.exception("LingBot draft worker refresh failed; speculative rounds are disabled")
            self.draft_failed = True
            return None

    def _force_accept(self, accepted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Replace the accept verdict with the configured rate.

        All-or-nothing per round, because that is what acceptance changes about
        *performance*: a rejected round forces the next tick to be a full round,
        and within a round the verify cost is K forward passes either way.

        **Stratified, not Bernoulli and not evenly spaced**, and both rejected
        alternatives failed in instructive ways:

        * A seeded Bernoulli realised 0.03 for a requested 0.25 over 31 rounds.
          Reproducible, but an instrument whose x-axis wanders is a bad one.
        * Spreading evenly (a Bresenham accumulator) realised the rate exactly and
          then **phase-locked with** ``spec_full_every``: at 0.75 every forced
          rejection landed on a tick where the periodic schedule wanted a full
          round anyway, so it was free, while at 0.9 they landed between them and
          each one added a round. 0.9 measured *slower* than 0.75. That artifact
          does not shrink with more ticks.

        So: exactly ``round(rate * BLOCK)`` acceptances per block of ``BLOCK``
        rounds, in a seeded order. Exact mix, no resonance.
        """
        if not self._forced_queue:
            block = 20
            takes = round(float(self._forced_rate) * block)
            draws = [True] * takes + [False] * (block - takes)
            self._forced_rng.shuffle(draws)
            self._forced_queue.extend(draws)
        take = self._forced_queue.pop()
        if take:
            self.rounds["forced_accept"] += 1
        value = self.config.spec_max_exec_steps if take else 0
        cut = torch.zeros((int(accepted.shape[0]),), device=accepted.device, dtype=torch.bool)
        return torch.full_like(accepted, value), cut, True

    def _gripper_of(self, x0: torch.Tensor) -> torch.Tensor | None:
        """The gripper command of the step about to be executed.

        One tick replans once, so exactly step 0 of the chunk is executed.
        """
        if not self.grippers:
            return None
        index = torch.as_tensor(self.grippers, device=x0.device, dtype=torch.long)
        return x0[:, 0, :].index_select(-1, index).detach()

    def _draft_rms(self, x0_draft: torch.Tensor, x0_teacher: torch.Tensor) -> float:
        """Per-dim RMS of draft vs teacher on the executed horizon.

        Gate 3's ``eps``: the draft needs <= 0.02 for any ``tau >= 0.05`` to accept.
        Free on every full round, and the only accuracy signal available online.
        """
        dims = torch.as_tensor(sorted(self.pose + self.grippers), device=x0_draft.device, dtype=torch.long)
        horizon = self.config.spec_max_exec_steps
        delta = (x0_draft - x0_teacher)[:, :horizon].index_select(-1, dims).to(torch.float32)
        return float(delta.pow(2).mean().sqrt().item())


__all__ = [
    "DecodeResult",
    "DraftBackend",
    "SpecDecoder",
    "SpecSession",
    "SpecStats",
    "action_group_slots",
    "build_x_t",
    "gripper_dims",
    "pose_dims",
    "prefix_shape",
    "radius_prefix_acceptance",
    "stitch_prefix",
    "truncate_on_gripper_switch",
    "x0_from_velocity",
]
