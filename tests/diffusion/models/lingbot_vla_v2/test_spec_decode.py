# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for LingBot-VLA 2.0 speculative decoding.

The draft itself runs on an iGPU in another process, which no test host has, so
the process boundary is where these stop: ``SpecDecoder`` takes its
``DraftBackend`` as an argument precisely so the schedule, the accept rule and
the session lifecycle can be tested in-process with a stub. The transport is
covered by ``spikes/lingbot_vla_v2/phase10_draft_worker.py --role selftest``,
which runs both backends against each other on the real devices, and the accept
algebra additionally by ``phase10_port_exactness.py``, which compares it with the
FLASH originals using ``torch.equal``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.lingbot_vla_v2.config import ACTION, LingbotVlaV2Config
from vllm_omni.diffusion.models.lingbot_vla_v2.draft_igpu import LingbotDraftHead
from vllm_omni.diffusion.models.lingbot_vla_v2.processor import JointGroup, RobotSpec, SourceSlice
from vllm_omni.diffusion.models.lingbot_vla_v2.spec_decode import (
    SpecDecoder,
    _expand_rows,
    gripper_dims,
    pose_dims,
    radius_prefix_acceptance,
    stitch_prefix,
    truncate_on_gripper_switch,
)

CHUNK = 6
ACTION_DIM = 6
STATE_DIM = 6
EVAL_H = 4


def robot_spec() -> RobotSpec:
    """Two joint groups, so the grippers are *not* where FLASH hardcoded them.

    ``arm.position`` is 3 real dims in a 4-wide slot and ``effector.position`` is
    2 dims after it, which puts the grippers at 4 and 5 and a *pose* dim at
    index... nowhere near 6. That asymmetry is the whole reason the dims are
    derived from the spec instead of copied from the reference implementation.
    """
    return RobotSpec(
        joints=(
            JointGroup(name="arm.position", max_dim=4, norm_type="identity"),
            JointGroup(name="effector.position", max_dim=2, norm_type="identity"),
        ),
        cameras=("cam",),
        camera_sources={"cam": "cam"},
        state_slices={"arm.position": (SourceSlice(ACTION, 0, 3),)},
        action_slices={
            "arm.position": (SourceSlice(ACTION, 0, 3),),
            "effector.position": (SourceSlice(ACTION, 3, 5),),
        },
        subtract_state={},
        norm_stats={},
    )


class FakeProcessor:
    """Enough of ``LingbotVlaV2Processor`` for the decoder, counting its calls."""

    def __init__(self) -> None:
        self.spec = robot_spec()
        self.preprocess_calls = 0
        self.state_calls = 0

    def preprocess(self, observation):
        self.preprocess_calls += 1
        return FakeFeatures(observation)

    def preprocess_state(self, observation):
        self.state_calls += 1
        return torch.full((1, STATE_DIM), float(observation["state"]))

    def postprocess(self, actions, features):
        assert isinstance(features, FakeFeatures)
        return {"action": actions.detach().cpu().numpy()[0]}


class FakeFeatures:
    def __init__(self, observation) -> None:
        self.observation = observation

    def to(self, *, device=None, dtype=None):
        return self

    def model_inputs(self):
        return {
            "images": torch.zeros(1, 1, 4, 3),
            "img_masks": torch.ones(1, 1, dtype=torch.bool),
            "lang_tokens": torch.zeros(1, 2, dtype=torch.long),
            "lang_masks": torch.ones(1, 2, dtype=torch.bool),
            "image_grid_thw": torch.ones(1, 1, 3, dtype=torch.long),
            "state": torch.full((1, STATE_DIM), float(self.observation["state"])),
        }


class FakeTransformer:
    """Returns whatever the test says the teacher and the verifier should say.

    ``predict_velocity`` reconstructs ``self.verify_target`` exactly: given
    ``x_t`` and ``t``, returning ``(x_t - target) / t`` makes
    ``x0_hat = x_t - t*v_t`` equal ``target``. So a test controls acceptance by
    choosing whether that target matches the draft, which is the same knob the
    real model turns, without needing a model.
    """

    def __init__(self) -> None:
        self.teacher = torch.zeros(1, CHUNK, ACTION_DIM)
        self.verify_target = torch.zeros(1, CHUNK, ACTION_DIM)
        self.velocity_calls = 0
        self.velocity_rows: list[int] = []
        self.prefix_fills = 0
        self.denoise_calls = 0

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, image_grid_thw):
        embs = torch.zeros(1, 5, 8)
        pad_masks = torch.ones(1, 5, dtype=torch.bool)
        att_masks = torch.zeros(1, 5, dtype=torch.bool)
        position_ids = torch.zeros(3, 1, 5, dtype=torch.long)
        return embs, pad_masks, att_masks, position_ids, None, None

    def prefix_forward(self, **kwargs):
        self.prefix_fills += 1
        return None, [(torch.zeros(1, 1, 5, 2), torch.zeros(1, 1, 5, 2))]

    def denoise_actions(self, *, state, prefix_pad_masks, prefix_position_ids, past_key_values, noise, num_steps):
        self.denoise_calls += 1
        return self.teacher.clone()

    def predict_velocity(self, *, state, prefix_pad_masks, prefix_position_ids, past_key_values, x_t, timestep):
        self.velocity_calls += 1
        rows = int(x_t.shape[0])
        # Every conditioning tensor must have been expanded to the same B*K, or
        # the batched verify is silently pairing a row with another row's prefix.
        # ``prefix_position_ids`` is ``[3,B,S]``, so its batch axis is 1.
        assert state.shape[0] == rows
        assert prefix_pad_masks.shape[0] == rows
        assert prefix_position_ids.shape[1] == rows
        for key, value in past_key_values:
            assert key.shape[0] == rows and value.shape[0] == rows
        self.velocity_rows.append(rows)
        t = timestep.reshape(rows, *([1] * (x_t.ndim - 1)))
        return (x_t - self.verify_target) / t


def spec_config(**overrides) -> LingbotVlaV2Config:
    settings = {
        "chunk_size": CHUNK,
        "max_action_dim": ACTION_DIM,
        "max_state_dim": STATE_DIM,
        "num_steps": 4,
        "spec_decode": True,
        "spec_max_exec_steps": EVAL_H,
        "spec_full_every": 2,
        "spec_t_list": [0.10, 0.05],
        "spec_tau": 0.15,
    }
    settings.update(overrides)
    return LingbotVlaV2Config(**settings)


class StubDraft:
    def __init__(self, chunk: torch.Tensor) -> None:
        self.chunk = chunk
        self.refreshes = 0
        self.drafts = 0
        self.fail = False

    def refresh(self, prefix_embs, state):
        self.refreshes += 1
        if self.fail:
            raise RuntimeError("draft worker died")
        return self.chunk.clone()

    def draft(self, state):
        self.drafts += 1
        if self.fail:
            raise RuntimeError("draft worker died")
        return self.chunk.clone()

    def close(self) -> None:
        pass


def build_decoder(*, draft_chunk: torch.Tensor, config: LingbotVlaV2Config | None = None):
    processor, transformer = FakeProcessor(), FakeTransformer()
    draft = StubDraft(draft_chunk)
    decoder = SpecDecoder(
        transformer=transformer,
        processor=processor,
        config=config or spec_config(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        draft=draft,
    )
    return decoder, transformer, processor, draft


def tick(decoder, value: float, *, reset: bool = False):
    return decoder.decode(
        {"state": value}, session_id="s", reset=reset, noise=torch.zeros(1, CHUNK, ACTION_DIM), num_steps=4
    )


# -- accept algebra ---------------------------------------------------------
def test_dims_come_from_the_spec_not_from_flash_indices():
    processor = FakeProcessor()
    assert pose_dims(processor) == [0, 1, 2]
    # The grippers are the `effector.position` slots, and index 6 -- FLASH's
    # hardcoded gripper -- is not one of them.
    assert gripper_dims(processor) == [4, 5]


def test_radius_acceptance_is_a_prefix_and_stops_at_the_first_disagreement():
    draft = torch.zeros(1, CHUNK, ACTION_DIM)
    hat = torch.zeros(1, 2, CHUNK, ACTION_DIM)
    accepted, _ = radius_prefix_acceptance(draft, hat, tau=0.1, dims=[0, 1, 2], eval_h=EVAL_H)
    assert int(accepted) == EVAL_H

    hat[0, 1, 2, 0] = 10.0  # one verify timestep disagrees at step 2
    accepted, dist = radius_prefix_acceptance(draft, hat, tau=0.1, dims=[0, 1, 2], eval_h=EVAL_H)
    assert int(accepted) == 2
    assert dist.shape == (1, 2, EVAL_H)

    hat[0, 0, 0, 0] = 10.0  # ... and now at step 0, so nothing is accepted
    accepted, _ = radius_prefix_acceptance(draft, hat, tau=0.1, dims=[0, 1, 2], eval_h=EVAL_H)
    assert int(accepted) == 0


def test_radius_acceptance_is_a_per_dim_rms_so_tau_does_not_move_with_dof():
    draft = torch.zeros(1, 1, 8)
    hat = torch.full((1, 1, 1, 8), 0.2)
    narrow, _ = radius_prefix_acceptance(draft, hat, tau=0.25, dims=[0, 1], eval_h=1)
    wide, _ = radius_prefix_acceptance(draft, hat, tau=0.25, dims=list(range(8)), eval_h=1)
    assert int(narrow) == int(wide) == 1


def test_stitch_takes_the_accepted_prefix_from_the_draft_and_the_rest_from_verify():
    draft = torch.ones(1, CHUNK, ACTION_DIM)
    tail = torch.zeros(1, CHUNK, ACTION_DIM)
    stitched = stitch_prefix(draft, tail, torch.tensor([2]))
    assert torch.equal(stitched[0, :2], draft[0, :2])
    assert torch.equal(stitched[0, 2:], tail[0, 2:])


# -- batched verify ---------------------------------------------------------
def test_expand_rows_is_b_major_and_handles_the_mrope_layout():
    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # row = b*K + k, so each source row is repeated K times before the next.
    assert torch.equal(
        _expand_rows(rows, 3), torch.tensor([[1.0, 2.0]] * 3 + [[3.0, 4.0]] * 3)
    )
    # K=1 must not copy: the sequential path shares the session's tensors.
    assert _expand_rows(rows, 1) is rows
    # ``prefix_position_ids`` is [3,B,S] -- expanding the wrong axis would give
    # [3*K,B,S] and pass a shape check while pairing rows with wrong positions.
    position_ids = torch.zeros(3, 2, 5, dtype=torch.long)
    assert _expand_rows(position_ids, 4, dim=1).shape == (3, 8, 5)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_batched_verify_agrees_with_sequential_and_costs_one_call(k):
    t_list = [0.10, 0.05, 0.08, 0.03][:k]
    draft_chunk = torch.full((1, CHUNK, ACTION_DIM), 0.5)

    results = {}
    for batched in (False, True):
        decoder, transformer, _, _ = build_decoder(
            draft_chunk=draft_chunk,
            config=spec_config(spec_t_list=t_list, spec_verify_batched=batched),
        )
        transformer.verify_target = draft_chunk.clone()
        tick(decoder, 0.0)  # full round primes the session
        transformer.velocity_calls, transformer.velocity_rows = 0, []
        result = tick(decoder, 1.0)
        results[batched] = (result.stats.accepted, result.actions["action"], transformer.velocity_rows)

    sequential, batched = results[False], results[True]
    assert sequential[0] == batched[0] == EVAL_H
    np.testing.assert_allclose(sequential[1], batched[1])
    # K sequential calls at batch 1, against one call at batch K.
    assert sequential[2] == [1] * k
    assert batched[2] == ([1] * k if k == 1 else [k])


def test_batched_verify_keeps_each_row_on_its_own_timestep():
    """A row/timestep swap is invisible unless ``x0_hat`` depends on ``t``."""
    t_list = [0.11, 0.07, 0.03]
    decoder, transformer, _, _ = build_decoder(
        draft_chunk=torch.zeros(1, CHUNK, ACTION_DIM),
        config=spec_config(spec_t_list=t_list, spec_verify_batched=True),
    )

    # Make the teacher answer "whatever t you asked me at", so x0_hat[0,k] must
    # come back filled with t_list[k] exactly when the rows line up.
    def velocity(*, state, prefix_pad_masks, prefix_position_ids, past_key_values, x_t, timestep):
        t = timestep.reshape(-1, *([1] * (x_t.ndim - 1)))
        return (x_t - t.expand_as(x_t)) / t

    tick(decoder, 0.0)
    transformer.predict_velocity = velocity
    session = decoder.sessions["s"]
    zeros = torch.zeros(1, CHUNK, ACTION_DIM)
    x0_hat = decoder._verify(session, torch.zeros(1, STATE_DIM), zeros, zeros)

    assert x0_hat.shape == (1, len(t_list), CHUNK, ACTION_DIM)
    for index, t in enumerate(t_list):
        assert torch.allclose(x0_hat[0, index], torch.full((CHUNK, ACTION_DIM), t), atol=1e-6)


def test_gripper_switch_truncates_the_accepted_prefix_at_the_flip():
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    chunk[0, 3:, 5] = 1.0  # the right gripper closes at step 3
    truncated, cut = truncate_on_gripper_switch(
        chunk, torch.tensor([EVAL_H]), gripper_prev=torch.zeros(1, 2), dims=[4, 5], threshold=0.5
    )
    assert int(truncated) == 3
    assert bool(cut)

    # No flip inside the accepted prefix -> nothing to cut.
    truncated, cut = truncate_on_gripper_switch(
        chunk, torch.tensor([2]), gripper_prev=torch.zeros(1, 2), dims=[4, 5], threshold=0.5
    )
    assert int(truncated) == 2
    assert not bool(cut)


# -- schedule and sessions --------------------------------------------------
def test_accepted_rounds_follow_the_periodic_schedule_and_reuse_the_cache():
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    decoder, transformer, processor, draft = build_decoder(draft_chunk=chunk)
    transformer.verify_target = chunk.clone()  # the verifier agrees with the draft

    kinds = [tick(decoder, index).stats.kind for index in range(6)]
    # spec_full_every=2: one full round, then two speculative rounds.
    assert kinds == ["full", "spec", "spec", "full", "spec", "spec"]
    assert transformer.prefix_fills == 2  # the prefix is filled once per full round
    assert draft.refreshes == 2 and draft.drafts == 4
    # A speculative round never runs the image processor: it has the cached
    # prefix and only needs the new state.
    assert processor.preprocess_calls == 2 and processor.state_calls == 4


def test_a_rejected_round_forces_the_next_tick_to_re_ground():
    decoder, transformer, _, _ = build_decoder(draft_chunk=torch.zeros(1, CHUNK, ACTION_DIM))
    transformer.verify_target = torch.full((1, CHUNK, ACTION_DIM), 5.0)  # far from the draft

    kinds = [tick(decoder, index).stats.kind for index in range(4)]
    assert kinds == ["full", "spec", "full", "spec"]
    assert decoder.rounds["reject"] == 2


def test_a_rejected_round_still_returns_the_verifier_s_answer():
    """The fallback path is not a failure path: it returns mean_k(x0_hat)."""
    target = torch.full((1, CHUNK, ACTION_DIM), 5.0)
    decoder, transformer, _, _ = build_decoder(draft_chunk=torch.zeros(1, CHUNK, ACTION_DIM))
    transformer.verify_target = target

    tick(decoder, 0.0)
    result = tick(decoder, 1.0)
    assert result.stats.accepted == 0
    assert np.allclose(result.actions["action"], target[0].numpy(), atol=1e-5)


def test_reset_drops_the_session_so_the_next_tick_is_a_full_round():
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    decoder, transformer, _, _ = build_decoder(draft_chunk=chunk)
    transformer.verify_target = chunk.clone()

    assert tick(decoder, 0.0).stats.kind == "full"
    assert tick(decoder, 1.0).stats.kind == "spec"
    # A reset mid-schedule re-grounds even though a speculative round was due.
    assert tick(decoder, 2.0, reset=True).stats.kind == "full"
    decoder.reset("s")
    assert "s" not in decoder.sessions


def test_sessions_are_independent():
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    decoder, transformer, _, _ = build_decoder(draft_chunk=chunk)
    transformer.verify_target = chunk.clone()

    assert decoder.decode({"state": 0.0}, session_id="a", reset=False, noise=None, num_steps=4).stats.kind == "full"
    assert decoder.decode({"state": 0.0}, session_id="b", reset=False, noise=None, num_steps=4).stats.kind == "full"
    assert decoder.decode({"state": 1.0}, session_id="a", reset=False, noise=None, num_steps=4).stats.kind == "spec"
    assert set(decoder.sessions) == {"a", "b"}


def test_a_dead_draft_worker_degrades_to_full_rounds_instead_of_failing():
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    decoder, transformer, _, draft = build_decoder(draft_chunk=chunk)
    transformer.verify_target = chunk.clone()

    tick(decoder, 0.0)
    draft.fail = True
    result = tick(decoder, 1.0)
    assert result.stats.kind == "full"  # the tick still answers
    assert decoder.draft_failed
    assert [tick(decoder, index).stats.kind for index in range(3)] == ["full", "full", "full"]


# -- spec_reground ----------------------------------------------------------
def test_reground_grounds_every_tick_and_stops_deferring_full_rounds():
    """``spec_full_every`` is inert: nothing is stale, so nothing expires."""
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    decoder, transformer, processor, draft = build_decoder(
        draft_chunk=chunk, config=spec_config(spec_reground=True, spec_full_every=2)
    )
    transformer.verify_target = chunk.clone()  # the verifier agrees with the draft

    kinds = [tick(decoder, index).stats.kind for index in range(6)]
    # One full round to create the session, and then never again -- against
    # ["full","spec","spec","full",...] for the cached scheme at full_every=2.
    assert kinds == ["full"] + ["spec"] * 5
    # The prefix is filled on *every* tick, which is what this costs.
    assert transformer.prefix_fills == 6
    assert processor.preprocess_calls == 6
    # And `preprocess_state` is never reached: the state comes out of the same
    # `preprocess` that built the prefix, so there is no second path for it.
    assert processor.state_calls == 0
    # The draft is refreshed off the fresh embeddings, not asked for a chunk
    # against a projection it made last tick.
    assert draft.refreshes == 6 and draft.drafts == 0
    assert transformer.denoise_calls == 1  # the full round only; nothing rejected


def test_reground_rejection_denoises_in_its_own_tick_instead_of_executing_x0_tail():
    """The quality fix, and the reason this scheme costs what it costs.

    The cached scheme's rejected tick returns ``x0_tail`` -- one near-terminal
    step's estimate off the draft it just rejected -- and defers a full round to
    the next tick. Re-grounding pays the Euler loop on the spot and returns the
    teacher's actual answer.
    """
    teacher = torch.full((1, CHUNK, ACTION_DIM), 7.0)
    tail = torch.full((1, CHUNK, ACTION_DIM), 5.0)

    outcomes = {}
    for reground in (False, True):
        decoder, transformer, _, _ = build_decoder(
            draft_chunk=torch.zeros(1, CHUNK, ACTION_DIM),
            config=spec_config(spec_reground=reground, spec_full_every=1000),
        )
        transformer.teacher = teacher.clone()
        transformer.verify_target = tail.clone()  # far from the draft: accept 0
        tick(decoder, 0.0)
        before = transformer.denoise_calls
        result = tick(decoder, 1.0)
        outcomes[reground] = (result.stats, result.actions["action"], transformer.denoise_calls - before)

    cached_stats, cached_action, cached_denoises = outcomes[False]
    regrounded_stats, regrounded_action, regrounded_denoises = outcomes[True]

    assert cached_stats.accepted == regrounded_stats.accepted == 0
    # Cached: no Euler loop in this tick, and the executed chunk is x0_tail.
    assert cached_denoises == 0 and not cached_stats.fell_back
    np.testing.assert_allclose(cached_action, tail[0].numpy(), atol=1e-5)
    # Re-grounded: one Euler loop in this tick, and the teacher's own answer.
    assert regrounded_denoises == 1 and regrounded_stats.fell_back
    assert regrounded_stats.regrounded
    np.testing.assert_allclose(regrounded_action, teacher[0].numpy(), atol=1e-5)


def test_reground_does_not_denoise_when_only_the_gripper_guard_cut():
    """A truncated accept is still real accepted steps, so there is nothing to redo."""
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    chunk[0, 3:, 5] = -1.0  # the right gripper crosses zero at step 3
    decoder, transformer, _, _ = build_decoder(
        draft_chunk=chunk, config=spec_config(spec_reground=True, spec_full_every=1000)
    )
    transformer.verify_target = chunk.clone()  # the pose dims agree exactly

    tick(decoder, 0.0)
    before = transformer.denoise_calls
    stats = tick(decoder, 1.0).stats

    assert stats.accepted == 3 and stats.gripper_cut
    assert not stats.fell_back
    assert transformer.denoise_calls == before


def test_reground_reuses_the_prefix_it_built_when_the_draft_worker_dies():
    """The failure path must not ground twice: it already holds a fresh prefix."""
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    decoder, transformer, _, draft = build_decoder(
        draft_chunk=chunk, config=spec_config(spec_reground=True, spec_full_every=1000)
    )
    transformer.verify_target = chunk.clone()

    tick(decoder, 0.0)
    assert transformer.prefix_fills == 1
    draft.fail = True
    result = tick(decoder, 1.0)

    assert result.stats.kind == "full"  # the tick still answers
    assert decoder.draft_failed
    assert transformer.prefix_fills == 2  # one grounding for this tick, not two
    assert [tick(decoder, index).stats.kind for index in range(2)] == ["full", "full"]


# -- forced acceptance ------------------------------------------------------
@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_forced_acceptance_drives_the_schedule_and_still_runs_verify(rate):
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    decoder, transformer, _, _ = build_decoder(
        draft_chunk=chunk, config=spec_config(spec_force_accept_rate=rate)
    )
    # Deliberately disagreeing: under a forced rate the comparison must not
    # decide anything, or the mode would measure the draft instead of the
    # schedule.
    transformer.verify_target = torch.full((1, CHUNK, ACTION_DIM), 5.0)

    kinds = [tick(decoder, index).stats.kind for index in range(6)]
    spec_rounds = kinds.count("spec")
    if rate == 1.0:
        assert kinds == ["full", "spec", "spec", "full", "spec", "spec"]
    else:
        assert kinds == ["full", "spec", "full", "spec", "full", "spec"]
    # The latency has to stay real: K verify passes per speculative round either
    # way. That is the whole premise of the sweep. Counted in *rows*, not calls,
    # so the invariant holds for both the batched and the sequential path.
    assert sum(transformer.velocity_rows) == spec_rounds * len(decoder.config.spec_t_list)


def test_forced_acceptance_survives_the_gripper_guard():
    """The regression this file missed first time round.

    An untrained draft's gripper dims cross the threshold on almost every step,
    so the post-verify gripper truncation cuts the accepted prefix back to 0. When
    the forced rate was applied *before* that guard, every arm of the sweep --
    rate 0, 0.5 and 1 -- came out identical to "reject always", and the whole
    acceptance curve read flat at 1.66x. The override has to be the last word.
    """
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    # The last executed gripper command is 0 (the teacher's), so a draft that
    # commands -1 crosses the threshold at step 0 -- the worst case, and the one
    # the real random-weight draft hit: truncation to 0, which then forces a full
    # round and makes every rate look like "reject always".
    chunk[0, :, 5] = -1.0
    decoder, transformer, _, _ = build_decoder(
        draft_chunk=chunk, config=spec_config(spec_force_accept_rate=1.0)
    )
    transformer.verify_target = chunk.clone()

    kinds = [tick(decoder, index).stats for index in range(4)]
    assert [stats.kind for stats in kinds] == ["full", "spec", "spec", "full"]
    assert all(stats.accepted == EVAL_H for stats in kinds if stats.kind == "spec")
    assert not any(stats.gripper_cut for stats in kinds)


def test_forced_acceptance_realises_the_requested_rate():
    chunk = torch.zeros(1, CHUNK, ACTION_DIM)
    decoder, transformer, _, _ = build_decoder(
        draft_chunk=chunk, config=spec_config(spec_force_accept_rate=0.5, spec_full_every=1000)
    )
    transformer.verify_target = torch.full((1, CHUNK, ACTION_DIM), 5.0)

    tick(decoder, 0.0)
    rounds = [tick(decoder, index).stats for index in range(300)]
    # Measured over speculative rounds only: a rejected round makes the *next*
    # tick a full round, which is the mechanism being priced, not a draw.
    draws = [stats.accepted > 0 for stats in rounds if stats.kind == "spec"]
    assert len(draws) > 100
    # Stratified in blocks of 20, so the realised rate *is* the requested one to
    # within one block -- a seeded Bernoulli realised 0.03 for a requested 0.25
    # over 31 rounds, and spreading evenly instead phase-locked with
    # spec_full_every and made 0.9 measure slower than 0.75.
    assert abs(sum(draws) / len(draws) - 0.5) <= 1.0 / len(draws)
    assert decoder.rounds["forced_accept"] == sum(draws)


# -- config -----------------------------------------------------------------
def test_spec_settings_are_validated_only_when_the_switch_is_on():
    LingbotVlaV2Config(spec_t_list=[])  # off: never looked at
    with pytest.raises(ValueError, match="at least one verify timestep"):
        LingbotVlaV2Config(spec_decode=True, spec_t_list=[])
    with pytest.raises(ValueError, match=r"lie in \(0, 1\)"):
        LingbotVlaV2Config(spec_decode=True, spec_t_list=[1.5])
    with pytest.raises(ValueError, match="spec_full_every"):
        LingbotVlaV2Config(spec_decode=True, spec_full_every=0)
    with pytest.raises(ValueError, match="spec_max_exec_steps"):
        LingbotVlaV2Config(spec_decode=True, spec_max_exec_steps=1000)
    with pytest.raises(ValueError, match="spec_force_accept_rate"):
        LingbotVlaV2Config(spec_decode=True, spec_force_accept_rate=1.5)


# -- the head ---------------------------------------------------------------
def test_draft_head_keeps_the_shape_gate_4_priced():
    """Pins the measured shape: 0.72 ms on the iGPU, 2.31 M per-tick parameters.

    The per-tick count excludes ``prefix_proj``/``prefix_kv`` because those run
    once per full round, not once per tick, and the iGPU timing that justified
    this design was taken on that subset. A change here invalidates the number
    the config's docstring quotes.
    """
    head = LingbotDraftHead(chunk_size=50, action_dim=55, state_dim=55, prefix_width=2560)
    per_tick = sum(
        tensor.numel()
        for name, tensor in head.named_parameters()
        if not name.startswith(("prefix_proj", "prefix_kv"))
    )
    assert per_tick == 2_311_168
    assert sum(tensor.numel() for tensor in head.parameters()) == 3_752_960

    prefix = torch.zeros(1, 286, 2560)
    cached = head.kv_from_projection(head.project(prefix))
    chunk = head(torch.zeros(1, 55), cached)
    assert chunk.shape == (1, 50, 55)
