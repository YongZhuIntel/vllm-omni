# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Robot observation <-> model tensor conversion for LingBot-VLA 2.0.

The kernel takes six tensors; a robot sends camera frames, a state vector and an
instruction. Everything between them lives here, and it is all *configuration*:

* the **robot config** says which slice of the robot's own state vector is which
  joint group, and which camera topic feeds which model camera slot;
* the **norm stats** say what units the policy was trained in.

Neither is recoverable from the checkpoint, and getting either wrong does not
crash — it produces plausible-looking actions in the wrong units. So upstream's
mapping semantics are reproduced exactly rather than approximated.

Upstream equivalents: ``lingbotvla.data.vla_data.utils.FeatureTransform`` and the
``prepare_*`` helpers in ``lingbotvla.data.vla_data.transform``, plus
``deploy/lingbot_vla_v2_policy.py``'s ``resize_image`` / ``_prepare_model_input``.
``FeatureTransform`` also serves training — augmentation, future frames, depth
teachers, action padding masks, EE-pose conversion, dataset-side action horizons —
and none of that is here. What is here is graded bit-exact against it by
``spikes/lingbot_vla_v2/phase2_processor_parity.py``.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.lingbot_vla_v2.config import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    LingbotVlaV2Config,
)

__all__ = [
    "JointGroup",
    "LingbotVlaV2Processor",
    "RobotFeatures",
    "RobotSpec",
    "SourceSlice",
    "load_hf_processor",
]

# The stat each normalization mode reads. Upstream computes all of them for every
# feature, so an unused mode costs nothing but a wrong name is silent.
_NORM_STATS_USED: dict[str, tuple[str, ...]] = {
    "identity": (),
    "sincos": (),
    "meanstd": ("mean", "std"),
    "std": ("std",),
    "minmax": ("min", "max"),
    "minmax_woclip": ("min", "max"),
    "bounds_98": ("q02", "q98"),
    "bounds_98_woclip": ("q02", "q98"),
    "bounds_99": ("q01", "q99"),
    "bounds_99_woclip": ("q01", "q99"),
}


def _state_key(joint: str) -> str:
    return f"{OBS_STATE}.{joint}"


def _action_key(joint: str) -> str:
    return f"{ACTION}.{joint}"


# ----------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------
def _stat(stats: Mapping[str, torch.Tensor], name: str, value: torch.Tensor) -> torch.Tensor:
    """One statistic, aligned to ``value``'s action horizon.

    Action stats are computed at the training ``chunk_size`` and are then ``[chunk,
    dim]``; a deployment asking for a shorter chunk gets the leading rows rather
    than a broadcast (which would silently apply step 0's bounds to every step).
    """
    stat = stats[name]
    if stat.ndim == 2 and value.ndim == 2:
        if stat.shape[-1] != value.shape[-1]:
            raise ValueError(f"norm stat '{name}' has width {stat.shape[-1]}, value has {value.shape[-1]}.")
        if value.shape[0] > stat.shape[0]:
            raise ValueError(
                f"norm stat '{name}' covers a horizon of {stat.shape[0]} but the chunk is "
                f"{value.shape[0]}; recompute the stats with a larger chunk_size."
            )
        stat = stat[: value.shape[0]]
    return stat


def _normalize(value: torch.Tensor, stats: Mapping[str, torch.Tensor], norm_type: str) -> torch.Tensor:
    """Raw robot units -> the policy's normalized space.

    The stats are float64 (they come from JSON), so every branch promotes to
    float64 exactly as upstream's numpy stats do. The caller casts back.
    """
    if norm_type == "identity":
        return value
    if norm_type == "sincos":
        return torch.cat([torch.cos(value), torch.sin(value)], dim=-1)
    if norm_type == "meanstd":
        mean = _stat(stats, "mean", value)
        return (value - mean) / (_stat(stats, "std", value) + 1e-6)
    if norm_type == "std":
        return value / (_stat(stats, "std", value) + 1e-6)
    if norm_type in ("minmax", "minmax_woclip"):
        low, high = _stat(stats, "min", value), _stat(stats, "max", value)
        scaled = (value - low) / (high - low + 1e-6) * 2 - 1
        return torch.clamp(scaled, min=-1, max=1) if norm_type == "minmax" else scaled
    if norm_type in ("bounds_98", "bounds_98_woclip", "bounds_99", "bounds_99_woclip"):
        keys = ("q02", "q98") if "98" in norm_type else ("q01", "q99")
        low, high = _stat(stats, keys[0], value), _stat(stats, keys[1], value)
        scaled = (value - low) / (high - low + 1e-6) * 2.0 - 1.0
        # The clipped variants bound at 1.5, not 1.0: the training distribution's
        # 1st/99th percentile is not its support.
        return scaled if norm_type.endswith("_woclip") else torch.clamp(scaled, -1.5, 1.5)
    raise ValueError(f"unknown normalization type {norm_type!r}")


def _unnormalize(value: torch.Tensor, stats: Mapping[str, torch.Tensor], norm_type: str) -> torch.Tensor:
    """The inverse of :func:`_normalize`; the clipped variants invert unclipped."""
    if norm_type == "identity":
        return value
    if norm_type == "sincos":
        if value.shape[-1] % 2:
            raise ValueError(f"sincos state has odd width {value.shape[-1]}")
        half = value.shape[-1] // 2
        return torch.atan2(value[..., half:], value[..., :half])
    if norm_type == "meanstd":
        return value * (_stat(stats, "std", value) + 1e-6) + _stat(stats, "mean", value)
    if norm_type == "std":
        return value * (_stat(stats, "std", value) + 1e-6)
    if norm_type in ("minmax", "minmax_woclip"):
        low, high = _stat(stats, "min", value), _stat(stats, "max", value)
        return (value + 1) / 2.0 * (high - low + 1e-6) + low
    if norm_type in ("bounds_98", "bounds_98_woclip", "bounds_99", "bounds_99_woclip"):
        keys = ("q02", "q98") if "98" in norm_type else ("q01", "q99")
        low, high = _stat(stats, keys[0], value), _stat(stats, keys[1], value)
        return (value + 1.0) / 2.0 * (high - low + 1e-6) + low
    raise ValueError(f"unknown normalization type {norm_type!r}")


# ----------------------------------------------------------------------------
# The deployment spec
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceSlice:
    """``source[..., start:end]`` — one contribution to a joint group."""

    source: str
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class JointGroup:
    """One row of the training config's ``data.joints``.

    ``max_dim`` is the *slot* width in the model's state/action vector, which the
    policy's cross-embodiment layout fixes; a robot with fewer joints pads into it
    and masks the tail off. A group the robot does not report at all still
    occupies its slot, filled with zeros.
    """

    name: str
    max_dim: int
    norm_type: str = "identity"


@dataclass(frozen=True)
class RobotSpec:
    """Everything about a robot that the checkpoint does not know.

    Built from the two files a LingBot deployment already has — a robot config
    (``configs/robot_configs/<robot>.yaml``) and its norm stats
    (``assets/norm_stats/<robot>.json``) — plus the ``data`` block of the training
    config, which fixes the joint slot layout and camera order.
    """

    # Joint slots, in the order they are concatenated into the state vector.
    joints: tuple[JointGroup, ...]
    # Model camera slots, in the order the prefix attends to them.
    cameras: tuple[str, ...]
    # Model camera slot -> the robot's own key for that frame.
    camera_sources: Mapping[str, str] = field(default_factory=dict)
    # Joint name -> the raw state slices concatenated to build it. A joint absent
    # here is not reported by this robot.
    state_slices: Mapping[str, tuple[SourceSlice, ...]] = field(default_factory=dict)
    action_slices: Mapping[str, tuple[SourceSlice, ...]] = field(default_factory=dict)
    # Joints whose action is a delta on the current state.
    subtract_state: Mapping[str, bool] = field(default_factory=dict)
    # Feature key ("action.arm.position") -> {stat name: float64 tensor}.
    norm_stats: Mapping[str, Mapping[str, torch.Tensor]] = field(default_factory=dict)
    # Square side the cameras are resized to before the Qwen-VL image processor.
    image_size: int = 256

    def __post_init__(self) -> None:
        names = [joint.name for joint in self.joints]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate joint groups: {names}")
        for mapping, kind in ((self.state_slices, "state"), (self.action_slices, "action")):
            unknown = set(mapping) - set(names)
            if unknown:
                raise ValueError(f"{kind} slices for unknown joint group(s): {sorted(unknown)}")
        missing = set(self.cameras) - set(self.camera_sources)
        if missing:
            raise ValueError(f"camera slot(s) with no source key: {sorted(missing)}")

    # -- geometry ---------------------------------------------------------
    @property
    def joint_dim(self) -> int:
        """Width of the concatenated state/action vector, before padding."""
        return sum(joint.max_dim for joint in self.joints)

    def norm_width(self, key: str) -> int:
        """The real (unpadded) width of a feature, per its norm stats.

        Upstream reads it off ``mean``, which is the only stat guaranteed to be
        present for every mode, and is how it splits a padded chunk back apart.
        """
        stats = self.norm_stats.get(key)
        if stats is None or "mean" not in stats:
            raise KeyError(f"no 'mean' norm stat for {key!r}; cannot recover its width")
        return int(stats["mean"].shape[-1])

    # -- constructors -----------------------------------------------------
    @classmethod
    def passthrough(cls, cameras: Sequence[str], state_dim: int, image_size: int = 256) -> RobotSpec:
        """A spec with no unit conversion: one identity joint group, cameras named
        after themselves.

        For warmup and for numerics work (Phase 0 ran the model this way), where
        the state is already in the model's normalized space and the returned chunk
        is meant to stay there. Not for driving a robot.
        """
        return cls(
            joints=(JointGroup(name="state", max_dim=state_dim, norm_type="identity"),),
            cameras=tuple(cameras),
            camera_sources={camera: camera for camera in cameras},
            state_slices={"state": (SourceSlice(OBS_STATE, 0, state_dim),)},
            action_slices={"state": (SourceSlice(ACTION, 0, state_dim),)},
            subtract_state={"state": False},
            norm_stats={},
            image_size=image_size,
        )

    @classmethod
    def from_dicts(
        cls,
        robot_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        norm_stats: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> RobotSpec:
        """Parse upstream's own config dicts.

        ``robot_config`` is a ``configs/robot_configs/*.yaml`` body,
        ``data_config`` the ``data:`` block of a training config, ``norm_stats``
        the ``norm_stats`` object of the stats JSON (defaulting to the path named
        by the robot config).
        """
        norm_type = _kv_list(data_config.get("norm_type", []))
        joints = tuple(
            JointGroup(name=name, max_dim=int(max_dim), norm_type=norm_type.get(name, "identity"))
            for name, max_dim in _kv_list(data_config.get("joints", [])).items()
            # A slot of width 0 is how upstream disables a joint group entirely.
            if int(max_dim) > 0
        )
        cameras = tuple(f"{OBS_IMAGES}.{camera}" for camera in data_config.get("cameras", []))

        state_slices, _ = _parse_features(robot_config.get("states", []), OBS_STATE)
        action_slices, subtract = _parse_features(robot_config.get("actions", []), ACTION)
        camera_sources = _parse_cameras(robot_config.get("images", []))

        unknown_cameras = set(camera_sources) - set(cameras)
        if unknown_cameras:
            raise ValueError(
                f"robot config maps camera(s) the training config does not list: "
                f"{sorted(unknown_cameras)}; known slots are {list(cameras)}"
            )

        stats = {
            key: {
                name: torch.as_tensor(np.asarray(value, dtype=np.float64), device="cpu")
                for name, value in entry.items()
            }
            for key, entry in (norm_stats or {}).items()
        }
        for joint in joints:
            feature_presence = (
                (_state_key, joint.name in state_slices),
                (_action_key, joint.name in action_slices),
            )
            for prefix, present in feature_presence:
                key = prefix(joint.name)
                if not present or joint.norm_type == "identity":
                    continue
                if key not in stats:
                    raise ValueError(f"{key} is normalized as {joint.norm_type!r} but has no norm stats")
                missing = set(_NORM_STATS_USED[joint.norm_type]) - set(stats[key])
                if missing:
                    raise ValueError(f"{key} norm stats lack {sorted(missing)} for {joint.norm_type!r}")

        return cls(
            joints=joints,
            cameras=cameras,
            camera_sources=camera_sources,
            state_slices=state_slices,
            action_slices=action_slices,
            subtract_state=subtract,
            norm_stats=stats,
            image_size=int(data_config.get("img_size", 256)),
        )

    @classmethod
    def from_files(
        cls,
        robot_config: str | Path,
        data_config: str | Path,
        norm_stats: str | Path | None = None,
    ) -> RobotSpec:
        """As :meth:`from_dicts`, reading upstream's files.

        ``data_config`` may be a whole training config (its ``data`` block is
        used). ``norm_stats`` defaults to the robot config's own ``norm_stats``
        entry, resolved relative to the robot config's repo root — the path there
        is written relative to the LingBot checkout.
        """
        import yaml

        robot_path = Path(robot_config)
        robot = yaml.safe_load(robot_path.read_text())
        data = yaml.safe_load(Path(data_config).read_text())
        data = data.get("data", data)

        stats_path = norm_stats or robot.get("norm_stats")
        if stats_path is None:
            raise ValueError(f"{robot_path} names no norm_stats and none was given")
        stats_path = Path(stats_path)
        if not stats_path.is_absolute() and not stats_path.exists():
            # "assets/norm_stats/<robot>.json", relative to the checkout root.
            stats_path = robot_path.parents[2] / stats_path
        payload = json.loads(stats_path.read_text())
        return cls.from_dicts(robot, data, payload.get("norm_stats", payload))


def _kv_list(entries: Iterable[Any]) -> dict[str, Any]:
    """Upstream's ``[{name: value}, ...]`` config lists, as a dict.

    The training-config loader stringifies each entry, so ``joints`` and
    ``norm_type`` arrive as ``"{'arm.position': 14}"`` there and as real dicts
    when the YAML is read directly. Accept both.
    """
    out: dict[str, Any] = {}
    for entry in entries:
        if isinstance(entry, str):
            entry = ast.literal_eval(entry)
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise ValueError(f"expected a single-key mapping, got {entry!r}")
        ((name, value),) = entry.items()
        out[name] = value
    return out


def _parse_features(entries: Iterable[Any], prefix: str) -> tuple[dict[str, tuple[SourceSlice, ...]], dict[str, bool]]:
    """A robot config's ``states:`` / ``actions:`` list -> per-joint slices.

    Two forms per entry: a bare string names a feature the robot reports under
    the model's own key, and a mapping gives ``origin_keys`` — either one source
    key, or an ordered list of ``{source: {start, end}}`` slices to concatenate.
    """
    slices: dict[str, tuple[SourceSlice, ...]] = {}
    subtract: dict[str, bool] = {}
    for entry in entries:
        if isinstance(entry, str):
            joint = entry.split(f"{prefix}.")[-1]
            slices[joint] = ()  # reported verbatim; no slicing
            subtract[joint] = False
            continue
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise ValueError(f"expected a single-key mapping, got {entry!r}")
        ((target, info),) = entry.items()
        joint = target.split(f"{prefix}.")[-1]
        info = dict(info or {})
        if info.pop("convert_from_state", False):
            raise NotImplementedError(
                f"{target}: convert_from_state derives actions from the *next* state, "
                "which only exists in a dataset, not in a live observation."
            )
        relative_type = info.pop("relative_type", "quaternion_local")
        subtract[joint] = bool(info.pop("subtract_state", False))
        if subtract[joint] and "quaternion" in relative_type:
            raise NotImplementedError(
                f"{target}: relative_type={relative_type!r} needs the SE(3) pose "
                "conversions in upstream's ee_pose_transform, which are not ported."
            )
        origin = info.get("origin_keys")
        if isinstance(origin, str):
            slices[joint] = (SourceSlice(origin, 0, -1),)  # whole vector
        elif isinstance(origin, Sequence):
            parsed = []
            for item in origin:
                ((source, bounds),) = dict(item).items()
                parsed.append(SourceSlice(source, int(bounds["start"]), int(bounds["end"])))
            slices[joint] = tuple(parsed)
        else:
            raise ValueError(f"{target}: origin_keys must be a key or a list of slices")
    return slices, subtract


def _parse_cameras(entries: Iterable[Any]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, str):
            sources[entry] = entry
            continue
        ((target, info),) = dict(entry).items()
        origin = info if isinstance(info, str) else dict(info)["origin_keys"]
        if not isinstance(origin, str):
            raise ValueError(f"{target}: a camera maps from exactly one source key")
        sources[target] = origin
    return sources


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
@dataclass
class RobotFeatures:
    """One observation, as the kernel wants it: batch-of-one, padded, normalized.

    Every field is statically shaped (the prefix length is baked into the
    checkpoint's align-query layout, and text is padded to
    ``tokenizer_max_length``), so a batch is a ``torch.cat`` along dim 0.
    """

    images: torch.Tensor  # (1, cameras, patches, patch_dim) float32
    img_masks: torch.Tensor  # (1, cameras) bool
    lang_tokens: torch.Tensor  # (1, tokenizer_max_length) int64
    lang_masks: torch.Tensor  # (1, tokenizer_max_length) bool
    state: torch.Tensor  # (1, max_state_dim) float32, normalized
    image_grid_thw: torch.Tensor  # (1, cameras, 3) int64
    # Which of the padded state/action slots are real joints. Not model inputs —
    # ``postprocess`` needs them to split the chunk back apart.
    state_mask: torch.Tensor  # (max_state_dim,) bool
    action_mask: torch.Tensor  # (max_action_dim,) bool

    def model_inputs(self) -> dict[str, torch.Tensor]:
        """The six ``sample_actions`` keyword arguments."""
        return {
            "images": self.images,
            "img_masks": self.img_masks,
            "lang_tokens": self.lang_tokens,
            "lang_masks": self.lang_masks,
            "state": self.state,
            "image_grid_thw": self.image_grid_thw,
        }

    def to(self, *, device: Any = None, dtype: torch.dtype | None = None) -> RobotFeatures:
        """Move the model inputs; float tensors also cast, masks and ids do not."""

        def move(tensor: torch.Tensor, cast: bool) -> torch.Tensor:
            target = dtype if cast and tensor.is_floating_point() else None
            return tensor.to(device=device, dtype=target)

        return RobotFeatures(
            images=move(self.images, True),
            img_masks=move(self.img_masks, False),
            lang_tokens=move(self.lang_tokens, False),
            lang_masks=move(self.lang_masks, False),
            state=move(self.state, True),
            image_grid_thw=move(self.image_grid_thw, False),
            state_mask=self.state_mask,
            action_mask=self.action_mask,
        )


class LingbotVlaV2Processor:
    """``robot_obs`` dict <-> model tensors.

        >>> features = processor.preprocess(robot_obs)
        >>> chunk = model.sample_actions(**features.model_inputs(), noise=noise)
        >>> processor.postprocess(chunk, features)   # {"action": (chunk, 14)}

    ``preprocess`` returns normalized model inputs; ``postprocess`` takes the
    normalized chunk back to the robot's own keys and units.
    """

    def __init__(
        self,
        spec: RobotSpec,
        config: LingbotVlaV2Config,
        *,
        tokenizer: Any,
        image_processor: Any,
    ) -> None:
        if len(spec.cameras) > config.max_cameras:
            raise ValueError(f"{len(spec.cameras)} camera slots exceed max_cameras={config.max_cameras}")
        if spec.joint_dim > config.max_state_dim:
            raise ValueError(f"joint slots total {spec.joint_dim}, wider than max_state_dim={config.max_state_dim}")
        self.spec = spec
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    # -- input ------------------------------------------------------------
    def preprocess(self, robot_obs: Mapping[str, Any]) -> RobotFeatures:
        """``{"images": {key: HWC uint8}, "state": [...], "prompt": str}`` -> tensors."""
        # vLLM constructs diffusion pipelines under a target-device context.
        # Preprocessing stays on CPU; the pipeline moves the completed inputs.
        with torch.device("cpu"):
            return self._preprocess(robot_obs)

    def _preprocess(self, robot_obs: Mapping[str, Any]) -> RobotFeatures:
        for required in ("images", "state", "prompt"):
            if required not in robot_obs:
                raise KeyError(f"robot_obs is missing {required!r}")

        raw = {OBS_STATE: _as_float_tensor(robot_obs["state"])}
        state, state_mask = self._build_vector(raw, self.spec.state_slices, _state_key, self.config.max_state_dim)
        _, action_mask = self._build_vector(
            raw, self.spec.action_slices, _action_key, self.config.max_action_dim, values=False
        )
        images, img_masks, grid = self._build_images(robot_obs["images"])
        lang_tokens, lang_masks = self._build_language(robot_obs["prompt"])
        return RobotFeatures(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state.unsqueeze(0),
            image_grid_thw=grid,
            state_mask=state_mask,
            action_mask=action_mask,
        )

    def _build_vector(
        self,
        raw: Mapping[str, torch.Tensor],
        slices: Mapping[str, tuple[SourceSlice, ...]],
        key_of: Any,
        width: int,
        values: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather, normalize and pad every joint slot into one flat vector.

        The mask is built alongside because a slot can be partly real (a 12-DoF
        arm in a 14-wide slot) or entirely absent (a joint group this robot does
        not have), and only the real entries survive ``postprocess``.
        """
        chunks: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for joint in self.spec.joints:
            parts = slices.get(joint.name)
            if parts is None:
                chunks.append(torch.zeros(joint.max_dim, dtype=torch.float32))
                masks.append(torch.zeros(joint.max_dim, dtype=torch.bool))
                continue
            if values:
                value = _gather(raw, parts)
                key = key_of(joint.name)
                if joint.norm_type != "identity":
                    value = _normalize(value, self.spec.norm_stats[key], joint.norm_type)
                real = value.shape[-1]
            else:
                # Action slots are only needed for their mask here: the chunk does
                # not exist yet. Its real width is the one the stats were computed
                # at, which is also how upstream splits the chunk back apart.
                key = key_of(joint.name)
                real = self.spec.norm_width(key) if key in self.spec.norm_stats else sum(part.width for part in parts)
                value = torch.zeros(real, dtype=torch.float32)
            if real > joint.max_dim:
                raise ValueError(f"{key_of(joint.name)} is {real} wide but its slot is {joint.max_dim}")
            chunks.append(F.pad(value, (0, joint.max_dim - real)))
            masks.append(F.pad(torch.ones(real, dtype=torch.bool), (0, joint.max_dim - real)))
        vector = torch.cat(chunks, dim=-1).to(torch.float32)
        mask = torch.cat(masks, dim=-1)
        return F.pad(vector, (0, width - vector.shape[-1])), F.pad(mask, (0, width - mask.shape[-1]))

    def _build_images(self, frames: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resize each camera to a square, patchify it, and fill absent slots.

        A camera slot is never dropped — the prefix length is fixed by the
        checkpoint's align-query layout — so a missing frame keeps its slots and
        is masked off instead.
        """
        unknown = set(frames) - set(self.spec.camera_sources.values())
        if unknown:
            raise ValueError(
                f"observation has camera key(s) no slot maps from: {sorted(unknown)}; "
                f"known sources are {sorted(set(self.spec.camera_sources.values()))}"
            )

        processed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for camera in self.spec.cameras:
            source = self.spec.camera_sources[camera]
            if source in frames:
                processed[camera] = self._process_frame(frames[source])
        if not processed:
            raise ValueError(f"none of {sorted(self.spec.camera_sources.values())} is present")

        # Absent slots copy the first present camera's *shape* and grid, filled
        # with -1.0 — the bottom of the image processor's [-1, 1] output range.
        template_pixels, template_grid = next(iter(processed.values()))
        pixels, grids, masks = [], [], []
        for camera in self.spec.cameras:
            if camera in processed:
                frame_pixels, frame_grid = processed[camera]
                if frame_pixels.shape != template_pixels.shape:
                    raise ValueError(
                        f"{camera} patchifies to {tuple(frame_pixels.shape)}, but "
                        f"{tuple(template_pixels.shape)} is expected; cameras must share a grid"
                    )
            else:
                frame_pixels = torch.full_like(template_pixels, -1.0)
                frame_grid = template_grid
            pixels.append(frame_pixels)
            grids.append(frame_grid)
            masks.append(camera in processed)
        return (
            torch.stack(pixels, dim=0).unsqueeze(0),
            torch.tensor(masks, dtype=torch.bool).unsqueeze(0),
            torch.stack(grids, dim=0).to(torch.long).unsqueeze(0),
        )

    def _process_frame(self, frame: Any) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.as_tensor(np.asarray(frame), device="cpu")
        if image.ndim != 3:
            raise ValueError(f"expected one HWC frame, got shape {tuple(image.shape)}")
        if image.shape[-1] not in (1, 3):
            raise ValueError(f"expected a channel-last frame, got shape {tuple(image.shape)}")
        image = image.permute(2, 0, 1).contiguous().to(torch.float32)
        side = self.spec.image_size
        if image.shape[-2:] != (side, side):
            # Square, ignoring aspect ratio: that is what the policy was trained
            # on. Bit-identical to torchvision's ``Resize((side, side))``, which
            # is what upstream's deploy path uses.
            image = F.interpolate(
                image.unsqueeze(0),
                size=(side, side),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            ).squeeze(0)
        # The image processor's do_rescale divides by 255, so a frame that arrived
        # already scaled to [0, 1] (as lerobot decodes PNG-backed cameras) has to
        # go back to [0, 255] first or it collapses to ~-1.
        if float(image.max()) <= 2.0:
            image = (image * 255).round().clamp(0, 255).to(torch.uint8)
        out = self.image_processor(image)
        grid = torch.as_tensor(out["image_grid_thw"], device="cpu").reshape(-1, 3)[0]
        return out["pixel_values"].cpu(), grid

    def _build_language(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.use_chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=False
            )
        else:
            # The v1 (PaliGemma-shaped) prompt wrapper, kept because the flag is
            # part of a checkpoint's identity rather than a preference.
            text = prompt if prompt.startswith("<bos>") else f"<bos>{prompt}"
            text = text if text.endswith("\n") else f"{text}\n"
        tokenized = self.tokenizer(
            [text],
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return (
            tokenized["input_ids"].cpu(),
            tokenized["attention_mask"].to(device="cpu", dtype=torch.bool),
        )

    # -- output -----------------------------------------------------------
    def postprocess(self, actions: torch.Tensor, features: RobotFeatures) -> dict[str, np.ndarray]:
        """Normalized chunk -> the robot's own keys, in the robot's own units.

        Returns one array per source key named by the robot config — for a config
        whose actions all slice out of ``"action"``, that is a single
        ``(chunk, robot_dim)`` array laid out exactly like the robot's own command
        vector.
        """
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(f"postprocess handles one request; got batch {actions.shape[0]}")
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"expected a (chunk, dim) chunk, got {tuple(actions.shape)}")
        if actions.shape[-1] != features.action_mask.shape[-1]:
            raise ValueError(f"chunk is {actions.shape[-1]} wide, mask is {features.action_mask.shape[-1]}")

        chunk = actions.detach().to(torch.float32).cpu()[:, features.action_mask]
        state = features.state.detach().to(torch.float32).cpu()[0][features.state_mask]

        # Split the packed real dims back into joint groups, in slot order.
        states = self._split(state, self.spec.state_slices, _state_key)
        chunks = self._split(chunk, self.spec.action_slices, _action_key)

        out: dict[str, torch.Tensor] = {}
        for joint in self.spec.joints:
            if joint.name not in chunks:
                continue
            key = _action_key(joint.name)
            value = chunks[joint.name]
            if joint.norm_type != "identity":
                value = _unnormalize(value, self.spec.norm_stats[key], joint.norm_type)
            if self.spec.subtract_state.get(joint.name):
                if joint.name not in states:
                    raise ValueError(f"{key} is a delta on a state this robot does not report")
                reference = states[joint.name]
                if joint.norm_type != "identity":
                    reference = _unnormalize(reference, self.spec.norm_stats[_state_key(joint.name)], joint.norm_type)
                value = value + reference
            out[joint.name] = value
        return self._to_source_keys(out)

    def _split(
        self,
        packed: torch.Tensor,
        slices: Mapping[str, tuple[SourceSlice, ...]],
        key_of: Any,
    ) -> dict[str, torch.Tensor]:
        """Undo ``_build_vector``: the masked-out padding is already gone, so each
        joint group takes its real width off the front, in slot order."""
        out: dict[str, torch.Tensor] = {}
        offset = 0
        for joint in self.spec.joints:
            parts = slices.get(joint.name)
            if parts is None:
                continue
            key = key_of(joint.name)
            if joint.norm_type == "sincos":
                raise NotImplementedError(
                    f"{key}: sincos doubles a feature's width, which the norm stats do "
                    "not record; splitting the packed vector back apart is ambiguous."
                )
            width = self.spec.norm_width(key) if key in self.spec.norm_stats else sum(part.width for part in parts)
            out[joint.name] = packed[..., offset : offset + width]
            offset += width
        return out

    def _to_source_keys(self, by_joint: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Reassemble the robot's own vectors from the joint groups.

        Each source key collects the slices that were taken out of it, ordered by
        where they sat in the original vector, so the result is laid out exactly
        like the command vector the robot sent its state in.
        """
        collected: dict[str, list[tuple[int, torch.Tensor]]] = {}
        for joint_name, value in by_joint.items():
            offset = 0
            for part in self.spec.action_slices[joint_name]:
                if part.end < 0:  # a whole-vector source; nothing to reassemble
                    collected.setdefault(part.source, []).append((part.start, value))
                    break
                piece = value[..., offset : offset + part.width]
                collected.setdefault(part.source, []).append((part.start, piece))
                offset += part.width
        return {
            source: torch.cat([piece for _, piece in sorted(pieces, key=lambda item: item[0])], dim=-1)
            .to(torch.float32)
            .numpy()
            for source, pieces in collected.items()
        }


def _as_float_tensor(value: Any) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(np.asarray(value), device="cpu")
    return tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)


def _gather(raw: Mapping[str, torch.Tensor], parts: tuple[SourceSlice, ...]) -> torch.Tensor:
    """Concatenate the configured slices of the robot's raw vectors."""
    if not parts:
        raise ValueError("a joint group reported verbatim needs no gathering")
    pieces = []
    for part in parts:
        if part.source not in raw:
            raise KeyError(f"observation has no {part.source!r}")
        source = raw[part.source]
        pieces.append(source if part.end < 0 else source[..., part.start : part.end])
    return torch.cat(pieces, dim=-1)


def load_hf_processor(qwen3vl_path: str) -> tuple[Any, Any]:
    """The Qwen3-VL tokenizer and image processor, as upstream builds them.

    ``padding_side="right"`` matters: the prefix is laid out as
    ``[cameras..., prompt, align queries]`` at a fixed length, so prompt padding
    has to sit at the tail of the prompt span.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(qwen3vl_path, padding_side="right")
    return processor.tokenizer, processor.image_processor
