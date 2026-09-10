#!/usr/bin/env python3
"""Grade the vendored processor against upstream on the *real* eval observations.

``phase2_processor_parity.py`` already grades the same six kernel inputs, but on
synthetic inputs: a 256x256 random-noise frame, a ``standard_normal(14)`` state
and the short prompt ``"pick up the object"``. Every one of those dodges a code
path the open-loop evaluation actually takes:

* the bundle's frames are **240x320**, so the resize is a no-op in phase2 and
  live here;
* the real state has an idle arm pinned at exactly ``0.0``, which normalizes to
  the ``q01`` floor -- a boundary the random state never lands on;
* the real prompts are full RobotWin instructions ("Raise the bottle with white
  neck and red top from the table using the correct arm, the left arm."), long
  enough to exercise truncation against ``tokenizer_max_length``, where
  ``"pick up the object"`` is short enough to only exercise padding.

If the processor agrees here too, the 0.6 MAE is not a preprocessing bug and the
remaining suspect is the checkpoint/data fit itself.

    PYTHONPATH=. python spikes/lingbot_vla_v2/open_loop_processor_real.py \
        --dataset /llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bootstrap
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2_processor_parity import (  # noqa: E402
    MODEL_INPUT_KEYS,
    _compare,
    _upstream_configs,
)

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _observations(bundle, index: int, image_size: int = 256) -> tuple[dict, dict]:
    """One bundle row, in upstream's and the vendored processor's input shapes.

    Upstream's square resize lives in the *deploy policy*
    (``LingbotVlaV2Policy.resize_image``), not in ``FeatureTransform``, so it has
    to be replayed here. Skipping it does not merely change pixels: the Qwen3-VL
    image processor's own ``smart_resize`` then rounds the native 240x320 frame to
    256x320, i.e. 320 patches per camera instead of 256, and the whole prefix
    changes length. ``img_size`` defaults to 256 in the training dataset builder
    (``lingbotvla/data/dataset.py``) and in the deploy policy alike, and the
    RoboTwin data config overrides neither.
    """
    from torchvision.transforms import Resize

    resize = Resize((image_size, image_size))
    frames = {key: bundle["images"][index, camera] for camera, key in enumerate(CAMERA_KEYS)}
    state = bundle["states"][index].astype(np.float32)
    prompt = str(bundle["prompts"][index])
    upstream = {
        **{
            key: resize(
                torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).contiguous().float()
            )
            for key, frame in frames.items()
        },
        "observation.state": torch.from_numpy(state),
        "task": prompt,
    }
    vendored = {"images": frames, "state": state, "prompt": prompt}
    return upstream, vendored


def run(args: argparse.Namespace) -> int:
    bootstrap.setup(verbose=False)

    from lingbotvla.data.vla_data.utils import FeatureTransform
    from transformers import AutoProcessor

    from vllm_omni.diffusion.models.lingbot_vla_v2 import (
        LingbotVlaV2Config,
        LingbotVlaV2Processor,
        RobotSpec,
    )

    root = Path(args.lingbot_root)
    robot_path = root / "configs/robot_configs/robotwin.yaml"
    data_path = root / "configs/vla/robotwin/robotwin.yaml"
    stats_path = root / "assets/norm_stats/robotwin.json"
    qwen_path = Path(args.qwen3vl_path)

    hf_processor = AutoProcessor.from_pretrained(qwen_path.as_posix(), padding_side="right")
    data_config, upstream_model_config = _upstream_configs(data_path)
    upstream_processor = FeatureTransform(
        robot_path.as_posix(),
        data_config,
        upstream_model_config,
        hf_processor,
        chunk_size=50,
        norm_stats_path=stats_path.as_posix(),
        use_future_image=False,
    )
    config = LingbotVlaV2Config(qwen3vl_path=qwen_path.as_posix())
    spec = RobotSpec.from_files(robot_path, data_path, stats_path)
    vendored_processor = LingbotVlaV2Processor(
        spec,
        config,
        tokenizer=hf_processor.tokenizer,
        image_processor=hf_processor.image_processor,
    )

    bundle = np.load(args.dataset)
    passed = True
    for index in range(len(bundle["states"])):
        prompt = str(bundle["prompts"][index])
        print(f"\n=== sample {index}  frame={bundle['frame_indices'][index]}  {prompt[:58]!r}")
        upstream_obs, vendored_obs = _observations(bundle, index)
        upstream_features = upstream_processor.apply(upstream_obs, policy_eval=True)
        vendored_features = vendored_processor.preprocess(vendored_obs)
        vendored_inputs = vendored_features.model_inputs()
        for key in MODEL_INPUT_KEYS:
            passed &= _compare(key, upstream_features[key], vendored_inputs[key][0])

        # The chunk the model would emit is graded too: a normalization or slot
        # mismatch on the way *out* looks exactly like a bad prediction.
        chunk = torch.from_numpy(
            np.random.default_rng(index).standard_normal((50, 55)).astype(np.float32)
        )
        upstream_output = upstream_processor.unapply(
            {
                "state": upstream_features["state"],
                "actions": chunk,
                "state_joint_mask": upstream_features["state_joint_mask"],
                "action_joint_mask": upstream_features["action_joint_mask"],
            }
        )
        vendored_output = vendored_processor.postprocess(chunk, vendored_features)
        passed &= _compare(
            "action", upstream_output["action"], torch.from_numpy(vendored_output["action"])
        )

    print(
        "\nPASS - processor agrees with upstream on every real observation."
        if passed
        else "\nFAIL - processor diverges on real observations."
    )
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="open-loop NPZ bundle")
    parser.add_argument("--lingbot-root", default=bootstrap.LINGBOT_SRC.as_posix())
    parser.add_argument("--qwen3vl-path", default=bootstrap.QWEN3VL_PATH.as_posix())
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
