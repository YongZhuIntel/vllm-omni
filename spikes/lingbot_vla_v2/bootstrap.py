"""Phase 0 spike bootstrap: make the upstream LingBot-VLA-2.0 source importable
from inside the vLLM-Omni environment.

This is a throwaway scaffold for the go/no-go experiment described in the port
plan. It does NOT touch vllm_omni; it only wires sys.path and reuses the
CPU/CUDA-free import patches that the OpenVINO export repo already proved out
(``export_common/export_patches.py``).

The two things this has to survive:

  * the upstream tree pins ``transformers==4.57.3`` while vLLM-Omni runs
    transformers 5.x — every symbol the model imports has to still resolve;
  * the upstream tree is CUDA/training coupled (lerobot / wandb / flash_attn at
    import time, Triton ``group_gemm`` autotune probing ``torch.cuda``) — the
    export patches stub all of that out.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Defaults match this machine's layout; override via env for other checkouts.
EXPORT_REPO = Path(
    os.environ.get(
        "LINGBOT_EXPORT_REPO",
        "/llm/zhuyong/lingbovla/my/frameworks.robotics.embodied-intelligence.lingbot-vla-v2",
    )
)
LINGBOT_SRC = Path(os.environ.get("LINGBOT_VLA_SRC", EXPORT_REPO / "lingbot-vla-v2"))
QWEN3VL_PATH = Path(os.environ.get("QWEN3VL_PATH", EXPORT_REPO / "qwen3vl_base_config"))
CKPT_DIR = Path(os.environ.get("LINGBOT_CKPT", "/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b"))


def _prepend(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def _defuse_flash_attn_stub() -> None:
    """transformers 5.x crashes if ``flash_attn`` *imports* but has no distribution.

    ``export_patches._STUB_TOP_LEVEL`` includes ``flash_attn``, which was the right
    call under transformers 4.57 but is actively harmful here: transformers 5.8's
    ``is_flash_attn_2_available()`` does

        is_available, ver = _is_package_available("flash_attn", return_version=True)
        is_available = is_available and "flash-attn" in [
            p.replace("_", "-") for p in PACKAGE_DISTRIBUTION_MAPPING["flash_attn"]
        ]

    With the stub in place the first term is True, so the second is evaluated and
    ``PACKAGE_DISTRIBUTION_MAPPING["flash_attn"]`` raises KeyError (the stub has no
    installed distribution). Without the stub the ``and`` short-circuits and FA is
    correctly reported absent.

    So: drop flash_attn from the stub roots (every upstream import of it is guarded
    by ``is_flash_attn_available()``), and defensively give the mapping empty entries
    in case something else fabricates the module.
    """
    import export_patches as ep

    ep._STUB_TOP_LEVEL = tuple(
        name for name in ep._STUB_TOP_LEVEL if name not in ("flash_attn", "flash_attn_2_cuda")
    )

    from transformers.utils import import_utils as tf_import_utils

    for pkg in ("flash_attn", "flash_attn_interface"):
        tf_import_utils.PACKAGE_DISTRIBUTION_MAPPING.setdefault(pkg, [])


# transformers 4.57 -> 5.x renames, for names the upstream tree still imports.
# Both are used ONLY by ``lingbotvla/models/loader.py`` (the training-side
# from-scratch loader); the inference path never calls it, it goes through
# ``load_state_dict``. Everything else the model imports (Qwen2 / Qwen3-VL
# modeling internals, GradientCheckpointingLayer, FlashAttentionKwargs,
# ALL_ATTENTION_FUNCTIONS, eager_attention_forward, ...) still resolves under 5.8.
_TF_TOPLEVEL_ALIASES = {"AutoModelForVision2Seq": "AutoModelForImageTextToText"}

# Symbols renamed inside individual transformers modules: (module, old_name, new_name).
# These are plain (non-lazy) modules, so a setattr sticks.
_TF_SUBMODULE_ALIASES = [
    # Only reached because the v2 model file imports the v1 (Qwen2.5-VL) module for
    # AdaRMSNorm / FlowMatching. A real port should cut that dependency instead.
    ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2RMSNorm", "Qwen2_5_VLRMSNorm"),
]


def _shim_transformers_v5() -> list[str]:
    """Re-add the transformers 4.57 symbols that 5.x removed.

    Note the trap: ``setattr(transformers, name, value)`` does NOT survive in
    transformers 5.x. The package installs a ``_LazyModule`` into ``sys.modules``
    and a later lazy attribute access (e.g. importing ``AutoProcessor``) swaps in a
    *new* module object with a fresh ``__dict__`` — silently dropping any attribute
    we injected. Verified on 5.8.0: after a plain setattr,
    ``sys.modules["transformers"] is transformers`` becomes False.

    So patch the **class** (``_LazyModule.__getattr__``), which survives module
    re-creation, and resolve the alias on demand.
    """
    import contextlib

    from transformers.utils import import_utils as tf_import_utils

    applied = []
    lazy_cls = tf_import_utils._LazyModule
    if not getattr(lazy_cls, "_lingbot_spike_patched", False):
        original_getattr = lazy_cls.__getattr__

        def __getattr__(self, name):  # noqa: N807
            alias = _TF_TOPLEVEL_ALIASES.get(name)
            if alias is not None and getattr(self, "__name__", None) == "transformers":
                return original_getattr(self, alias)
            return original_getattr(self, name)

        lazy_cls.__getattr__ = __getattr__
        lazy_cls._lingbot_spike_patched = True
        applied.append("_LazyModule.__getattr__ alias: " + ", ".join(_TF_TOPLEVEL_ALIASES))

    import importlib

    for mod_name, old_name, new_name in _TF_SUBMODULE_ALIASES:
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, old_name):
            setattr(mod, old_name, getattr(mod, new_name))
            applied.append(f"{mod_name.rsplit('.', 1)[-1]}.{old_name} -> {new_name}")

    from transformers import modeling_utils

    if not hasattr(modeling_utils, "no_init_weights"):
        # Removed outright in 5.x. A no-op context manager is correct here: we
        # always overwrite the randomly-initialized weights with the checkpoint.
        @contextlib.contextmanager
        def no_init_weights(*args, **kwargs):
            yield

        modeling_utils.no_init_weights = no_init_weights
        applied.append("modeling_utils.no_init_weights -> no-op")

    if not hasattr(tf_import_utils, "is_safetensors_available"):
        # Dropped in 5.x (safetensors is now a hard dependency of transformers).
        tf_import_utils.is_safetensors_available = lambda: True
        applied.append("import_utils.is_safetensors_available -> True")

    # XPU-specific trap. transformers 5.8:
    #
    #   def is_flash_attn_available():
    #       return (is_flash_attn_4_available() or is_flash_attn_3_available()
    #               or is_flash_attn_2_available() or is_torch_npu_available()
    #               or is_torch_xpu_available())
    #
    # i.e. it is True on *any* XPU box, because transformers routes XPU through its
    # own FA kernels. Upstream reads it as "the flash_attn PyPI package is
    # importable" and does an unguarded ``from flash_attn.layers.rotary import
    # apply_rotary_emb`` behind it (qwenvl_in_vla.py:31-33) -> ModuleNotFoundError.
    # The same code imports fine on a CPU-only host, which is why the OpenVINO
    # export repo never hit this. We force eager/SDPA anyway, so report False.
    from transformers import modeling_flash_attention_utils as tf_fa

    if tf_fa.is_flash_attn_available():
        tf_fa.is_flash_attn_available = lambda: False
        applied.append("modeling_flash_attention_utils.is_flash_attn_available -> False (XPU)")

    return applied


# Training-only third-party packages that upstream ``__init__`` chains import at
# module load but that the vLLM-Omni env does not ship. ``lingbotvla/ops/__init__``
# reaches ``lingbotvla.data.data_loader`` (torchdata) purely as import fallout — no
# inference code path touches it. Only the ones actually absent get stubbed, so a
# genuinely installed package is never shadowed.
_TRAINING_ONLY_DEPS = (
    "torchdata",
    "torchcodec",
    "datasets",
    "av",
    "imageio",
    "jsonlines",
    "blobfile",
    "h5py",
    "zstandard",
    "peft",
    "qwen_vl_utils",
    "decord",
)


def _absent_training_deps() -> tuple[str, ...]:
    import importlib.util

    absent = []
    for name in _TRAINING_ONLY_DEPS:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            absent.append(name)
    return tuple(absent)


def setup(verbose: bool = True):
    """Wire sys.path + install the CUDA-free import stubs. Call before importing
    anything from ``lingbotvla``. Returns the ``export_patches`` module."""
    for path in (EXPORT_REPO / "export_common", LINGBOT_SRC, EXPORT_REPO / "converter"):
        if not path.exists():
            raise FileNotFoundError(f"spike bootstrap: missing {path}")
        _prepend(path)

    # QWEN3VL_PATH is read by build_model / the config merge; export it for the
    # child code paths that call os.environ.get directly.
    os.environ.setdefault("QWEN3VL_PATH", str(QWEN3VL_PATH))
    os.environ.setdefault("LINGBOT_VLA_SRC", str(LINGBOT_SRC))

    import export_patches as ep

    _defuse_flash_attn_stub()
    shims = _shim_transformers_v5()
    extra = _absent_training_deps()
    ep.prepare_imports(extra_stubs=extra)
    if verbose:
        for shim in shims:
            print(f"[bootstrap] tf5 shim  : {shim}")
        if extra:
            print(f"[bootstrap] extra stub: {', '.join(sorted(extra))}")
        print(f"[bootstrap] export repo : {EXPORT_REPO}")
        print(f"[bootstrap] lingbot src : {LINGBOT_SRC}")
        print(f"[bootstrap] qwen3vl cfg : {QWEN3VL_PATH}")
        print(f"[bootstrap] checkpoint  : {CKPT_DIR}")
    return ep


def import_modeling(verbose: bool = True):
    """``build_model._import_modeling()`` plus the post-import transformers-5 fixes.

    Returns ``(modeling_module, LingbotVLAV2Config)``.
    """
    import build_model

    M, cfg_cls = build_model._import_modeling()
    notes = _fix_tied_weights_keys() + _restore_qwen3vl_proxies() + _adapt_get_rope_index(M)
    for note in notes:
        if verbose:
            print(f"[bootstrap] tf5 shim  : {note}")
    return M, cfg_cls


def _adapt_get_rope_index(M) -> list[str]:
    """``Qwen3VLModel.get_rope_index`` gained a required ``mm_token_type_ids`` arg.

    4.57 derived the image/video token positions internally from ``input_ids``;
    5.x makes the caller pass a per-token modality tensor (text=0, image=1,
    video=2). Upstream's ``QwenvlWithExpertV2Model.build_prefix_position_ids``
    (modeling_lingbot_vla_v2.py:265-273) still uses the old signature, so rebuild
    the tensor from the token ids and forward it.

    NOTE this is the one shim so far that is *behavioural*, not a rename: it feeds
    the mrope position-id computation. A port must diff these position ids against
    a transformers-4.57 reference — the OpenVINO export repo's
    ``validation/make_reference_bundle.py`` bundle is the natural baseline.
    """
    import torch

    model_cls = M.QwenvlWithExpertV2Model
    if getattr(model_cls, "_lingbot_spike_rope_patched", False):
        return []

    def build_prefix_position_ids(self, input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None):
        config = self.qwenvl.config
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids = torch.where(input_ids == image_token_id, 1, mm_token_type_ids)
        video_token_id = getattr(config, "video_token_id", None)
        if video_token_id is not None:
            mm_token_type_ids = torch.where(input_ids == video_token_id, 2, mm_token_type_ids)
        position_ids, _ = self.qwenvl.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        return position_ids

    model_cls.build_prefix_position_ids = build_prefix_position_ids
    model_cls._lingbot_spike_rope_patched = True
    return ["QwenvlWithExpertV2Model.build_prefix_position_ids -> pass mm_token_type_ids"]


def _restore_qwen3vl_proxies() -> list[str]:
    """transformers 4.57's ``Qwen3VLForConditionalGeneration`` exposed convenience
    proxies onto the inner ``Qwen3VLModel``; 5.x dropped them.

    The v2 model reaches through ``self.qwenvl.visual`` in ``get_image_features``
    (modeling_lingbot_vla_v2.py:228) and for the vision-tower eval/param walks. Its
    other reach-throughs (``self.qwenvl.model.language_model``,
    ``self.qwenvl.model.get_rope_index``) go via ``.model`` and still resolve, so
    ``visual`` is the only one to re-add.
    """
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import Qwen3VLForConditionalGeneration

    if hasattr(Qwen3VLForConditionalGeneration, "visual"):
        return []
    Qwen3VLForConditionalGeneration.visual = property(lambda self: self.model.visual)
    return ["Qwen3VLForConditionalGeneration.visual property re-added"]


def _fix_tied_weights_keys() -> list[str]:
    """``_tied_weights_keys`` changed from list[str] (4.x) to dict (5.x).

    The upstream subclasses override it with the 4.x form::

        class Qwen3VLForConditionalGeneration(...):
            _tied_weights_keys = ["lm_head.weight"]

    which makes transformers 5.8's ``post_init()`` blow up in
    ``get_expanded_tied_weights_keys`` with ``'list' object has no attribute
    'keys'``. Restore the 5.x mapping (same meaning: lm_head is tied to the input
    embedding).

    Note we cannot read the mapping back off the HF base classes here: the upstream
    patches rebind ``transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM`` to
    the lingbot subclass, so the "base" would hand back the same 4.x list. The two
    literal mappings below are transformers 5.8's own values.
    """
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2ForCausalLM
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import Qwen3VLForConditionalGeneration

    tied = (
        (Qwen3VLForConditionalGeneration, {"lm_head.weight": "model.language_model.embed_tokens.weight"}),
        (Qwen2ForCausalLM, {"lm_head.weight": "model.embed_tokens.weight"}),
    )
    notes = []
    for subclass, mapping in tied:
        if isinstance(subclass.__dict__.get("_tied_weights_keys"), list):
            subclass._tied_weights_keys = dict(mapping)
            notes.append(f"{subclass.__name__}._tied_weights_keys list -> dict")
    return notes


def env_report() -> dict:
    import torch
    import transformers

    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "xpu_available": xpu,
    }
    if xpu:
        props = torch.xpu.get_device_properties(0)
        info["xpu_device"] = props.name
        info["xpu_total_gib"] = round(props.total_memory / 2**30, 1)
    return info
