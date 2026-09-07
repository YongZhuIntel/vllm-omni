# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How far is our action chunk from a PyTorch fp32 reference, in the OpenVINO metric?

The export repo grades its IR with ``validation/validate_e2e_split.py``: same
observation, same fixed initial noise, PyTorch fp32 ``sample_actions`` as the
reference, and the metric taken on the final ``(1, 50, 55)`` chunk. It reports

    OpenVINO FP16   cosine 0.998544   mae 1.608e-02   mse 8.716e-04
    OpenVINO INT8   cosine 0.995633   mae 2.882e-02   mse 2.604e-03

This reproduces that measurement for the vLLM-Omni kernel so the two can be put
side by side. ``metric_stats`` below is a **verbatim** copy of
``validate_e2e_split.py:31-49`` -- a differently-defined MAE is not a smaller
MAE, and the whole point of this script is comparability.

What is deliberately *not* measured here
----------------------------------------
Task accuracy. This protocol feeds uniform-random pixels and a zero state; it
cannot tell you whether the policy completes ``adjust_bottle``. That number is
``open_loop_eval.py`` (mae 0.0112 against dataset ground truth, physical units)
and it must be re-run after any dtype change. See ``PHASE7_NUMERICS.md``.

Reference choice
----------------
The export repo's reference is *upstream* fp32; ours is the *vendored* kernel at
fp32 on CPU. ``phase1_parity.py`` puts those two within ~1e-7 of each other over
45 stages, i.e. four orders of magnitude below the 1e-2 effects under study, so
the substitution is free and it buys a clean dtype isolation: reference and
candidate are then the same code, differing only in dtype and device.

That also means we need no analogue of the export repo's "fp16-source vs
fp32-source" diagnostic. It exists there to separate the fp16 dtype floor from
OpenVINO's own implementation error, because those are two different codebases.
Here there is one codebase, so the measured number *is* the dtype floor.

Device and dtype are still confounded on XPU (fp32 does not fit in 23.9 GiB, so
there is no ``xpu:float32`` row to subtract). Add ``cpu:bfloat16`` to bound it:
phase 0 already puts CPU bf16 at mae 4.22e-2 against XPU bf16's 4.62e-2, so the
device contributes ~9% and the dtype carries the rest.

    PYTHONPATH=. python spikes/lingbot_vla_v2/phase7_numeric_parity.py \
        --model /tmp/lingbot-open-loop-eager

    # dynamic-range audit: does fp16's 65504 ceiling clip anything?
    PYTHONPATH=. python spikes/lingbot_vla_v2/phase7_numeric_parity.py \
        --model /tmp/lingbot-open-loop-eager \
        --candidates xpu:bfloat16 --activation-audit

    # Grade the opt-in Inductor path against the same fp32 reference.
    PYTHONPATH=. python spikes/lingbot_vla_v2/phase7_numeric_parity.py \
        --model /tmp/lingbot-vla-v2-perf \
        --candidates xpu:float16 xpu:float16:compiled \
        --noise-seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for path in (str(HERE), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import bootstrap  # noqa: E402
from phase5_latency import build  # noqa: E402

# The export repo's protocol constants, from ``make_reference_bundle.py`` and
# ``wrappers.make_prefix_example``: 3 cameras of 224x224 uniform-random pixels
# under generator seed 0, a fixed prompt, and a zero state.
NUM_CAMS = 3
IMAGE_SIZE = 224
IMAGE_SEED = 0
PROMPT = "pick up the object"

# ``make_reference_bundle.py`` seeds the *global* RNG with 0; ``phase0_torch_spike.py``
# used a private generator seeded 1234. Same protocol otherwise, so both are
# offered -- ``phase0`` reproduces the artifacts already on disk.
NOISE_PROTOCOLS = {"export": 0, "phase0": 1234}

FP16_MAX = 65504.0

# Reported by the export repo, for context in the output table. Not measured here.
REFERENCE_ROWS = {
    "OpenVINO FP16": (0.998544, 1.608e-02, 8.716e-04, 1.846e-01, 1.291e-01),
    "OpenVINO INT8": (0.995633, 2.882e-02, 2.604e-03, 3.135e-01, 2.089e-01),
}


def metric_stats(ref, out):
    """Verbatim from ``validation/validate_e2e_split.py``. Do not "improve" it."""
    ref32 = ref.astype(np.float32)
    out32 = out.astype(np.float32)
    diff = ref32 - out32
    abs_diff = np.abs(diff)
    ref_flat = ref32.reshape(ref32.shape[0], -1)
    out_flat = out32.reshape(out32.shape[0], -1)
    dots = np.sum(ref_flat * out_flat, axis=1)
    ref_norm = np.linalg.norm(ref_flat, axis=1)
    out_norm = np.linalg.norm(out_flat, axis=1)
    cosine = dots / np.clip(ref_norm * out_norm, 1e-12, None)
    return {
        "mae": float(abs_diff.mean()),
        "mse": float(np.square(diff).mean()),
        "max_abs": float(abs_diff.max()),
        "p99_abs": float(np.percentile(abs_diff, 99)),
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
    }


def timestep_drift(dtype: torch.dtype, num_steps: int) -> float:
    """Where the Euler clock actually lands, given ``sample_actions`` keeps it in model dtype.

    ``modeling_lingbot_vla_v2.py:1231-1294`` accumulates ``time = time + dt`` at
    ``dtype`` rather than recomputing ``1 - i/n``, so the loop does not end at 0.
    Free to compute and it isolates one of the three bf16 error sources.
    """
    dt = torch.tensor(-1.0 / num_steps, dtype=dtype)
    time_t = torch.tensor(1.0, dtype=dtype)
    for _ in range(num_steps):
        time_t = time_t + dt
    return float(time_t)


def make_inputs(model: Any, processor: Any, protocol: str, seed: int) -> dict[str, torch.Tensor]:
    """The export repo's exact inputs, at fp32. Candidates cast from these.

    Generating at fp32 and casting is identical to generating at the target
    dtype: ``make_prefix_example`` itself ends in ``.to(dtype)``.

    Only the *noise* varies with ``seed``; the observation is fixed, exactly as
    in ``make_reference_bundle.py``.
    """
    import wrappers

    # ``make_prefix_example`` only reaches ``fm.config.tokenizer_max_length`` and
    # the processor's two sub-processors, so our objects duck-type into it.
    class _Shim:
        config = model.config

    hf_processor = type(
        "_P", (), {"tokenizer": processor.tokenizer, "image_processor": processor.image_processor}
    )()
    images, img_masks, lang_tokens, lang_masks, image_grid_thw = wrappers.make_prefix_example(
        _Shim(), hf_processor, num_cams=NUM_CAMS, size=IMAGE_SIZE, prompt=PROMPT,
        dtype=torch.float32, seed=IMAGE_SEED,
    )
    config = model.config
    state = torch.zeros(1, config.max_state_dim, dtype=torch.float32)
    if protocol == "export":
        torch.manual_seed(seed)
        noise = torch.randn(1, config.chunk_size, config.max_action_dim)
    else:
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(1, config.chunk_size, config.max_action_dim, generator=generator)
    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
        "image_grid_thw": image_grid_thw,
        "noise": noise,
    }


def cast_inputs(inputs: dict, device: torch.device, dtype: torch.dtype) -> dict:
    """Floats cast, ids and masks do not -- the same rule ``RobotFeatures.to`` uses."""
    return {
        key: tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else None)
        for key, tensor in inputs.items()
    }


class ActivationAudit:
    """Max |activation| per leaf module, to answer whether fp16's ceiling clips.

    Audit the **bf16** run for this: bf16 carries fp32's exponent range, so its
    magnitudes are the true ones. Auditing an fp16 run instead only tells you
    whether it *already* overflowed, which is the same question asked too late.
    """

    def __init__(self, model: torch.nn.Module):
        self.peaks: dict[str, float] = {}
        self.nonfinite: dict[str, int] = {}
        self._handles = []
        for name, module in model.named_modules():
            if next(module.children(), None) is not None:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _args, output):
            for tensor in torch.utils._pytree.tree_flatten(output)[0]:
                if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                    continue
                finite = torch.isfinite(tensor)
                bad = int((~finite).sum())
                if bad:
                    self.nonfinite[name] = self.nonfinite.get(name, 0) + bad
                peak = float(tensor.where(finite, torch.zeros((), dtype=tensor.dtype,
                                                              device=tensor.device)).abs().max())
                self.peaks[name] = max(self.peaks.get(name, 0.0), peak)

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def report(self, top: int = 12) -> dict:
        ranked = sorted(self.peaks.items(), key=lambda kv: kv[1], reverse=True)[:top]
        worst = ranked[0][1] if ranked else 0.0
        print(f"\n  activation audit -- fp16 ceiling {FP16_MAX:.0f}, "
              f"observed peak {worst:.1f} ({FP16_MAX / max(worst, 1e-9):.1f}x headroom)")
        for name, peak in ranked:
            flag = "  <-- OVERFLOWS fp16" if peak >= FP16_MAX else ""
            print(f"    {peak:12.1f}  {name}{flag}")
        if self.nonfinite:
            print("  NON-FINITE activations:")
            for name, count in sorted(self.nonfinite.items(), key=lambda kv: -kv[1])[:top]:
                print(f"    {count:8d}  {name}")
        return {
            "peak": worst,
            "headroom": FP16_MAX / max(worst, 1e-9),
            "top": [{"module": n, "peak": p} for n, p in ranked],
            "nonfinite": self.nonfinite,
        }


def sample(model: Any, inputs: dict, device: torch.device, num_steps: int) -> np.ndarray:
    with torch.inference_mode():
        actions = model.sample_actions(**inputs, num_steps=num_steps)
    if device.type == "xpu":
        torch.xpu.synchronize()
    return actions.float().cpu().numpy()


def run_config(
    model_dir: Path, device: torch.device, dtype: torch.dtype, num_steps: int,
    protocol: str, seeds: list[int], audit: bool, compiled: bool,
    attention_precision: str = "fp32", compiled_prefix: bool = False,
) -> tuple[dict[int, np.ndarray], dict | None]:
    """Build once, sample every noise seed, tear down.

    Rebuilt per *config* rather than per sample so fp32-on-CPU and bf16-on-XPU
    are never resident together, and so N seeds cost N samples rather than N
    builds -- the build is 4-16 s, the sample 1-15 s.
    """
    processor, model = build(model_dir, device, dtype, num_steps, None)
    model.qwenvl_with_expert.attention_precision = attention_precision
    if compiled_prefix:
        model.prefix_forward = torch.compile(
            model.prefix_forward,
            backend="inductor",
            dynamic=False,
            fullgraph=True,
        )
    if compiled:
        model.predict_velocity = torch.compile(
            model.predict_velocity,
            backend="inductor",
            dynamic=False,
            fullgraph=True,
        )
    # Only the first seed is audited: the hooks fire on every module of a 6B
    # model, and the peak magnitudes do not depend on the noise draw.
    auditor = ActivationAudit(model) if audit else None
    audit_report = None
    chunks: dict[int, np.ndarray] = {}
    for seed in seeds:
        inputs = cast_inputs(make_inputs(model, processor, protocol, seed), device, dtype)
        t0 = time.perf_counter()
        chunks[seed] = sample(model, inputs, device, num_steps)
        print(f"[phase7] {device.type}:{str(dtype).split('.')[-1]} seed {seed} "
              f"sampled in {time.perf_counter() - t0:.1f}s")
        if auditor is not None:
            auditor.close()
            audit_report = auditor.report()
            auditor = None
    if auditor is not None:
        auditor.close()
    del model, processor, auditor
    gc.collect()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    return chunks, audit_report


def checkpoint_fingerprint(model_dir: Path) -> str:
    """What weights this prepared directory actually resolves to.

    The prepared dirs under ``/tmp`` are symlink farms, and
    ``run_open_loop_eval.sh --model-root`` will happily re-point one at a
    different checkpoint. Keying the fp32 golden cache on the directory *path*
    alone would then serve a foundation-model reference for a RoboTwin run
    without saying so, which is exactly the failure that cost this port a day.
    """
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if not shards:
        return "no-shards"
    resolved = shards[0].resolve()
    return f"{resolved}:{resolved.stat().st_size}"


def reference(args, model_dir: Path, seeds: list[int]) -> dict[int, np.ndarray]:
    """The fp32 CPU golden, one chunk per noise seed, cached across runs.

    ~25.5 GB of host RAM to build, so the cache is worth having even though a
    single sample is only ~15 s.
    """
    path = Path(args.reference)
    fingerprint = checkpoint_fingerprint(model_dir)
    cached: dict[int, np.ndarray] = {}
    if path.exists() and not args.refresh_reference:
        stored = np.load(path)
        same = (
            int(stored["num_steps"]) == args.num_steps
            and str(stored["protocol"]) == args.protocol
            # Caches written before the fingerprint existed have no entry, so
            # they cannot be trusted and are rebuilt rather than assumed good.
            and "checkpoint" in stored.files
            and str(stored["checkpoint"]) == fingerprint
        )
        if same:
            cached = {int(k.removeprefix("seed_")): stored[k]
                      for k in stored.files if k.startswith("seed_")}
            print(f"[phase7] reference: {path} (cached seeds {sorted(cached)})")
        else:
            print(f"[phase7] reference: {path} was built under a different protocol or "
                  f"checkpoint -- rebuilding")
    missing = [s for s in seeds if s not in cached]
    if missing:
        print(f"[phase7] building the fp32 CPU reference for seeds {missing} (~25.5 GB RAM) ...")
        fresh, _ = run_config(model_dir, torch.device("cpu"), torch.float32,
                              args.num_steps, args.protocol, missing, audit=False, compiled=False)
        cached.update(fresh)
        np.savez(path, num_steps=args.num_steps, protocol=args.protocol,
                 model_dir=str(model_dir), checkpoint=fingerprint,
                 **{f"seed_{s}": chunk for s, chunk in cached.items()})
        print(f"[phase7] wrote {path}")
    return {s: cached[s] for s in seeds}


def aggregate(per_seed: list[dict]) -> dict:
    """Collapse one ``metric_stats`` per seed into one row.

    Means for the averaged quantities, a max for ``max_abs`` (a max of maxima is
    still a max), and the MAE range kept explicitly -- with a single noise draw
    there is no way to tell a real 2x from a lucky one.
    """
    maes = [m["mae"] for m in per_seed]
    return {
        "mae": float(np.mean(maes)),
        "mae_min": float(np.min(maes)),
        "mae_max": float(np.max(maes)),
        "mse": float(np.mean([m["mse"] for m in per_seed])),
        "max_abs": float(np.max([m["max_abs"] for m in per_seed])),
        "p99_abs": float(np.mean([m["p99_abs"] for m in per_seed])),
        "cosine_mean": float(np.mean([m["cosine_mean"] for m in per_seed])),
        "cosine_min": float(np.min([m["cosine_min"] for m in per_seed])),
        "seeds": len(per_seed),
    }


def print_table(rows: list[tuple[str, dict]]) -> None:
    header = (f"{'path':<26} {'cosine':>10} {'MAE':>11} {'MSE':>11} "
              f"{'max abs':>11} {'p99 abs':>11}  {'MAE range':>21}")
    print("\n" + header)
    print("-" * len(header))
    for name, values in REFERENCE_ROWS.items():
        cos, mae, mse, mx, p99 = values
        print(f"{name:<26} {cos:10.6f} {mae:11.3e} {mse:11.3e} {mx:11.3e} {p99:11.3e}"
              f"  {'(1 noise draw)':>21}")
    print("-" * len(header))
    for name, m in rows:
        span = f"{m['mae_min']:.3e}-{m['mae_max']:.3e}"
        print(f"{name:<26} {m['cosine_mean']:10.6f} {m['mae']:11.3e} {m['mse']:11.3e} "
              f"{m['max_abs']:11.3e} {m['p99_abs']:11.3e}  {span:>21}")


def main() -> int:
    args = parse_args()
    bootstrap.setup(verbose=False)
    model_dir = Path(args.model)

    seeds = args.noise_seeds or [NOISE_PROTOCOLS[args.protocol]]
    ref = reference(args, model_dir, seeds)
    stacked = np.stack([ref[s] for s in seeds])
    print(f"[phase7] reference {stacked.shape}  rms {np.sqrt((stacked.astype(np.float64) ** 2).mean()):.4f}  "
          f"abs max {np.abs(stacked).max():.4f}  seeds {seeds}")

    rows: list[tuple[str, dict]] = []
    report: dict[str, Any] = {"protocol": args.protocol, "num_steps": args.num_steps,
                              "model": str(model_dir), "seeds": seeds, "candidates": {}}
    for spec in args.candidates:
        parts = spec.split(":")
        if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != "compiled"):
            raise ValueError(f"candidate must be device:dtype or device:dtype:compiled, got {spec!r}")
        device_name, dtype_name = parts[:2]
        compiled = len(parts) == 3
        device, dtype = torch.device(device_name), getattr(torch, dtype_name)
        chunks, audit_report = run_config(
            model_dir,
            device,
            dtype,
            args.num_steps,
            args.protocol,
            seeds,
            args.activation_audit,
            compiled,
            args.attention_precision,
            args.compile_prefix,
        )
        stats = aggregate([metric_stats(ref[s], chunks[s]) for s in seeds])
        stats["timestep_end"] = timestep_drift(dtype, args.num_steps)
        if audit_report is not None:
            stats["activation_audit"] = audit_report
        rows.append((f"ours {device_name}:{dtype_name}{' compiled' if compiled else ''}", stats))
        report["candidates"][spec] = stats

    print_table(rows)
    print("\ntimestep the Euler clock actually ends on (0.0 is exact):")
    for name, stats in rows:
        print(f"  {name:<26} {stats['timestep_end']:+.6f}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\n[phase7] wrote {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="prepared vLLM-Omni model directory")
    parser.add_argument("--candidates", nargs="+", default=["xpu:bfloat16", "xpu:float16"],
                        help="device:dtype pairs, e.g. xpu:bfloat16 cpu:bfloat16")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--protocol", choices=sorted(NOISE_PROTOCOLS), default="export",
                        help="noise seeding mechanism: 'export' matches make_reference_bundle.py")
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=None,
                        help="noise draws to average over (default: the protocol's own seed). "
                             "The observation is fixed; only the noise varies.")
    parser.add_argument("--reference", default=str(HERE / "phase7_golden_fp32.npz"))
    parser.add_argument("--refresh-reference", action="store_true")
    parser.add_argument("--activation-audit", action="store_true",
                        help="per-module peak |activation|; run it on a bf16 candidate")
    parser.add_argument("--attention-precision", choices=("fp32", "fp16"), default="fp16",
                        help="attention accumulation precision for candidates (default: fp16)")
    parser.add_argument("--compile-prefix", action="store_true",
                        help="compile the Prefix walk for each candidate")
    parser.add_argument("--out", default=None, help="write the full report as JSON")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
