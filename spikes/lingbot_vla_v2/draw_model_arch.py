# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the LingBot-VLA 2.0 architecture figures used by ``MODEL_ARCH.md``.

    python spikes/lingbot_vla_v2/draw_model_arch.py [--checkpoint DIR] [--outdir DIR]

Five figures land in ``spikes/lingbot_vla_v2/figures/``:

===========================  =========================================================
``arch_overview.png``        the whole request: prefix pass, KV cache, denoise loop
``arch_joint_layer.png``     layer *i* of both towers, and the one attention they share
``arch_vision_tower.png``    inside the ViT: patch embed, block, merger, deepstack taps
``arch_token_moe.png``       inside the action expert's MLP: router + grouped experts
``arch_masks.png``           token layout and the two attention masks, **rendered by
                             calling the model's own ``make_att_2d_masks``**
``arch_params.png``          where the 6.38 G parameters actually sit
===========================  =========================================================

Labels are English because the container ships no CJK font (``fc-list`` is empty
and matplotlib only has DejaVu/STIX); a Chinese label would render as tofu boxes.

matplotlib is an opt-in dependency repo-wide (same as ``--plots`` in
``open_loop_eval.py``), so this is a spike script rather than an example.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

# Run from anywhere: a script's sys.path[0] is its own directory, so the repo
# root has to be added explicitly for the ``vllm_omni`` imports further down.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

DEFAULT_OUTDIR = Path(__file__).resolve().parent / "figures"

# draw.io's default palette: light fills with a matching darker stroke. Chosen so
# the figures stay legible in greyscale print and on a dark README background.
INPUT = ("#FFF2CC", "#B8912F")  # observations
VISION = ("#DAE8FC", "#5B7FB0")  # anything ViT
TEXT = ("#D5E8D4", "#6E9E5A")  # anything Qwen3-VL text tower
CACHE = ("#FFE6CC", "#C98A00")  # the KV cache
EXPERT = ("#F8CECC", "#A84A46")  # action expert / MoE
NORM = ("#E1D5E7", "#8A6A9E")  # norms, projections, small ops
OUTPUT = ("#D6D6D6", "#666666")  # actions leaving the model
PLAIN = ("#FFFFFF", "#999999")

STAGE_A_BG = ("#F4F8FF", "#5B7FB0")
STAGE_B_BG = ("#FFF6F5", "#A84A46")

ARROW = "#3B3B3B"
MONO = "DejaVu Sans Mono"

# Measured on robbyant/lingbot-vla-v2-6b with config.read_safetensors_shapes:
# 1708 tensors, 6375.91 M parameters. Recompute with --checkpoint.
PARAMS_M = {
    "Qwen3-VL text tower (36 layers)": 3633.51,
    "Expert routed MoE experts (36x32)": 1358.95,
    "Vision tower ViT (24 layers)": 415.35,
    "VLM embed_tokens (151936x2560)": 388.96,
    "Expert attention (36 layers)": 283.34,
    "Align/resampler heads (never run)": 120.68,
    "Expert AdaRMSNorm gamma/beta": 85.10,
    "Expert shared experts": 58.39,
    "Task projections + query tables": 28.84,
    "Action IO heads + MoE routers": 2.79,
}
# Fraction of each group that a single denoise step actually executes.
ACTIVE_FRACTION = {
    "Expert routed MoE experts (36x32)": 4 / 32,  # top-4 of 32
    "Expert attention (36 layers)": 1.0,
    "Expert AdaRMSNorm gamma/beta": 1.0,
    "Expert shared experts": 1.0,
    "Action IO heads + MoE routers": 1.0,
}


# ---------------------------------------------------------------------------
# Drawing primitives. All coordinates are "percent of figure width", and the y
# axis uses the same unit, so a square is square and 1 unit = width/100 inches.
# ---------------------------------------------------------------------------
def canvas(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * height / width)
    ax.axis("off")
    return fig, ax


def box(
    ax,
    x,
    y,
    w,
    h,
    title=None,
    body=None,
    *,
    color=PLAIN,
    title_size=10.5,
    body_size=8.3,
    lw=1.5,
    ls="solid",
    mono=False,
    alpha=1.0,
    title_y=None,
    zorder=2,
):
    """A rounded box spanning exactly ``[x, x+w] x [y, y+h]``."""
    face, edge = color
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.9",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
            zorder=zorder,
        )
    )
    font = {"family": MONO} if mono else {}
    if title and body:
        ax.text(
            x + w / 2,
            title_y if title_y is not None else y + h - 1.5,
            title,
            ha="center",
            va="top",
            fontsize=title_size,
            fontweight="bold",
            zorder=zorder + 1,
            **font,
        )
        ax.text(
            x + w / 2,
            y + (h - 3.2) / 2,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            linespacing=1.5,
            zorder=zorder + 1,
            **font,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title or body,
            ha="center",
            va="center",
            fontsize=title_size if title else body_size,
            fontweight="bold" if title else "normal",
            linespacing=1.5,
            zorder=zorder + 1,
            **font,
        )


def arrow(ax, p0, p1, *, color=ARROW, lw=1.8, ls="solid", rad=0.0, head=True, zorder=5):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        zorder=zorder,
        arrowprops={
            "arrowstyle": "-|>" if head else "-",
            "linewidth": lw,
            "linestyle": ls,
            "color": color,
            "shrinkA": 0,
            "shrinkB": 0,
            "connectionstyle": f"arc3,rad={rad}",
            "mutation_scale": 16,
        },
    )


def note(ax, x, y, text, *, size=8.0, ha="center", va="center", color="#444444", style="italic", rot=0, weight=None):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, style=style, rotation=rot, fontweight=weight)


def token_strip(ax, x, y, w, h, segments, *, size=7.4):
    """A proportional bar over a token sequence: ``segments = [(count, label, color)]``.

    Segments too narrow to hold their label get a leader line *upwards* instead,
    at two alternating heights so that neighbouring thin segments (the 8-token
    query blocks) do not overprint each other.
    """
    total = sum(count for count, _, _ in segments)
    cursor = x
    narrow_seen = 0
    for count, label, color in segments:
        seg_w = w * count / total
        face, edge = color
        ax.add_patch(
            FancyBboxPatch(
                (cursor, y),
                seg_w,
                h,
                boxstyle="round,pad=0,rounding_size=0.25",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.1,
                zorder=3,
            )
        )
        if seg_w > w * 0.06:
            ax.text(cursor + seg_w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, zorder=4)
        else:
            lead = 0.9 + 1.9 * (narrow_seen % 2)
            narrow_seen += 1
            mid = cursor + seg_w / 2
            ax.plot([mid, mid], [y + h, y + h + lead], color="#777777", lw=0.8, zorder=3)
            ax.text(mid, y + h + lead + 0.2, label, ha="center", va="bottom", fontsize=size, zorder=4)
        cursor += seg_w


# ---------------------------------------------------------------------------
# Figure 1: the whole request
# ---------------------------------------------------------------------------
def figure_overview(path: Path) -> None:
    fig, ax = canvas(15, 22.5)  # ylim 0..150

    ax.text(
        50,
        146.5,
        "LingBot-VLA 2.0  (lingbot-vla-v2-6b)  —  one action-chunk request",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    note(
        ax,
        50,
        142.3,
        "6.38 G parameters  ·  Qwen3-VL-4B backbone + Qwen2-shaped action expert  ·  flow matching, 10 Euler steps"
        "\nShapes are the released RoboTwin deployment: 3 cameras @256, 72 text tokens, chunk 50, action dim 55",
        size=9.2,
    )

    # ---- Stage A backdrop -------------------------------------------------
    box(ax, 0.5, 70, 99, 67, color=STAGE_A_BG, ls=(0, (6, 4)), lw=1.6, zorder=0, body=" ")
    note(ax, 3, 135, "STAGE A   Perception & context   —   runs ONCE per request", size=10.5, ha="left",
         style="normal", weight="bold", color=VISION[1])

    # inputs
    box(ax, 4, 122, 28, 9, "3 camera frames", "HWC uint8, any size\nbilinear+antialias resize to 256x256\n"
        "cam_top / wrist_left / wrist_right", color=INPUT)
    box(ax, 36, 122, 28, 9, "Language instruction", 'chat template -> tokenizer\nright-pad / truncate to 72 ids\n'
        '"pick up the bottle"', color=INPUT)
    box(ax, 68, 122, 28, 9, "Robot state", "14 joints (2 arms x (6 arm + 1 gripper))\nper-group bounds_99 normalise\n"
        "scatter into a 55-wide padded vector", color=INPUT)

    arrow(ax, (18, 122), (18, 117.5))
    arrow(ax, (50, 122), (50, 117.5))

    # encoders
    box(ax, 4, 106, 28, 11, "LingbotVisionTower  (ViT)",
        "Conv3d patch-embed 16x16 -> 1024\n+ interpolated 2-D pos-emb\n24 x [LN -> MHSA(16h, 2-D RoPE) -> LN -> MLP]\n"
        "2x2 merge -> 64 tokens x 2560 per camera\n415 M   (Fig. 3)", color=VISION)
    box(ax, 36, 106, 28, 11, "embed_tokens",
        "151936 x 2560 lookup table\nshared with <vision_start>/<vision_end>\nno lm_head (deleted at release)\n389 M",
        color=TEXT)
    box(ax, 68, 106, 28, 11, "Learned task-query tables",
        "depth / video align_embs  [256, 2560]\nmean-pool 256 rows -> 8 rows\ncurrent & future shared_task_proj\n"
        "fuse depth+video query (5120->2560)\n29 M  -  these are prefix TOKENS", color=NORM)

    arrow(ax, (18, 106), (18, 102.5))
    arrow(ax, (50, 106), (50, 102.5))
    arrow(ax, (82, 106), (82, 102.5))

    # prefix assembly
    box(ax, 4, 88, 92, 14, "embed_prefix   —   assemble the 286-token VLM prefix", " ", color=PLAIN)
    token_strip(
        ax, 7, 92.5, 86, 3.6,
        [
            (66, "cam_top:  <vis_start> + 64 patches + <vis_end>", VISION),
            (66, "cam_wrist_left:  66", VISION),
            (66, "cam_wrist_right:  66", VISION),
            (72, "language  72", TEXT),
            (8, "cur_depth 8", NORM),
            (8, "fut_depth 8", NORM),
        ],
    )
    note(ax, 50, 90.2, "prefix_len = 3x66 + 72 + 8 + 8 = 286   ·   only the inner 64 patch rows count as 'visual' "
                       "(the two boundary tokens advance mRoPE like text)", size=7.8)

    arrow(ax, (50, 88), (50, 85.5))

    # VLM tower
    box(ax, 4, 71.5, 92, 14, "Qwen3-VL-4B text tower   —   36 layers x hidden 2560   ·   3.63 G",
        "per layer:  RMSNorm -> q/k/v (32 Q / 8 KV heads, head_dim 128, per-head q-norm & k-norm)\n"
        "            -> interleaved 3-axis mRoPE (sections 24/20/20) -> attention -> o_proj -> +residual\n"
        "            -> RMSNorm -> SwiGLU MLP 2560 -> 9728 -> 2560 -> +residual\n"
        "deepstack: the ViT features tapped at vision layers 5/11/17 are ADDED onto the visual rows "
        "of text layers 0/1/2\nmask: causal over all 286 rows (vlm_causal=True)            (Fig. 2 for the "
        "layer internals, Fig. 5 for the mask)", color=TEXT, body_size=8.6)

    arrow(ax, (50, 71.5), (50, 68.5))

    # KV cache
    box(ax, 4, 61, 92, 7.5, "KV cache   —   36 layers x [286 tokens x 8 KV heads x 128]   (~5 MB in fp16)",
        "written once by the prefix pass, then FROZEN and re-read by all 10 denoise steps\n"
        "fixed size: no paged KV, no scheduler, no continuous batching -> this is a diffusion stage, not an AR stage",
        color=CACHE, body_size=8.6)

    # ---- Stage B backdrop -------------------------------------------------
    box(ax, 0.5, 7.5, 99, 51.5, color=STAGE_B_BG, ls=(0, (6, 4)), lw=1.6, zorder=0, body=" ")
    note(ax, 3, 57.6, "STAGE B   Flow-matching denoise   —   runs 10 x   (the only loop in the model)",
         size=10.5, ha="left", style="normal", weight="bold", color=EXPERT[1])

    # state bypass: straight from the input box down into embed_suffix
    arrow(ax, (96, 126.5), (98.5, 126.5), head=False, lw=1.4, ls=(0, (4, 3)), color="#8a6a9e")
    arrow(ax, (98.5, 126.5), (98.5, 56.2), head=False, lw=1.4, ls=(0, (4, 3)), color="#8a6a9e")
    arrow(ax, (98.5, 56.2), (20, 56.2), head=False, lw=1.4, ls=(0, (4, 3)), color="#8a6a9e")
    arrow(ax, (20, 56.2), (20, 54.2), lw=1.4, ls=(0, (4, 3)), color="#8a6a9e")
    note(ax, 97.4, 95, "robot state bypasses the VLM entirely", size=7.6, rot=90, color="#8a6a9e")

    box(ax, 4, 38, 44, 16, "embed_suffix   ->   51 suffix tokens x 768",
        "state -> state_proj 55->768   =  1 state token\n"
        "x_t   -> action_in_proj 55->768  (+)  tau(t) 768 sinusoidal\n"
        "      -> Linear 1536->768 -> SiLU -> Linear 768->768  = 50 action tokens",
        color=NORM, body_size=8.0)
    token_strip(ax, 7, 39.0, 38, 2.4, [(1, "state", NORM), (50, "50 action tokens", EXPERT)], size=7.2)

    box(ax, 52, 46, 38, 7, "x_t   [50, 55]", "t = 1.0  ->  x ~ N(0, I)   (or a seeded generator)", color=EXPERT)
    arrow(ax, (52, 49.5), (48.5, 49.5))

    # action expert
    box(ax, 4, 16, 86, 20,
        "Action expert   —   36 layers x hidden 768   ·   1.79 G stored,  ~0.60 G executed per step", " ",
        color=EXPERT, title_y=34.7)
    arrow(ax, (24, 38), (24, 36.3))

    inner = [
        (30.3, "1.  AdaRMSNorm(x, t)     out = (1 + gamma(t)) * RMSNorm(x) * w + beta(t)      "
               "gamma, beta = Linear 768->768 of the time embedding", NORM),
        (26.1, "2.  Joint attention      q/k/v 768 -> 32 Q heads / 8 KV heads x 128 (GQA, with bias)      "
               "K,V = [ 286 cached prefix  ||  51 own ] = 337", EXPERT),
        (21.9, "3.  AdaRMSNorm(x, t)     same FiLM conditioning again, before the MLP", NORM),
        (17.7, "4.  Token-MoE            router 768->32 in fp32 -> sigmoid -> top-4 of 32 -> renorm -> x4.0   "
               "+ always-on shared expert        (Fig. 4)", EXPERT),
    ]
    for y, text, color in inner:
        box(ax, 7, y, 80, 3.5, body=text, color=color, body_size=7.9, lw=1.1, mono=False)

    note(ax, 47, 16.9, "every layer: residual around (1,2) and around (3,4)   ·   both norms are time-conditioned, "
                       "which is how the expert knows where it is on the denoising trajectory", size=7.6)

    # KV cache feeds the expert attention
    arrow(ax, (96, 64.7), (97, 64.7), head=False, lw=1.6, color=CACHE[1])
    arrow(ax, (97, 64.7), (97, 27.85), head=False, lw=1.6, color=CACHE[1])
    arrow(ax, (97, 27.85), (87, 27.85), lw=1.6, color=CACHE[1])
    note(ax, 95.8, 44, "K/V read 10x", size=7.4, rot=90, color=CACHE[1])

    arrow(ax, (50, 16), (50, 15.1))
    box(ax, 10, 8.8, 60, 6.2, "action_out_proj  768 -> 55  on the last 50 rows   =>   velocity  v_t  [50, 55]",
        "Euler:   x_t  <-  x_t + dt * v_t   with dt = -1/10 ;   t <- t + dt\n(t: 1 = pure noise  ->  0 = action)",
        color=NORM, body_size=8.4)

    # loop back: down the left gutter into embed_suffix again
    arrow(ax, (10, 11.9), (2.5, 11.9), head=False, lw=1.8, color=EXPERT[1])
    arrow(ax, (2.5, 11.9), (2.5, 46), head=False, lw=1.8, color=EXPERT[1])
    arrow(ax, (2.5, 46), (4, 46), lw=1.8, color=EXPERT[1])
    note(ax, 3.9, 29, "x10", size=9.0, rot=90, color=EXPERT[1], weight="bold", style="normal")

    arrow(ax, (50, 8.8), (50, 7.6))
    box(ax, 10, 1.2, 80, 6.2, "actions  [50, 55]   ->   postprocess   ->   [50, 14]",
        "un-normalise per joint group, drop the padded dims, return under the robot's own key "
        "-> DiffusionOutput(output={'actions': ...})", color=OUTPUT, body_size=8.2)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: one layer of each tower, and the attention they share
# ---------------------------------------------------------------------------
def figure_joint_layer(path: Path) -> None:
    fig, ax = canvas(15, 11.5)  # ylim 0..76.7

    ax.text(50, 73.5, "Layer  i  —  the two towers are walked in lockstep and share ONE attention",
            ha="center", va="center", fontsize=15, fontweight="bold")
    note(ax, 50, 70.0,
         "LingbotJointModel.forward:  for each of the 36 layers, both towers project their own q/k/v, the two are "
         "concatenated along the sequence\naxis, one attention is taken over the union, and each tower consumes its "
         "own slice of the output.  Shared attention, separate weights.", size=8.6)

    lx, lw_ = 3, 42
    rx, rw = 55, 42

    box(ax, lx, 62, lw_, 4.6, body="VLM hidden  [286, 2560]", color=TEXT, body_size=9.5)
    box(ax, rx, 62, rw, 4.6, body="Expert hidden  [51, 768]", color=EXPERT, body_size=9.5)

    # time embedding source
    box(ax, 79, 55.5, 18, 4.2, body="tau(t)  [768]  sinusoidal", color=NORM, body_size=8.2)

    rows = [
        (55.0, 5.0,
         "RMSNorm (fp32 reduce)\nq_proj 2560->4096   k/v_proj 2560->1024\nper-head q_norm / k_norm (RMS over 128)",
         "AdaRMSNorm(x, tau(t))\nq_proj 768->4096   k/v_proj 768->1024   (+bias)\nno q/k norm"),
    ]
    box(ax, lx, 48.5, lw_, 6.2, body=rows[0][2], color=TEXT, body_size=8.0)
    box(ax, rx, 48.5, rw, 6.2, body=rows[0][3], color=EXPERT, body_size=8.0)
    arrow(ax, (lx + lw_ / 2, 62), (lx + lw_ / 2, 54.7))
    arrow(ax, (rx + rw / 2, 62), (rx + rw / 2, 54.7))
    arrow(ax, (79, 57.6), (rx + rw / 2 + 8, 54.7), rad=-0.2, color=NORM[1])

    # the shared band
    box(ax, lx, 36.5, rx + rw - lx, 10.0,
        "ONE joint attention",
        "q = cat([q_vlm, q_expert], dim=seq)    k, v likewise    ->  shared interleaved 3-axis mRoPE "
        "(one position basis for both towers)\n"
        "eager GQA: repeat_interleave K/V 8->32 heads, softmax(q k^T / sqrt(128) + mask), mask fill = -2.38e38 "
        "(never -inf: an all-masked row must stay uniform)\n"
        "prefix pass: only the VLM half exists, fill_kv_cache=True      denoise step: only the expert half exists, "
        "K/V = cat(cache, own)", color=CACHE, body_size=8.0)
    arrow(ax, (lx + lw_ / 2, 48.5), (lx + lw_ / 2, 46.7))
    arrow(ax, (rx + rw / 2, 48.5), (rx + rw / 2, 46.7))

    box(ax, lx, 28.5, lw_, 5.2, body="slice [0 : 286]  ->  o_proj 4096->2560\n+ residual", color=TEXT, body_size=8.2)
    box(ax, rx, 28.5, rw, 5.2, body="slice [286 : 337]  ->  o_proj 4096->768\n+ residual", color=EXPERT, body_size=8.2)
    arrow(ax, (lx + lw_ / 2, 36.5), (lx + lw_ / 2, 33.9))
    arrow(ax, (rx + rw / 2, 36.5), (rx + rw / 2, 33.9))

    box(ax, lx, 22.5, lw_, 4.2, body="RMSNorm", color=TEXT, body_size=8.5)
    box(ax, rx, 22.5, rw, 4.2, body="AdaRMSNorm(x, tau(t))", color=EXPERT, body_size=8.5)
    arrow(ax, (lx + lw_ / 2, 28.5), (lx + lw_ / 2, 26.9))
    arrow(ax, (rx + rw / 2, 28.5), (rx + rw / 2, 26.9))
    # tau(t) also conditions the second norm: routed down the right gutter, clear of the boxes
    arrow(ax, (97, 57.6), (98.6, 57.6), head=False, color=NORM[1], ls=(0, (4, 3)), lw=1.3)
    arrow(ax, (98.6, 57.6), (98.6, 24.6), head=False, color=NORM[1], ls=(0, (4, 3)), lw=1.3)
    arrow(ax, (98.6, 24.6), (97, 24.6), color=NORM[1], ls=(0, (4, 3)), lw=1.3)

    box(ax, lx, 13.5, lw_, 7.2, "SwiGLU MLP   (dense)",
        "down( SiLU(gate(x)) * up(x) )\n2560 -> 9728 -> 2560     74.7 M / layer", color=TEXT, body_size=8.2)
    box(ax, rx, 13.5, rw, 7.2, "Token-MoE   (all 36 layers)",
        "32 routed experts (SwiGLU 512, top-4) + shared (704)\n768 -> 768     39.4 M stored / 6.3 M active per layer",
        color=EXPERT, body_size=8.2)
    arrow(ax, (lx + lw_ / 2, 22.5), (lx + lw_ / 2, 20.7))
    arrow(ax, (rx + rw / 2, 22.5), (rx + rw / 2, 20.7))

    box(ax, lx, 6.5, lw_, 5.2, body="+ residual\n+ deepstack ViT feature (layers 0/1/2 only)", color=TEXT,
        body_size=8.2)
    box(ax, rx, 6.5, rw, 5.2, body="+ residual", color=EXPERT, body_size=8.2)
    arrow(ax, (lx + lw_ / 2, 13.5), (lx + lw_ / 2, 11.9))
    arrow(ax, (rx + rw / 2, 13.5), (rx + rw / 2, 11.9))

    note(ax, 50, 3.0,
         "Per-layer cost:  VLM 100.9 M params, run once   ·   Expert 49.8 M stored / 16.6 M active, run 10 times.  "
         "After layer 35: VLM RMSNorm, expert RMSNorm (not AdaRMSNorm - the release's model.norm carries only a "
         "weight).", size=8.2)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: inside the vision tower
# ---------------------------------------------------------------------------
def figure_vision_tower(path: Path) -> None:
    fig, ax = canvas(15, 9.6)  # ylim 0..64

    ax.text(50, 61, "Inside LingbotVisionTower  —  one camera frame  ->  64 tokens x 2560",
            ha="center", va="center", fontsize=15, fontweight="bold")
    note(ax, 50, 57.3,
         "Qwen3-VL ViT leaf modules, with the geometry (interpolated pos-emb, RoPE table, cu_seqlens) cached per "
         "grid_thw - robot cameras have a fixed\nresolution, so the cache hits on every request after the first.",
         size=8.6)

    box(ax, 2, 44.5, 17, 8.5, "frame", "256 x 256 x 3\nuint8, rescaled /255\nQwen3-VL image norm", color=INPUT,
        body_size=8.2)
    arrow(ax, (19, 48.8), (22, 48.8))
    box(ax, 22, 44.5, 20, 8.5, "patch_embed", "Conv3d(3 -> 1024)\nkernel = stride = (2, 16, 16)\n"
        "256 patches (16x16 grid)", color=VISION, body_size=8.2)
    arrow(ax, (42, 48.8), (45, 48.8))
    box(ax, 45, 44.5, 22, 8.5, "+ pos_embed", "2304-entry table, bicubic-\ninterpolated to this grid\n"
        "(fast_pos_embed_interpolate)", color=VISION, body_size=8.2)
    arrow(ax, (67, 48.8), (70, 48.8))
    box(ax, 70, 44.5, 28, 8.5, "2-D RoPE table", "rot_pos_emb(grid_thw) -> (cos, sin)\n"
        "applied inside every block\nboth cached per grid", color=NORM, body_size=8.2)

    arrow(ax, (32, 44.5), (32, 41))

    box(ax, 6, 22, 56, 19, "24 x Qwen3VLVisionBlock   (hidden 1024, 16 heads x 64)", " ", color=VISION,
        title_y=39.5)
    box(ax, 9, 32.5, 50, 4.0, body="LayerNorm(eps=1e-6)", color=PLAIN, body_size=8.0, lw=1.0)
    box(ax, 9, 27.6, 50, 4.4,
        body="Attention:  qkv 1024->3072 (bias) -> 16 heads -> 2-D RoPE (fp32) -> full attention -> proj 1024",
        color=PLAIN, body_size=7.6, lw=1.0)
    box(ax, 9, 22.8, 50, 4.2, body="+res  ->  LayerNorm  ->  MLP 1024 -> 4096 -> 1024 (GELU, bias)  ->  +res",
        color=PLAIN, body_size=7.6, lw=1.0)
    note(ax, 31.5, 20.3, "all 256 patches of one frame see each other\n(cu_seqlens keeps frames separate)",
         size=7.6, ha="right")

    # deepstack taps
    box(ax, 66, 22, 32, 19, "deepstack taps  (layers 5 / 11 / 17)",
        "after those 3 blocks the hidden state is also\nrun through its OWN patch merger\n"
        "(use_postshuffle_norm=True)\n\n"
        "the 3 results are ADDED onto the visual rows\nof TEXT layers 0 / 1 / 2 - an early, multi-scale\n"
        "injection of vision into the language tower", color=CACHE, body_size=8.0, title_y=39.5)
    arrow(ax, (62, 31.5), (66, 31.5), color=CACHE[1])

    arrow(ax, (34, 22), (34, 18.5))

    box(ax, 10, 8.5, 48, 10, "merger  (Qwen3VLVisionPatchMerger)",
        "LayerNorm(1024) -> view 2x2 spatial shuffle -> 4096\nLinear 4096->4096 -> GELU -> Linear 4096->2560\n"
        "256 patches  ->  64 tokens x 2560", color=VISION, body_size=8.2)
    arrow(ax, (58, 13.5), (62, 13.5))
    box(ax, 62, 8.5, 36, 10, "64 image tokens x 2560",
        "wrapped as <vision_start> + 64 + <vision_end> = 66\nrows in the prefix, x3 cameras\n"
        "invisible cameras are masked out, not skipped", color=TEXT, body_size=8.2)

    note(ax, 50, 4.0, "415 M params, executed once per request.  Cameras are batched: the 3 frames go through the "
                      "tower as one packed sequence.", size=8.2)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: inside the token MoE
# ---------------------------------------------------------------------------
def figure_token_moe(path: Path) -> None:
    fig, ax = canvas(15, 10.4)  # ylim 0..69.3

    ax.text(50, 66, "Inside TokenMoeBlock  —  the action expert's MLP, in all 36 layers",
            ha="center", va="center", fontsize=15, fontweight="bold")
    note(ax, 50, 62.3,
         "Routing is per TOKEN (51 of them per step), not per request.  The gate runs in true fp32 with autocast "
         "disabled: in bf16 the\nsigmoid scores of near-tied experts flip the top-4 selection, which changes the "
         "output discontinuously.", size=8.6)

    box(ax, 32, 54, 36, 5.0, body="hidden  [51, 768]   ->  flatten to [51, 768]", color=EXPERT, body_size=9.0)

    arrow(ax, (40, 54), (24, 49.5), rad=0.12)
    arrow(ax, (50, 54), (50, 49.5))
    arrow(ax, (60, 54), (80, 49.5), rad=-0.12)

    # router
    box(ax, 3, 30, 40, 19.5, "Router   (0.02 M / layer)",
        "gate: Linear(768 -> 32, no bias)\n"
        "     forced fp32, autocast disabled\n\n"
        "scores = sigmoid(logits)            [51, 32]\n"
        "choice = scores + e_score_correction_bias\n"
        "         (loss-free balancing bias: biases WHICH\n"
        "          experts win, never their weights)\n"
        "top-4 -> w = scores.gather(top4)\n"
        "w /= w.sum()      ->      w *= 4.0", color=NORM, body_size=8.0, title_y=48.0)

    # experts
    box(ax, 46, 24, 32, 25.5, "32 routed experts   (37.7 M / layer)",
        "stored GROUPED, as the release ships them:\n"
        "  gate_proj [32, 512, 768]\n"
        "  up_proj   [32, 512, 768]\n"
        "  down_proj [32, 768, 512]\n\n"
        "expert_e(x) = down_e( SiLU(gate_e x) * up_e x )\n\n"
        "'dense' kernel (default): every expert runs on\nevery token, 3 einsums, then the weights\n"
        "(zero for unselected pairs) contract the\nexpert axis away\n\n"
        "'gather' kernel: 8x fewer FLOPs but 32 tiny\nlaunches x 36 layers x 10 steps -> 3.7x SLOWER",
        color=EXPERT, body_size=7.7, title_y=48.0)

    box(ax, 81, 33, 16, 16.5, "shared expert",
        "always on, for\nevery token\n\nSwiGLU\n768 -> 704 -> 768\n\n1.6 M / layer", color=VISION, body_size=8.0,
        title_y=48.0)

    arrow(ax, (43, 39), (46, 39), color=NORM[1])
    note(ax, 44.5, 44.6, "top-4 ids + weights", size=7.0, rot=90, color=NORM[1])

    arrow(ax, (62, 24), (62, 20))
    arrow(ax, (89, 33), (89, 22), head=False)
    arrow(ax, (89, 22), (70, 22), head=False)
    arrow(ax, (70, 22), (70, 20))

    box(ax, 30, 13.5, 40, 6.0, body="out = sum_k w_k * expert_k(x)   +   shared(x)", color=EXPERT, body_size=9.0)
    arrow(ax, (50, 13.5), (50, 10))
    box(ax, 30, 5.0, 40, 5.0, body="reshape back to [1, 51, 768]", color=EXPERT, body_size=9.0)

    note(ax, 50, 1.5,
         "Per layer: 39.4 M stored, 6.3 M executed (4/32 routed + shared).  Across 36 layers: 1.42 G stored, "
         "0.23 G active - this is why a 6.4 G model denoises at ~33 ms/step.", size=8.2)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: token layout and the two attention masks, from the real code
# ---------------------------------------------------------------------------
def figure_masks(path: Path) -> None:
    import torch

    from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import make_att_2d_masks

    prefix_len, suffix_len = 286, 51

    # Prefix pass: vlm_causal=True fills att_masks with True -> pure causal.
    prefix_pad = torch.ones(1, prefix_len, dtype=torch.bool)
    prefix_att = torch.ones(1, prefix_len, dtype=torch.bool)
    prefix_mask = make_att_2d_masks(prefix_pad, prefix_att)[0].numpy()

    # Denoise step: suffix rows see every valid prefix column, plus the suffix's
    # own block structure ([state | actions], att_masks = [1, 1, 0, 0, ...]).
    suffix_pad = torch.ones(1, suffix_len, dtype=torch.bool)
    suffix_att = torch.zeros(1, suffix_len, dtype=torch.bool)
    suffix_att[:, :2] = True
    suffix_block = make_att_2d_masks(suffix_pad, suffix_att)[0]
    prefix_cols = torch.ones(1, suffix_len, prefix_len, dtype=torch.bool)
    denoise_mask = torch.cat([prefix_cols, suffix_block.unsqueeze(0)], dim=2)[0].numpy()

    fig = plt.figure(figsize=(15, 6.4))
    fig.suptitle("The two attention masks  (rendered by calling the real make_att_2d_masks)",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.905, "dark = attended,  light = masked out (filled with BIG_NEG = -2.3819763e38, never -inf)",
             ha="center", fontsize=9, style="italic", color="#555555")

    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.62], left=0.06, right=0.98,
                          top=0.79, bottom=0.20, wspace=0.30)
    shade = dict(cmap="Blues", interpolation="nearest", vmin=0, vmax=1.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(prefix_mask, **shade)
    ax1.set_title("STAGE A  ·  prefix pass\n286 x 286, causal (vlm_causal=True)", fontsize=11)
    ax1.set_xlabel("key / value position", fontsize=9)
    for edge in (66, 132, 198, 270, 278):
        ax1.axhline(edge - 0.5, color="#C1272D", lw=0.8, ls="--")
        ax1.axvline(edge - 0.5, color="#C1272D", lw=0.8, ls="--")
    for pos, label in ((33, "cam_top 66"), (99, "wrist_L 66"), (165, "wrist_R 66"), (234, "language 72"),
                       (274, "cur 8"), (282, "fut 8")):
        ax1.text(-8, pos, label, ha="right", va="center", fontsize=7.6)
    ax1.set_xticks([0, 66, 132, 198, 270])
    ax1.tick_params(axis="x", labelsize=8)
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(denoise_mask, aspect="auto", **shade)
    ax2.set_title("STAGE B  ·  one denoise step\n51 queries x 337 keys  =  [286 cached prefix | 51 own]", fontsize=11)
    ax2.set_xlabel("key / value position", fontsize=9)
    ax2.set_ylabel("query (suffix) position", fontsize=9)
    ax2.axvline(285.5, color="#C1272D", lw=1.6)
    label_bg = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.6)
    ax2.text(143, 25, "every suffix row sees\nall 286 prefix columns", ha="center", va="center", fontsize=8.4,
             bbox=label_bg)
    ax2.text(311, 25, "own\nblock", ha="center", va="center", fontsize=8.4, bbox=label_bg)
    ax2.set_yticks([0, 25, 50])
    ax2.set_yticklabels(["0  state", "25", "50"], fontsize=8)
    ax2.set_xticks([0, 100, 200, 286, 337])
    ax2.tick_params(axis="x", labelsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(denoise_mask[:, prefix_len:], **shade)
    ax3.set_title("zoom: the 51 x 51 own block\n(this is the only structure there is)", fontsize=11)
    ax3.set_xlabel("suffix key position", fontsize=9)
    ax3.axhline(0.5, color="#C1272D", lw=1.0)
    ax3.axvline(0.5, color="#C1272D", lw=1.0)
    ax3.text(-3, 0, "state", ha="right", va="center", fontsize=8)
    ax3.text(-3, 26, "50 action\ntokens", ha="right", va="center", fontsize=8)
    ax3.annotate("row 0 is blank from col 1 on:\nthe state token cannot\nsee the actions",
                 xy=(26, 0), xytext=(26, 14), ha="center", fontsize=8, color="#C1272D",
                 bbox=dict(facecolor="white", edgecolor="#C1272D", lw=0.8, pad=2.4),
                 arrowprops=dict(arrowstyle="-|>", color="#C1272D", lw=1.2))
    ax3.set_xticks([0, 25, 51])
    ax3.tick_params(axis="x", labelsize=8)
    ax3.set_yticks([])

    fig.text(0.5, 0.045,
             "Block structure of the suffix: att_masks = [1, 1, 0, 0, ...].  The state token opens one block, the "
             "first action token opens another, and the remaining 49 share it -\nso the state token cannot see the "
             "actions (the white notch in the zoom), while the 50 action tokens attend to each other bidirectionally: "
             "the whole chunk is predicted at once, not autoregressively.",
             ha="center", fontsize=8.6)

    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: parameter distribution
# ---------------------------------------------------------------------------
def measure_params(checkpoint: str) -> dict[str, float]:
    """Recompute PARAMS_M from a checkpoint's safetensors headers."""
    import re

    from vllm_omni.diffusion.models.lingbot_vla_v2.config import read_safetensors_shapes

    shapes = read_safetensors_shapes(checkpoint)
    totals = dict.fromkeys(PARAMS_M, 0.0)

    def bucket(name: str) -> str:
        if "align_head" in name or "resampler" in name:
            return "Align/resampler heads (never run)"
        if name.startswith("model.qwenvl_with_expert.qwenvl.model.visual"):
            return "Vision tower ViT (24 layers)"
        if "language_model.embed_tokens" in name:
            return "VLM embed_tokens (151936x2560)"
        if name.startswith("model.qwenvl_with_expert.qwenvl.model.language_model"):
            return "Qwen3-VL text tower (36 layers)"
        if "qwen_expert" in name:
            if ".mlp.experts." in name:
                return "Expert routed MoE experts (36x32)"
            if "shared_expert" in name:
                return "Expert shared experts"
            if "self_attn" in name:
                return "Expert attention (36 layers)"
            if re.search(r"layernorm\.|\.norm\.", name):
                return "Expert AdaRMSNorm gamma/beta"
            return "Action IO heads + MoE routers"  # gate + e_score_correction_bias
        if "align_emb" in name or "task_proj" in name:
            return "Task projections + query tables"
        return "Action IO heads + MoE routers"

    for name, shape in shapes.items():
        count = 1
        for dim in shape:
            count *= dim
        totals[bucket(name)] += count / 1e6
    return totals


def figure_params(path: Path, params: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    labels = list(params)
    stored = np.array([params[k] for k in labels])
    active = np.array([params[k] * ACTIVE_FRACTION.get(k, 0.0) for k in labels])
    order = np.argsort(stored)
    labels = [labels[i] for i in order]
    stored, active = stored[order], active[order]

    colors = []
    for label in labels:
        if "never run" in label:
            colors.append("#BBBBBB")
        elif label.startswith("Expert") or "IO heads" in label:
            colors.append(EXPERT[0])
        elif "ViT" in label:
            colors.append(VISION[0])
        elif "Task projections" in label:
            colors.append(NORM[0])
        else:
            colors.append(TEXT[0])

    y = np.arange(len(labels))
    ax.barh(y, stored, color=colors, edgecolor="#555555", linewidth=0.9, label="stored")
    ax.barh(y, active, height=0.40, color="#C0392B", alpha=0.85, label="executed per denoise step")
    total = stored.sum()
    for i, (value, act) in enumerate(zip(stored, active, strict=True)):
        text = f"  {value:,.1f} M   ({100 * value / total:.1f}%)"
        if act > 0:
            text += f"   |  active {act:,.1f} M"
        ax.text(value, i, text, va="center", fontsize=8.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, stored.max() * 1.52)
    ax.set_xlabel("parameters (millions)")
    ax.set_title(f"Where the {total:,.0f} M parameters sit  —  and what a single denoise step touches",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.5, 0.012,
             "The VLM half (4.44 G, 70%) runs once; the expert half (1.79 G) runs 10 times but only ~0.60 G of it "
             "is executed per step, because the MoE selects 4 of 32 experts.",
             ha="center", fontsize=8.8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, type=Path)
    parser.add_argument("--checkpoint", default=None, help="recompute the parameter table from this checkpoint")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    params = measure_params(args.checkpoint) if args.checkpoint else dict(PARAMS_M)

    figure_overview(args.outdir / "arch_overview.png")
    figure_joint_layer(args.outdir / "arch_joint_layer.png")
    figure_vision_tower(args.outdir / "arch_vision_tower.png")
    figure_token_moe(args.outdir / "arch_token_moe.png")
    figure_masks(args.outdir / "arch_masks.png")
    figure_params(args.outdir / "arch_params.png", params)
    for name in sorted(p.name for p in args.outdir.glob("arch_*.png")):
        print(f"wrote {args.outdir / name}")


if __name__ == "__main__":
    main()
