# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many of the 32 experts does one denoise step actually select?

Decides whether a grouped MoE kernel can reduce weight traffic. If nearly all
experts are hit by at least one of the 51 suffix tokens, top-4 routing saves
FLOPs but reads exactly the same weights, and the DRAM floor is unchanged.

It also answers the obvious follow-up — "if only 4 of 32 experts are active per
token, why move all 32?" — by measuring the quantity that actually sets weight
traffic: the number of *distinct* experts in the union over the tokens processed
together. An expert's weights are read once per invocation, not once per token,
so per-token sparsity only becomes byte sparsity when few enough tokens share an
invocation. The group sweep at the end measures where that crossover is, and
what splitting the 51 tokens to chase it would cost.
"""
import sys, torch, torch.nn.functional as F
sys.path[:0] = ["/llm/zhuyong/lingbovla/my/vllm-omni", "/llm/zhuyong/lingbovla/my/vllm-omni/spikes/lingbot_vla_v2"]
from pathlib import Path
from phase5_latency import build, observation

device, dtype = torch.device("xpu"), torch.float16
proc, model = build(Path("/tmp/lingbot-vla-v2-perf"), device, dtype, None, None)
model.qwenvl_with_expert.attention_backend = "eager"
model.qwenvl_with_expert.attention_precision = "fp16"

stats = []
selections = []  # the raw [T, top_k] choices, for the group sweep
def hook(mod, args, out):
    hidden = args[0]
    flat = hidden.reshape(-1, hidden.shape[-1])
    with torch.amp.autocast("xpu", enabled=False):
        logits = F.linear(flat.float(), mod.gate.weight.float())
    scores = logits.sigmoid() if mod.router_activation == "sigmoid" else F.softmax(logits, 1, dtype=torch.float)
    _, sel = torch.topk(scores + mod.e_score_correction_bias.unsqueeze(0), mod.top_k, dim=-1)
    stats.append((flat.shape[0], int(sel.unique().numel()), mod.num_experts))
    selections.append(sel.cpu())

from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import TokenMoeBlock
n = 0
for m in model.modules():
    if isinstance(m, TokenMoeBlock):
        m.register_forward_hook(hook); n += 1
print(f"hooked {n} TokenMoeBlock modules")

obs = observation(proc.spec, 0)
inputs = proc.preprocess(obs).to(device=device, dtype=dtype).model_inputs()
with torch.inference_mode():
    model.sample_actions(**inputs)

tok = {s[0] for s in stats}
E = stats[0][2]
hits = [s[1] for s in stats]
print(f"\ninvocations: {len(stats)}  (36 layers x 10 steps = 360)")
print(f"tokens per invocation: {sorted(tok)}   experts E={E}, top_k=4")
print(f"distinct experts selected per invocation: min={min(hits)} max={max(hits)} mean={sum(hits)/len(hits):.1f}")
full = sum(1 for h in hits if h == E)
print(f"invocations touching ALL {E} experts: {full}/{len(hits)} ({full/len(hits)*100:.1f}%)")
print(f"invocations touching >=30: {sum(1 for h in hits if h>=30)}/{len(hits)}")

# --- Why "only move the active experts" does not work -----------------------
#
# Weight bytes per invocation are (distinct experts in the union) x 2.36 MB,
# where 2.36 MB = 3 * 768 * 512 * 2 bytes for one expert's gate/up/down. So the
# question is how the union grows with the number of tokens sharing the read.
PER_EXPERT_MB = 3 * 768 * 512 * 2 / 1e6

print(f"\nunion of top-4 choices vs how many tokens share one weight read")
print(f"(one expert = {PER_EXPERT_MB:.2f} MB of gate+up+down, fp16)\n")
print(f"{'tokens/group':>13} {'groups':>7} {'distinct experts':>17} {'MB read':>9} {'vs dense':>9}")
print("-" * 62)
T = selections[0].shape[0]
dense_mb = 32 * PER_EXPERT_MB
for g in (1, 2, 4, 8, 16, 26, T):
    tot_distinct, tot_groups = 0.0, 0
    for sel in selections:
        for start in range(0, T, g):
            chunk = sel[start:start + g]
            tot_distinct += chunk.unique().numel()
            tot_groups += 1
    mean_distinct = tot_distinct / tot_groups
    # Per invocation: every group re-reads its own experts from DRAM.
    groups_per_inv = (T + g - 1) // g
    mb = mean_distinct * PER_EXPERT_MB * groups_per_inv
    print(f"{g:>13} {groups_per_inv:>7} {mean_distinct:>17.1f} {mb:>9.1f} {mb/dense_mb:>8.2f}x")

print(f"\ndense today: all 32 experts, one read = {dense_mb:.1f} MB per invocation")
print("Sparsity only reduces bytes when a group is small, but splitting the 51")
print("tokens re-reads weights per group and the product gets worse, not better.")
print("The dense path is already the byte-minimal grouping for T=51.")
