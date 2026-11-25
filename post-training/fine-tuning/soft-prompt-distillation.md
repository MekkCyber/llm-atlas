# Composable Soft-Prompt Distillation (CSPD)
*Depth — distill each RL expert into a soft prompt over a frozen base, then compose experts by prompt concatenation.*

**TL;DR:** RL post-training produces domain experts (math, code, agent), and consolidating them into one model with weight merging or multi-teacher on-policy distillation re-pays the consolidation cost every time a new expert arrives and erodes capabilities outside the target domains. CSPD (2026) keeps the pre-RL base frozen, distills each expert independently into its own soft prompt, and composes experts by concatenating prompts — no joint training when a new expert is added.

**Prereqs:** [README](README.md), [_rl](../_rl.md)
**Related:** [model-souping](../../pre-training/model-souping.md), [dpo](../dpo.md)

---

## What it is

An **input-space adaptation** method for continual post-training. A soft prompt — a learnable sequence of vectors prepended to the embedding layer — captures what an RL expert added over the base. Each expert gets exactly one prompt. To serve several experts jointly, concatenate their prompts; no retraining required.

## How it works

Per-expert training:

1. Freeze the pre-RL base model.
2. Initialize a soft-prompt tensor (sequence length $L$, hidden size $d$).
3. Distill from the RL expert: for each prompt in the expert's domain, minimize KL between frozen-base-with-soft-prompt and the RL expert's outputs.
4. Stop when the prompt recovers ≥90% of the expert's RL gain.

Composition at inference:

- Multi-expert serving = concatenate the trained soft prompts, in any order (order has minor effect per the paper's ablations).
- Routing is implicit through the input distribution — prompts specialize to their domains during distillation.

CSPD is tested across 23 math/code/agent settings and six held-out benchmarks.

## Why it matters

Three concrete wins over weight merging and multi-teacher on-policy distillation (MOPD):

1. **Adding an expert is additive, not consolidation.** No re-training of the full model when a new expert arrives; just train one more prompt.
2. **Base capabilities are preserved.** The frozen base means capabilities outside expert domains don't erode.
3. **Serving overhead is small.** Throughput stays within 1.5% of the unadapted backbone and runs ~2.4× faster than rank-4 LoRA.

In the multi-expert benchmark, the full composition ties MOPD (67.83 vs. 67.79) and generalizes better on held-out tasks. This is a credible path to industrial-scale continual post-training.

## Gotchas & tricks

- **One soft prompt suffices in 22/23 cases**, but failures do happen — complex experts may need multi-prompt stacks or deeper (prefix-tuning style) prompts.
- **Prompt length is a capacity knob.** The paper uses modest lengths ($L$ on the order of 50–100 tokens); longer prompts rarely help and cost context.
- **Catastrophic composition failures** can appear if two experts' domains overlap and conflict — the paper reports no such cases in its mix but notes the risk.
- Distillation data must cover the expert's domain; distilling on an off-distribution corpus gives a weak prompt.

## Sources

- Paper: Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training — Yang et al., 2026 — [arXiv:2611.27376](https://arxiv.org/abs/2611.27376)
