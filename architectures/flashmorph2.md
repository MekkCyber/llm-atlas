# FlashMorph2
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Converts a pretrained Transformer into a **hybrid full-attention + linear-attention** model under a fixed budget of full-attention layers. The key insight is to jointly learn the per-layer selection scores and model weights **on-policy**: sampled discrete hybrid architectures are rolled out end-to-end on their own hidden states, fixing the two off-policy mismatches in the original FlashMorph. Beats FlashMorph on RULER, LongBench, and recall-intensive tasks across Qwen3 1.7B / 4B / 8B, with layer selection finishing in <4 GPU-hours for 8B.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [_sparse-attention.md](_sparse-attention.md)
**Related:** [moba.md](moba.md), [mla.md](mla.md)

---

## What it is

A layer-selection and fine-tuning recipe: given a dense pretrained Transformer and a target full-attention budget (say "keep 8 of 32 layers full"), pick which layers to keep full and convert the rest to a linear-attention variant (GLA, RWKV-style, or similar). FlashMorph2's contribution is a selection procedure that produces strictly better choices than per-layer scoring heuristics.

## How it works

Two off-policy problems in FlashMorph v1:

- **Architecture off-policy.** FlashMorph scored each candidate layer *in isolation* ("if I converted this one layer, what happens?"). The right question is "what happens when the final mixture is in place."
- **Activation off-policy.** The scoring forward pass used the dense model's activations, not the activations the final hybrid would produce.

FlashMorph2 fixes both:

1. **Discrete architecture sampling.** Maintain a distribution over per-layer *keep-full / convert* decisions. Sample a full discrete hybrid.
2. **End-to-end rollout on its own hidden states.** Run the sampled hybrid forward (and backward) on the training batch. The activations at every layer come from the sampled hybrid, not the dense model.
3. **Joint update.** Backpropagate through a straight-through gate on each per-layer decision, updating both the selection logits and the hybrid's weights (converted-layer parameters, kept-layer deltas).
4. **Short budget.** With a modest batch of samples per step, selection converges in under 4 GPU-hours for 8B.

## Why it matters

- **Hybrid attention is becoming the long-context default.** Jamba, Zamba, and a growing set of 2026 designs mix full attention with Mamba-/linear-attention layers. FlashMorph2 makes the mix choice cheap and principled rather than hand-tuned.
- **On-policy fixes the obvious bug.** Isolated-layer scoring is a known worst-case for architecture search; FlashMorph2 is a straightforward application of on-policy AutoML to the hybridisation problem.
- **Preserves most of the dense base.** Because selection is conditioned on the final hybrid's own activations, the kept layers learn a mild compensation for the converted layers' errors — cheaper than full fine-tuning.

## Gotchas & tricks

- **Full-attention budget matters.** Below a threshold (empirically ~¼ of layers), recall on RULER-style tasks collapses — no selection recipe saves you.
- **Converted-layer family matters.** FlashMorph2 is agnostic between linear-attention variants, but numerical results depend on which one you pick (GLA vs RWKV vs Mamba-2). The paper reports multiple linear-attention variants on Qwen3.
- **Not a free retrofit.** Still needs a fine-tuning pass on the sampled hybrid — short, but required. The ~4 GPU-hour figure is for selection *on top of* the fine-tune cost.
- **Scale untested past 8B.** Qwen3 1.7B / 4B / 8B are the reported scales; large-model behaviour is open.

## Sources

- Paper: *On-Policy Morphing into Hybrid Attention Models* — Xuanda Wang, Jianbin Zheng, Huafeng Kuang, Wei Tang, Yu Cheng (ByteDance Seed / NTU), 2026 — primary source.
- Paper: *FlashMorph* — prior version, ByteDance, 2025 — baseline FlashMorph2 fixes.
- Related: Jamba, Zamba — architectures that bake in the hybrid structure instead of converting.
