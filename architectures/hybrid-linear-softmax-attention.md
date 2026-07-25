# Hybrid Linear–Softmax Attention
*Depth — mixing linear attention with periodic softmax "anchor" layers to keep O(N) scaling without losing full-rank interactions.*

**TL;DR:** Pure linear attention scales as O(N) but produces a low-rank token-mixing operator that misses interactions softmax attention captures naturally. Hybrid Linear–Softmax Attention interleaves linear-attention layers with a *periodic minority* of full-softmax "anchor" layers (typically 25%, e.g. 3:1 linear:softmax within each block) so most compute stays sub-quadratic while every few layers still restore full-rank cross-token mixing. Scaled from scratch (not by linearizing a pretrained softmax stack) it matches full-softmax video DiTs at long sequence lengths.

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md), [block-attention-residuals](block-attention-residuals.md)

---

## What it is

A layer-level architecture recipe rather than a new attention kernel. Every attention layer in the transformer is one of two flavours:

- **Linear-attention layer** (gated linear attention or similar): O(N) time and memory, but token-mixing is fundamentally rank-limited by the fixed feature map.
- **Softmax anchor layer**: ordinary multi-head softmax attention, O(N²) but full-rank in its interactions.

The layers are interleaved at a fixed ratio — e.g., 3 linear layers between every softmax layer — so cost is dominated by linear, and quality by the softmax anchors that refresh full-rank representations.

## How it works

The design has three knobs:

1. **Ratio.** How many linear layers per softmax layer. Video-DiT-scale studies find ~25% softmax (i.e. 3:1) to be the quality/efficiency sweet spot — enough anchors to restore full-rank mixing, few enough to keep long-sequence scaling.
2. **Placement.** Anchors are periodic rather than clustered so that no long stretch of the network is starved of full-rank mixing. Early-only or late-only anchor placements underperform periodic.
3. **From-scratch vs linearized training.** Linearizing a pretrained softmax model (replacing softmax layers with linear ones and finetuning) reliably loses quality. Training the hybrid *from scratch* lets the linear layers learn representations that anticipate the anchor refresh, closing most of the gap.

The linear layers typically use gated linear attention (a state-space-adjacent form) so the "carrier" is expressive; the anchors do the heavy interaction work.

## Why it matters

Long-sequence generative modeling (video, long-context text, high-resolution image) is dominated by attention cost. Full linearization loses too much quality; full softmax doesn't scale. A hybrid stacks the desirable half of each: near-linear scaling with a small, well-placed softmax budget bought back the quality. In practice this lets a single H100 generate 480p video at competitive VBench scores in low-teens seconds.

## Gotchas & tricks

- **Don't linearize; train from scratch.** Post-hoc linearization systematically underperforms; the linear layers only pull their weight when trained *with* the anchors in place.
- **Ratio depends on modality.** 3:1 is a good default for video DiT; text-only stacks sometimes prefer fewer softmax layers, and highly interaction-heavy tasks may need more.
- **Anchor placement matters more than count.** Two anchors evenly spread beat two clustered at the top.
- **Downstream layers can starve.** Deep-layer effective rank drops if refreshed features aren't reused; pairing with [block-attention-residuals](block-attention-residuals.md) is the natural remedy.
- **KV cache design changes.** Anchors need a full softmax KV cache; linear layers need only their recurrent state. Serving stacks must handle both.

## Sources

- Paper: *SANA-Video 2.0: Hybrid Linear Attention with Attention Residuals for Efficient Video Generation* — Chen et al., NVIDIA, 2026 — [arXiv:2607.21553](https://arxiv.org/abs/2607.21553).
