# Hybrid Attention (Full + Linear Layer Mix)
*Depth — Transformer variants that keep full softmax attention in a subset of layers and replace the rest with linear attention.*

**TL;DR:** Long-context serving costs are dominated by O(n²) attention. Fully swapping to linear attention destroys long-range retrieval; keeping every layer full is expensive. **Hybrid attention** models keep k full-attention layers and convert the remaining $L{-}k$ to linear — the trick is *which* layers to keep. Recent work like **FlashMorph** frames the layer-selection problem as a budget-constrained optimization over learnable gates on morphable layers, and finds converged layer masks with **20M tokens of fine-tuning** vs. 234M for prior methods.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [mla.md](mla.md), [transformer-block.md](transformer-block.md)

---

## What it is

Attention comes in two dominant complexity classes:
- **Full attention** — softmax over all pairs, O(n²) time, O(n) memory with KV cache. Preserves exact long-range retrieval.
- **Linear attention** — kernel or state-space variants (Mamba, RetNet, GLA, Linear Attention with feature maps), O(n) time, O(1) recurrent state. Cheap, but retrieval and in-context recall degrade sharply.

Hybrid models split the difference: keep a small number of full-attention layers where they matter most, and use linear attention everywhere else. The key question — and the source of most of the recent variants — is which layers to keep full, and how to identify them cheaply.

## How it works

A modern hybridization recipe (FlashMorph-style):

1. **Morphable layers.** Each layer is initialized as a *gated mixture* of full and linear attention: $\text{layer}(x) = g \cdot \text{FullAttn}(x) + (1{-}g) \cdot \text{LinAttn}(x)$, with $g \in [0, 1]$ learnable per layer.
2. **Budget constraint.** Fix a target: "keep k layers as full attention." Add a regularizer (e.g. $\ell_1$ on the gate vector, or a top-k mask straight-through estimator) that drives gates to 0 or 1.
3. **Fine-tune on retrieval-heavy synthetic data.** Long-context retrieval tasks — needle-in-a-haystack, multi-hop, key-value recall — reveal which layers must remain quadratic. Non-retrieval layers converge their gate to 0 (linear); retrieval-critical ones converge to 1 (full).
4. **Read out the mask.** After convergence, take the top-k layers by gate value and hardwire the model with full attention there and linear elsewhere.

Alternative recipes (with fixed masks, e.g. every-Kth-layer, first-and-last-N-layers, hand-designed) all lose to learned selection when the model is long-context tuned.

## Why it matters

- **Linear serving cost with recoverable long-context quality.** Hybrid models get most of the throughput/memory of pure-linear at most of the retrieval quality of pure-full. The exact tradeoff depends on which layers you keep.
- **Post-hoc conversion is now cheap.** FlashMorph's 20M-token fine-tune (vs. 234M for the prior best conversion) makes hybridization a practical *post-training step* rather than a bet you make at pre-training time.
- **Composes with orthogonal cheap-attention.** [MLA](mla.md), GQA, sliding-window can each be stacked onto the full-attention layers a hybrid model preserves.

## Gotchas & tricks

- **The "which layers" answer is not model-agnostic.** Different architectures / pretraining data yield different retrieval-critical layer distributions. Rerun the selector per model.
- **Retrieval-heavy synthetic data is essential.** Fine-tuning on generic pretraining data doesn't stress the retrieval-specific layers enough to distinguish them; use structured needle-in-a-haystack curricula.
- **Linear attention flavor matters.** Mamba-style state-space, GLA-style gated linear, RetNet-style retention all have different memory/recall tradeoffs. The selector's answer depends on which one is on the other side of the switch.
- **Don't hybridize small models blindly.** Long-context matters less for small models with short training contexts; the quality loss from any linearization can exceed the throughput gain.

## Sources

- Paper: *Morphing into Hybrid Attention Models* (FlashMorph) — Fudan / ByteDance Seed / CUHK, 2026 — [arXiv:2606.30562](https://arxiv.org/abs/2606.30562).
- Predecessor: *Hymba* — Nvidia, 2024 — parallel hybrid design.
- Related: *Mamba*, *RetNet*, *Gated Linear Attention* — the linear-attention half of a hybrid.
