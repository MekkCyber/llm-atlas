# FlashMorph

*Depth — a budget-constrained global subset optimizer for Transformer→hybrid attention conversion.*

**TL;DR:** When converting a dense Transformer into a hybrid model (some layers keep full attention, others become linear attention), *which* layers stay full is the dominant lever on quality. FlashMorph frames this as a **budget-constrained subset optimization** problem — you have $k$ "full attention" slots out of $L$ layers, and you must pick the set that maximizes downstream performance under global interactions. It replaces heuristic per-layer scoring with a scalable global selector.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [_hybrid-attention.md](_hybrid-attention.md)
**Related:** [mla.md](mla.md)

---

## What it is

Prior Transformer→hybrid conversion methods score each layer's importance independently (attention entropy, ablation loss, retrieval-hit rate) and keep the top-$k$. That treats layer importance as isolated — but empirically the effect of one full-attention layer depends strongly on what the neighboring layers are. Layer 12 might look critical in isolation and useless if layer 11 is already full.

FlashMorph reformulates the choice as: *among all $\binom{L}{k}$ subsets, which one gives the best downstream quality?* Then it makes that search tractable.

## How it works

### Objective

Given an $L$-layer Transformer and a budget $k$, find the subset $S \subseteq \{1, \ldots, L\}$ with $|S| = k$ that maximizes model quality $Q$ when layers in $S$ keep full attention and layers outside $S$ become linear attention. Naively $\binom{L}{k}$ evaluations are needed — a lot for $L = 80$, $k = 8$.

### Making search tractable

FlashMorph uses a **fast layer-selection procedure** that avoids re-training and full-forward evaluation for every candidate subset. The key primitives:

- Cheap proxy for subset quality that captures cross-layer interactions rather than summing independent per-layer scores.
- Greedy-with-look-ahead style search (or a similar approximation) that prunes obviously-bad subsets early.
- Reuse of activations across candidate subsets so that swapping one layer between full and linear doesn't require a full re-forward.

The result is a search cost measured in a handful of forward-pass equivalents rather than $\binom{L}{k}$.

### Where linear attention plugs in

FlashMorph is agnostic to the linear-attention family used for the "replaced" layers — it could be RetNet-style linear attention, GLA, or Mamba/SSM cells. What FlashMorph decides is *which* layers get replaced; how they are replaced is a separate design choice.

## Why it matters

- **The selection choice is worth several points of quality** at the same budget compared to fixed-pattern and per-layer scoring baselines. That's the whole point: with the same throughput target, FlashMorph gives you a better model.
- **Post-hoc conversion without full retraining** is much more attractive than pretraining a new hybrid. If you already have a strong dense Transformer, FlashMorph is the shortest path to a hybrid variant.
- **Names a real problem.** Prior hybrid work implicitly assumed layer importance is additive. FlashMorph makes the counterexamples explicit and gives a method that handles them.

## Gotchas & tricks

- **The proxy metric is a knob.** Any subset-quality proxy is imperfect; expect the ranking of subsets to differ across proxies. Validate the chosen subset with a real downstream eval before committing.
- **Budget interacts with linear-attention choice.** A weaker linear attention (fewer states, worse recall) demands a larger $k$; FlashMorph doesn't tell you how to pick $k$.
- **Search is one-shot.** After you pick the subset, you still need light fine-tuning to close the gap between the linear layers and their dense originals.
- **Not a drop-in for pretraining.** FlashMorph solves the *conversion* problem. If you're pretraining a hybrid from scratch, you're better off with a fixed pattern and co-training the linear layers than running FlashMorph inside the training loop.

## Sources

- Paper: *Morphing into Hybrid Attention Models* — Fudan University / ByteDance Seed / CUHK, 2026 — [arXiv:2606.30562](https://arxiv.org/abs/2606.30562). Introduces FlashMorph.
- Related: *Jamba* (2024), *Zamba* (2024) — hybrid architectures with fixed patterns that motivated the selection question.
