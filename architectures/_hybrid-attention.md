# Hybrid Attention

*Taxonomy — mixing full attention and linear-attention layers in a single Transformer stack.*

**TL;DR:** Long-context inference is dominated by softmax attention's quadratic cost. Fully replacing attention with linear or state-space alternatives has quality gaps; **hybrid** stacks keep a small subset of layers as full attention and swap the rest for cheap alternatives. Every variant asks the same question — *which* layers stay full? — and answers it differently: fixed patterns (Jamba, Zamba), post-hoc scoring (Mamba-conversion baselines), or budget-constrained global search ([FlashMorph](flashmorph.md)).

**Related taxonomies:** [_moe.md](_moe.md) — the same "keep some, replace others" pattern, applied to FFN routing rather than attention.
**Depth files covered here:** [flashmorph](flashmorph.md)

---

## The problem

Softmax attention costs $O(N^2)$ in tokens; linear / state-space variants cost $O(N)$ but lose some in-context capability (recall, long-range copying, in-context learning). Neither is universally better. Empirically, a **small fraction** of full-attention layers preserves most of the quality gap while capturing most of the efficiency win — but only if the *right* layers stay full.

## The shared pattern

All hybrids fix a **budget** $k$ — how many full-attention layers you're allowed to keep out of $L$ total — then decide which $k$. Everything else is architecture and training details. The interesting variance across methods is the selection strategy:

- **Fixed placement.** Choose $k$ layer indices by hand (first-and-last, every $L/k$, alternating). Simple; ignores model-specific structure.
- **Layerwise scoring.** For each layer, compute a proxy score (attention entropy, ablation loss, retrieval-head hit rate); keep the top-$k$.
- **Global subset search.** Treat the choice as a budget-constrained subset optimization; account for layer interactions.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Fixed placement (Jamba, Zamba-style) | Hand-picked indices for full-attention layers | Zero search cost; may be suboptimal | Training a new hybrid from scratch |
| Layerwise scoring | Independent per-layer importance | Fast; ignores interactions between layers | Small budgets on well-behaved models |
| [FlashMorph](flashmorph.md) | Budget-constrained global subset optimization | Best quality at given budget; extra search compute | Converting an existing Transformer to hybrid |
| Full linear (no full-attention layers) | Everything is linear / SSM | Cheapest; largest quality drop on recall tasks | When capability tradeoff is acceptable |

## How to choose

If you are **converting** an existing dense Transformer (Llama, Qwen) into a hybrid without full retraining, use a global subset selector like [FlashMorph](flashmorph.md) — the interaction effects between layers are large enough that greedy layer scoring keeps the wrong set. If you are **pretraining** a hybrid from scratch, a well-tested fixed pattern is fine because you get to co-train the linear layers alongside the full ones. Full-linear architectures are attractive for extreme long-context serving; expect a real capability tax on recall-heavy tasks.

## Adjacent but distinct

- **[MLA](mla.md)** — reduces KV cache without changing the attention operator. Orthogonal to hybrid selection.
- **Sliding-window / streaming attention** — bounds attention range instead of replacing the operator. Complements hybrids.
- **Sparse attention (Longformer-style)** — same "cheap operator" bucket as linear attention but with different quality tradeoffs.

## Sources

- Paper: *Morphing into Hybrid Attention Models (FlashMorph)* — Fudan / ByteDance Seed / CUHK, 2026 — [arXiv:2606.30562](https://arxiv.org/abs/2606.30562).
- Paper: *Jamba: A Hybrid Transformer-Mamba Language Model* — AI21 Labs, 2024.
- Paper: *Zamba: A Compact 7B SSM Hybrid Model* — Zyphra, 2024.
