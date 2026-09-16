# Attention Variants

*Taxonomy — attention-mechanism variants designed primarily to shrink or restructure the KV cache while keeping quality.*

**TL;DR:** Modern LLM attention almost never runs as full per-head multi-head attention (MHA) — the KV cache dominates serving cost at long context. Variants shrink the cache along three orthogonal axes: **share across heads** (MQA/GQA), **project into a shared latent** (MLA), or **cache only values and reconstruct keys on demand** (GVA). Additional variants restrict the *reach* of attention (sliding-window, sparse). This taxonomy names the axes, links depth files, and gives the modern default.

**Related taxonomies:** n/a
**Depth files covered here:** [multi-head-attention](multi-head-attention.md) · [mla](mla.md) · [gva](gva.md) · [sliding-window-attention](sliding-window-attention.md)

---

## The problem

Standard MHA caches, for every token and every layer, one K and one V vector per head. At long context (32K–1M) that cache dominates GPU memory during serving, capping concurrency and context length before any FLOP concern applies. Every variant in this taxonomy is fighting that constraint while trying not to lose quality.

## The shared pattern

Every variant answers three questions:

1. **What still exists per head?** — Full per-head K/V (MHA), shared K/V across a group (GQA/MQA), or none at all (MLA/GVA).
2. **What replaces it in the cache?** — A shared latent (MLA), a shared value plus reconstruction (GVA), or a shared K/V pair (MQA/GQA).
3. **How is position carried?** — Baked into the cached K (MHA/GQA/MQA), or split into a **decoupled RoPE channel** stored separately (MLA, GVA).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [multi-head-attention](multi-head-attention.md) | Full per-head K and V, cached per token per layer | Largest cache | Small models, short context, teaching baselines |
| MQA (no depth file yet) | One K/V shared across all query heads | Quality drop at scale | Latency-critical short-context serving |
| GQA (no depth file yet) | K/V shared across groups of query heads | Modest quality vs MHA | The pragmatic modern default for open weights |
| [mla](mla.md) | Cache one shared low-rank latent per token; each head reconstructs K/V from it | Needs a bespoke kernel and a decoupled-RoPE slice | Long-context serving where cache is the bottleneck |
| [gva](gva.md) | Cache values only; reconstruct content keys from values with a learned linear map | Reconstruction-quality tradeoff at scale; kernel work in flight | Memory-bound serving where a further ~45% cut over GQA matters |
| [sliding-window-attention](sliding-window-attention.md) | Each token attends only to a local window; interleave with occasional full-attention layers | Loses information beyond the window unless bridged | Cheap long-context bases with retrieval / tool use bolted on |

## How to choose

- **Modern open-weights default: GQA.** It's the "no thinking required" choice — a modest quality drop vs MHA and roughly proportional cache savings.
- **Long-context serving where cache dominates: MLA.** DeepSeek-V2/V3 have shown ~50× cache reduction with quality on par with MHA. Kernel support is now available in FlashInfer and vendor stacks, so the deployment tax is smaller than it was.
- **A further cut on top of GQA, if you accept early-stage tooling: GVA.** Cuts persistent cache by another ~45% vs GQA at essentially the same accuracy in 350M-scale results; kernels are being open-sourced.
- **Small-model + tool-use "compact reasoner" bases: sliding-window with a few full-attention layers.** Keeps most layers cheap, bridges long-range with the full-attention layers or with retrieval.

Variants stack: a base can use sliding-window + GQA + decoupled-RoPE; MLA and GVA are alternatives to the "K/V cache axis" rather than to each other's siblings.

## Adjacent but distinct

- **Sparse attention (BigBird, Longformer-random, block-sparse):** restricts *which* pairs attend, not what is cached. Sits in a different taxonomy and rarely composes cleanly with cache-shrinking variants.
- **Linear / state-space attention (Mamba, Linear Attention):** removes the softmax and the quadratic term entirely; a different family, tracked separately in `architectures/`.

## Sources

- *GQA: Training Generalized Multi-Query Transformer Models* — Ainslie et al. — 2023.
- *DeepSeek-V2 / V3* — DeepSeek — 2024–2025 — introduces MLA.
- *Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction* — Tripathi et al. — 2026 — [arXiv:2609.13285](https://arxiv.org/abs/2609.13285).
- *Longformer / Sliding-Window Attention* — Beltagy et al. — 2020 — the sliding-window primitive.
