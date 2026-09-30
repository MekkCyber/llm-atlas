# Attention sparsity

*Taxonomy — how attention layers avoid doing full O(N²) work while keeping model quality.*

**TL;DR:** Every attention-sparsity variant reduces work by *not* computing something dense attention would. What differs is where the decision is made (compile-time, per-model, per-head, per-query) and which quantity is skipped (whole QK tiles, top-k rows, post-score V projections). The modern default for LLMs is a mix of GQA-style KV compression, sliding-window locality, and (increasingly) a data-dependent gate on top of a dense scorer.

**Related taxonomies:** [_normalization](_normalization.md), [_moe](_moe.md)
**Depth files covered here:** [mla](mla.md) · [mass-alloc-attention](mass-alloc-attention.md) · [dca](../fundamentals/dca.md)

---

## The problem

Standard causal attention costs O(N²) time and O(N²) score memory. At 128K–1M-token contexts, both blow up wall-clock and HBM budgets. Real attention distributions are extremely concentrated — most causal interactions carry negligible normalized mass — so the dense compute is largely wasted, but only *after* you see the distribution.

## The shared pattern

Every variant answers three questions:

1. **When is the decision made?** Compile time (fixed pattern), per-head at training time (learned pattern), or per-query at run time (data-dependent).
2. **What is dropped?** Whole QK tiles, KV projections, post-score V-mul-P work, or nothing (only KV storage compressed).
3. **Is training and inference symmetric?** If yes, you avoid the "sparse at serve, dense at train" gap that hurts model quality.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Sliding-window / local attention | Fixed local receptive field per token | Loses long-range info; needs global tokens or hybridization | Very long contexts with mostly local structure |
| Block-sparse (Longformer, BigBird) | Fixed pattern of blocks + a few global rows | Pattern must be picked in advance | Retrieval-style long-context tasks |
| Top-k attention | Score everything, keep the top-k Vs per query | Still forms the full QK tile | Small-vocab probe tasks; not typical LLMs |
| [MLA](mla.md) | Low-rank KV projection compresses the cache | Adds a factorization; not free at short context | Serving-throughput-bound long-context inference |
| GQA / MQA (no depth file yet) | Multiple Q heads share one KV head | Small quality loss vs MHA at big cache savings | Modern default for LLM serving |
| [MassAlloc (MALA)](mass-alloc-attention.md) | Runtime gate on post-score work driven by the softmax normalizer | Score matrix still O(N²) memory | Long-context training and decoding, MoE stacks |
| [DCA](../fundamentals/dca.md) | Chunked long-context attention with global anchors | Chunk boundary artifacts | Extreme-context serving on models trained short |
| Native sparse attention (Gemini-style, no depth file yet) | Learned per-head hybrid of local + landmark | Training complexity | Frontier long-context reasoning models |

Link techniques with a depth file; leave others as plain text until a depth file lands.

## How to choose

- **Serving long context on a model you didn't train?** Compress the KV cache (MLA if you retrain, DCA-style chunking if you don't) — this is where quality-preserving gains live.
- **Training-time compute pressure on long context?** Data-dependent gating on the post-score path (MALA) is the current best answer if you want a *single* knob and no architectural change.
- **Building a new model?** Start with GQA + local windows for the cheap wins, then layer MLA or MALA for the last multiplier.
- **Never** ship a training-serving asymmetry (dense at train, top-k at serve) without measuring — the last mile of quality is exactly there.

## Adjacent but distinct

- **MoE** ([_moe](_moe.md)) — sparsity in the *FFN*, not attention. Combines cleanly with any of the above.
- **Speculative decoding** — cuts inference tokens generated, not attention cost per token.
- **KV quantization** — orthogonal cache-size axis; use with any variant here.

## Sources

- MLA — DeepSeek-V2 tech report — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- Longformer — Beltagy et al., 2020 — [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
- MALA — Shi et al., 2026 — [arXiv:2609.32712](https://arxiv.org/abs/2609.32712)
