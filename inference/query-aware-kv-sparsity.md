# Query-aware KV sparsity (Quest, MoBA, SnapKV, …)
*Depth — score and select KV-cache positions per query during decode, instead of keeping the full cache.*

**TL;DR:** Long-context inference is dominated by KV-cache access. Query-aware KV sparsity methods (Quest, MoBA, SnapKV) score cache positions against the current query and read only the top-K. Quality recovers most full-attention accuracy at a fraction of the KV bandwidth. Their accuracy at fixed sparse budget improves further when the backbone carries a complementary memory mechanism (e.g. RAT+).

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [mla](../architectures/mla.md), [rat-plus](../architectures/rat-plus.md)

---

## What it is

A family of inference-side techniques that approximate full attention by *selecting* a subset of KV-cache positions per query, rather than compressing or evicting them ahead of time. The selection is dynamic and query-conditioned: each new token's attention reads only the K positions most relevant to *that* query.

## How it works

Common scaffolding across variants:

1. **Score every cached position against the current query.** Variants differ in the scorer — Quest uses page-level upper bounds, MoBA uses block-relevance heuristics, SnapKV uses observation-window pooling.
2. **Pick top-K** (typically K ≪ context length). K is the "budget".
3. **Run attention over the K selected positions only.** Saves bandwidth (most KV not loaded into compute) and FLOPs.

Variants:

- **Quest** — page-granular selection with upper-bound scoring; cheap, no learning.
- **MoBA** — block selection learned from training signal; tighter but model-coupled.
- **SnapKV** — uses a recent observation window to predict which past positions matter for the current query.

## Why it matters

- Long-context inference costs are I/O-bound; query-aware sparsity cuts the dominant cost without changing the model.
- Composable with model-side compression (MLA): MLA reduces KV size per position, query-aware sparsity reduces positions accessed per query — they multiply.
- Backbone choices matter. Adding an exponentially-decaying memory module ([RAT+](../architectures/rat-plus.md)) consistently lifts query-aware sparsity accuracy across Quest/MoBA/SnapKV at every sparse budget — the recurrent state subsidises information from skipped positions.

## Gotchas & tricks

- **K is task-dependent.** Needle-in-a-haystack tasks need K large enough to cover at least one relevant position; QA over short contexts tolerates aggressive sparsity.
- **Page/block granularity affects accuracy.** Coarser blocks save scoring cost but miss small relevant spans.
- **Combine carefully with prefix-caching.** Prefix-cached KV is dense; sparse decode on top of dense prefill is fine, but reusing across queries needs scorer invalidation.

## Sources

- Paper: *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference* — Tang et al., 2024.
- Paper: *MoBA: Mixture of Block Attention* — 2024/2025.
- Paper: *SnapKV: LLM Knows What You Are Looking For Before Generation* — Li et al., 2024.
- Paper: *Augmenting Attention with Exponentially Decaying Memory Improves Query-Aware KV Sparsity* — Wei & Gulcehre — 2026 — [arXiv:2605.28640](https://arxiv.org/abs/2605.28640) — backbone-side amplifier across Quest/MoBA/SnapKV.
