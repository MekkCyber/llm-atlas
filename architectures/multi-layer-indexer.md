# Multi-layer Indexer (MLI) — Overlapping Selection and Attention in DeepSeek Sparse Attention
*Depth — cross-layer reuse of the sparse-attention indexer to parallelize selection with attention.*

**TL;DR:** DeepSeek Sparse Attention (DSA) uses an **indexer** to pick which KV positions attention actually reads. The indexer currently runs **before** its layer's attention, serially. **MLI** reuses layer `l − 1`'s indexer output for layer `l`'s attention, so the current indexer can run **in parallel** with the attention it would have blocked. No retraining, no new weights; composes with IndexCache. On DeepSeek-V3.2 at retention 1/2, **up to +14.5% decode speed** over IndexCache. At 128K context + retention 1/4, **1.32× unmodified decode throughput** (vs. 1.28× for IndexCache at 1/8).

**Prereqs:** [mla.md](./mla.md), [multi-head-attention.md](./multi-head-attention.md)
**Related:** [../case-studies/deepseek-v3.md](../case-studies/deepseek-v3.md) · [deepseek-moe.md](./deepseek-moe.md)

---

## What it is

DSA, introduced with DeepSeek-V3.2, is a learned sparse-attention variant. Each attention layer has an auxiliary **indexer** — a small network that scores KV positions and selects a subset ("retention ratio") that the dense attention then reads. The indexer makes DSA's selection *content-aware* rather than hand-coded (sliding window, strided).

The cost shape:

```
layer l:  [indexer_l] → selected positions → [attention_l]  (serial)
```

The indexer is cheap in FLOPs but blocks attention from starting. At decode time, where each step is small, this serial dependence is a nontrivial fraction of latency.

**MLI's observation:** DSA's selected positions evolve **slowly** across layers — the selection at layer `l − 1` and at layer `l` are highly overlapping. So the current layer's attention can safely use the *previous* layer's indexer output as its mask, while the current layer's indexer computes concurrently for the *next* layer.

## How it works

```
layer l-1:   indexer_{l-1} ─┐
                            └─► attention_{l-1}
                            └─► attention_l    (reuses indexer_{l-1})
layer l:     indexer_l    ──┐   (runs in parallel with attention_l)
                            └─► attention_l+1
```

**Pipeline the two.** Attention at layer $l$ consumes the indexer output from layer $l-1$. Indexer at layer $l$ runs *during* attention at layer $l$ to produce the mask for layer $l+1$. The first and last layers need a small cold-start adjustment (use layer 0's indexer twice, or run it offline).

**No weight changes.** The indexer and attention weights are unchanged; the only change is the scheduling. DSA models trained without MLI serve correctly under MLI.

**Composes with IndexCache.** IndexCache caches selection across decode steps for the same layer; MLI pipelines selection across layers. Together: cache in time *and* overlap in space.

## Why it matters

DSA is the sparse-attention default inside DeepSeek's stack and is spreading as other labs adopt MLA + DSA. Making the indexer **free at decode time** — by overlapping it with the attention it would have blocked — eliminates the main latency cost sparse attention was supposed to *save*. Small, retrofit-friendly, no retraining.

## Gotchas & tricks

- The quality of the approximation depends on layer-to-layer selection stability. Early-stack layers select differently from late-stack layers; the paper's +14.5% assumes per-layer calibration of overlap.
- Prefill is mostly unaffected — the win is at decode, where each step is short and serial dependencies hurt most.
- Pipelining changes the memory footprint — two consecutive layers' indexer outputs now live concurrently.
- Not a correctness change (DSA is already an approximation), but MLI makes it a slightly *older* approximation — monitor retention-sensitive long-context eval.

## Sources

- Paper: *Multi-layer Indexer: Overlapping Selection and Attention in DeepSeek Sparse Attention* — Dilbaz, Krishna, Kamble, Bikel, 2026 — https://arxiv.org/abs/2611.03768
- Related: DeepSeek-V3.2 tech report (origin of DSA); IndexCache (origin of across-step selection caching).
