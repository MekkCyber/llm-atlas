# KV-Cache Eviction

*Depth — dropping unimportant key-value pairs from the attention cache during decoding to bound memory.*

**TL;DR:** Long-CoT decoding grows the KV cache linearly; eviction caps it at a fixed budget by dropping the K/V entries judged least useful. Older eviction methods consistently underperformed *selection*-based sparse attention (which keeps the full cache, just attends to a subset) on reasoning tasks. VaSE (USC / U. Chicago, 2026) shows two fixable causes — outlier value states whose eviction causes repetition-loop collapse, and overly deterministic eviction that kills cache diversity — and closes the gap.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [../architectures/mla.md](../architectures/mla.md)
**Related:** [kv-cache-quantization.md](./kv-cache-quantization.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Two ways to bound KV-cache memory during long decoding:

| Approach | Cache size | What's lost |
| --- | --- | --- |
| **Selection (sparse attention)** | Full cache kept | Compute saved by attending to subset |
| **Eviction** | Bounded cache | Evicted tokens are permanently gone |

Selection has the better accuracy but a non-static memory footprint, which matters for serving. Eviction gives the *static* footprint serving wants — at, until recently, a real accuracy cost on reasoning benchmarks.

## How it works

Standard eviction-score eviction at each decode step:

```
score_t[i] = attention_score(Q_t, K_cache[i])    # or aggregated over recent steps
evict = argmin(score_t)
drop K_cache[evict], V_cache[evict]
```

VaSE adds two interventions on top:

1. **Value-aware protection.** Identify K/V pairs whose V vector has anomalously large magnitude — empirically a small fraction. Evicting these triggers catastrophic failure (the model enters repetition loops). Protect them from eviction regardless of attention score. The mechanism: large-magnitude V values dominate the attention-weighted sum even when their attention weight is small; removing them creates a large unrecoverable shift in the layer output.
2. **Stochastic eviction.** Instead of greedy top-K eviction, sample which entries to drop in proportion to their scored "evictability". This trades a small per-step accuracy hit for cache *diversity* across rollouts — important when serving reasoning models with self-consistency or majority voting.

Both interventions are training-free and FlashAttention2-compatible.

## Why it matters

- **Static memory for reasoning models.** With VaSE at 4× compression, eviction now matches or beats selection on six reasoning benchmarks (Qwen3 family). Pre-VaSE this was a deal-breaker tradeoff; post-VaSE eviction is the default for fixed-budget serving.
- **Compose with quantization.** Eviction reduces cache *length*; quantization reduces cache *width per entry*. Apply both: 4× length reduction × 4× width reduction = 16× memory at modest accuracy cost.
- **No retraining.** Drop-in for any pretrained transformer with a standard attention cache.

## Gotchas & tricks

- **Outlier V is the dominant failure mode.** Before VaSE, "why is eviction destroying my reasoning model" almost always traced to a few large-magnitude V states. Always log per-token V norms before tuning eviction policies.
- **Determinism vs diversity.** Greedy eviction is fine for single-pass generation; stochastic eviction wins when downstream uses majority voting / self-consistency over K samples (each rollout gets a slightly different evicted set, decorrelating errors).
- **Recent-window protection.** Most working evictors protect the last N tokens from eviction regardless of score — local context is almost never safe to drop. This is orthogonal to VaSE's outlier-value protection.
- **Selection vs eviction is task-dependent.** Long-form generation: eviction. Multi-document QA where retrieval order matters: selection is still safer.

## Sources

- *Value-Aware Stochastic KV Cache Eviction for Reasoning Models* — Chang et al., USC / U. Chicago, 2026 — [arXiv:2606.03928](https://arxiv.org/abs/2606.03928) — primary source for value-aware protection and stochastic eviction.
- *H2O: Heavy-Hitter Oracle for Efficient Generative Inference* — Zhang et al., 2023 — attention-score-based eviction baseline.
- *StreamingLLM* — Xiao et al., 2023 — attention-sink + recent-window eviction lineage.
