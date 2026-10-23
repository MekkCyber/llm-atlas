# Stratified Sparse Attention (SSA)
*Depth — combine deterministic top-k retrieval with score-guided sampling of the residual tail to close the sparse-to-dense gap.*

**TL;DR:** Standard sparse attention at extreme sparsity falls into two camps: **deterministic retrieval** (top-k keys) throws away all residual attention mass, while **score-guided sampling** is unbiased but noisy. **SSA** does both — deterministic top-k for the mass-heavy tier, importance-sampled with inclusion-probability correction for the tail. Up to **2.96× decode speedup over FlashInfer at 1M context**, with consistent accuracy gains over either baseline alone on RULER, LongBench, LOFT, and AIME across 3B–27B models.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md), [dca](../fundamentals/dca.md)
**Related:** [mla](../architectures/mla.md), [agent-kv-cache](agent-kv-cache.md), [block-sparse-attention](block-sparse-attention.md)

---

## What it is

At long context (100K+ tokens), the KV cache dominates serving cost and compute. Sparse attention mitigates by only scoring a subset of keys. Two existing families:

- **Deterministic retrieval** (top-k): compute cheap scores for all keys, pick the top-k by score, run attention on those. Loses the long tail; the discarded mass biases the softmax.
- **Score-guided sampling**: sample keys with probability proportional to their scores, re-weight by 1 / inclusion-probability (importance sampling). Unbiased but variance-heavy at small sample budgets.

Neither works well at *extreme* sparsity (keep only 1–2% of keys). SSA stratifies the key set into two tiers and treats each with the right tool.

---

## How it works

For a query *q* and keys `K_0, …, K_N`:

1. **Score all keys cheaply.** Use a proxy score `s_i` (e.g. a low-dim projection dot product; same proxy as prior top-k methods).
2. **Deterministic tier.** Pick the top-k keys by `s_i`. These are included with probability 1.
3. **Sampled tier.** For the remaining keys, sample `m` of them with probability proportional to `s_i`. Each sampled key is included with known inclusion probability `p_i`.
4. **Unbiased attention.** Compute attention using the top-k deterministically and the sampled set with importance-corrected weights `1 / p_i`. The combined estimator is unbiased for the full-dense attention output.

The key insight: the top-k tier captures most of the softmax mass with zero variance, and the sampled tier captures the residual tail with unbiased noise. Variance of the overall estimator is much lower than sampling alone because the mass-heavy keys are not sampled — they are kept deterministically.

---

## Why it matters

At 1M-token context, dense attention is impractical; sparse attention is the only option, and extreme sparsity (<2%) is where the gap to dense accuracy is largest. SSA closes most of that gap without extra model-side cost: it is a drop-in change to the kernel, not a training recipe.

End-to-end wins:
- Up to **2.96× decode speedup** over FlashInfer at 1M context.
- Consistent accuracy improvement over deterministic-only and sampling-only baselines at matched sparsity across four benchmarks and five model sizes.

Also: unbiasedness is a theoretical property the pure top-k family cannot claim. For reasoning tasks where the attention tail matters, the unbiased estimator is particularly important.

---

## Gotchas & tricks

- **Score proxy accuracy sets the ceiling.** If `s_i` is a bad predictor of true attention weights, both tiers suffer. Use a proxy consistent with the model's attention structure (per-head, not shared).
- **Sampling without replacement.** Sampling with replacement is simpler but wastes budget on duplicates; the paper uses without-replacement with Horvitz-Thompson style correction.
- **Interaction with speculative decoding.** SSA composes with speculative decoding, but check that the draft model and target model use consistent sparse patterns — otherwise acceptance rates drop.
- **Blockified version.** For hardware efficiency, apply the deterministic/sampled split at block granularity, not per-key.

---

## Sources

- Paper: *Stratified Sparse Attention: Reducing the Sparse-to-Dense Gap at Extreme Sparsity* — Sahil Joshi, Prithvi Dixit, Aryan Katiyar, Agniva Chowdhury, Joseph E. Gonzalez, Anshumali Shrivastava, Aditya Desai, Kumar Krishna Agrawal — Rice / UC Berkeley / Stanford / IIT Bombay, 2026 — [arXiv:2610.28441](https://arxiv.org/abs/2610.28441).
- Related: Native Sparse Attention; Quest; HashAttention; MInference.
