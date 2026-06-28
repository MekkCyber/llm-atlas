# InfoKV
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An entropy-aware KV-cache compression criterion. Prior schemes rank tokens by attention scores; InfoKV adds *predictive uncertainty* (per-token next-token entropy) because high-entropy tokens turn out to influence the distant future far more than attention alone reveals. Improves long-context reasoning at the same compression rate.

**Prereqs:** [_kv-cache-compression.md](_kv-cache-compression.md), [multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [dca.md](../fundamentals/dca.md)

---

## What it is

KV-cache compression evicts a subset of past key/value tensors to bound memory in long prefilling and long decoding. The dominant signal in prior work (H2O, SnapKV, etc.) is *attention weight*: tokens that receive a lot of attention from recent queries get kept. InfoKV's claim is that attention catches *local* relevance and misses *long-range* influence.

The paper defines **Forward Influence** — how much evicting a token actually perturbs the model's outputs further down the sequence. Empirically: high-attention tokens dominate Forward Influence over short distances; high *predictive-uncertainty* (high-entropy) tokens dominate it over long distances.

## How it works

- For each token, combine two signals:
  - **Attention score**, as in prior KV-cache compression schemes.
  - **Predictive uncertainty**: the entropy of the model's next-token distribution at that position, computed once during prefilling.
- The entropy signal is also collected **per layer** to track how representation uncertainty evolves with depth.
- A combined importance score (attention + layer-wise entropy) drives the eviction policy in both the prefill and decoding phases.
- Drop-in: no retraining of the backbone, no architecture changes.

## Why it matters

- Long-CoT reasoning models (R1, R1-distill, o1-likes) generate huge token traces; KV-cache is the dominant memory cost at serve time. A better-quality compression at the same budget moves the cost curve.
- Validated on **Llama-3.1, Llama-3.2, and DeepSeek-R1** long-context reasoning benchmarks — consistently beats attention-only KV compression in both long prefill and long decode.
- Cheap to add: entropy is already computed at every step; no extra forward passes.

## Gotchas & tricks

- Entropy is informative but noisy; layer-wise smoothing matters more than the paper's headline ablation suggests.
- The "high-entropy tokens are important" finding is the inverse of common intuition — surprised tokens are pivots, not noise.
- Combining with FP8/INT4 KV-cache quantization is orthogonal; you can stack.

## Sources

- Paper: *Information-Aware KV Cache Compression for Long Reasoning* — Xiao, Birch, Lin, 2026 — arXiv:2606.26875.
