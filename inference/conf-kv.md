# Conf-KV: Confidence-Aware KV Cache Eviction

*Depth — a KV-cache manager that uses the model's per-step next-token confidence to set a dynamic cache budget, combined with mixed FP16/INT8 storage.*

**TL;DR:** Most eviction policies pick a static budget (sliding window of $K$, top-$K$ by attention mass). Conf-KV computes a confidence scalar from each decode step's next-token distribution and uses it to vary the budget *per step* — keep more context when uncertain, prune harder when confident. Within the budget, tokens are ranked by accumulated attention × recency, with a protected recent window. Pairs naturally with mixed-precision (FP16 recent / INT8 historical) storage and pyramidal per-layer budgets.

**Prereqs:** [_kv-cache-eviction.md](_kv-cache-eviction.md)
**Related:** [../quantization/_number-formats.md](../quantization/_number-formats.md)

---

## What it is

A per-step adaptive KV-cache eviction policy. Drops the assumption that the right cache budget is constant — instead, makes it a function of the model's current uncertainty.

## How it works

At each decoding step:

1. **Compute confidence.** Take the next-token distribution, summarize into a scalar — e.g., $1 - H(p)/H_{\max}$ for a normalized inverse-entropy score, or top-1 probability. High value = confident.
2. **Choose step budget.** Map confidence to a budget $K_t$ — confident steps get a smaller $K_t$, uncertain steps get a larger one. Linear or piecewise-linear mapping; the exact shape is a hyperparameter.
3. **Rank tokens.** Score each cached token by a composite of accumulated attention mass and recency. Always keep a protected recent window of $W$ tokens (typically 32–128) for local coherence.
4. **Evict.** Keep top-$K_t$ tokens by composite score (plus the protected window). Evict the rest.
5. **Store the kept tokens** in mixed precision — recent window in FP16, historical block in INT8 (the bulk).

Blockwise online-softmax attention reads the mixed-precision store directly. A pyramidal per-layer variant assigns smaller budgets to upper layers (their attention is usually more diffuse), shaving total memory further.

## Why it matters

- **Free signal exploited.** Confidence is computed on every decode step already (for sampling). Prior eviction policies threw it away.
- **Long-context recall.** On Needle-in-a-Haystack to 32K tokens: 91.4% retrieval vs 53.8% for sliding-window and 80.6% for H2O. The dynamic budget catches the "uncertain → keep more" regime where the model is actively searching distant context.
- **Practical memory win.** At <512-token sliding-window memory footprint, Conf-KV+INT8 stays within 1.5–2.1 PPL of full KV across four model families up to 4K context.
- **Agentic tasks.** On 75 VisualWebArena tasks: 95.3% of full-KV success at 2.8× lower peak memory — agentic workloads have variable uncertainty over a long trajectory, which is exactly where the dynamic budget pays off.

## Gotchas & tricks

- The confidence → budget mapping needs calibration per model family. Different base models have different entropy profiles; reuse a calibration set from the deployment domain.
- The accumulated attention rank can become stale on very long traces — old tokens that earned attention early may not matter later. Combining with recency in the composite (not pure attention) prevents the worst cases.
- INT8 storage can hurt extreme-precision needs (exact code, exact numbers). If you see degradation, raise the protected window to cover the recent code/numeric region.
- Pyramidal budgets help most on deep models (>32 layers); shallow models gain little.
- The policy is training-free — drops into existing vLLM-class servers as a KV-cache manager swap.

## Sources

- Paper: *Conf-KV: Confidence-Aware KV Cache Eviction with Mixed-Precision Storage for Long-Horizon LLM Inference* — 2026 — [arXiv 2605.24786](https://arxiv.org/abs/2605.24786).
