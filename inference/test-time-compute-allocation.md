# Test-Time Compute Allocation
*Depth — given a global inference-token budget across many queries, where do you spend the next 1000 tokens?*

**TL;DR:** Test-time scaling research mostly tunes *per-query* knobs (CoT length, samples per query, search depth). When you actually have a fixed cluster running a real workload, the question is *global*: which queries deserve more compute, which should be abandoned, and what's the equilibrium? CLEAR (2026) formulates this as a constrained optimization problem with a closed-form solution: the optimal allocation equalizes **marginal utility** across all queries via a single **global shadow price**. The operational move is *rational abandonment* — drop hopeless queries, redirect their budget to ones near their emergence threshold.

**Prereqs:** [post-training/reasoning/length-penalty](../post-training/reasoning/length-penalty.md)
**Related:** [post-training/reasoning/long2short](../post-training/reasoning/long2short.md)

---

## What it is

A real inference workload has N queries arriving with heterogeneous difficulty, all sharing a budget of T tokens. The naive policy is uniform: T/N tokens each. This is *provably suboptimal* whenever per-query utility curves differ — which they always do.

Per-query utility: how much accuracy improvement do you get from the k-th extra token on query i? Typical shape:

```
utility_i(k):  flat near 0 → sharp surge near i's emergence threshold → flat plateau
```

CLEAR models this with a *shifted-surge* function (s-curve with a query-specific shift parameter).

## How it works

Constrained optimization: choose allocation `(k_1, k_2, …, k_N)` to maximize `Σ utility_i(k_i)` subject to `Σ k_i ≤ T`.

Lagrangian:

```
L = Σ utility_i(k_i) − λ · (Σ k_i − T)
```

First-order condition: at the optimum, `utility_i'(k_i) = λ` for every query `i` that gets nonzero budget. **All marginal utilities equalize** at a common value λ — the **global shadow price** of one extra token.

Operational consequences:

- **Rational abandonment.** If query i's *maximum* marginal utility is less than λ, allocate zero. The query is below the emergence threshold and won't benefit from this budget level.
- **Concentrated spend.** Budget concentrates on queries currently near their emergence threshold (where marginal utility is high).
- **λ is an online quantity.** With a streaming workload, λ adapts: as new queries arrive, the shadow price shifts.

CLEAR (Constrained Latent-utility Equilibrium Allocation for Reasoning) implements this with the shifted-surge utility model and a per-query difficulty estimator. The estimator is cheap (a small classifier on the prompt + a few rollout samples).

## Why it matters

- **Up to 3× global accuracy improvement** vs. uniform allocation in compute-scarce regimes, on standard reasoning task streams. The improvement comes mostly from abandoning hopeless queries.
- **Principled answer to a previously heuristic question.** Test-time-compute work has had per-query knobs (length penalties, adaptive sampling); CLEAR provides the *cross-query* allocator they were missing.
- **Inference cost is the new training cost.** As serving dominates total LLM cost, "spend tokens where they matter" is a deployment-time lever as important as model size or quantization.

## Gotchas & tricks

- **Utility estimation is the hard part.** The shifted-surge fit needs per-query difficulty signal. If your estimator is noisy, expect noisy allocations. Cheapest signals: prompt-only difficulty classifier, early-rollout accuracy.
- **Abandonment must be user-visible somehow.** Dropping a query silently is a bad UX. Practical deployments fall back to a cheaper model or a "don't know" answer.
- **Static vs. streaming.** Static λ (recompute per batch) works in batch serving. Streaming workloads need an online λ-update (PI-control-style works in practice).
- **Per-query knobs are complementary.** Length penalty / [long2short](../post-training/reasoning/long2short.md) shape utility *curves*; CLEAR allocates *across* them. Use both.

## Sources

- Paper: *The Shadow Price of Reasoning: Economic Perspective on Optimal Budget Allocation for LLMs* (CLEAR) — Zhu et al., 2026 — [arXiv:2606.03092](https://arxiv.org/abs/2606.03092) — primary source.
