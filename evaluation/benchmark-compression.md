# Benchmark Compression (BestRand)
*Depth — pick a small, stable subset of a benchmark that preserves full-set score at a fraction of the cost.*

**TL;DR:** Running every item of large LLM benchmarks is expensive; a compressed subset (10–20% of items) suffices if it's chosen well. **Random sampling** is nearly unbiased in expectation but has a fat tail of unlucky draws that estimate the full score very poorly. **BestRand** (Ling et al., 2026) keeps random sampling's simplicity and cost but kills the tail: draw $K$ random subsets, score each with a cheap stability proxy, and return the one that minimises the proxy. On 70+ text / multimodal / agent benchmarks it reports lower high-error risk, lower mean error, and lower construction cost than IRT- and clustering-based compressors.

**Prereqs:** [mmlu](mmlu.md), [ifeval](ifeval.md)
**Related:** [decontamination](../data/decontamination.md)

---

## What it is

Benchmark compression asks: given a benchmark $B$ with $N$ items and a (growing) catalogue of models $\mathcal M$, pick a subset $S \subseteq B$ of size $n \ll N$ such that the per-model score on $S$ predicts the per-model score on $B$ well — in expectation *and* in the worst case.

Three families have been tried:

| Family | Idea | Example |
| --- | --- | --- |
| **Random** | Uniform sample of items | Baseline |
| **IRT / clustering** | Learn item difficulty / discrimination from model-response matrix, pick items that span the latent space | Anchor Points, IRT compression |
| **Multi-draw selection** | Draw several random subsets, pick the one that minimises a stability criterion | BestRand |

## How it works

Ingredients:

- $K$ random subsets $S_1, \ldots, S_K$ of size $n$, each drawn from $B$ without replacement.
- A **cheap stability proxy** $\pi(S)$ computed only from the model-response matrix already available (no new inference). BestRand's proxy is the cross-model-family *variance of estimated full-score on $S$* for a held-out set of reference models — i.e. how much estimates swing when you look at $S$ through different model lenses. Lower is more stable.

Pipeline:

1. For $k = 1, \ldots, K$: draw $S_k$, compute $\pi(S_k)$ using only the already-available model-response matrix.
2. Return $S^\star = \arg\min_k \pi(S_k)$.

Cost: $K$ scalar proxy computations — no model inference, since the proxy uses the pre-collected model-response matrix. $K \in [100, 1000]$ is typical.

## Why it matters

- **Negative result for the complex compressors.** On 70+ benchmarks, BestRand often matches or beats IRT- and clustering-based compressors at a fraction of the construction cost. Claim: the sophisticated methods' main gain over plain random was *luck of the draw*, and multi-draw selection captures the same win.
- **Deployable cheaply.** The proxy uses already-collected model-response data, so compressing a new benchmark costs essentially nothing beyond already-run evals.
- **Fat-tail mitigation matters more than bias reduction.** The dominant failure mode of random sampling isn't the mean estimation error, it's the rare very-bad subset. BestRand directly targets that.

## Gotchas & tricks

- **Reference-model set matters.** The stability proxy depends on a held-out set of reference models. If reference models cluster on one family, the proxy over-fits that family's score surface. Diversify.
- **$K$ has sharply diminishing returns.** Going from $K=100$ to $K=1000$ changes the picked subset but not the full-score estimation much. $K = 200$ is a safe default.
- **Doesn't fix a bad benchmark.** If $B$ is contaminated, has duplicates, or has brittle items, compression inherits the problems — see [decontamination](../data/decontamination.md).
- **Subset size $n$ should be the real tradeoff.** Too small → estimator variance dominates whatever the method; too large → compression isn't worth it. Start at $n \approx 0.1 N$ and sweep.
- **Not identifiability-preserving.** Models that are close on $B$ may rank differently on $S^\star$. Compression is for *point estimation* of score, not for leaderboard sorting of near-tied models.

## Sources

- Paper: *Random Is Almost Enough? Stable and Low-Cost Benchmark Compression via BestRand* — Ling, Zhong, Yang, Yang, Li, Huang — CMU, 2026 — [arXiv:2611.00883](https://arxiv.org/abs/2611.00883).
- Background: *Efficient Benchmarking of Language Models* — Polo et al., 2024 — IRT-based compressor (tiny-benchmarks family).
- Background: *Anchor Points* — Vivek et al., 2024 — clustering-based subset selection.
