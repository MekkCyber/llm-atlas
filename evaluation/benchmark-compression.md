# Benchmark compression (BestRand)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A low-cost method for estimating a model's full-benchmark score from a small subset of items. The core observation is that the *error distribution* across random subsets is long-tailed — most random subsets are fine, a few are catastrophically off. BestRand (2026) derives this analytically and then selects, from many random candidates, the subset whose predicted error is smallest.

**Prereqs:** [mmlu](mmlu.md)
**Related:** [codeforces-benchmark](codeforces-benchmark.md), [livecodebench](livecodebench.md)

---

## What it is

Modern LLM evaluation runs the same model across 50+ benchmarks; benchmarks like MMLU, GSM8K, BBH have thousands of items. Evaluating every checkpoint at full scale is wasteful. A natural shortcut is "score the model on a 100-item subset and extrapolate" — the question is *which* subset.

BestRand gives a principled answer: don't curate the subset by item difficulty, keep the random-sampling mechanism, but select among random candidates using an analytic error approximation.

## How it works

1. **Theory.** For a benchmark of N items with a known model-agnostic difficulty prior, approximate the sampling distribution of (subset mean − full mean) with a closed-form expression. The approximation shows the distribution has heavier-than-Gaussian tails — the source of the catastrophic outliers.
2. **Candidate pool.** Draw K random subsets of the target size s (e.g. K = 10,000, s = 100).
3. **Score each candidate** with the analytic predicted-error expression (no model inference needed at this step).
4. **Pick the best-predicted subset** and run the model only on it.

The method inherits the simplicity and unbiasedness of random sampling while trimming the right tail of the error distribution.

## Why it matters

Benchmark compute is a real bottleneck for anyone doing RL post-training sweeps, quantization ablations, or fine-tune comparisons — hundreds of checkpoints × dozens of benchmarks. BestRand lets you drop benchmark cost by 10–20× without the "cherry-picked easy items" risk that hand-curated coresets carry. The analytic framing also means you can state a confidence bound on the estimate.

The paper validates on 70+ benchmarks across text, multimodal, and agentic LLM eval — and extends to VLA evaluation, where full-benchmark runs are especially expensive.

## Gotchas & tricks

- **Difficulty prior quality matters.** The analytic predictor needs a reasonable prior on per-item difficulty. Papers often ship this now (IRT parameters, per-item accuracy of a reference model).
- **Still per-benchmark.** The subset is benchmark-specific; you can't reuse a 100-item MMLU subset for GSM8K.
- **New frontier models.** If a target model is far outside the difficulty prior's reference set (much stronger or weaker), the error bound degrades. Refresh the prior.

## Sources

- Paper: Random Is Almost Enough? Stable and Low-Cost Benchmark Compression via BestRand — Ling et al., 2026 — [arXiv:2611.00883](https://arxiv.org/abs/2611.00883)
