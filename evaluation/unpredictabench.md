# UnpredictaBench — distributional sampling benchmark
*Depth — measure how well an LLM samples from a target distribution, scored by Kolmogorov–Smirnov.*

**TL;DR:** Diversity is not distribution matching. UnpredictaBench tests whether an LLM can produce N samples that *match* a target distribution — canonical statistical, induced by a stochastic program, or specified in natural language. The KS@N metric is the rate at which a Kolmogorov–Smirnov test fails to reject the model's samples vs. ground truth at sample size N. No model exceeds 40% at KS@100; many sit near 0.

**Prereqs:** [rlvr](../post-training/rlvr.md)
**Related:** [mmlu](mmlu.md), [subtlememory](subtlememory.md)

---

## What it is

A benchmark with 448 problems, each defining a *target probability distribution*. Categories:

- **Canonical statistical** — Gaussian, exponential, Poisson, etc., described in natural language.
- **Stochastic programs** — a short program (`flip a coin three times, count heads`) whose induced distribution is the target.
- **Natural-language scenarios** — random processes described as text ("pick a random tourist destination weighted by popularity").

For each problem the model is asked for N samples; UnpredictaBench checks whether they're distributionally indistinguishable from a ground-truth sample.

## How it works

- **KS@N metric.** Generate N samples from the model and N from the ground truth (or its analytic CDF). Run a two-sample Kolmogorov–Smirnov test at a fixed significance. KS@N reports the rate across problems at which the test *fails to reject* — i.e. the model's samples are statistically indistinguishable.
- **N as a difficulty knob.** Small N is easy (a few lucky samples pass); large N is hard. The standard reported metric is KS@100.
- **Per-category breakdowns** isolate whether the model can sample canonical distributions but fails on stochastic programs, etc.

## Why it matters

- Direct prerequisite for the increasingly common "LLM as Monte-Carlo engine" use case (econ agents, persona panels, synthetic data generation). Distribution mismatch ⇒ biased downstream conclusions.
- Reveals that mode collapse, not just lack of diversity, is the real failure: reasoning models do *somewhat* better, but no model approaches calibrated sampling.
- KS is a classical metric — results are interpretable beyond LLM-internal conventions.

## Gotchas & tricks

- **Sampling temperature ≠ distribution shape.** Cranking temperature broadens the output but doesn't make it match a target distribution; UnpredictaBench rewards calibration, not entropy.
- **Continuous vs. discrete handling.** KS is defined for continuous CDFs; discrete distributions use a small-jitter variant or per-category χ² — the benchmark spec controls which.
- **Ground-truth sample size matters.** For natural-language scenarios where the "target" is itself estimated, KS power depends on how many ground-truth samples are available.

## Sources

- Paper: *UnpredictaBench: A Benchmark for Evaluating Distributional Randomness in LLMs* — Abaskohi, Dabiriaghdam, Luo, Wen, Wang, Carenini, West — 2026 — [arXiv:2606.06622](https://arxiv.org/abs/2606.06622)
