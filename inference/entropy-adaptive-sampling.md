# Entropy-Adaptive Sampling (FLEET)
*Depth — trajectory-memory best-of-N sampling driven by per-token entropy.*

**TL;DR:** Temperature sampling for best-of-N is **memoryless** — it doesn't know what previous samples looked like, so as N grows an increasing fraction of samples become semantic duplicates and pass@N returns diminish. **FLEET** represents each generation as a sparse trajectory through **high-entropy decision-point states**, aggregates trajectories seen so far into per-token utility scores, and adjusts the logits of new generations to steer away from repeats. Matches repeated-sampling accuracy with 3× fewer samples; under a fixed budget lifts **LiveCodeBench pass@32 from 59.9% to 66.2%**.

**Prereqs:** *(none in current KG; see below)*
**Related:** [livecodebench](../evaluation/livecodebench.md)

---

## What it is

A sampling procedure that plugs into standard autoregressive generation and drives diversity through a per-token utility signal derived from previous samples' outcomes. It's a training-free wrapper over temperature sampling that turns memoryless best-of-N into a memory-aware search.

## How it works

Define a **decision point** as a token position where the logit distribution's entropy exceeds a preset threshold. Non-decision tokens are effectively forced by the previous context and contribute nothing to sample diversity; ignore them.

For each generated sequence, the **trajectory** is the sequence of decision-point tokens (a sparse skeleton of the full generation). After scoring each sample (compile, pass tests, etc.), aggregate over past trajectories to compute a **per-token utility score** for each decision-point token that has been seen — how often it appears in high-scoring vs low-scoring trajectories.

For the next sample, at each decision-point token position, adjust the logits by a term proportional to the utility score of candidate tokens observed in past samples. Tokens that led to high scores get boosted; tokens that led to low scores are suppressed. Tokens that haven't been seen yet are left unchanged (implicit exploration).

Greedy variant: fully deterministic; a single calibration pass sets the entropy threshold and utility scaling.

## Why it matters

Test-time-compute scaling is a first-class inference lever for reasoning models. Naive best-of-N under-delivers because the effective sample count is far below N once duplicates are counted. FLEET is a plug-in that recovers most of the "true" pass@N gain — 3× wall-clock speedup for equal accuracy — while adding no model changes and only minimal pipeline changes.

On complex coding tasks where sample variance is high and duplicate rate is worst, FLEET's improvement is largest.

## Gotchas & tricks

- Threshold on entropy is the main hyperparameter. Too low: everything counts as a decision point and the utility signal is diluted. Too high: only rare bifurcations are tracked and diversity boost is weak. The paper's single-calibration-pass sets this from a small prompt set.
- Utility scores are per-prompt in the strictest sense; sharing across similar prompts requires an embedding-based lookup, which the paper leaves as future work.
- Complementary to speculative decoding and to reasoning-length control — FLEET works on the sample side, not the per-token throughput side.

## Sources

- Paper: *FLEET: From Logits Entropy to Enhanced Trajectories in Text Generation* — Streltsov & Vitko, 2026 — [arXiv:2609.27657](https://arxiv.org/abs/2609.27657)
