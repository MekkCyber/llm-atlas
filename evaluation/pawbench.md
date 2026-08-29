# PAWBench — Probabilistically Aligned World Modeling
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A video-generator benchmark that treats a world model as a **stochastic sampler of world dynamics** rather than a "single plausible video" model. Given a fixed initial frame and action, repeatedly sample rollouts, discretize each rollout's final outcome into a physical event, and compare the *empirical distribution* over events against a reference distribution. Complements per-video plausibility metrics with a distributional fidelity axis. 50 scenarios, 11 systems evaluated; no model consistently matches both the reference frequencies and the range of valid behaviors.

**Prereqs:** [README.md](README.md)
**Related:** [../data/decontamination.md](../data/decontamination.md)

---

## What it is

Standard video-generation evals score one video at a time: is it sharp, temporally consistent, physically plausible? That misses the fact that many real physical processes have multiple valid outcomes (a bouncing ball can settle in several places; a spilled liquid can go left or right depending on unseen micro-conditions). A world model should reproduce not one plausible trajectory but the *distribution* of possible behaviors.

PAWBench defines **probabilistic alignment** as: the model's distribution over rollout outcomes matches a reference distribution over physically valid outcomes under the same initial observation and action. PAWEval is the outcome-level scoring protocol that turns repeated rollouts into empirical histograms and scores them against the reference.

## How it works

1. **Scenario spec.** 50 scenarios, each with an initial frame + action + a reference distribution over physical outcome categories (from human labelers or ground-truth simulators).
2. **Rollout sampling.** For each scenario, sample N=20+ videos from the model under evaluation.
3. **Outcome discretization.** Classify each rollout's final state into one of the scenario's discrete outcome categories (via a task-specific classifier / VLM judge).
4. **Distributional score.** Compare empirical outcome frequencies against the reference (KL, TVD, or coverage-plus-frequency composite). Penalizes both mode collapse (missing valid outcomes) and mis-calibration (right outcomes, wrong frequencies).
5. **Intervention study.** Test whether prompt rephrasing, initial noise sampling, or fine-tuning shifts the empirical distribution toward the reference.

## Why it matters

- **Names a failure mode current metrics miss.** A visually pristine world model that always produces the same outcome from a stochastic scenario is useless for planning or model-based RL — PAWBench flags it.
- **Sets a foundation for world-model-as-simulator.** If the point of a video generator is to be a rollout engine for downstream agents, distributional fidelity is the metric that actually matters.
- **Empirically bounded.** All 11 tested systems fail to match reference distributions well, so the benchmark is currently informative rather than saturated.

## Gotchas & tricks

- **Classifier reliability is a confound.** If the outcome classifier is noisy, empirical histograms are noisy too — cross-check with human agreement on a subset.
- **N matters.** With only 5 rollouts you can't distinguish "correct rare-outcome frequency" from noise. N=20+ per scenario is a practical floor.
- **Reference distributions are hard to get.** Human-elicited distributions are subjective; simulator-derived ones assume a physics model. The paper's approach is a mix; treat scores as ordinal, not absolute.
- **Not a replacement for FVD.** Distributional alignment and per-video quality are orthogonal — a model can be well-calibrated on outcomes but ugly frame-by-frame. Report both.

## Sources

- Paper: *PAWBench: How Far Are We from Probabilistically Aligned World Modeling?* — Pu et al. (Shanghai AI Lab / HKU), 2026 — arXiv:2608.27345.
