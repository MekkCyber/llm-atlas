# PAWBench: Probabilistic Alignment for World Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Video "world models" are usually graded one rollout at a time — does this clip look plausible? — which conflates *individual* plausibility with *distributional* alignment. PAWBench introduces **probabilistic alignment** as a distribution-level criterion: given the same initial observation and action, a world model should reproduce the *distribution* of physically valid futures. PAWEval, the accompanying protocol, replays each scenario many times, converts rollouts into empirical outcome distributions, and scores them against reference distributions across 50 scenarios and 11 current systems.

**Prereqs:** [README.md](README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A benchmark and evaluation protocol for **video generators framed as world models**. Distinct from FVD-style plausibility metrics: PAWBench doesn't ask whether one video looks real, it asks whether the *set* of videos a model would generate matches the *set* of physically valid outcomes.

Two artifacts:

- **PAWBench** — 50 physical scenarios, each with defined outcome categories and reference occurrence probabilities.
- **PAWEval** — the protocol that turns repeated stochastic generations into an empirical outcome distribution and scores it against the reference.

## How it works

For each scenario $s$ with initial observation $o_0$ and action $a$:

1. **Repeat generation.** Sample $N$ videos from the model under fixed $(o_0, a)$, varying only prompt (optional), noise seed, and internal stochasticity.
2. **Outcome classification.** Assign each generated video to one of the scenario's outcome categories (via a scenario-specific classifier or human labels).
3. **Empirical distribution.** Compute the empirical distribution $\hat{p}$ over categories.
4. **Compare to reference.** Score $\hat{p}$ against the reference $p^*$ using distributional metrics (e.g. TV distance, KL, or the paper's outcome-level protocol).

**Reference distributions** encode which outcomes should be common vs. rare under real physics, not just which are possible.

## Why it matters

Across 50 scenarios × 11 current systems, **no model consistently recovers reference probabilities** while covering the full range of valid outcomes. The gap is now quantified: strong single-clip generators can be terrible probabilistic samplers.

Reframes what "world model" means. A world model that always shows the ball landing on the *most likely* side of the table is not a world model — it's a mode-collapsed clip generator. PAWBench gives the field a target beyond plausibility: **calibrated sampling of futures**.

Interventions the paper tests — language prompts, initial-noise sampling, targeted training — all move the model's distribution but none close the gap.

## Gotchas & tricks

- **Sample count matters.** $N$ must be large enough for the empirical distribution to be non-degenerate; small $N$ makes all systems look distributionally bad.
- **Reference-distribution provenance.** Where the reference comes from (physics simulator vs. human elicitation vs. real recordings) affects which systems win. Report it explicitly.
- **Not a substitute for FVD.** A model can be probabilistically aligned but individually implausible (blurry, artifacted) — use both.
- **Category granularity.** Coarse outcome bins hide mode collapse inside a bin; fine bins make the metric noisy. The 50-scenario battery calibrates bin granularity per scenario.

## Sources

- Paper: *PAWBench: How Far Are We from Probabilistically Aligned World Modeling?* — Pu et al., 2026 — [arXiv:2608.27345](https://arxiv.org/abs/2608.27345)
