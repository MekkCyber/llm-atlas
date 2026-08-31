# PAWBench — Probabilistic Alignment for World Models
*Depth — a distributional benchmark for video generators pitched as world models.*

**TL;DR:** Video-generator "world models" are almost always evaluated on single-video plausibility (FVD, human ratings). But physical processes are inherently stochastic — dropping a ball ten times gives ten different bounces. PAWBench asks a harder question: when you sample many rollouts from the same initial condition, does the empirical distribution over outcomes match reality? Across 50 scenarios and 11 systems, no model does.

**Prereqs:** [README.md](README.md)
**Related:** [../agents/_world-models.md](../agents/_world-models.md), [../multimodal/flow-matching.md](../multimodal/flow-matching.md)

---

## What it is

A benchmark suite with two components:

- **PAWBench** — 50 curated scenarios where a physical process can unfold in more than one valid way (a die roll, an unstable stack falling, a diver's flip direction), each with a reference distribution over valid outcomes.
- **PAWEval** — a protocol that converts repeated video rollouts from a model into empirical outcome distributions, then compares them to the reference (KL, TV, or per-outcome frequency error).

Both target **probabilistic alignment** — the requirement that a world model reproduce not one plausible trajectory but the *distribution* of possible trajectories under the same initial observation and action.

## How it works

For each scenario:

1. Fix an initial observation (video prefix) and action prompt.
2. Sample `N` completions from the model under evaluation.
3. Auto-classify each completion into one of the scenario's outcome categories (via a strong VLM judge with human spot-checks).
4. Compute the empirical distribution `p̂(outcome)`.
5. Compare to the human-labeled reference `p*(outcome)` with a distributional metric.

Aggregate across scenarios to score the model. Ablations vary text prompts, initial noise sampling, and further training — none of them consistently closes the gap.

## Why it matters

If world models are meant to power planning, simulation, or offline RL, single-shot plausibility is the wrong bar. A model that always predicts "the die shows 1" scores well on plausibility (each individual video is fine) but is useless for anticipating outcomes. PAWBench formalizes the gap and shows current systems collapse to a narrow mode rather than modeling stochastic physics.

## Gotchas & tricks

- Judge choice matters — a weak VLM classifier washes out subtle outcome differences; use a strong VLM with per-scenario few-shot examples.
- `N` must be large enough for the distribution to stabilize; the paper uses tens per scenario.
- Prompt engineering can *shift* the distribution but rarely widens it — models are entropy-limited, not prompt-limited.
- Complement rather than replacement for FVD / CLIP-based scores — distribution and per-video quality are orthogonal axes.

## Sources

- Paper: *PAWBench: How Far Are We from Probabilistically Aligned World Modeling?* — Pu, Zhuo, Paul, Zhou, et al., 2026 — [arXiv:2608.27345](https://arxiv.org/abs/2608.27345)
