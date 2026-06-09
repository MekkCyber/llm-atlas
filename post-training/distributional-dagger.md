# Distributional DAgger
*Depth — RL-from-rich-feedback recast as distributional imitation learning with a forward cross-entropy objective.*

**TL;DR:** Most RL post-training relies on a scalar reward and a value model for credit assignment. Distributional DAgger reframes the same problem as imitation learning against a blackbox expert *distribution* (rather than just its argmax), using a forward cross-entropy objective whose sequence-level gradient propagates future expert-student disagreement back to earlier decisions. Provides monotonic-policy-improvement and regret guarantees; outperforms RLVR and self-distillation baselines on scientific reasoning, code, and math.

**Prereqs:** [_rl](_rl.md), [rlvr](rlvr.md), [grpo](grpo.md)
**Related:** [rejection-sampling](rejection-sampling.md), [orm](reasoning/orm.md)

---

## What it is

A reframing of RL-from-rich-feedback. Where standard RL maximises a scalar reward via policy gradients, distributional DAgger treats the expert (often a stronger LLM or a verifier-augmented oracle) as defining a *distribution* over correct continuations and trains the student to match it under its own state visitation.

The DAgger lineage (Dataset Aggregation, Ross et al.) handles distribution shift in imitation learning by querying the expert at student-visited states. The "distributional" variant generalises beyond argmax-imitation to whole-distribution matching, and adds a forward CE objective whose theoretical guarantees follow.

## How it works

At each policy update:

1. Roll out the student to get a trajectory `τ`.
2. Query the (blackbox) expert at each state in `τ` for its conditional distribution over continuations.
3. Compute the forward cross-entropy loss between student and expert distributions, summed across the trajectory.
4. The sequence-level gradient — being a cross-entropy over the trajectory — naturally back-propagates downstream disagreement to earlier tokens, doing credit assignment without a value model.

Theory:

- **Monotonic policy improvement.** Each update is guaranteed to not make the student strictly worse against the forward-CE objective.
- **Regret bound.** Cumulative regret vs. the expert is bounded by a function of expert/student gap; matches DAgger-family guarantees.

## Why it matters

- Long-horizon credit assignment without a critic. Critics are noisy and require their own training; the forward-CE objective sidesteps them.
- Cleanly handles "rich feedback" — situations where the expert can give a *distribution*, not just a scalar reward.
- Beats RLVR and RL-with-self-distillation baselines on scientific reasoning, coding, and math, per the paper's experiments.

## Gotchas & tricks

- **Expert query budget.** Distributional queries are expensive if the expert is another LLM; per-trajectory sampling rate is a hyperparameter.
- **Distribution support mismatch.** When the student visits states the expert assigns near-zero density, forward CE blows up; standard fix is a clipping or smoothing term.
- **Composes with RLVR.** Use RLVR for verifiable signal and distributional DAgger when a strong expert is available; they're not mutually exclusive.

## Sources

- Paper: *Reinforcement Learning from Rich Feedback with Distributional DAgger* — Agrawal, Fein-Ashley, Rashidinejad — 2026 — [arXiv:2606.05152](https://arxiv.org/abs/2606.05152)
- Foundational: *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* — Ross, Gordon, Bagnell, 2011 — original DAgger.
