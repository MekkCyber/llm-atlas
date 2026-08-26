# R²-OPD (Reasoning-Progress-Aware OPD)
*Depth — filtering on-policy distillation rewards by rank-disagreement with a reasoning-progress signal.*

**TL;DR:** On-policy distillation (OPD) treats every token-level teacher reward as ground truth, but teacher signal often disagrees with *actual reasoning progress* — a step that makes real progress can still get a low reward for deviating from the teacher's own outputs. R²-OPD builds two within-trajectory rankings — one from teacher-derived rewards, one from an independent progress reward — and suppresses distillation supervision on spans where they disagree.

**Prereqs:** [orm](orm.md), [prm](prm.md), [long-cot-rl](long-cot-rl.md)
**Related:** [../rejection-sampling](../rejection-sampling.md), [../grpo](../grpo.md)

---

## What it is

A per-span filter on top of standard on-policy distillation for reasoning. Instead of trusting the teacher-derived reward on every reasoning span, R²-OPD cross-checks against an independent progress estimator and drops supervision on disputed spans.

## How it works

- For each rollout, segment into reasoning spans (contiguous chains-of-thought).
- **Ranking A:** teacher-derived reward per span (from the OPD teacher: log-prob delta, KL-based, or reward-model score).
- **Ranking B:** independent *reasoning-progress reward* per span — a separate signal for whether the span advances toward the correct answer (verifier-derived, PRM-like, or delta-in-solved-probability).
- Compare the two rankings within the same trajectory. On spans where they *disagree* (e.g., teacher ranks high but progress ranks low, or vice versa), zero out or downweight the distillation reward.
- Spans where both rankings agree keep full teacher supervision — the OPD loss is applied as usual.

## Why it matters

OPD's failure mode is subtle: the student can drift into "sounds like the teacher" behavior at the expense of solving the problem. Filtering by rank *disagreement* is a lightweight, hyperparameter-light way to make sure teacher signal only reinforces spans that also help reasoning progress — the two objectives get reconciled without abandoning the teacher.

## Gotchas & tricks

- Rank comparison (not scalar comparison) sidesteps scale mismatch between the two reward types.
- Choice of progress reward matters — a noisy PRM can create spurious disagreements. Verifier-derived progress (solved / step-verifier) is cleaner than a learned PRM.
- Only bites where the teacher and progress signals actually disagree; on easy problems the two rankings align and R²-OPD reduces to plain OPD.

## Sources

- Paper: *Beyond Imitation: Filtering On-Policy Distillation by Reasoning Progress* — Yang et al., 2026 — [arXiv:2608.19408](https://arxiv.org/abs/2608.19408)
