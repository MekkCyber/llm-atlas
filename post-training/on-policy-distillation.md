# On-Policy Distillation
*Depth — reverse-KL distillation from student rollouts that transfers compositional skills but not new facts.*

**TL;DR:** On-policy distillation samples continuations from the student, then nudges the student's logits toward the teacher's via **reverse KL** on those on-policy sequences. This dynamic transfers reasoning *skills* (compositional patterns the student can generalise) across unseen structures but transfers almost **no new factual knowledge**. Positions on-policy distillation as a *behavioural* training tool and separates it cleanly from knowledge-injection (which needs SFT/continued-pretraining on labelled text).

**Prereqs:** [_post-training.md](_post-training.md), [dpo.md](dpo.md), [ppo.md](ppo.md)
**Related:** [grpo.md](grpo.md) · [rlvr.md](rlvr.md)

---

## What it is

A student–teacher training recipe that treats the student's own generations as the data distribution the loss operates on. For each prompt $x$:

1. Sample $y \sim \pi_{\text{student}}(\cdot \mid x)$.
2. Query the teacher on $(x, y)$ to get a reference distribution.
3. Minimise $\mathrm{KL}(\pi_{\text{student}} \,\|\, \pi_{\text{teacher}})$ along $y$ — **reverse** KL, since the expectation is under the student.

Reverse KL is **mode-seeking**: it concentrates student probability mass where the teacher assigns probability, but allows the student to drop teacher modes it has no need for. Combined with sampling from the student, this makes the loss stay in the student's own output manifold — unlike SFT, which can push the student toward sequences it can't produce fluently.

## How it works

- The student sampling step produces *coverage* over sequences it will plausibly generate in deployment.
- The teacher provides *per-position* soft targets on exactly those sequences.
- Reverse KL is the right discrepancy because forward KL (SFT-like) wastes capacity matching teacher modes the student rarely visits.

The empirical finding of the originating paper: this dynamic transfers *procedure* (chain-of-thought shape, tool-use pattern, decomposition structure) but barely moves student factual-recall probes. The student's representation space learns to execute what the teacher does, not to remember what the teacher knows.

## Why it matters

- **Clean decision rule for practitioners.** New procedure (CoT structure, tool call pattern, planning style) → use on-policy distillation. New facts (recent events, domain-specific knowledge) → use SFT on labelled data or continued pretraining.
- **Recalibrates "small matches large via distillation" claims.** Many such claims silently include knowledge probes in their eval, inflating what distillation delivers. The dissociation shows those gains come from procedure match, not knowledge transfer.
- **Shared primitives with policy-gradient RL.** On-policy sampling, student-centric loss — a bridge between distillation and RL that's useful when thinking about agent post-training.

## Gotchas & tricks

- Reverse KL can collapse the student to a single mode if the teacher is multimodal and under-sampled; mix in a small forward-KL / SFT term to retain diversity.
- Teacher queries dominate cost — cache teacher logits across epochs when the student doesn't drift too much.
- Factual gaps show up subtly: the student *acts* like it knows but produces confident wrong facts. Combine with a knowledge probe before concluding "distillation worked."
- Prompt distribution matters: the student samples under some prompt mix; distillation only transfers skill on prompt types in that mix.

## Sources

- Paper: *On-Policy Distillation Teaches New Skills but Not New Knowledge* — Tang & Yang, 2026 — [arXiv:2610.09639](https://arxiv.org/abs/2610.09639).
