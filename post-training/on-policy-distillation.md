# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation runs the student policy to generate trajectories, then supervises each student-generated token with the teacher's token-level distribution for that same prefix. Unlike classical sequence-level distillation (teacher trajectories, teacher labels), OPD trains the student on *its own* behavioral distribution, so the supervision targets the states the student actually visits at inference. It is especially popular for agentic post-training, where the teacher provides dense per-token guidance along cheap-to-generate student rollouts.

**Prereqs:** [_post-training](../post-training/_post-training.md), [_rl](../post-training/_rl.md)
**Related:** [grpo](../post-training/grpo.md), [rejection-sampling](../post-training/rejection-sampling.md), [branch-opd](../post-training/branch-opd.md), [pr-opd](../post-training/pr-opd.md)

---

## What it is

A distillation regime where the student generates the trajectories and the teacher provides token-level targets on them. Lies between pure SFT on teacher trajectories (off-policy; mismatched state distribution) and RL (on-policy; sparse reward). OPD keeps RL's on-policy state distribution and swaps the sparse reward for a dense teacher distribution.

## How it works

1. Sample a prompt / task.
2. **Student rolls out** a trajectory token-by-token with its current policy.
3. For each decision step, compute the teacher's next-token distribution conditioned on the same prefix.
4. Apply a token-level loss between student and teacher logits (KL, forward or reverse, is standard).
5. Optionally interleave with reward signals (success / verifier) if available.

In agentic settings, the "token-level decision" extends to within-turn decisions (which tool, which argument). The student rollout can hit tool calls and multi-turn environment state like a real deployment; the teacher labels every decision along the realized trajectory.

## Why it matters

- **State match.** The student is supervised exactly where it will act, so there's no train/inference distribution gap.
- **Dense signal.** Every token carries supervision, not just the final reward. This dominates sparse-reward RL for tasks where a competent teacher exists.
- **Cheap teacher labels.** Only requires a forward pass of the teacher per student token, not a teacher rollout per prompt.

Weakness: a single student rollout only realizes one continuation at every decision, so supervision coverage is bottlenecked by the number of rollouts. Variants like [BranchOPD](branch-opd.md) attack exactly this.

## Gotchas & tricks

- Reverse-KL vs forward-KL changes failure modes: forward-KL spreads mass toward teacher support (safer, less sharp); reverse-KL concentrates on teacher's modes (sharper, more mode-collapse-prone).
- Student entropy collapses fast — temperature schedules on the student's rollout policy are standard.
- If the student drifts far from the teacher's support early, teacher logits become uninformative at those states. A short SFT warmup on teacher trajectories, or a GRPO warm-start, mitigates.
- For agent tasks, decide whether supervision is per-token or per-decision-boundary. Mixing them silently reweights the loss.

## Sources

- Paper: *Branching On-Policy Distillation* — anon, 2026 — [arXiv:2610.13481](https://arxiv.org/abs/2610.13481) (frames OPD cleanly as the baseline it extends)
- Paper: *PR-OPD: Privileged Representation On-policy Self-Distillation* — Yang et al., 2026 — [arXiv:2609.36642](https://arxiv.org/abs/2609.36642)
- Classical reference: *Policy Distillation* — Rusu et al., 2015 — [arXiv:1511.06295](https://arxiv.org/abs/1511.06295)
