# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Supervise a student LLM on trajectories sampled from *its own* policy, using a teacher LLM to score or relabel those trajectories. This transfers the teacher's **reasoning behavior** — not the teacher's answers on specific problems — and generalizes best when teacher and student share model *origin* (same base family). Multi-teacher OPD produces a mixture-dependent seesaw between capabilities.

**Prereqs:** [rejection-sampling.md](./rejection-sampling.md), [_post-training.md](./_post-training.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md) · [grpo.md](./grpo.md) · [rlvr.md](./rlvr.md)

---

## What it is

Classical (off-policy) distillation samples from the teacher and trains the student to imitate. **On-policy distillation** flips the sampling side: the student generates the trajectory, the teacher provides the training signal (either a token-level KL against the teacher distribution, or a relabelled response the student is trained to match). The student always sees its own errors.

## How it works

Two common variants:

```
# KL-OPD
for prompt in batch:
    y ~ π_student(· | prompt)                    # student rollout
    loss = KL( π_teacher(· | prompt, y_<t) || π_student(· | prompt, y_<t) )

# Relabel-OPD (rejection-sampling flavour)
for prompt in batch:
    y ~ π_student(· | prompt)                    # student rollout
    y* = teacher_correction(prompt, y)           # teacher rewrites/improves
    loss = -log π_student(y* | prompt)           # SFT on relabelled trajectory
```

Neither variant requires that the teacher can *solve* the prompt — the interesting empirical finding from Li et al. (2026) is that teacher-unsolved problems are still useful, because the training signal is the teacher's *behavior* on the state the student actually visits.

## Why it matters

- **Transfers behavior, not answers.** In-domain performance transfers with training difficulty barely mattering — same-origin OPD generalizes across languages, reasoning horizons, and adjacent domains even when trained on only one.
- **Origin matters more than domain.** Same-origin (teacher and student from the same base family) transfers broadly; cross-origin OPD mostly overfits to the trained distribution. Practical implication: distill Qwen from Qwen, DeepSeek from DeepSeek — mixing families loses transferability.
- **Multi-teacher does not compound.** Because per-prompt routing cannot confine each teacher's influence, combining teachers produces a **mixture-dependent seesaw**: gains in one teacher's capability come at the cost of another. Naive `weight1·teacher1 + weight2·teacher2` is not additive in downstream performance.
- **Compute picture.** Cheaper than RL (no reward-model calls, no critic) but pricier than SFT (student rollouts on every step). Comparable to GRPO in wall-clock when teacher scoring is a forward pass.

## Gotchas & tricks

- **Distribution-shift is the point.** The whole reason on-policy > off-policy for distillation is that the student sees its own error distribution. If you cache student rollouts and reuse them across epochs, you defeat the purpose.
- **Teacher must cover the student's trajectory space.** If the student wanders into states the teacher has never seen, the teacher's signal is unreliable. This is worse cross-origin (why cross-origin OPD generalizes poorly).
- **Multi-teacher: prefer sequential curricula.** A student trained on teacher A then teacher B often outperforms simultaneous mixing, because the seesaw is at least ordered.
- **Compatible with RL.** OPD and RLVR are complementary: OPD to transfer behavior, RLVR to sharpen on verifiable rewards. Sequential is common.

## Sources

- Paper: *Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models* — Li et al., 2026 — the origin-effect and multi-teacher seesaw study.
- Paper: *Distilling the Knowledge in a Neural Network* — Hinton et al., 2015 — off-policy distillation baseline.
- Paper: *On-Policy Distillation of Language Models* — Agarwal et al., 2023 — earlier OPD formulation.
