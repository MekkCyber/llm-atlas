# On-Policy Reverse Distillation (OPRD)
*Depth — weak-to-strong training in which the teacher's policy-divergence signal is evaluated on the student's own rollouts and gated by a verifier.*

**TL;DR:** Standard forward-KL distillation caps the student at the teacher's ability. OPRD asks a different question: "how does the teacher's policy diverge from its reference *on trajectories the student itself produces*?", then amplifies verifier-supported updates in that direction. It keeps the stability profile of on-policy RL while enabling weak-to-strong generalisation — the student surpasses the weaker teacher on math/code benchmarks.

**Prereqs:** [grpo.md](grpo.md), [ppo.md](ppo.md), [rlvr.md](rlvr.md)
**Related:** [dpo.md](dpo.md), [rejection-sampling.md](rejection-sampling.md)

---

## What it is

A distillation objective sitting between behavior cloning and RL. Ordinary distillation minimises `KL(teacher ‖ student)` (or the reverse) on **teacher-drawn** samples; OPRD instead uses **student-drawn** samples and pulls the student in the direction the teacher itself would deviate from a fixed reference — but only on tokens a verifier supports.

## How it works

Let the teacher policy be `π_T`, its reference (typically the pre-fine-tune base) `π_ref`, the student `π_S`, and a verifier `V`. For each student rollout `y ∼ π_S(·|x)`:

1. **On-policy sample.** Generate `y` from the student. No teacher rollouts.
2. **Teacher signal.** Compute the log-ratio `Δ = log π_T(y|x) − log π_ref(y|x)` — i.e. the direction in which the teacher's policy has moved away from its reference on this trajectory.
3. **Verifier gate.** Evaluate `V(x, y)` (task-level correctness or a reward-model score). Amplify the gradient step along the teacher-divergence direction where `V` supports it; suppress it where `V` doesn't.
4. **Policy update.** Take a KL-clipped step — same clipping and reference-KL machinery as PPO/GRPO — using this composite direction.

Two properties fall out:
- If the teacher's divergence from its reference reflects a *correct* skill (verifier agrees), OPRD copies that skill in a way that respects the student's own capabilities on the student's own outputs — so the student can *exceed* the teacher on tokens where its own capacity is higher.
- If the teacher's divergence is wrong (verifier disagrees), the update is damped rather than trusted blindly.

## Why it matters

- **A concrete weak-to-strong recipe** that plugs into existing RL post-training loops — same clipping, same reference-KL, no reward model retrain.
- **Preserves policy-optimization stability.** OPRD is on-policy and verifier-gated, so it behaves like GRPO with a teacher-derived advantage rather than a group-relative one; the failure modes of forward-KL distillation (student pinned to teacher failures) don't apply.
- **Scales to multi-teacher and successive-transfer** regimes: the paper reports gains in both.
- **Load-bearing for scalable oversight.** Weak-to-strong is the empirical proxy for "the human supervisor is weaker than the model"; a stable on-policy version lowers the cost of that whole line of experiments.

## Gotchas & tricks

- **Verifier quality caps gains.** OPRD amplifies exactly what the verifier accepts; a noisy verifier turns the method back into vanilla forward-KL distillation with extra steps.
- **Reference-model choice matters.** The teacher's `π_ref` should be close to the student in family — otherwise the divergence direction is dominated by artefacts of the reference gap.
- **Amplification schedule.** Too aggressive a step in the divergence direction breaks stability; standard PPO clipping applies.
- **Compute cost is dominated by rollouts + teacher scoring**, not by loss computation; batch the teacher forward passes over the rollout buffer.

## Sources

- Park, Bae, Jung, Ko, Choi, Kim, Cameron, Courville, Yun — *Eliciting Weak-to-Strong Generalization with On-Policy Reverse Distillation* — 2026 — https://arxiv.org/abs/2609.08798
- Burns et al., *Weak-to-Strong Generalization* — 2023 — https://arxiv.org/abs/2312.09390
