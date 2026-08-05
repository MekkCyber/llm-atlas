# On-Policy Self-Distillation (OPSD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-training regime where a student model generates rollouts and is then supervised, on those own rollouts, by a "teacher" that has access to privileged information (a longer context, a solution rationale, a tool result, a ground-truth answer). The student learns from its own trajectory but corrected with teacher information — combining RL-style on-policy data with distillation-style supervision.

**Prereqs:** [_rl](_rl.md), [_rewards](_rewards.md)
**Related:** [rejection-sampling](rejection-sampling.md), [grpo](grpo.md), [dpo](dpo.md)

---

## What it is

Regular knowledge distillation uses fixed teacher completions — off-policy data, so distribution shift eats into gains as the student diverges. Pure on-policy RL (PPO/GRPO) is on-distribution but the reward signal is coarse. **On-policy self-distillation** splits the difference:

1. Student generates a completion under its current policy $\pi_\theta$.
2. A **privileged teacher** — often the same model with extra context (gold answer, extra thinking budget, retrieved evidence, a slightly stronger checkpoint) — scores or re-rolls that completion.
3. Student is updated to match the teacher's *per-token* distribution on the student-generated prefix.

The result: on-distribution data (like RL) with dense per-token supervision (like distillation).

## How it works

Given a student policy $\pi_\theta$ and a teacher $\pi_T$ (with extra info $z$), and a student-generated trajectory $y_{1:T}$ conditioned on prompt $x$:

$$
\mathcal{L}_{\text{OPSD}}(\theta) = \sum_{t=1}^{T} \mathrm{KL}\!\left( \pi_T(\cdot \mid x, z, y_{<t}) \,\Big\|\, \pi_\theta(\cdot \mid x, y_{<t}) \right)
$$

Note: the *rollout* is student-generated (on-policy), but the *target* at each token comes from the teacher, which sees privileged info $z$ that the student does not.

Popular concrete instantiations:
- **Long-CoT self-distillation** — teacher has more thinking budget than the student.
- **Rationale-conditioned** — teacher sees the ground-truth answer; student is supervised on the resulting per-token distribution.
- **Retrieval-augmented teacher** — teacher can look things up; student learns to "guess like an informed teacher."
- **Off-policy-with-on-policy-mix** — combine student rollouts with a fraction of teacher rollouts.

## Why it matters

- **Dense signal, on-distribution.** Solves both the coarse-reward problem of RLHF/RLVR and the distribution-mismatch problem of off-policy distillation.
- **Cheap.** A single forward pass through the teacher per rollout — no reward model, no advantage estimation.
- Increasingly used inside reasoning-training recipes (long-CoT distillation, verifier-conditioned distillation) at frontier scale.

## Gotchas & tricks

- **Privilege illusion** — the student can learn behaviour that only makes sense *because* the teacher had extra info, then confidently reproduce it at inference when that info is gone. See [dapd](dapd.md) for a concrete mitigation.
- **Teacher-student capability gap** — if the teacher is too strong, the target distribution is unreachable and gradients explode; if too close, there's nothing to learn. A gap of 1–2 capability levels tends to work best.
- **KL vs cross-entropy target** — KL against teacher soft-labels is standard, but hard-label (top-1 argmax) cross-entropy is simpler and often within noise for small teacher-student gaps.
- **Do not train on tokens the teacher itself is uncertain about.** Mask out positions where the teacher's own entropy exceeds a threshold — otherwise you distill noise.
- **On-policy freshness.** Re-generate rollouts every $k$ steps ($k$ small, e.g. 1–4). Stale rollouts drift off-policy and the whole point is lost.

## Sources

- Paper: *DAPD: Dual-Anchored Policy Distillation* — arXiv:2608.01735, 2026 — names and mitigates the "privilege illusion" failure mode.
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — SFT-on-teacher-CoT plus RL is a related pattern.
- Paper: *Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning* — 2024 — early formalisation.
