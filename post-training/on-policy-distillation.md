# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A knowledge-distillation variant where the student is supervised on *its own* sampled trajectories rather than a fixed corpus. For each student rollout, the teacher re-scores the student's tokens and the student is pushed toward the teacher's distribution on that trajectory. Fixes the exposure-bias problem of offline distillation (student never sees sequences a weaker-sampler would actually produce) and is the standard distillation recipe for compressing RL-trained teachers.

**Prereqs:** [dpo.md](./dpo.md) (student/teacher view), [_rl.md](./_rl.md) (on-policy semantics), [rejection-sampling.md](./rejection-sampling.md).
**Related:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md), [ride.md](./ride.md), [cot-reward-model.md](./cot-reward-model.md).

---

## What it is

Offline distillation trains the student to match the teacher on sequences the *teacher* generated. The student never sees sequences it would actually produce, so once deployed it drifts into regions where the teacher's distribution is a poor target. OPD closes this gap: at every step of training, the student samples its own trajectory, the teacher evaluates that trajectory in parallel, and the loss is a per-token forward-KL (or cross-entropy on teacher next-token logits).

The pattern is identical in shape to on-policy RL — rollout, score, update — with the teacher's distribution replacing the reward.

---

## How it works

For each prompt $q$ sample a response $o \sim \pi_{\text{student}}(\cdot \mid q)$. Teacher-force the same prefix through the teacher to get $\pi_{\text{teacher}}(\cdot \mid q, o_{<t})$ at every position. The loss is:

$$
L_{\text{OPD}} = \mathbb{E}_{o \sim \pi_s} \; \sum_t \mathrm{KL}\!\left( \pi_{\text{teacher}}(\cdot \mid q, o_{<t}) \,\|\, \pi_{\text{student}}(\cdot \mid q, o_{<t}) \right)
$$

In practice this is implemented as a cross-entropy against the teacher's full next-token distribution (not just the sampled token), or a top-$K$ truncation when teacher logits are expensive. The student still only backprops through *its* forward pass; the teacher is frozen and only serves next-token targets.

Variants:
- **Output-space extrapolation** — supervise on $\pi_t + \alpha (\pi_t - \pi_{\text{base}})$ to let the student *exceed* the teacher. Noisy when the LM head attenuates representation changes anisotropically (see [ride.md](./ride.md)).
- **Verified-trajectory OPD (OASIS, 2026)** — only supervise rollouts where the final answer is label-verified; replaces written reference solutions with model-generated attempts as the teacher's context. Preserves effectiveness as model scale grows.
- **Self-OPD (UniEvo-VL, 2026)** — a unified multimodal model plays both roles, using its understanding head to critique its generation head's outputs.

---

## Why it matters

- **RL compression.** Running RL on a small student is often unstable or data-starved. OPD compresses a large RL-trained teacher into a cheap student without re-running RL.
- **Fixes exposure bias.** Classical SFT-style distillation gives the student teacher-shaped data; OPD gives it student-shaped data with teacher-shaped targets, matching the deployment distribution.
- **No reward model needed.** The teacher's distribution *is* the reward, so OPD doesn't require a separate RM, verifier, or preference dataset.

---

## Gotchas & tricks

- **Imitation gap scales with capability gap.** When the teacher sees privileged context (reference solutions, hints) the student can't use, OPD supervises hallucinations. Verified-scaffold variants (OASIS) are a direct fix.
- **Rollout cost.** On-policy means the student samples every step. Combine with mini-batching and short responses when teacher throughput is the bottleneck.
- **Teacher–student entropy mismatch.** A sharp RL teacher distilled into a cold student can collapse exploration. Keep a small KL term to the pre-OPD student checkpoint, mirroring the RL $\pi_{\text{ref}}$ trick.
- **The teacher plateaus the student.** Standard OPD caps the student at teacher capability. Representation-residual methods ([ride.md](./ride.md)) and output-space extrapolation push past it; both introduce stability issues.

---

## Sources

- Paper: *The Teacher Is a Direction, Not a Destination: Extrapolating RL-Induced Representation Residuals in On-Policy Distillation* — Li et al., 2026 — formalizes OPD vs. output-space extrapolation; introduces RIDE.
- Paper: *UniEvo-VL* — Wu et al., 2026 — self-OPD in a unified MLLM.
- Paper: *Overcoming Scaling Limits in On-Policy Self-Distillation for LLM Reasoning (OASIS)* — Hossain et al., 2026 — verified-scaffold OPD for math reasoning.
