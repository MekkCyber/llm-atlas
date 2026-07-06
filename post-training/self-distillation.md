# On-policy self-distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use the current policy's own dense output distribution as a distillation target while training toward a task objective. It accelerates in-domain specialization but **fails to prevent forgetting** and can *collapse* out-of-distribution — a systematic negative result for the "denser signals = better learning" reflex in continual post-training. Mix on-policy self-distillation with off-policy replay or a reference-model KL to avoid collapse.

**Prereqs:** [grpo](grpo.md), [_rl](_rl.md)
**Related:** [_post-training](_post-training.md), [rlvr](rlvr.md), [dpo](dpo.md)

---

## What it is

Sparse RL rewards (a scalar per rollout) provide low-bandwidth training signal per step. **On-policy self-distillation** (SDPO in Wang et al., 2026) tries to enrich the signal by adding a per-token distillation term:

$$
L_{\mathrm{SDPO}}(\theta) = L_{\mathrm{task}}(\theta) + \lambda \cdot \mathrm{KL}\big(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta\big)
$$

where $\pi_{\theta_{\text{old}}}$ is a recent copy of the policy — not a fixed pretraining reference. Every token now has a dense target (the old policy's next-token distribution) alongside the task gradient.

The intuition: PPO / GRPO already use $\pi_{\theta_{\text{old}}}$ implicitly as a ratio denominator; SDPO makes it an *explicit* distillation teacher, densifying the signal per token.

---

## How it works

- **Rollouts** come from the current policy (on-policy).
- **Task loss** is standard GRPO / DPO / SFT-like, whatever the primary objective is.
- **Distillation loss** is a KL from a stopped-gradient copy of the policy at a small delay (e.g. one gradient step behind, or a rolling checkpoint every $k$ steps).
- **Combined** as a weighted sum. Reference-KL to the pretraining base is optionally dropped, since the self-distillation term is doing similar regularization work in-domain.

The regime that fails: **continual post-training**, where new tasks arrive sequentially and the policy is expected to retain old capabilities. SDPO concentrates the policy on the current task's mode — the self-teacher shifts every step, so long-term regularization to the base disappears.

---

## Why it matters

- **In-domain sample efficiency wins.** Wang et al. show SDPO learns the current-task distribution faster than plain GRPO — the dense per-token target genuinely helps early.
- **Forgetting and OOD collapse.** But retention on prior tasks degrades faster, and OOD performance can collapse entirely. The paper's message: "denser $\neq$ better" — denser signals from the current policy *concentrate* it, they don't *broaden* it.
- **Practitioner steer.** For continual regimes, don't drop the reference-model KL. Mix on-policy self-distillation with off-policy replay from prior tasks. Hybrid recipes beat pure SDPO on the paper's continual suite.
- **Distinguishes self-distillation from distillation-from-teacher.** Fixed-teacher distillation (e.g. from a larger frozen model) does *not* have this pathology — the target is external and doesn't drift.

---

## Gotchas & tricks

- **Delay of the self-teacher matters.** A 0-step delay collapses to the identity (no gradient). Too-large delay (e.g. keep last epoch) turns into vanilla knowledge distillation from a stale checkpoint. Wang et al. settle on 1–$k$ step delay tuned per task.
- **Reference-KL removal is the trap.** SDPO was tempting because the self-distillation term "looked like" it was providing regularization. It isn't — the self-teacher moves. Always keep some $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ term or explicit replay if you want retention.
- **On-policy only.** SDPO's failure mode goes away if you also include off-policy replay or a fraction of on-teacher-off-policy distillation. Consider hybrid schedulers.
- **Not the same as CoT self-distillation.** Reasoning-model self-distillation ($x \to \text{CoT} \to y$ trajectories fine-tuning the same model) is a *data-augmentation* recipe with a different failure mode. Don't confuse with SDPO.

---

## Sources

- Paper: *Denser $\ne$ Better: Limits of On-Policy Self-Distillation for Continual Post-Training* — Wang et al., 2026 — [arXiv:2607.01763](https://arxiv.org/abs/2607.01763).
- Related: [grpo](grpo.md), [dpo](dpo.md).
