# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Distillation variant where the student trains on **its own rollouts**, with the teacher providing token-level supervision (log-probability match, score, or correction) **along the student's trajectory** rather than the teacher's. Fixes the exposure-bias problem of offline SFT distillation and the sparse-reward problem of pure RL.

**Prereqs:** [grpo.md](./grpo.md), [rejection-sampling.md](./rejection-sampling.md), [ppo.md](./ppo.md)
**Related:** [critique-guided-opd.md](./critique-guided-opd.md) · [multi-teacher-opd.md](./multi-teacher-opd.md) · [action-level-distillation.md](./action-level-distillation.md) · [rlvr.md](./rlvr.md) · [_distillation.md](./_distillation.md)

---

## What it is

Classical SFT distillation ("offline distillation") copies teacher outputs: the student is trained on `(prompt, teacher-response)` pairs. The student never sees its own mistakes during training, so at inference time it compounds errors on sequences the teacher would never have produced. This is exposure bias in a distillation setting.

OPD inverts the data loop: the **student generates the rollout**, and the teacher annotates it. Supervision is applied *along the student's path*, so the student learns to recover from states it actually visits. The teacher never shapes the trajectory, only grades it.

## How it works

The generic OPD loop:

1. Sample a rollout from the current student: $o \sim \pi_\theta(\cdot \mid q)$.
2. Query the teacher on the student's rollout. The query shape determines the OPD variant:
   - **Logit-level OPD:** teacher provides per-token distributions $\pi_T(\cdot \mid q, o_{<t})$. Loss = token-level KL $\sum_t \mathrm{KL}(\pi_T \| \pi_\theta)$. Needs white-box teacher + shared tokenizer.
   - **Score-level OPD:** teacher scores the whole rollout; the score becomes an RL advantage. Looks like RLVR with a model-based verifier (the teacher).
   - **Critique-level OPD:** teacher writes a critique that identifies the mistake; a revision is sampled; the revision is scored (see [critique-guided-opd.md](./critique-guided-opd.md)).
   - **Action-level OPD:** teacher's chosen *action* at the student's state is the target (see [action-level-distillation.md](./action-level-distillation.md)).
3. Compute an advantage from the teacher signal and apply a PPO-style clipped update, usually under GRPO's group-relative baseline.
4. Repeat; the rollout distribution shifts with the student.

## Why it matters

- **Exposure-bias free.** Supervision is applied where the student actually generates, not where the teacher would have.
- **Dense signal.** Logit-level OPD gives a token-level signal where RL only gives a trajectory-level reward.
- **Scales distillation past the SFT ceiling.** Several recent reasoning-RL papers show OPD improves beyond standard SFT-on-teacher-traces because the student can keep learning from its own failures.
- **Works with black-box teachers** in the score / critique / action variants — practically important when the teacher is an API-only frontier model.

## Gotchas & tricks

- **Mode collapse under logit KL.** Minimizing forward KL $\mathrm{KL}(\pi_T \| \pi_\theta)$ makes the student *cover* the teacher — safe but can be mode-wide. Reverse KL $\mathrm{KL}(\pi_\theta \| \pi_T)$ is mode-seeking and sometimes preferred in reasoning distillation.
- **Rollout drift.** The student's rollout distribution moves fast early; using a slightly off-policy rollout buffer helps variance, at the cost of a PPO-style importance ratio.
- **Teacher cost.** Logit-level OPD runs the teacher at full length on every rollout — often more expensive than the student. Score-only OPD runs the teacher once per rollout.
- **Agreement signal is reward-hackable.** Score-level OPD can be gamed by producing outputs the teacher scores high but that are not actually correct — the same failure mode as learned reward models.

## Sources

- Paper: *Recovering Reliable Supervision via Critique-Guided On-Policy Distillation* — Yu et al., 2026 — the critique-guided variant.
- Paper: *Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation* — Tian et al., 2026 — the multi-teacher variant.
- Paper: *ActionOPD: Distilling Decisions, Not Tokens* — 2026 — action-level variant.
- Paper: *More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic RL* — Zhong et al., 2026 — benchmarks OPD variants against GRPO.
