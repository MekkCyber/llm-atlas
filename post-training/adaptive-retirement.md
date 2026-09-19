# Adaptive Retirement (RetireOPD)
*Depth — an on-line criterion for turning off on-policy distillation before it starts to hurt.*

**TL;DR:** Multi-turn agent RL often layers on-policy distillation (OPD) from a privileged teacher on top of GRPO to get dense supervision. But teacher signal is *stage-dependent*: helpful early, actively harmful late. Adaptive Retirement watches (a) the student-teacher discrepancy and (b) the student's success rate as fractions of the teacher's, and *the student itself drops the teacher* once the discrepancy stops shrinking and success reaches a target fraction of the teacher's. From then on, training is RL alone. No hand-tuned distillation schedule.

**Prereqs:** [on-policy-distillation](on-policy-distillation.md), [grpo](grpo.md)
**Related:** [rlvr](rlvr.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md), [rejection-sampling](rejection-sampling.md)

---

## What it is

A control rule for joint RL + OPD training that decides *when to disable* the OPD term without human intervention. It replaces the two common conventions:

- **Full-schedule OPD.** Distill throughout training; the "late-stage drag" observed in RetireOPD says this hurts on agentic tasks.
- **Hand-tuned anneal.** Linear or step decay of the OPD coefficient — but the right schedule is task-dependent and rarely known.

Adaptive Retirement is data-driven: the student decides.

---

## How it works

RetireOPD has two stages:

1. **Skill-conditioned teacher training.** Train a teacher with *decoupled privileged skills* — env rewards drive its skill training rather than fixed distillation targets. This produces a teacher whose privileged supervision is more reliable than a raw oracle.
2. **Joint RL + OPD with retirement.** Train the skill-free student with GRPO + OPD from the teacher. Track:
   - $\Delta_t = \mathrm{KL}(\pi_T \| \pi_S)$ — the student-teacher discrepancy over recent batches.
   - $\rho_t = \text{success}(\pi_S) / \text{success}(\pi_T)$ — student success as a fraction of teacher's.

**Retirement fires** when both conditions hold:
- $\Delta_t$ has *stopped shrinking* (plateau over a window).
- $\rho_t \ge \tau$ for a target fraction $\tau$ (e.g. 0.9).

Once retired, OPD is off; training continues with GRPO alone.

The criterion is model- and task-agnostic — it uses only quantities the training loop already tracks.

---

## Why it matters

Distillation-into-RL is the modern recipe for agent training (Kimi-K1.5, DeepSeek-R1) and it works well *until* it doesn't. The failure mode — teacher signal that used to help now drags a competent student toward teacher artifacts — was known but not addressed with a general control rule. Adaptive Retirement is a drop-in criterion for any joint OPD+RL loop.

The privileged-info result (Zhang et al. 2026) makes this even more relevant: since privileged references add modest value, retiring the teacher earlier is often the right choice.

---

## Gotchas & tricks

- **Both conditions matter.** Discrepancy plateau alone can trigger on a bad teacher; success fraction alone can trigger on an easy task. Require both.
- **Choose $\tau$ conservatively for open-ended tasks.** For coding agents where "success" is verifiable, $\tau \approx 0.9$ is safe. For softer rewards, use a wider window or a rank-based criterion.
- **Once retired, don't re-enable.** The paper trains RL-only after retirement; a naive re-enablement on a new phase can flip signals.
- **Teacher decoupling is load-bearing.** RetireOPD's teacher is *trained with env rewards on privileged skills*, not just prompted with them. A fixed prompted teacher is less reliable and produces a worse retirement point.

---

## Sources

- Paper: *RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning* — Yu et al., 2026 — [arXiv:2609.20784](https://arxiv.org/abs/2609.20784).
- Context: *What Does Privileged Information Add to On-Policy Self-Distillation?* — Zhang et al., 2026 — [arXiv:2609.20612](https://arxiv.org/abs/2609.20612).
