# Distilled Reinforcement Learning

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Distilled RL folds teacher supervision *inside* the RL objective, so the student pulls toward the teacher on tokens where the teacher was right (as scored by the RL reward), and ignores it elsewhere. This gets around the two well-known failure modes of on-policy distillation (OPD): similar teachers add nothing new; substantially different teachers destabilize the student. It also fixes a weakness of pure RL — the credit assignment problem under coarse outcome supervision — by giving fine-grained per-token guidance where the teacher agrees with the reward.

**Prereqs:** [grpo.md](./grpo.md), [ppo.md](./ppo.md), [_rl.md](./_rl.md)
**Related:** [rejection-sampling.md](./rejection-sampling.md) · [rlvr.md](./rlvr.md) · [_post-training.md](./_post-training.md)

---

## What it is

A composite post-training objective. On each rollout, two supervision signals are computed:

- The usual RL signal (GRPO-style advantage from the terminal reward).
- A per-token KL to a teacher's logit distribution on the same trajectory.

The two are combined so that the teacher-KL term is **gated by reward-consistency** — token positions where the teacher's high-probability action would have hurt the terminal reward get their teacher-KL down-weighted; positions where the teacher was consistent get it up-weighted.

## How it works

For each rollout $o$ under policy $\pi_\theta$:

1. Compute standard GRPO advantages $A_i$ for the group (see [grpo.md](./grpo.md)).
2. Compute per-token teacher divergence $D_t = \mathrm{KL}(\pi_\theta(\cdot\mid s_t) \| \pi_\text{teach}(\cdot \mid s_t))$.
3. Compute a **reward-consistency gate** $g_t$ from the rollout's advantage sign and the teacher's local action distribution — token positions where teacher probability is highest on tokens that correlate with high reward get $g_t \approx 1$; positions where the teacher would nudge away from high-reward tokens get $g_t \approx 0$.
4. Combine:

    $$L = -\mathbb{E}[A_i \cdot L_\text{PPO-ratio}] + \lambda \cdot \mathbb{E}[g_t \cdot D_t] + \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_\text{ref})$$

The last term is the standard reference-model anchor. The middle term is the gated teacher signal.

## Why it matters

Two problems collapse into one clean objective:

- Pure RL wastes signal on trajectories where a strong teacher would already know the right next token.
- Pure OPD wastes signal on trajectories where the teacher is wrong (or from a different family), because it unconditionally imitates.

Reward-gated teacher KL uses each source of supervision where it's most reliable. The paper reports gains over both pure RL and pure OPD, with cross-family teacher/student transfer as the standout — previously OPD's biggest failure mode.

## Gotchas & tricks

- The gate $g_t$ is critical. Ungated teacher-KL just recovers OPD and inherits its weaknesses.
- Reward-conditioning does introduce a chicken-and-egg risk: if the student's early reward estimates are noisy, the gate is noisy. Warmup with pure RL for a fraction of training helps.
- Teacher rollouts aren't required — only the teacher's *logit distribution* on the student's tokens is needed, which is a cheaper API/local-inference call than generating full teacher trajectories.

## Sources

- Paper: *Distilled Reinforcement Learning for LLM Post-training* — Chen Wang, Zhaochun Li, Jionghao Bai, Yining Zhang, Hexuan Deng, Ge Lan, Yue Wang (Nankai U. / Zhongguancun Academy / BIT / Zhejiang U. / CAS / HIT), 2026 — [arXiv:2607.17247](https://arxiv.org/abs/2607.17247) · [HF](https://huggingface.co/papers/2607.17247)
