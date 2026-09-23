# On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a student LLM on its *own* rollouts, but replace the reward with per-token supervision from a teacher: at every position of a student trajectory, take the KL divergence between the student's next-token distribution and the teacher's next-token distribution. Cheap to sample (student policy stays fixed for the rollout), high-signal (teacher labels every token), and stays on-policy — no distribution mismatch between what the student is trained on and what it will generate at inference.

**Prereqs:** [rlvr.md](./rlvr.md), [_rl.md](./_rl.md), [_post-training.md](./_post-training.md)
**Related:** [rejection-sampling.md](./rejection-sampling.md) · [grpo.md](./grpo.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md)

---

## What it is

Standard SFT uses off-policy data (human text, distilled from a teacher's own rollouts) — the student is trained on trajectories it *would not have* produced, so a train-inference gap remains. Standard RL is on-policy but uses a scalar reward, which is high-variance and needs many rollouts.

**On-policy distillation (OPD)** sits between them. For a student policy $\pi_S$ and a teacher $\pi_T$:

1. **Sample** a trajectory $y_{1:T} \sim \pi_S(\cdot \mid x)$ (on-policy student rollout).
2. **Score** at each position with the teacher's next-token distribution $\pi_T(\cdot \mid x, y_{<t})$.
3. **Update** the student to minimize a token-level divergence (usually reverse KL) between $\pi_S(\cdot \mid x, y_{<t})$ and $\pi_T(\cdot \mid x, y_{<t})$.

The student learns to imitate the teacher *along its own trajectories*, not the teacher's trajectories.

## How it works

The reverse-KL objective at a fixed prefix:

$$
\mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_S}\left[\sum_t \mathrm{KL}\!\left(\pi_S(\cdot \mid x, y_{<t}) \,\|\, \pi_T(\cdot \mid x, y_{<t})\right)\right]
$$

In practice you compute the KL only at a **sparse subset of tokens** — running the teacher on every token in a long CoT is expensive. The subset is chosen by a **selector** that combines a *usefulness* score (how much this token could change the trajectory) with, per the 2026 IER work, an *information-efficiency* score that accounts for the noise in estimating the gradient from a sampled next token.

The **Information-Efficiency Ratio (IER)** decomposes the sampled-gradient error into signal and noise components using an information-geometric view; tokens with high IER give reliable updates at low cost. A candidate-set approximation makes IER practical to compute alongside the standard usefulness scores.

## Why it matters

- **On-policy, dense supervision.** Combines RL's on-policy-ness with SFT's dense per-token signal.
- **Cheap.** Sparse selection lets 0.1–1% of tokens carry the training signal, dropping teacher-inference cost by 100–1000× at little to no quality loss.
- **Emerging as the scalable alternative to RL** for reasoning-model post-training, especially when a strong teacher exists (e.g. distilling R1-style CoT into smaller students).

## Gotchas & tricks

- **Reverse KL is not forward KL.** Reverse KL is mode-seeking (the student concentrates on modes the teacher considers plausible); forward KL is mode-covering. Reverse KL is standard for OPD because you want a decisive student, not a fuzzy imitator.
- **Selector matters as much as objective.** A bad selector concentrates budget on trivially-correct or nearly-random tokens; the IER work shows selectors that account for gradient noise (not just token usefulness) recover full-OPD performance at 1% of the compute.
- **Teacher-student capability gap.** Too-large gap and the student never matches the teacher's next-token distribution → reverse KL saturates. Too-small gap and OPD is wasted compute vs. SFT.
- **On-policy drift.** The student's rollout distribution shifts during training, so old teacher logits go stale. Fresh sampling per update, not a big pre-computed OPD dataset.
- **Related to online DPO / RL from AI feedback** in spirit — teacher stands in for a reward model — but the *token-level* signal is what makes OPD distinct from preference-based methods.

## Sources

- Paper: *On-Policy Distillation of Language Models* — Agarwal et al., Google DeepMind, 2023 — introduces the on-policy KL-distillation objective for LMs.
- Paper: *1% of Tokens Can Be Enough: On Gradient Estimation in On-Policy Distillation* — Sheng et al., 2026 — Information-Efficiency Ratio for sparse-token selection; matches full OPD at 0.1–1% token budgets. [arXiv:2609.24432](https://arxiv.org/abs/2609.24432)
- Code: [github.com/BruceSheng1202/IER-OPD](https://github.com/BruceSheng1202/IER-OPD)
