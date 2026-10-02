# Rubric Rewards (IRT aggregation)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For open-ended RL tasks, a **rubric** of yes/no criteria is often cheaper than a scalar reward model. Standard aggregation sums the satisfied criteria's points; distinct verdict patterns collapse to the same scalar, and criterion importance is baked into fixed weights. Rubric Response Theory (RRT, Microsoft 2026) replaces the sum with a **2-parameter Item Response Theory likelihood** — treating verdict patterns as evidence about a scalar latent quality — and learns criterion difficulty / discrimination online during training. +1.7 points over GRPO on average; +2.8 to +5.6 on hard criteria; half the judge budget at the same quality.

**Prereqs:** [grpo.md](./grpo.md), [_rewards.md](./_rewards.md).
**Related:** [rlvr.md](./rlvr.md), [cot-reward-model.md](./cot-reward-model.md).

---

## What it is

Classical **rubric-as-reward** training: a judge (LLM or human) evaluates a rollout against $C$ criteria, each yielding a 0/1 verdict. Reward is $r = \sum_c w_c \cdot v_c$ with fixed $w_c$ (often all 1).

Two failure modes:
- **Pattern aliasing.** Two rollouts with different verdict patterns $(v_1, \ldots, v_C)$ but the same sum get the same reward, even if one pattern reflects a much higher-quality answer.
- **Fixed weights.** Which criterion is informative changes across the policy distribution — an "easy" criterion stops discriminating once the policy satisfies it reliably. Fixed $w_c$ can't adapt.

RRT addresses both.

---

## How it works

**Model.** For a scalar latent quality $\theta$ (the "rubric target"), each criterion $c$ has difficulty $b_c$ and discrimination $a_c$. Probability of a positive verdict:

$$
P(v_c = 1 \mid \theta) = \sigma(a_c (\theta - b_c))
$$

A rollout with verdict pattern $(v_1, \ldots, v_C)$ has likelihood

$$
L(\theta) = \prod_c P(v_c = 1 \mid \theta)^{v_c} \cdot (1 - P(v_c = 1 \mid \theta))^{1 - v_c}
$$

RRT uses the log-likelihood (or MAP $\hat\theta$) as the reward signal — maximizes the local signal-to-noise ratio for quality given the current rubric.

**Response Parameter Network (RPN).** A learned network reads criterion text + prompt and outputs $(a_c, b_c)$. The RPN is updated by **online expectation-maximization** from current rollout verdicts, so as the policy improves, the RPN tracks which criteria still discriminate.

**Adaptive judging (Fisher selection).** Pick the subset of criteria that maximize the Fisher information about $\theta$ for the current rollout. Cuts judge calls roughly in half.

---

## Why it matters

- **Replaces hand-tuned criterion weights.** The IRT model learns per-criterion importance from data, continuously.
- **Judge-budget savings.** With adaptive Fisher selection and a frozen RPN, half-budget RRT stays within 0.1 points of full-judging GRPO. Judge calls are often the dominant RL cost on open-ended tasks.
- **Reported gains.** Qwen3.5-4B policy, macro criterion score across Medical, Science, Rubrics-as-Rewards Science, RubricBench: +1.7 over GRPO. On hard / very-hard criteria: +2.8 to +5.6 over GRPO.
- **Theory-grounded.** IRT is the standard psychometric model for exactly this scenario (binary verdicts, latent trait). The RL twist is online updating of its parameters.

---

## Gotchas & tricks

- **Monotonicity assumption.** IRT presumes each criterion is a monotone indicator of the latent target. Criteria that *trade off* against each other break the model.
- **RPN warmup.** Early training, the RPN hasn't seen enough verdicts to estimate $(a_c, b_c)$; falls back toward uniform weighting. Not a problem in practice if the rubric is reasonable.
- **Online EM can thrash on distribution shift.** Big policy jumps change which criteria discriminate; use a small learning rate for the RPN or batch EM updates.
- **Judge still has to be reliable per criterion.** Noisy or biased per-criterion verdicts propagate into $\theta$. RRT doesn't rescue a bad judge.
- **Interpretable logs.** Logged $(a_c, b_c)$ pairs after training are directly interpretable (difficulty, discrimination), unlike opaque RM scores — nice side effect.

---

## Sources

- Paper: *Rubric Rewards from Item Response Theory* — Yazdani, Souri, Zhou, Chawla, Shahriari, Som, Song, Microsoft, 2026. arXiv:2609.35646.
- Reference: *Item Response Theory for Psychologists* — Embretson & Reise, 2000 — the classical IRT model RRT adapts.
