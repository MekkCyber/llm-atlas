# On-Policy Distillation
*Depth — token-level distillation on trajectories the student itself produced.*

**TL;DR:** Sample rollouts from the *current* student, then train the student with token-level distillation loss against a teacher's distribution on those same rollouts. Sits between vanilla SFT (teacher-only trajectories) and pure RL (outcome reward only): dense per-token supervision, but on the on-policy state distribution the student actually visits. Used in modern reasoning post-training alongside RLVR.

**Prereqs:** [ppo](ppo.md), [grpo](grpo.md), [rlvr.md](rlvr.md)
**Related:** [rejection-sampling](rejection-sampling.md), [cot-reward-model](cot-reward-model.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md), [reasoning/length-penalty](reasoning/length-penalty.md)

---

## What it is

Given a student policy `π_s` and a teacher policy `π_t` (typically larger or already RLVR-trained), on-policy distillation (OPD) does:

1. Sample a rollout `τ = (x, y_1, …, y_T)` from `π_s`.
2. For every token, compute the KL between `π_t(· | x, y_{<t})` and `π_s(· | x, y_{<t})`.
3. Take a gradient step on `π_s` that minimizes that KL.

The distinguishing move is step 1: the *state distribution* is the student's, not the teacher's. That fixes the covariate shift that plagues offline distillation (student never sees the states it will visit at inference time).

## How it works

At each token position the loss is a forward or reverse KL against the teacher:

$$
\mathcal{L}_{\text{OPD}}(x, y) = \sum_{t=1}^{T} \mathrm{KL}\bigl(\pi_t(\cdot \mid x, y_{<t}) \,\Vert\, \pi_s(\cdot \mid x, y_{<t})\bigr)
$$

The gradient is dense — every token contributes signal — which is why OPD often gets more useful updates per rollout than outcome-reward RL on the same trajectories.

Practical recipes usually stack OPD with an RL objective. Group-relative frameworks (GRPO) can add a per-token distillation term on top of the group-baseline advantage; RLVR pipelines can alternate OPD steps with verifier-reward steps.

The "self-evolving" variant (see *Seed*, 2026) drops the fixed teacher: it treats *successful* on-policy trajectories as hindsight demonstrations and distills their behavioral effect back into `π_s`, closing the loop without an external teacher model.

## Why it matters

OPD is best understood as an **exploration catalyst** (Demystifying OPD, 2026): it does not raise the capability ceiling, it steers the student toward correct reasoning paths inside a ceiling that RLVR would eventually reach anyway. Two consequences:

- **Prompt diversity > per-prompt sample count.** OPD gains scale with how many *distinct* prompts you distill on, not how many samples per prompt.
- **Signal quality > teacher scale.** A too-large teacher can *hurt* if the distributional gap makes the token loss misalign with task correctness.

## Gotchas & tricks

- **Student-Teacher Mismatch.** A much larger teacher's per-token distribution can be dominated by tokens the student can't reach; the KL then pushes the student off the correct-answer manifold. Advantage clipping on the OPD term dampens it.
- **Length Exploitation.** An aggregated token-level loss lets the student *shorten* or *pad* responses to game the objective — a length-dependent shortcut, not reasoning. Log-scale compression of the per-token loss removes the incentive.
- **Reward-shaped OPD ≠ OPD.** If you use only the teacher's *argmax token* rather than its distribution, you have soft SFT on student rollouts, not OPD — coarser signal, weaker gains.
- **Determinism.** Reproducibility requires pinning both models' inference config; small floating-point differences in the teacher's distribution swamp the KL.

## Sources

- Paper: *Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations* — 2026 — names OPD as an exploration catalyst and identifies the two pathologies.
- Paper: *Seed: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning* — Wu et al., 2026 — teacher-free variant using hindsight skills from successful trajectories.
- Related: [rlvr.md](rlvr.md), [rejection-sampling.md](rejection-sampling.md).
