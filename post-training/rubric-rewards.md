# Rubric rewards with dynamic credit redistribution
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For long-horizon agents in the *outcome-blind* setting — no programmatic checker, no ground-truth success signal — the standard fallback is a multi-criteria rubric scored once per trajectory. A single scalar over tens of steps is a lousy RL signal. **DRACO** (Gandhi et al., 2026) grows rubrics *during* training to track the policy's capability, scores each per trajectory, and redistributes those scores over the steps that "activated" each criterion in closed form. The result: dense, differentiated per-step advantages for GRPO — no learned attribution module, no verifier.

**Prereqs:** [_rl](_rl.md), [_rewards](_rewards.md), [grpo](grpo.md)
**Related:** [rlvr](rlvr.md), [prm](reasoning/prm.md), [cot-reward-model](cot-reward-model.md)

---

## What it is

Two problems compound in outcome-blind long-horizon agent RL:

1. **No verifier.** Rule-based reward isn't available: the task can't be checked by regex or unit tests. That kills RLVR.
2. **Credit assignment.** Even if you have a trajectory-level judge (rubric-scored by an LLM), one scalar over 30 steps gives every step the same gradient. That kills fine-grained learning.

DRACO addresses both without training a separate attribution model.

## How it works

Three moving parts, chained per training step:

### 1. Dynamic rubric generation

A rubric is a set of yes/no criteria over the trajectory ("did the agent read the user's email before sending?", "did it verify the destination?"). Rather than fix rubrics up front (which capability-locks the curriculum), DRACO **generates rubrics per training generation**, conditioned on the current policy's failure modes on recent rollouts. The rubric set drifts as the policy improves — it stays near the learnable frontier.

### 2. Trajectory-level scoring

An LLM judge scores the rubric against each completed trajectory, producing per-criterion yes/no plus an annotation of *which steps* touched each criterion. This is the same rubric-judge pattern used in agentic evals, just wired into training.

### 3. Closed-form redistribution → per-step advantages

For a trajectory with $T$ steps and rubric criteria $\{c_i\}$, each criterion $c_i$ has:
- a score $s_i \in \{0, 1\}$,
- a set of *responsible* steps $S_i \subseteq \{1, \ldots, T\}$ (from the judge's annotation).

The per-step reward for step $t$ is a normalized redistribution:

$$
r_t = \sum_i s_i \cdot \mathbb{1}[t \in S_i] / |S_i|
$$

Then group-relative advantages (GRPO's estimator) are computed over $r_t$ across rollouts sharing the same prompt. The redistribution is closed-form — no trained attribution head.

## Why it matters

**Outcome-blind RL that works.** On AppWorld, DRACO gains **+15.9 pts over the base** and **+5.3 pts over GRPO trained with a sparse ground-truth reward** — despite not using verifiers itself. On Tau-Bench (OOD), +5.3 pts over base with no frontier judge.

**Beats verifier-based training in some settings.** The rubric signal is dense; the ground-truth-reward signal is sparse. Dense-but-noisy can beat sparse-but-correct once trajectory length passes some threshold. That's the load-bearing observation.

**No new trained pieces.** The redistribution is analytical, so DRACO doesn't add an attribution model with its own overfitting/reward-hacking failure mode.

## Gotchas & tricks

- **Rubric drift.** If the judge's rubrics drift faster than the policy learns them, the reward signal becomes non-stationary. In practice, cap rubric refresh frequency to once per generation.
- **Judge cost.** LLM-graded rubrics per trajectory can dominate rollout cost; a smaller judge with a distilled rubric prompt is a common shortcut.
- **The redistribution assumes each criterion is well-localized.** For criteria whose "responsible steps" span the whole trajectory, the per-step differentiation collapses back to a scalar.
- **Not a substitute for verifiers where you have them.** RLVR still wins when a rule verifier exists; DRACO is the recipe for when it doesn't.

## Sources

- Paper: *DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training* — Gandhi, Goyal, Kate, Rizk, IBM Research, 2026 — [arXiv:2609.04094](https://arxiv.org/abs/2609.04094).
- Code: https://github.com/IBM/draco
