# Progress Advantage
*Depth — the implicit step-level signal that falls out of any KL-regularized RL post-training run.*

**TL;DR:** Process reward models for agent settings are usually intractable to build — long horizons, irreversible actions, stochastic environments make both human and Monte-Carlo annotation infeasible. Progress advantage shows you don't need one: under a stochastic Markov decision process with KL-regularized RL post-training, the log-probability ratio between the RL-trained policy and its frozen reference policy **exactly recovers the optimal advantage function**. It's annotation-free, domain-agnostic, and computed as a byproduct of training. Validated as a drop-in replacement for trained PRMs in test-time scaling, uncertainty quantification, and failure attribution.

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md), [dpo.md](dpo.md)
**Related:** [reasoning/prm.md](reasoning/prm.md), [_rewards.md](_rewards.md), [cot-reward-model.md](cot-reward-model.md)

---

## What it is

A way to read step-level credit out of an already-trained RL policy, *for free*. Concretely, at any partial trajectory state $s$ the implicit advantage is

$$
A_{\text{progress}}(s) = \log \pi_\theta(s) - \log \pi_{\text{ref}}(s)
$$

— the same log-ratio DPO uses to express the reward, here reinterpreted as a process reward signal. Under a KL-regularized policy gradient with reference $\pi_{\text{ref}}$, the optimal policy's log-ratio over the reference is the optimal advantage function. So the byproduct of standard RL post-training already contains the step-by-step credit signal that a trained PRM would provide.

## How it works

The construction needs only the artifacts every RL post-training run already produces:

1. Train policy $\pi_\theta$ from reference $\pi_{\text{ref}}$ with a KL-regularized RL objective (GRPO, PPO, RLVR — all qualify).
2. At inference, score any partial trajectory by summing the per-token log-ratios up to that point. No additional model, no annotation, no rollouts beyond the candidate trajectory itself.
3. Use that scalar as a step-level signal — for best-of-N reranking, confidence estimation, or attributing failure to a specific step.

Because the construction is **annotation-free** and **domain-agnostic**, it transfers across model families and task types without retraining a verifier.

## Why it matters

- **Kills the case for PRMs in agent settings.** The dominant blocker for agentic RL has been "we'd want PRM-style step rewards but can't build a PRM." Progress advantage shows the RL artifact already contains them.
- **Free.** Zero extra training compute, zero annotation effort, computed from existing forward passes.
- **Generalizes.** Same object works for test-time scaling, uncertainty quantification, and failure attribution — three loosely-related tasks that previously each needed bespoke machinery.
- **Surpasses dedicated trained reward models** on the paper's five benchmarks across four model families, despite needing no task-specific training.

## Gotchas & tricks

- Requires the reference $\pi_{\text{ref}}$ to be kept around at inference — same constraint as DPO. Cheap if you already had it for KL regularization.
- The signal degrades if the reference is too far from the current policy (very long RL runs without periodic re-anchoring).
- Sign matters: positive log-ratio = state more likely under the RL policy = "progress"; negative = the policy is *less* likely to be there than the reference, often a failure-attribution clue.
- Doesn't replace a verifier for *outcome* reward. It's a process-level reweighting of trajectories the policy already finds plausible.

## Sources

- Paper: *Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents* — Oh, Li, Park, Yeh, Mallick, Li, 2026 — [arXiv:2606.26080](https://arxiv.org/abs/2606.26080)
- Related: *Direct Preference Optimization* — Rafailov et al., 2023 — same log-ratio object, used as the reward.
