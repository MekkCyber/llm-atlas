# BudgetMix RL — Interaction-Budget Curriculum
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-time fix for multi-turn agent RL where the learned policy is **brittle to the deployment turn budget**. Standard agentic RL trains with a fixed maximum turn count T, and the resulting policy's performance can degrade sharply at smaller T. BudgetMix RL **randomizes T per rollout** during training, exposing the policy to a wide range of interaction budgets so its Pass@(K,T) surface stays flat across the deployment range.

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md)
**Related:** [_rl.md](./_rl.md) · [../evaluation/pass-at-kt.md](../evaluation/pass-at-kt.md) · [rl-prompt-curation.md](./rl-prompt-curation.md)

---

## What it is

Agent RL trains with a cap on the number of turns a trajectory can take. The cap is both a compute knob and a signal: at test time the policy will see *some* budget, and if that budget is tighter than training, the policy may never have learned to compress. BudgetMix RL treats the budget T itself as a training-time variable — different rollouts in the same RL batch see different T. The result is a policy whose capability does not fall off a cliff when deployed with a different budget.

## How it works

**Sample T per rollout.** For each prompt in the RL batch, draw $T_i$ from a distribution over interaction budgets — e.g. $T_i \sim \text{Uniform}\{T_{\min}, \ldots, T_{\max}\}$, or a mixture biased toward the deployment budget.

**Enforce the budget as a hard cap.** Terminate the trajectory at $T_i$ turns whether the task is solved or not. Reward is computed on whatever the policy produced by then (success/failure or verifier score).

**Compute advantages per budget bucket.** Within a group sharing a prompt, compare rollouts at the *same* budget — don't let a long-budget success inflate the advantage of a short-budget failure. GRPO's group-relative baseline is applied per budget bucket.

**(Optional) condition on T.** Add the chosen $T_i$ to the prompt ("you have at most T turns") so the policy can adapt its planning explicitly.

## Why it matters

Deployment constraints often tighten after training — latency budgets, user patience, tool-rate quotas all cap T at serving time. A policy trained at T = 32 and served at T = 8 can underperform a policy trained with budget randomization over [4, 32]. BudgetMix is the simplest possible remedy, costs nothing per step, and composes cleanly with any policy-gradient method (PPO, GRPO).

## Gotchas & tricks

- The distribution over T matters: pure uniform wastes samples on budgets that will never ship. A mixture that emphasizes the deployment budget while retaining a tail is usually best.
- If the reward structure gives *per-step* bonuses (e.g. a judge scoring intermediate turns), BudgetMix changes the reward scale — normalize per bucket.
- Pairs naturally with evaluation reported as the full Pass@(K,T) surface rather than a single cell — otherwise the budget robustness BudgetMix buys is invisible in the headline number.

## Sources

- Paper: *What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents?* — Zhu et al., 2026 — https://arxiv.org/abs/2611.06483
