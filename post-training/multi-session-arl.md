# Multi-Session Agentic RL (Mosaic)
*Depth — two-level credit assignment for trajectories that span multiple sessions.*

**TL;DR:** Most agentic RL methods assume one continuous session per trajectory, but production agents already compact context and spawn subagents, breaking that assumption. Mosaic is a critic-free method for multi-session agentic RL that assigns advantage at two levels: at the trajectory level, it conditions advantage on session count given the outcome; at the session level, it credits each session by its estimated marginal contribution to the final reward.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md), [partial-rollouts.md](../systems/partial-rollouts.md)
**Related:** [rlvr.md](rlvr.md), [ppo.md](ppo.md)

---

## What it is

A fix for the training/inference mismatch created by context-management techniques. Inference-time compaction (fold older context into summary) and subagent forking (hand a sub-task to a child agent) produce trajectories with discontinuous sessions; naive RL either flattens them (losing structure) or re-samples the whole trajectory per session (wasting rollouts).

## How it works

For a multi-session trajectory `τ = {S_1, …, S_N}` with final outcome reward `R(τ)`:

1. **Trajectory-level advantage.** Compute group-relative advantage conditioning on `N`: compare `R(τ)` to the mean reward of trajectories with the *same session count* `N`. Prevents the policy from gaming session count.
2. **Session-level advantage.** Estimate each session `S_i`'s marginal contribution by sampling counterfactual completions that skip or truncate `S_i` (practical approximations: leave-one-out from a batch of parallel rollouts).
3. Combine: per-token advantage inside `S_i` = trajectory-level advantage × session-level weight.
4. Apply standard critic-free (GRPO-style) policy update.

## Why it matters

Agent harnesses increasingly route around context limits with compaction and spawning. Training signal that collapses these to a single trajectory is wasteful, and training signal that treats each session as independent breaks long-horizon credit. Two-level credit is the straightforward middle path.

## Gotchas & tricks

- Needs enough parallel trajectories per prompt to estimate the session-count conditional baseline reliably.
- Subagent sessions have *their own* internal credit; Mosaic as described operates at the parent-agent level — child credit propagates via the parent's session marginal.
- Reported wins on BrowseComp with Qwen3.5-4B, both compaction and subagent regimes.

## Sources

- Paper: *Mosaic: Multi-Session Agentic Reinforcement Learning* — Xu et al., HKUST(GZ) / Huawei / HKUST, 2026 — [arXiv:2610.22174](https://arxiv.org/abs/2610.22174)
