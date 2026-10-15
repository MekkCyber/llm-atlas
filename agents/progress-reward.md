# Progress Reward (Tev PPRS)
*Depth — dense per-step rewards for long-horizon agent RL, read from a hidden latent-world state.*

**TL;DR:** Long-horizon agent RL historically breaks on **sparse terminal rewards**: a trajectory of 50–200 tool calls returns one scalar at the end, giving almost no credit-assignment signal. **PPRS** (Per-step Progress Reward Signal), introduced with Tev-Gym, fixes this by *generating* each environment with a hidden ground-truth "latent world" that the reward model can read, letting it score **progress toward the goal at every step**, not just at termination. The dense signal unlocks ordinary GRPO/PPO on tasks previously treated as sparse-reward.

**Prereqs:** [../post-training/_rewards.md](../post-training/_rewards.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md) · [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md) · [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md) · [reflective-agent-training.md](./reflective-agent-training.md)

---

## What it is

An alternative to **outcome-only** rewards for agent RL. Instead of `r = 1` on success / `r = 0` otherwise at the end of the trajectory, PPRS emits a scalar at each step measuring how much that step moved the agent toward the goal. The signal is *verifiable* in the same sense RLVR is: it is computed by a rule-based scorer against a hidden ground-truth state, not by a learned reward model.

## How it works

**Environment construction.** Each Tev-Gym environment is sampled from a hidden *latent world*: the generator produces (i) an observable environment the agent interacts with, and (ii) a hidden state that encodes the goal structure (sub-task graph, required artifacts, acceptance predicates).

**Per-step scoring.** After each agent action `a_t` and environment response, PPRS reads the hidden state and emits:

```
r_t = w · Δgoal_state(t) − c · step_cost
```

where `Δgoal_state` is the delta in a goal-progress function the generator defines (fraction of sub-goals satisfied, distance in the sub-task DAG, etc.). Progress is monotone-non-decreasing in principle, but *regressions* (undoing a sub-goal) get negative `Δgoal_state`.

**Training.** Standard GRPO / PPO over multi-step trajectories, with the per-step reward replacing the usual terminal reward. Because the signal is dense, the group-normalized advantage carries real information at every token.

## Why it matters

- Breaks the long-horizon RL bottleneck: credit assignment across 50–200 tool calls no longer depends on a terminal binary signal.
- Keeps the RLVR property — verifier is rule-based, not a learned RM, so it can't be reward-hacked by surface features.
- Scales environment supply: the latent-world generator decouples environment synthesis from human authoring.
- Reported result: Tev-27B trained with PPRS matches GPT-6 Astra and Claude Opus 5 at Pass@5 on **ALE-CM** and generalizes to **Terminal-Bench** subsets.

## Gotchas & tricks

- **Goal-progress function design is the whole game.** A reward that saturates too fast lets the agent farm partial credit without finishing. A reward that gates most of its mass at termination recreates the sparse-reward problem it was supposed to fix.
- **Step cost `c` matters.** Without it, the agent is incentivized to take infinitely many micro-progress steps.
- **Only works when you can *build* the environment.** Benchmarks collected from humans (not generated) don't have a latent-world oracle; you'd need a reward model trained to approximate progress, which reintroduces reward hacking.
- **Regression penalties can destabilize.** Negative `Δgoal_state` on undo actions can make the agent risk-averse and refuse exploratory moves; clip the negative side if seen.

## Sources

- Paper: *Tev: Scaling Long-Horizon Agentic Environments with Dense Progress Rewards* — Hu, Singh, Peng, Chang, Lee — Adobe Research / UCLA, 2026 — arXiv:2610.18868.
