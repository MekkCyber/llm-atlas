# LLM-as-Environment-Engineer

*Depth — self-improving RL where the policy redesigns its own training environment between rollout phases.*

**TL;DR:** Curriculum design is a human bottleneck for agentic RL — picking which environments to train on, in what order, with what difficulty. **LLM-as-Environment-Engineer** turns this into a reasoning task the policy itself handles: after each rollout phase, the model analyzes its own failure trajectories, emits structured edits to the environment config (difficulty, agent count, layout), the env is rebuilt, and RL continues. On the MAPF-FrozenLake multi-agent benchmark, a Qwen3-4B trained this way outperforms much larger proprietary baselines and a fixed-environment baseline.

**Prereqs:** [grpo](grpo.md), [rl-prompt-curation](rl-prompt-curation.md)
**Related:** [rlvr](rlvr.md), [_rl](_rl.md)

---

## What it is

A close relative of automatic curriculum learning, but the curriculum designer **is the policy being trained**, not an outer-loop heuristic. The same model that's learning to solve tasks is the model that's redesigning what tasks to learn from.

## How it works

**Outer loop** (per training phase):

1. **Rollout.** Run current policy on the current env config; collect trajectories, mark failures.
2. **Failure analysis.** Prompt the policy with batches of failure trajectories + the current env config. Ask: *"What about this env makes the failures happen? What config changes would produce more learnable failures?"*
3. **Env edit.** Policy outputs structured edits to the env config (e.g. JSON patches: increase difficulty, change agent layout, adjust reward density). A verifier checks edits are well-formed and bounded.
4. **Env rebuild.** The simulator instantiates the new env.
5. **RL step.** Standard GRPO (or other policy-gradient) on the new env. Repeat.

**Structured edits** are key: free-form text would be unreliable. The action space for env edits is a fixed schema (which knobs can change, what ranges) that the policy fills in.

**Why it works.** When the model itself identifies what's blocking progress, the curriculum naturally targets the model's actual learning frontier — not a human-guessed difficulty schedule. The policy effectively self-supervises its own training distribution.

## Why it matters

- **Curriculum design becomes self-driving.** No more hand-tuned env-difficulty schedules; the policy points at its own frontier.
- **Beats much larger proprietary models.** On MAPF-FrozenLake, a 4B policy with self-designed envs outperforms baselines orders of magnitude larger trained on fixed envs. Curriculum efficiency > raw scale, for this domain.
- **Generalizes the "data curation by the policy" pattern** that R-Zero, R1-style RL, and rejection sampling have been using on the data side — now applied to environments.
- **Composable with agentic RL stacks.** Drops in next to any policy-gradient trainer; just need an env that exposes a structured config knob.

## Gotchas & tricks

- **Verifier on edits is non-negotiable.** A policy that emits malformed edits will brick the env. The structured-schema check is the floor.
- **Edit-space bounds.** Without bounds the policy can edit itself into trivial envs that maximize reward but teach nothing. Bound the action space and gate progress on actual difficulty increase.
- **Cold start.** Early in training, the policy can't analyze its own failures usefully. Use a fixed env or a human curriculum for the first few thousand steps, then hand over to the policy.
- **Reward-hacking risk.** If the env edits feed back into the reward function, the policy can hack — edit the env to make any output "correct." Keep the reward function fixed; only let the policy change the *task distribution*.
- **MAPF-FrozenLake is a clean lab benchmark.** Generalization to richer envs (real code execution, browser agents) is open.

## Sources

- Paper: *From Trainee to Trainer: LLM-Designed Training Environment for RL with Multi-Agent Reasoning* — Chen, Li, Li, Liu, Guo, LARK Lab @ HKUST (GZ), 2026 — [arXiv:2606.17682](https://arxiv.org/abs/2606.17682).
