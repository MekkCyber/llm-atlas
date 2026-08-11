# Environment curriculum for agent training
*Depth — the Hierarchical Difficulty Curriculum (HDC) for multimodal agent RL.*

**TL;DR:** When training agents by RL over pools of environments, environment *count* alone plateaus. HDC organizes the environment pool as a curriculum with two orthogonal difficulty axes — **harness weakening** (reducing scaffolding assistance) and **state-scale progression** (growing the state-space the agent must navigate). Sequencing across both axes trains multimodal agents that scale beyond what naive environment-count scaling achieves.

**Prereqs:** [README.md](README.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [ability-aware-environment-selection.md](ability-aware-environment-selection.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

Modern agent RL trains a policy against a pool of environments (games, GUI worlds, tool-use sandboxes). The default has been "more environments = better agent"; empirically this saturates. HDC treats the environment pool the way pretraining treats the data mixture — as something to *curriculum-schedule*, not just enumerate.

## How it works

HDC organizes environments along two independent difficulty axes:

1. **Harness weakening.** How much the surrounding scaffolding does for the agent — pre-parsed observations, hint tools, action-menu constraints, high-level primitives. Early curriculum stages leave heavy scaffolding in; later stages strip it out so the agent must handle raw observations and low-level actions.
2. **State-scale progression.** How large the accessible state space is — number of screens / rooms / tabs, tool count, horizon length. Early stages restrict; later stages expand.

Training schedules move the environment sampling distribution along both axes over the RL run. The two axes are treated as *independent knobs* so you can, e.g., strip the harness while keeping state small, or expand state under a permissive harness.

## Why it matters

- **Environment-count scaling saturates.** More of the same difficulty class stops helping — the same conclusion Chinchilla drew for data.
- **Two-axis difficulty is more general than reward shaping.** Doesn't require reward-function surgery; the environment itself gets harder.
- **Composable with diversity selection.** Pairs naturally with [ability-aware-environment-selection.md](ability-aware-environment-selection.md), which addresses the orthogonal question of *which* environments to include.

## Gotchas & tricks

- **Harness weakening can be destabilizing.** Removing scaffolding too fast can collapse success rate to zero — curriculum step size matters.
- **State-scale progression can blow up context.** Large-state environments raise trajectory length and thus RL-step cost.
- **Not the same as PPO/GRPO curriculum on prompts.** Prompt curricula (like [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)) sequence *what problems* the agent sees; HDC sequences *what environment* it acts in.
- **Requires environment authorship.** You need environments that can be parameterized along both axes — off-the-shelf benchmarks usually can't.

## Sources

- Paper: *Beyond Simply Environment Scaling: Designing Effective Environment Distributions for Multimodal Agent Learning* — Zhu et al., CAS/UCAS, 2026 — arXiv:2608.03571.
