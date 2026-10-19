# Windowed Interaction RL (WIRL)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For long-horizon agent tasks whose feedback comes in discrete observations (an experiment finishes, a test suite runs, a build completes), WIRL groups all agent decisions **between two consecutive observations** into a "window" and gives every decision inside that window a shared outcome-based reward. It is the natural credit-assignment granularity for ML-engineering agents and other scientific-loop agents — not per-token, not per-full-episode.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md), [rlvr.md](./rlvr.md)
**Related:** [multi-session-rl.md](./multi-session-rl.md), [_rewards.md](./_rewards.md)

---

## What it is

The MLExpert paper identifies three granularities for credit assignment in long-horizon RL:

- **Per-token.** Standard LM RL. Credit broadcast across all tokens of a response.
- **Per-episode.** One reward at the end of the whole trajectory. Appropriate for short tasks.
- **Per-window.** Reward granularity matches the environment's *observation granularity*. For an ML-engineering agent, each experiment run yields a numeric result; everything the agent decided between two experiments belongs to the same credit window.

WIRL formalizes the per-window choice: a *window* is the span of agent decisions and tool calls between observation $o_k$ and $o_{k+1}$; the shared reward is a function of the pair $(o_k, o_{k+1})$ (e.g., did validation metric improve? did the model train without erroring?).

---

## How it works

1. **Segment the trajectory** at observation events. Each window $W_k$ is the set of agent tokens / tool calls between $o_k$ and $o_{k+1}$.
2. **Compute per-window reward**:

$$
r_k = f(o_{k+1}) - f(o_k) \quad \text{(e.g., validation-score delta)}
$$

optionally clipped and shaped. Negative deltas discourage changes that regressed the result.

3. **Advantage.** Group-relative across the batch's windows at the same depth (same "experiment number"), echoing GRPO's baseline but inside the window axis:

$$
A_k = \frac{r_k - \bar{r}_{\text{depth }k}}{\sigma_{r,\text{depth }k}}
$$

4. **Broadcast** $A_k$ to every token inside window $W_k$ and apply standard PPO-clipped policy gradient.

---

## Why it matters

- **Matches the scientific-loop structure.** ML engineering, hyperparameter search, iterative code debugging — all look like "think, act, observe, repeat." WIRL puts credit exactly where the information arrives.
- **Avoids per-episode sparsity.** On tasks that span dozens of experiments, broadcasting one terminal reward across the whole trajectory is too sparse; WIRL turns the problem into a sequence of smaller, denser RL subproblems.
- **Reported impact.** Underpins MLExpert, a 35B MoE agent that lifts the MLE-Bench Lite medal rate from 37.9% → 72.7% over its base model and transfers to agent-scaffold and algorithm-optimization tasks.
- **Reusable beyond ML-engineering.** Any domain with discrete feedback events (CI runs for a code agent, battle outcomes for a game agent, A/B test rounds for a growth agent) fits the same pattern.

---

## Gotchas & tricks

- **Observation granularity must be meaningful.** Observations that don't carry evaluative signal (a dependency installed) give flat rewards; the paper assumes observations map to measurable deltas.
- **Window boundaries are not sessions.** WIRL slices on *environment* events; multi-session-rl slices on *LLM* events (compaction, forking). They compose: a trajectory can have both nested session boundaries and window boundaries.
- **Shared reward within a window broadcasts credit widely.** Fine when the window is short (few decisions); when the agent spends 50k tokens between experiments, you may want to combine WIRL with within-window length normalization.
- **Combine with hierarchical SFT.** MLExpert warms up with hierarchical SFT (plan → steps → actions) before WIRL — the authors argue the hierarchical imitation stage is what makes the window-level signal usable.

---

## Sources

- Paper: *MLExpert: Towards a Generalist Agentic Model for Autonomous Machine Learning Engineering* — Shi et al., Shanghai AI Lab, 2026 — arXiv [2610.22344](https://arxiv.org/abs/2610.22344).
- Baseline: *GRPO* — [grpo.md](./grpo.md).
