# Multi-Session Agentic RL (Mosaic)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Most agentic-RL algorithms assume one LLM session per trajectory. But frontier agents compact context, swap models, or fork subagents — producing trajectories that span **multiple sessions**. Multi-session RL (introduced as *Mosaic*) is a critic-free formulation that estimates advantages at both the trajectory and session level, so credit flows across session boundaries without naive broadcasting of the final reward.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [rlvr.md](./rlvr.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

An agentic trajectory is modeled as an ordered sequence of **sessions**, each a self-contained LLM invocation with its own context window. Sessions are produced by:

- **Context compaction** — when the window fills, summarize + start a fresh session carrying the summary.
- **Subagent forking** — the main agent spawns child sessions for specific subtasks and collects their outputs.
- **Model switching** — different sessions may call different models (planner vs executor).

The RL problem: given a terminal reward on the whole trajectory, how do we assign credit to tokens inside each session?

---

## How it works

Mosaic uses a **two-level, critic-free** advantage estimator.

### 1. Trajectory-level advantage (across trajectories)

For a batch of $G$ trajectories per task, use the standard group-relative baseline:

$$
A_i^{\text{traj}} = \frac{R_i - \bar{R}}{\sigma_R}
$$

where $R_i$ is the terminal reward and $\bar{R}, \sigma_R$ are the group's mean and std. This is GRPO.

### 2. Session-level advantage shaping (within a trajectory)

Within each trajectory, assign each session $s$ a shaped advantage

$$
A^{\text{sess}}_s = A^{\text{traj}}_i \cdot w_s
$$

where $w_s$ is a session-level weight reflecting how much that session contributed to the final outcome. Candidate weights the paper evaluates:

- **Uniform** (baseline — equivalent to broadcasting).
- **Position-based** (later sessions see more of the outcome and get more credit).
- **Group-relative within-trajectory** (compare each session's intermediate signal to a within-trajectory baseline).

Group-relative within-trajectory weighting is reported strongest.

### 3. Standard PPO-clipped update

Each session's tokens get $A^{\text{sess}}_s$ as their advantage; PPO clip and KL-to-reference are applied as usual.

---

## Why it matters

- **Matches modern agent reality.** Context compaction and subagent forking are already standard in long-horizon agents; "one session = one trajectory" never held in production.
- **Beats broadcasting.** On BrowseComp with Qwen3.5-4B, Mosaic wins in both context-compaction and subagent-forking configurations and uses fewer sessions per trajectory — i.e., learns to avoid unnecessary compaction / forking.
- **Critic-free.** Keeps GRPO's main practical virtue (no value network) while adding the session dimension — a small addition on top of a GRPO rollout pipeline.

---

## Gotchas & tricks

- **Session boundary definition matters.** If sessions are too fine (every tool call) the baseline becomes noisy; too coarse and shaping degenerates to broadcasting. The paper uses natural compaction / fork events.
- **Rollout infra needs to track session trees.** Subagent forking produces a DAG, not a linear sequence. Advantage propagation must respect dependencies.
- **Compose with RLVR verifiers.** Terminal reward is still a scalar from the task verifier; the multi-session trick is about *distributing* it, not sourcing it.
- **Won't help pure single-session tasks.** Short tool-calling (one session end to end) sees no gain; the method targets the long-horizon regime.

---

## Sources

- Paper: *Mosaic: Multi-Session Agentic Reinforcement Learning* — Xu, Zhou, Long, Tan, Guo, 2026 — arXiv [2610.22174](https://arxiv.org/abs/2610.22174).
- Baseline: *GRPO* — [grpo.md](./grpo.md).
