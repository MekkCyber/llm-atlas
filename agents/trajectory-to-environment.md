# Trajectory-to-Environment
*Depth — reconstructing executable, verifiable agent environments from frozen tool-execution trajectories.*

**TL;DR:** A frozen agent trajectory is a demo; an environment is a re-queryable substrate that yields many verifiable tasks. Trajectory-to-environment replays the tool-execution history in an existing trajectory to reconstruct the workspace the agent ran in (restoring files before the agent edited them, filling in the missing files/deps via a completion agent), turning each trajectory into a reusable environment. Terminal-Universe (Qwen/Alibaba, 2026) uses this to generate 37.3k terminal environments and reports +11.9 pts on Terminal-Bench 2.1 via SFT.

**Prereqs:** [rlvr](../post-training/rlvr.md)
**Related:** [environment-evolution](environment-evolution.md), [_agent-training-environments](_agent-training-environments.md)

---

## What it is

An offline data-engine step. Input: a corpus of agent trajectories (bash / code-interpreter sessions, one per task) captured with full tool-call history. Output: a corpus of workspaces that can be booted, queried with new tasks, and verified by execution — one workspace per input trajectory, plus synthesized cross-workspace variants.

Distinct from **synthesized-from-scratch environments** (LLM-generates an env spec) and **co-evolved environments** (grown against a specific model's on-policy rollouts).

## How it works

Three stages:

1. **Workspace recovery.** Replay every file operation in the trajectory in reverse to restore each file to its pre-modification state. This yields a *partial* workspace — only the files the agent actually touched are known.
2. **Workspace completion.** A completion agent (an LLM given the trajectory's transcript and the partial workspace) fills in the missing files, dependencies, and directory scaffolding needed for the environment to boot and the original task to run.
3. **Task synthesis.**
   - *Breadth:* mine directional dependency edges between related workspaces and synthesize cross-workspace queries spanning multiple codebases (mimicking real developer flows).
   - *Depth:* extend the original single-turn query into a multi-round session with a user agent that iterates on feedback and requirement refinement.

Each recovered environment is verified by re-executing the reconstructed original task; only environments that pass this task-sufficiency check are kept.

## Why it matters

- **Environments are scarce; trajectories are cheap.** Public terminal-agent trajectories exist in bulk; realistic executable envs do not. This gap has bottlenecked agent RL and SFT.
- **Verifiable.** Recovered environments run real code, so downstream training gets execution-based rewards — the substrate RLVR needs.
- **Cross-workspace coverage.** Breadth mining reproduces the multi-repo reality of real developer tasks that single-trajectory synthesis misses.

## Gotchas & tricks

- **Task-sufficiency check is non-negotiable.** Without it you keep environments that superficially boot but do not actually verify the reconstructed task.
- **Completion agent can hallucinate.** Missing files invented by the completion agent may not match the trajectory's actual code paths; verify by re-execution.
- **Version drift.** Package versions from the trajectory may not resolve today. Pin an image / lockfile at recovery time.
- **Complements environment evolution.** Trajectory-to-env supplies the initial breadth of envs; off-policy environment evolution (see companion depth file) grows their difficulty over training generations.
- **Not just terminals.** The recipe generalizes to any tool-using agent whose tools have deterministic, replay-friendly state (browsers with recorded DOM, filesystem-backed IDEs).

## Sources

- Paper: *Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments* — Wu et al., Qwen / Alibaba, 2026 — https://arxiv.org/abs/2609.04148
- Related: *Environment Evolution for Terminal Agents* — Fan et al., 2026 — the off-policy difficulty-growth companion.
