# Environment synthesis from agent trajectories
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Training verifiable coding agents needs *executable environments* (repos, workspaces), not just logged trajectories. **Terminal-Universe** (Wu et al., 2026) reconstructs the environment each trajectory ran in by *replaying its file-operation history in reverse*: every logged tool call reveals the workspace state before that step. A completion agent then fills in unknown files and dependencies. From the recovered workspace, both the original task and new tasks can be synthesized. Applied to public terminal-agent trajectories, this produces 37.3k task-sufficient environments and lifts Terminal-Bench 2.1 by +11.9 pts after SFT.

**Prereqs:** [_post-training](../post-training/_post-training.md)
**Related:** [environment-curriculum](environment-curriculum.md), [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

Two very different things get called "training data" for agents:

- **Trajectories.** Frozen tool-call logs. Good for SFT/behavior cloning; each is a single frozen demonstration.
- **Environments.** Runnable workspaces the agent can act on and get execution feedback from. Good for RL (verifiable rewards) and multi-round synthesis.

Trajectory data is abundant (every dogfood log); environments are scarce (they're hard to synthesize from scratch, and companies don't publish their internal repos). Environment synthesis via trajectory inversion closes that gap by extracting environments *from the trajectories that used them*.

## How it works

Three phases:

### 1. Trajectory-to-partial-workspace via file-op replay

The tool-call history of a terminal trajectory logs `read`, `write`, `edit`, `mkdir`, `delete`, `mv`, `cp`, `chmod`. Each of these can be *inverted* to say what the workspace looked like immediately before that step. Replaying an entire trajectory in reverse produces a **partial** workspace — populated for every path the agent touched, unknown for paths it didn't.

### 2. Completion agent fills the gaps

A separate LLM agent inspects the partial workspace and synthesizes plausible content for unreferenced files (config files, dependencies, README fragments, source files the trajectory only read a snippet from). The result is a **runnable** workspace: `pip install` succeeds, tests can be discovered, the entrypoint the trajectory used exists.

### 3. Task synthesis over the recovered workspace

Two directions:

- **Reconstruct the original intent task.** Frame the workspace + trajectory outcome as a task the agent must reach.
- **Synthesize new tasks.** Along two axes:
  - *Breadth*: mine dependency relations between related workspaces and synthesize *cross-workspace* queries (e.g. "add a feature in repo A that consumes an interface from repo B").
  - *Depth*: extend the initial single-turn query into a **multi-round session** where a user agent supplies iterative feedback and requirement refinement.

Each recovered environment thus produces many verifiable tasks with execution feedback.

## Why it matters

**Environments become recoverable output** of the pipeline you already run. Every public trajectory dump — SWE-agent traces, Terminal-Bench logs, dogfood corpora — becomes upstream training data instead of a frozen artifact. That reframes the bottleneck: environment scarcity was a *choice* imposed by the way trajectory logging discards workspace context.

The Qwen3.5-27B result — **+11.9 pts single-round on Terminal-Bench 2.1**, **+13.8 pts multi-round on EvoCode-Bench v2** — is a lower bound on what SFT alone extracts; combined with RL on the same environments the ceiling is higher.

## Gotchas & tricks

- **Ambient state.** File-op replay misses environment state that wasn't touched by the trajectory (network config, installed packages, git origin). The completion agent has to invent these plausibly; some recovered environments will fail to build.
- **Cross-workspace synthesis assumes clean interfaces.** If the recovered dependency between repos requires shared runtime state, the synthesized cross-workspace query may be unsolvable.
- **Trajectory quality shapes environment quality.** A trajectory that stumbled around blindly reveals less structure than a targeted one; curate trajectories for coverage, not just count.
- **Verify before training on it.** A synthesized environment that doesn't run under the agent's tool loop is worse than no environment — it produces training signal for tools that never resolve.

## Sources

- Paper: *Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments* — Wu, Zhang, Zhang, Wang, Su, Chen, Wang, Wang, Shen, Zhou, Yang, Huang, Yang, Liu, 2026 — [arXiv:2609.04148](https://arxiv.org/abs/2609.04148).
