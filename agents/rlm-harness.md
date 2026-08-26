# Recursive Language Model (RLM) harness
*Depth — a persistent-state harness for long-horizon agents with recursive subagents, from Prime Agent.*

**TL;DR:** LLMs are sequential processors; long-horizon agency requires state, tools, and computation that live outside the model's active context. An RLM harness makes that plumbing first-class: a persistent IPython REPL as the programmable context surface, continual carry-over of histories/memories/skills across trajectories, and recursive subagents that speak agent-to-agent through the same REPL. The pitch is that harness quality has been silently capping model capability, and a well-designed membrane exposes what the model can actually do.

**Prereqs:** [../systems/partial-rollouts](../systems/partial-rollouts.md)
**Related:** [../post-training/rlvr](../post-training/rlvr.md)

---

## What it is

An open-source harness pattern — realized by Prime Agent — for running language models as long-horizon agents. The harness owns *execution, state, recovery, verification, and resource accounting*. It does not own *strategy* — decomposition, delegation, tool choice all stay with the model.

## How it works

- **Persistent IPython REPL** as the substrate. Every tool call, subagent invocation, and memory read/write happens through Python. The REPL persists across an agent turn *and* across trajectories.
- **Continual Harness.** State — histories, memories, skills, prompts, subagent specifications — is carried across trajectory boundaries. A trajectory that ends is not a reset; the next trajectory inherits everything the harness has learned to preserve.
- **Recursive subagents.** Subagents are first-class citizens with their own specs, prompts, and REPLs. They communicate **agent-to-agent** rather than always routing through the top-level model. A daemon-backed session model means subagents outlive individual calls.
- **Agents View.** A UI surface lets humans inspect and manage the daemon-backed sessions — pause, resume, edit state.
- **Standardized accounting.** Compute, tokens, tool calls, and errors are tracked uniformly so harness failures don't get mis-attributed to model failures.

## Why it matters

Most agent benchmarks conflate model quality and harness quality. Prime Agent's ARC-AGI-3 RHAE Best@1 result — 30% → 95.5% on the *same underlying model* — is the strongest single data point that harness ceilings have been the bottleneck. An open harness good enough to expose the true model ceiling turns model comparisons from noisy to legible.

## Gotchas & tricks

- The REPL as substrate is only as safe as its sandbox. Persistent state across trajectories means state leakage between users is a real risk if you skip isolation.
- Recursive subagent-to-subagent chatter can blow up token budgets; enforce a per-trajectory ceiling and a recursion depth cap.
- Continual state grows unboundedly — periodic *skill compression* (merge similar memories, prune unused skills) is required at scale.
- Not a training scheme by itself; pairs naturally with agent RL where trajectories become training data.

## Sources

- Paper: *Prime Agent: A Self-Improving RLM Harness* — Karten et al., 2026 (Prime Intellect) — [arXiv:2608.23552](https://arxiv.org/abs/2608.23552)
- Code: https://github.com/PrimeIntellect-ai/prime-agent
