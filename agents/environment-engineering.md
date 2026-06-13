# Environment Engineering

*Depth — designing the resources, constraints, and interfaces around an autonomous agent, rather than designing the agent's workflow itself.*

**TL;DR:** As LLM agents become strong enough to solve novel scientific tasks, the bottleneck shifts from "what prompt / workflow makes the agent do the right thing" to "what environment makes the agent's natural behavior productive". Environment engineering is the explicit design of four axes: **permissions** (what the agent can touch), **artifacts** (filesystem + version-controlled state), **budget** (compute / API spend caps), and **human-in-the-loop hooks** (cheap supervision and intervention). EurekAgent uses this template to set SOTA on math, kernel engineering, and ML tasks, including a new 26-circle packing result discovered for under $11 in API cost.

**Prereqs:** [agents/README](README.md)
**Related:** [code-as-action](code-as-action.md)

---

## What it is

A reframing of agentic system design: keep the agent loop conventional (model emits action, harness executes, result feeds back), and put the engineering effort into the *environment* that surrounds the loop. The environment shapes which behaviors are easy, which are hard, and which are impossible. Done well it amplifies productive behavior (open-ended exploration, systematic artifact management, multi-agent collaboration) and suppresses destructive behavior (reward hacking, runaway cost, hard-to-audit actions).

This is a counterpoint to the workflow-engineering trend — instead of orchestrating ever-more-elaborate planner / executor / critic graphs, invest in the substrate. The agent itself stays simple.

## How it works

Four axes:

1. **Permissions engineering.** Each step runs under a bounded permission set: read-only mounts for inputs, scratch directories for outputs, network egress disabled by default. Evaluation runs in isolated containers so the agent can't peek at held-out scores or modify the judge. The agent doesn't need to be "trustworthy" — it just can't reach the things that would be costly to corrupt.
2. **Artifact engineering.** The environment exposes a filesystem (often Git-backed) as the durable substrate for cross-step and cross-agent collaboration. Multi-agent workflows coordinate via commits, branches, and merges instead of natural-language handoffs. State is inspectable, diffable, and rewindable.
3. **Budget engineering.** Each task has explicit token / API / wall-clock budgets, and the agent is told what's left. This is what turns "open-ended exploration" into "budget-aware exploration" — the agent learns to choose between cheap probes and expensive deep dives.
4. **Human-in-the-loop engineering.** Lightweight intervention hooks (e.g., a "pause" file the human can drop into the workspace, a structured suggestion channel) make supervision cheap. The agent treats human input as just another tool result, so HITL doesn't break the autonomous loop.

The combination matters more than any one axis: permissions without budget invites runaway cost; artifacts without permissions invites corruption; HITL without artifacts has nothing to point at.

## Why it matters

- **Empirical**: SOTA on multiple math, kernel-engineering, and ML benchmarks from EurekAgent, including a new 26-circle packing result for <$11. That's both a quality result and a cost-effectiveness result.
- **Generalizable template**: the four axes are claim-by-claim independent of the underlying agent or task. Any agentic system — code, scientific discovery, computer use — can be analyzed along them.
- **Aligns with capability scaling**: as models get stronger, "tell the agent what to do" stops being the bottleneck. The next-mile work is what the agent can *act on*, which is exactly what environment engineering controls.

## Gotchas & tricks

- **Budget visibility is necessary.** Hiding the remaining budget produces either over-conservative or over-aggressive exploration. Make it part of the prompt or tool output.
- **Git-backed artifacts unlock surprisingly much.** Diffs, blames, and branches give the agent (and the human reviewer) cheap structural inspection of state evolution.
- **Permissions are the cheapest safety lever.** Locking down by default and unlocking only what the task needs is much easier than catching the agent doing something it shouldn't.
- **HITL hooks should be optional from the agent's side.** If the agent has to actively request human input, supervision becomes a bottleneck; if the human can intervene by leaving a note in the workspace, supervision parallelizes.

## Sources

- Paper: EurekAgent — Xin et al. (2026) — [arXiv:2606.13662](https://arxiv.org/abs/2606.13662)
