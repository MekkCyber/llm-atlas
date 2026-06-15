# Agent Environment Engineering

*Depth — shaping LLM agent behavior by designing the execution environment (permissions, artifacts, budgets, HITL) rather than the agent workflow.*

**TL;DR:** As LLM agents become more capable, the bottleneck for autonomous discovery shifts from *workflow design* (which prompts, which tool sequence, which planner pattern) to *environment design* — the permissions, file system, budgets, and human-in-the-loop hooks that shape what the agent can or can't do. **EurekAgent** operationalizes this along four axes and sets SOTA on math, kernel engineering, and ML research tasks — including a new 26-circle packing result discovered for **<$11** in API cost. The thesis: environment design *amplifies* productive behaviors (exploration, artifact management, inter-agent collaboration) and *suppresses* harmful ones (reward hacking, high-friction oversight).

**Prereqs:** [README.md](README.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../systems/ray.md](../systems/ray.md), [../safety/safety-case.md](../safety/safety-case.md)

---

## What it is

A design discipline for autonomous-agent systems: instead of tuning the agent's prompt / workflow / loop structure, tune the *environment* it runs in. The environment is treated as a first-class research artifact — what permissions, what artifact store, what budget enforcement, what human escalation hooks.

## How it works

EurekAgent identifies four pillars:

### 1. Permissions engineering

Bounded execution sandboxes and isolated evaluation harnesses. The agent runs in a constrained environment with clearly-scoped capabilities — file write to a sandbox, no network to production, isolated eval to prevent train/test contamination. This makes reward hacking detectable (and often impossible) because the agent can't reach the things that would let it cheat.

### 2. Artifact engineering

Filesystem + Git-based collaboration for multi-agent systems. Sub-agents write artifacts (code, hypotheses, partial results) to a shared filesystem; coordination is via real version-control primitives (branches, diffs, merges) rather than via in-context message passing. This scales further than pure prompt-based handoff and gives the discovery process an auditable history.

### 3. Budget engineering

Budget-aware exploration. The agent is given an explicit cost ceiling (API tokens, GPU-hours, wall-clock) and learns to allocate compute across hypotheses rather than running unbounded loops. Budget pressure forces decisions; without it, the agent will spend arbitrarily long on dead ends.

### 4. Human-in-the-loop engineering

Cheap human supervision and intervention. The environment exposes ergonomic hooks for humans to inspect intermediate state, approve risky actions, or redirect the search — without halting the agent or requiring a context dump. Friction in HITL is what makes oversight costly; engineering it down makes more supervision viable.

## Why it matters

- **Shifts the research frontier.** If most agent failures are environment-friction failures (permission tangles, lost artifacts, runaway costs, expensive oversight), workflow tuning is the wrong axis. Environment engineering is where marginal effort pays off.
- **Discovery becomes a cost question.** The 26-circle-packing result wasn't blocked by capability — it was blocked by cost-effective autonomy. Environment engineering directly attacks the cost axis.
- **Reward hacking is structurally suppressed.** Sandboxed permissions remove the *capability* to reward-hack rather than relying on the agent to choose not to.
- **Auditable by construction.** Git-backed artifact stores make every step of the agent's reasoning replayable; in-context-only agents do not.

## Gotchas & tricks

- **Permissions must be tight but reachable.** Too loose and reward hacking re-emerges; too tight and the agent can't do useful work. The sweet spot is task-shaped permissions, not blanket restrictions.
- **Budget pressure ≠ time-out.** A hard wall-clock limit isn't budget engineering — the agent needs *signal* about its budget state to allocate compute across hypotheses, not just to be killed.
- **HITL hooks degrade if rarely used.** If humans never look at intermediate artifacts, the environment becomes effectively un-supervised regardless of how easy intervention is. Practice the oversight loop.
- **Artifact merging is its own design problem.** Multi-agent Git workflows hit the same merge-conflict pain humans do — needs explicit conflict-resolution policies.
- **Environment is part of the safety case.** A safety case ([safety-case.md](../safety/safety-case.md)) for an autonomous agent should describe the environment as carefully as it describes the model — they are inseparable.

## Sources

- Paper: *Agent Environment Engineering is All You Need For Autonomous Scientific Discovery (EurekAgent)* — Xin et al., Tsinghua / Zhipu AI, 2026 — [arXiv:2606.13662](https://arxiv.org/abs/2606.13662).
- Related: [README.md](README.md), [../systems/ray.md](../systems/ray.md).
