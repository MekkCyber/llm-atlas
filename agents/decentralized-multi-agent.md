# Decentralized Multi-Agent Harness
*Depth — an orchestrator-free multi-agent loop scaling to 1,024 concurrent workers.*

**TL;DR:** A multi-agent harness where **no central orchestrator** exists — instead, concurrent workers execute a shared cooperation loop against three durable surfaces (a workspace of proposed/ongoing/completed work, a message interface, and a shared-context store). Each worker gathers context, self-assigns sub-tasks, acts, shares findings, verifies peers, and merges progress asynchronously. Introduced by Agensh (Microsoft Research, 2026), which scales cleanly from 1 to 1,024 agents and shows *number-of-agents* as a new scaling axis.

**Prereqs:** [README.md](./README.md), [../systems/README.md](../systems/README.md)
**Related:** [../systems/ray.md](../systems/ray.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [recursive-self-improvement.md](./recursive-self-improvement.md)

---

## What it is

A multi-agent orchestration pattern that removes the central dispatcher. Instead of "orchestrator hands out sub-tasks to worker pool", every worker runs the same cooperation loop and coordinates via three shared artifacts. The whole system is asynchronous, self-organizing, and — critically — has no orchestrator bottleneck to cap scaling.

Contrast with the standard multi-agent harness (AutoGen-style, LangGraph-style, MetaGPT-style, and most agent frameworks), which puts a single planner / router at the top and workers underneath. In those, the router's context-window and per-step latency cap effective parallelism at a few dozen workers.

## How it works

Every worker runs the same loop:

1. **Gather context** from the shared-context store and workspace.
2. **Claim a sub-task** from the workspace (or self-assign a new one when a gap is identified). Claim writes into the workspace so peers see it.
3. **Take action** — call tools, generate outputs, decompose further if needed.
4. **Share findings** into the shared-context store so future workers reuse them.
5. **Verify** peer work — pick claimed / completed items owned by other workers and cross-check outputs; disagreements are surfaced via the message interface.
6. **Merge** — when several workers have converged on a sub-task, one merges progress back into the workspace as completed.

The three shared surfaces are the whole coordination substrate: no scheduler decides who does what, no planner rewrites the plan. Emergent patterns (specialist workers, division of labor, standardized verification protocols) *appear as the org grows*, per Agensh's trajectory analysis.

## Why it matters

- **Number-of-agents is a scaling axis.** Agensh's ProgramBench results: mean pass rate 19.31% (1 worker) → 28.78% (128 workers, +49% relative); on pandoc 33.89% (1) → 55.06% (1,024). Larger orgs also reach a given pass rate earlier.
- **Removes the orchestrator bottleneck.** Central orchestrators cap out around 10s of concurrent workers; decentralized designs are limited only by the shared-workspace substrate.
- **Systems-first framing of multi-agent.** The interesting engineering surface moves from "prompt the orchestrator" to "design the shared workspace and message interface" — distributed-systems territory.

## Gotchas & tricks

- **Shared workspace ≈ Google Docs semantics.** All concurrency and consistency problems that plague collaborative editing (write conflicts, ordering, stale reads) apply. Agensh's paper doesn't give the substrate details; expect real-world implementations to need CRDT-like or transactional guarantees.
- **Verification is the load-bearing peer mechanism.** Without peer verification, workers drift and duplicate. Design incentives that reward verifying others, not just claiming new work.
- **Cost scales linearly with agents.** No free lunch — you pay $N$× the inference bill. Gains on top of a strong solo run are ~20–60% not 10×.
- **Convergence detection is unclear.** Deciding when a sub-task is "done" without a central controller is nontrivial; Agensh uses per-task test-pass signals, but for tasks without a crisp check this is an open problem.
- **Emergent-cooperation claims deserve inspection.** Trajectory logs *look* like cooperation but include a lot of redundant work. Measure task-throughput per token, not just aggregate pass rate.

## Sources

- Paper: *Agensh: Scaling Organizational Intelligence to 1,024 Agents* — Zhan, Song, Dong, Huang, Lian, Xia, Wei (Microsoft Research), 2026 — [arXiv:2609.26781](https://arxiv.org/abs/2609.26781).
