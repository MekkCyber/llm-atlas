# Agent Evaluation Infrastructure
*Depth — decoupling benchmark, harness, and environment so agent evaluations are composable, reproducible, and diagnosable.*

**TL;DR:** Agent evaluations have been fragmented — every group ships its own bundle of benchmark + harness + environment, so cross-paper results are hard to compare and adding a new benchmark or harness means rewriting execution logic. **AgentCompass** (Shanghai AI Lab, 2026) decomposes evaluation into three independent axes — **Benchmark** (the task), **Harness** (the agent scaffold: prompts, tools, control loop), **Environment** (the runtime that executes tool calls) — with a fault-tolerant asynchronous runtime and trajectory-analysis tools that catch nuanced failure modes like reward hacking. Ships 20+ benchmarks across five capability dimensions.

**Prereqs:** [../agents/agent-harness.md](../agents/agent-harness.md)
**Related:** [../agents/failure-attribution.md](../agents/failure-attribution.md)

---

## What it is

Agent evaluation infrastructure is the plumbing that runs agents against benchmarks reproducibly. The core insight in AgentCompass is that three axes must be independently swappable:

- **Benchmark** — the task definition, its inputs, its scoring function, its ground truth.
- **Harness** — the agent scaffold: prompt template, tool schemas, memory policy, control loop.
- **Environment** — the runtime that receives tool calls and returns observations (browsers, shells, MCP servers, simulators).

Historically these three have been fused: WebArena bundles its own harness with its own environment; SWE-bench evaluators embed a specific harness. The result is that "score on benchmark X" reflects a specific (benchmark, harness, environment) triple, and swapping any axis means rebuilding the eval.

## How it works

1. **Interface-first decomposition.** Each axis declares a small interface. Benchmarks emit tasks and score outcomes; harnesses expose a `step(observation) -> action` loop; environments expose `execute(action) -> observation`.
2. **Async fault-tolerant runtime.** Sweeps of `(benchmark × harness × environment × seed)` combinations run in parallel. When an individual run fails (agent crashes, environment times out, model returns garbage), the runtime isolates it and moves on rather than blocking the sweep.
3. **Native support for 20+ benchmarks across five capability dimensions.** Coverage of common agent tasks so a new harness can be measured immediately, and coverage of common harness patterns so a new benchmark plugs in without custom glue.
4. **Trajectory-analysis tools.** Post-run diagnostics that surface pathologies the final score would hide — reward hacking, tool misuse, infinite loops, silent history-truncation failures.
5. **Composability guarantee.** New benchmark = one plug. New harness = one plug. New environment = one plug. No axis-crossing rewrites.

## Why it matters

- **Cross-paper comparability.** When two papers can be run on the same (benchmark, environment) with different harnesses, their harness contribution becomes measurable in isolation. This is the missing rigor the agent literature has been paying for.
- **Cheaper new-benchmark launches.** A new benchmark can be evaluated against every existing harness on day one, rather than needing a bespoke evaluation harness first.
- **Diagnostics catch reward hacking early.** Trajectory analysis surfaces agents that game the metric without solving the task — a chronic issue in RL-trained agents that only reads on a metric line.
- **Reproducibility floor.** A shared open-source infrastructure lets researchers publish `(benchmark, harness, environment, seeds)` and expect others can rerun it.

## Gotchas & tricks

- **The three-axis decomposition doesn't hold everywhere.** Some benchmarks bake harness assumptions into their scoring (e.g., turn-count limits). Recover these as environment configs where possible; document the coupling where not.
- **Async runtime hides silent failures.** A run that "completed" with an empty output can be mistaken for a legitimate zero. Log terminal state per run and audit the distribution of terminal states, not just the score.
- **Model API rate limits become the bottleneck** for large sweeps. The runtime's fault tolerance should include model-side back-pressure, not just tool-side.
- **Trajectory diagnostics need calibration.** "Reward hacking" is easy to misidentify from a single failure signature; use aggregate patterns across runs before making a judgment.
- **Environment determinism matters.** Non-deterministic environments (real browsers, real shells) inflate variance. Prefer sandboxed / snapshotted environments for benchmark runs; use real environments only for stress tests.

## Sources

- Paper: *AgentCompass: A Unified Evaluation Infrastructure for Agent Capabilities* — Ding, Ge, Jiang, Chen et al., 2026 — [arXiv 2607.13705](https://arxiv.org/abs/2607.13705). Shanghai Artificial Intelligence Laboratory.
- Related: *Tracing Agentic Failure from the Flow of Success* — Yeh et al., 2026 — [arXiv 2607.12747](https://arxiv.org/abs/2607.12747). The step-level attribution counterpart to trajectory-level diagnostics.
