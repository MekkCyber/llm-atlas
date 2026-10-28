# Agent-harness benchmarks — scoring the runtime, not just the model
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Agent benchmarks usually blur "the model was wrong" with "the harness was wrong." An agent-harness benchmark treats the agent runtime itself as the system under test: it decomposes a run into harness components (tool invocation, memory, planning glue, retries, state restore, trajectory logging, failure attribution), runs executable or state-based tasks, and reports **two scores per run** — task success and harness reliability. Introduced by TraceTrial-Bench (Trial by Trace), which found that **31.1% of task-successful runs still fail at least one harness criterion**.

**Prereqs:** [../evaluation/README.md](../evaluation/README.md), [../agents/README.md](../agents/README.md)
**Related:** [../safety/safety-case.md](../safety/safety-case.md)

---

## What it is

A benchmark whose unit of evaluation is `(model, harness, task) → (task_score, harness_component_scores)`. Classical agent benchmarks only expose `task_score`, so when a leaderboard says "Model X beats Model Y at agentic coding" you can't tell whether the signal is model quality, harness quality, or an interaction. An agent-harness benchmark fixes the task set and the models and *varies the harness*, scoring harness components separately from the task.

The TraceTrial-Bench instance organizes harness reliability into **seven components**: tool invocation correctness, memory handling, planning/routing glue, retry logic, state restoration after failure, trajectory logging, and failure attribution. Each task comes with executable or state-based ground truth so component scores can be computed without an LLM judge.

## How it works

For each `(harness, model, task)`:

1. **Run the task** inside the harness, collecting a full trajectory (observations, tool calls, intermediate states).
2. **Trace reconstruction.** Replay the trajectory against the task's state oracle, matching recorded events to expected transitions.
3. **Multi-source evidence gathering.** For each harness component, assemble the trace lines, state snapshots, and error traces that bear on that component.
4. **Two-layer scoring.** Compute (a) task success from the final state and (b) a per-component harness score from the evidence.
5. **Failure attribution.** When task success fails, point at the harness component(s) that caused it; when task success passes but a component failed, flag the lucky-but-broken run.

Aggregated over many tasks and models, the benchmark reports both a per-harness macro score and the rate of "task-success-despite-harness-failure" runs.

## Why it matters

Harness quality is a hidden variable in every "which model is best at X" agent comparison. Making it a first-class, auditable dimension is a prerequisite for honest leaderboards, reproducible agent training (same model + different harness ≠ same result), and for reasoning about agent *safety cases*, where "retries don't drop evidence" and "state restore is reliable" are load-bearing claims.

## Gotchas & tricks

- **Executable ground truth is the hard part.** Component-level scoring only works when the task has state oracles; free-form conversational tasks are harder to grade this way.
- **High task-pass + low harness score is the interesting cell.** It flags lucky runs; those are the runs that will regress once small changes are made to the model or environment.
- **Not a replacement for task benchmarks.** A model that is better *because* a harness is better can still be the right choice for production — but you now know *why*.

## Sources

- Paper: *Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution* — TraceLite AI / CAS / Tsinghua / BIT, 2026 — [arXiv:2610.32285](https://arxiv.org/abs/2610.32285).
