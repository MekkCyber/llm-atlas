# Agentic Judge
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** LLM-as-a-judge answers "is this good?" with a scalar. **Agentic judging** answers with a *hierarchical evidence tree*: a parent agent decomposes the evaluation question into measurable subproblems, spawns specialized sub-agents (with tailored context and diagnostic tools) to reason over each, and aggregates their evidence into a final verdict. Introduced by HarnessEval-W (2026), applied to 18 world models across 330 cases with high alignment to human preference and per-rollout, per-subproblem diagnostics.

**Prereqs:** [../evaluation/README.md](README.md), [../agents/_agent-harness.md](../agents/_agent-harness.md)
**Related:** [../agents/harness-scaling.md](../agents/harness-scaling.md) · [../agents/README.md](../agents/README.md)

---

## What it is

A fixed rubric collapses many failure modes into one score. Agentic judging keeps them separate: instead of "is this world-model rollout good?" the judge asks "does physics evolve correctly? does causality hold? does object persistence hold? are affordances consistent?" — one sub-judge per axis, each with its own diagnostic tools, each producing a defensible verdict backed by inspectable evidence.

The verdict is a *tree*, not a scalar. Every score is reconstructable from the evidence chain that produced it — a big deal when a benchmark result is used to make research decisions.

## How it works

Four steps per evaluation case:

1. **Interpret the case.** The parent agent reads the evaluation question and context, and decides which subproblems apply — different rollouts stress different axes.
2. **Decompose.** The question is split into measurable subproblems (physics plausibility, causal chain, object persistence, world-state consistency, etc.).
3. **Spawn tool-equipped sub-agents.** Each subproblem gets a specialized sub-agent with (i) subproblem-tailored context, (ii) subproblem-specific diagnostic tools (frame comparators, physics checkers, object trackers), and (iii) a narrower judgement scope. Sub-agents run in parallel.
4. **Aggregate.** The parent validates each sub-agent's evidence and synthesizes a final verdict — a scalar if the harness demands one, plus the full evidence tree behind it.

The whole workflow is agentic in the same sense as any other agent harness: state, tool calls, retries, transparent logs. The "eval" and "agent" abstractions collapse into one.

## Why it matters

- Bench-as-agent is a general pattern. Any judgement task with a rich reasoning chain (world models, long-form generation, agent trajectories) is a candidate — HarnessEval-W is the first fleshed-out realization.
- Verdicts are *audit-able*. A rejected rollout comes with the subproblem that failed and the evidence for it — not just a low number.
- Verdicts align with human preference on 330 cases while producing fine-grained diagnostics; live-benchmark design means the case set grows as world models grow.

## Gotchas & tricks

- **Cost per case is high.** Each evaluation now spawns many sub-agents with their own tool calls. Batching, caching, and skipping already-decided subproblems all matter.
- **Sub-agent scope creep.** If a sub-agent's tools are too general it will start answering questions outside its scope and reproduce the same "vague scalar" failure the design was avoiding. Narrow tools > general tools.
- **Aggregation is where the bias lives.** The parent's aggregation policy (min? weighted mean? veto?) is a lot of the judgment. Publish it alongside the benchmark.
- **Live benchmarks drift.** As the case set grows, older scores become non-comparable. Version the case set and pin scores to versions.

## Sources

- Paper: *HarnessEval-W: Agentifying the Evaluation of Visual Worlds* — Chen, Sun, Gao et al. (30+ authors) — arXiv:2608.16859 — 2026.
- Contrast: classical *LLM-as-a-judge* — Zheng et al., 2023 (MT-Bench, Chatbot Arena) — the flat-rubric ancestor.
