# AgentSWE
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for the specific task of using a coding agent to *build another agent*. Three stages — create an agent from a natural-language requirement, edit an existing agent codebase, optimize an agent toward a measurable target — scored not on diffs but on the *behavior* of the produced agent on held-out test cases. Across 25 tasks, delivered agents consistently fall short of the spec; a large visible-vs-held-out generalization gap is first-class in the metrics.

**Prereqs:** [evaluation/README](../evaluation/README.md), [agents/README](../agents/README.md)
**Related:** [livecodebench](../evaluation/livecodebench.md)

---

## What it is

A code-agent benchmark where the artifact under test is itself an agent. The judgment is behavioral: run the produced agent, score it on cases the coding agent didn't see at build time. Distinct from SWE-bench and LiveCodeBench, which score produced code on unit tests directly rather than on the downstream behavior of a *built system*.

## How it works

25 tasks, each specified in natural language, structured in three stages:

1. **Create.** Given a spec, build an agent (from scratch or from a template).
2. **Edit.** Given an existing agent codebase, modify it to meet a new requirement.
3. **Optimize.** Given a measurable objective and a baseline agent, improve it on that objective.

Each task includes:

- A **visible dev set** — behavioral test cases the coding agent can run to iterate.
- A **held-out set** — behavioral test cases revealed only during scoring.

The headline metric separates **visible pass rate**, **held-out pass rate**, and the **generalization gap** (visible − held-out). A high gap means the coding agent overfit to the visible cases.

## Why it matters

- **Measures recursion.** "Agents building agents" is on many 2026-era roadmaps; AgentSWE provides sober evidence of the current gap.
- **Scores artifacts, not patches.** A diff can be correct and still produce a bad agent; AgentSWE catches that.
- **Visible/held-out split.** Separating in-distribution from out-of-distribution behavior exposes a failure mode (visible-case overfit) that unit-test benchmarks miss by design.

Reported finding: delivered agents consistently fail held-out requirements even when visible dev-set cases pass. Several recurring failure modes identified — visible-case overfit, dropped requirements, shallow optimization loops.

## Gotchas & tricks

- The behavioral scorer itself is a system: if a coding agent produces an agent that merely *passes the scorer's checks*, Goodhart's law applies. The paper mitigates with held-out variations.
- Optimize-stage tasks are sensitive to the objective's gameability — picking tight objectives matters.
- The generalization-gap metric is only interpretable alongside absolute scores; a 0-pp gap at 10% success is not a win.

## Sources

- Paper: *AgentSWE: Can Coding Agents Build the Agent You Actually Want?* — Wang et al., BAAI / CAS / RUC, 2026 — [arXiv:2610.13361](https://arxiv.org/abs/2610.13361)
