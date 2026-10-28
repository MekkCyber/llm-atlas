# Δτ-Bench — tool-use agents against a moving ecosystem
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Classical tool-use benchmarks (τ-Bench, BFCL) hold the tool catalog fixed. Δτ-Bench models **ecosystem churn**: six successive versions of a tool ecosystem with deprecations, replacements, and extensions, over **1,588 multi-turn tasks across 42 domains**. The strongest model/harness tops out at **42.3%** task success. The paper also introduces **Steward**, an ecosystem manager that incrementally maintains a skill library and dynamically provisions tools at execution time.

**Prereqs:** [../evaluation/README.md](../evaluation/README.md), [../agents/README.md](../agents/README.md)
**Related:** [agent-harness-benchmark.md](./agent-harness-benchmark.md)

---

## What it is

A benchmark where the environment is non-stationary: tools come and go between ecosystem versions, their argument shapes evolve, replacements have slightly different semantics. Agents are expected to adapt without full retraining. The benchmark is executable (state-based graders) and covers 42 domains with realistic multi-turn interactions — not single-call API grading.

## How it works

- **Ecosystem states.** Six snapshots of the tool catalog, each a strict delta over the previous: deprecations, additions, and in-place edits.
- **Task set.** 1,588 multi-turn tasks tagged with the ecosystem state they target; tasks that span states exercise adaptation explicitly.
- **State-based grading.** Each task has a terminal-state oracle rather than an answer string.
- **Steward (baseline runtime).** Keeps a persistent skill library keyed by tool interface signatures; when a tool changes, the affected skills are flagged and re-synthesized. At inference, Steward provisions only the tools relevant to the current step, keeping the agent's effective tool surface small.

## Why it matters

Agent production systems live in constantly-changing tool landscapes: API deprecations, schema versions, new integrations. Benchmarks that freeze the catalog systematically overstate real-world agent reliability. Δτ-Bench is the first evaluation that scores *adaptation* across ecosystem versions as a first-class property, and the Steward runtime shows that a persistent skill library plus just-in-time tool provisioning is a lightweight alternative to full re-training per catalog update.

## Gotchas & tricks

- **Executable tool stubs.** The ecosystem states are implemented as runnable mocks; getting determinism right across versions is the hard infra work.
- **"Harness" and "model" are entangled.** Steward is a *runtime*; the model inside it is interchangeable. Headline percentages are joint model-harness scores.
- **42.3% is the ceiling.** The headline number is the best configuration; many model/harness combinations do dramatically worse on the later ecosystem states where more tools have churned.

## Sources

- Paper: *Δτ-Bench: Pushing Agents Toward a Moving Capability* — Qiao, Pan, Chen, Feng, Wu, Fang, Zhu, Deng, Yilmaz (lead affil UCL), 2026 — [arXiv:2610.34238](https://arxiv.org/abs/2610.34238).
