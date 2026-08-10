# HarnessOpt-Bench
*Depth — benchmark for LLMs at optimizing their own agent harness.*

**TL;DR:** An LLM in an agentic system depends on its *harness* — prompts, tools, control flow, memory, orchestration code. HarnessOpt-Bench measures how well a frontier LLM can autonomously improve a harness end-to-end: an optimizer edits a seed harness under a fixed evaluation budget and is scored by normalized gain on a held-out test split. Two protocol conditions: shared coding harness vs. each optimizer's native harness.

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** [../evaluation/README.md](../evaluation/README.md), [../evaluation/livecodebench.md](../evaluation/livecodebench.md)

---

## What it is

A benchmark for a *capability*, not a model: "given a target agent, can the LLM improve its harness under expensive, stochastic evaluation?" Optimizers get a seed harness, graded eval feedback on the training partition, and a fixed target-eval budget. They edit the harness iteratively and nominate a final candidate, which is scored on a held-out test partition by its normalized gain over the seed.

A trusted execution environment enforces the evaluation boundary, meters target-agent resource use, and preserves candidate versions for audit — the eval is reproducible and can support later governance/audit workflows.

## How it works

**Protocol per run:**

```
Optimizer  ← LLM + coding harness (shared or native)
Inputs     ← (seed harness, graded eval feedback fn, target-eval budget)
Loop       ← optimizer edits harness → runs graded eval → repeats
Output     ← final nominated harness
Score      ← normalized gain over seed on held-out test partition
```

**Conditions.** Every optimizer is run under (a) a shared coding harness (equalizes scaffolding) and (b) its native harness (fair comparison at deployment). Comparing the two conditions per model tells you how much of the optimizer's score is model vs. scaffolding.

Evaluated across 5 frontier LLMs × 4 downstream tasks × 111 scored runs.

## Why it matters

- **Separates skill from scaffolding.** Optimizer choice separates results more than the coding harness it acts through — meaning "the LLM matters," but native harnesses aren't consistently superior. That's an actionable finding for teams shipping agent products.
- **Turns "improve the prompt" into a benchmark.** Autonomous prompt/tool/scaffolding optimization is becoming a standard part of agent products; HarnessOpt-Bench is a shared yardstick for it.
- **Audit-ready.** Preserved candidate versions and metered budgets make the run reconstructible, which matters as agentic systems accrue governance requirements.

## Gotchas & tricks

- Gains vary substantially across tasks and seeds — reporting single-task scores masks large variance.
- Fixed evaluation budget is load-bearing: too small and optimizers can't explore; too large and score reflects budget more than skill.
- Comparing native harnesses across models mixes optimizer skill with harness-designer skill; the shared coding condition is the fairer capability comparison.

## Sources

- Paper: *HarnessOpt-Bench: Evaluating LLMs at Harness Optimization* — Shanker et al., 2026 — [arXiv:2608.06301](https://arxiv.org/abs/2608.06301) — Scale AI
