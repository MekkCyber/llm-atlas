# R³-Bench: Resource-Rational Reasoning under Shared Budgets
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reasoning models are usually benchmarked *one problem at a time*, but real deployments have to *share a compute budget across many problems at once*. R³-Bench (HKUST-DSAIL, 2026) evaluates six-problem suites under a shared inference budget and finds a stark gap: across 72 main-table cells the "oracle" (best per-problem allocation) matches or exceeds current-practice mean **in all 72**, and is strictly higher in **71**. No single allocation policy dominates across math, competitive programming, and abstract reasoning.

**Prereqs:** [../post-training/reasoning/README.md](../post-training/reasoning/README.md)
**Related:** [aime.md](aime.md) · [math500.md](math500.md) · [livecodebench.md](livecodebench.md) · [../post-training/reasoning/length-penalty.md](../post-training/reasoning/length-penalty.md)

---

## What it is

Every existing reasoning benchmark is *per-problem*: given a problem and unlimited (or fixed-per-problem) tokens, produce an answer. R³-Bench changes that to *per-suite*: given a suite of six problems and one shared thinking budget, decide how much of the budget to spend on each and produce answers for all six.

Two numbers per (model, suite):

- **Contest mean** — what the model actually does with the budget.
- **Oracle mean** — what an oracle allocation (using retrospective per-problem difficulty) would achieve with the same budget.

The gap between them is the benchmark's diagnosis.

## How it works

1. **Six-problem suites** drawn from math (AIME/MATH500-style), competitive programming (LiveCodeBench-style), and abstract reasoning.
2. **Shared inference budget** per suite: a fixed total token count the model must allocate across the six problems as it chooses. The budget forces prioritization — you can't just throw the same length at every problem.
3. **Contest run.** The model runs on the suite as it sees fit, using its own allocation policy (which may be "just try each problem in order," "reserve budget for the easy-looking ones," or anything else).
4. **Oracle allocation.** For the same budget, compute the best-possible allocation post-hoc using per-problem difficulty as revealed by the model's per-problem cost/quality curves.
5. **Report both means** across 72 cells (6 models × 12 suite conditions in the main table).

## Why it matters

- If you deploy a reasoning model as a batch API or as an agent making many child calls, **budget allocation across sub-problems is the dominant cost lever**. R³-Bench isolates that skill and shows current models are visibly bad at it.
- Reframes an under-examined capability: "resource-rational reasoning" — the ability to decide *how hard to think* per problem, not just *how well to think*.
- Directly motivates training work on adaptive length control, early-exit reasoning, and budget-conditioned inference.

## Gotchas & tricks

- **Oracle allocation is retrospective.** It uses information the model can't have at decision time. The oracle is an upper bound, not a target — a large gap means "there is room," not "the model should have known."
- **Budget shapes matter.** Token budget, wall-clock budget, and per-problem *sample* budget behave differently. The paper reports token budgets; other shapes may reorder the ranking.
- **Domain-specific behavior.** No allocation policy dominates across math / code / abstract reasoning — a model's allocation strength may be domain-specific. Aggregate scores can hide this; look at the per-domain cells.
- **Not the same as best-of-N.** Best-of-N spends parallel samples on one problem; R³-Bench spends sequential samples across many problems. Both are "how to spend compute" questions; different levers.

## Sources

- Paper: *R³-Bench: LLMs Struggle with Resource-Rational Reasoning under Shared Budgets* — Peisong Wang, Zhiwei Ma, Bowen Liu, Feixue Liu, Aochuan Chen, Chenyi Zi, Hongchuan Zeng, Yuhan Li, Jia Li — arXiv:2608.16033 — 2026 (HKUST-DSAIL).
- Related benchmarks (source domains): AIME, MATH500, LiveCodeBench — see the per-benchmark depth files.
