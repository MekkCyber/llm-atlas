# Constraint-Wise Audit (Deep-Research Recursive Self-Improvement)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A deep-research agent architecture built on the asymmetry that **verifying a multi-constraint answer is much cheaper than discovering it**. An inner research loop gathers evidence and drafts a candidate answer; an outer self-improvement loop decomposes the answer into per-constraint checks, audits each against the evidence, and dispatches targeted follow-up research only for the constraints that failed. Introduced as AREX.

**Prereqs:** [README.md](./README.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md).
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) · [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)

---

## What it is

A two-level control structure for deep-research agents:

- **Inner loop (research).** Given the query, an agent searches, reads, extracts evidence, and produces a *provisional* answer.
- **Outer loop (audit).** The provisional answer is decomposed into its atomic constraints ("must be a female French novelist", "must have won the Prix Goncourt", "must have been born before 1950"). Each constraint is verified independently against the gathered evidence. Failed constraints are converted into targeted sub-queries; the inner loop runs again *only for those*.

The recursion terminates when every constraint is either verified or definitively unsatisfiable.

## How it works

1. **Query decomposition** at outer-loop entry: parse the user's request into a set of constraints $\{c_1, \dots, c_k\}$.
2. **Inner research pass**: existing deep-research pipeline (search, browse, extract) produces a candidate answer $a$ and a supporting evidence bundle $E$.
3. **Per-constraint verification**: for each $c_j$, an audit prompt asks "does $E$ demonstrate $a$ satisfies $c_j$?" and returns satisfied / unsatisfied / undetermined.
4. **Selective re-planning**: for each unsatisfied or undetermined $c_j$, generate a targeted sub-query and re-enter the inner loop scoped to that constraint.
5. **Terminate** when all constraints verified (return $a$) or all remaining constraints are provably unsatisfiable by any answer (return "no such entity" or the closest match with caveats).

The auditor and the researcher are typically the *same* LLM in different roles — the recursion is in the control flow, not the model.

## Why it matters

Deep-research agents currently either "think longer" (o1-style, uniform depth) or ReAct-style-loop until they hit a budget cap — neither routes compute to where it's needed. Constraint-wise audit is a **cheap verifier that gates an expensive planner**: the recursion depth adapts per constraint, and the outer loop's cost is dominated by verification (small), not re-search (large).

This architecture matches how humans do hard lookups: draft, spot-check each requirement, chase what fails. Expect it to become the default shape for multi-constraint retrieval agents.

## Gotchas & tricks

- **Constraint decomposition quality is critical.** If the decomposer produces overlapping or missing constraints, either the audit gives false positives (missing a required attribute) or the recursion never terminates (re-checking the same thing).
- **Verifier ≠ prover.** A "verified" constraint under LLM audit is only as reliable as the LLM — it can rubber-stamp an unfounded answer. Ground the auditor on retrieved evidence, not free-form recall.
- **Runaway recursion.** Cap the number of outer iterations. If a constraint remains "undetermined" after $N$ audits, return the answer with an explicit caveat rather than looping.
- **Not the same as chain-of-verification.** CoVe verifies claims *within* a single response; constraint-wise audit runs *between* research passes and drives re-planning, not just self-critique.

## Sources

- Paper: *AREX: Towards a Recursively Self-Improving Agent for Deep Research* — Shuqi Lu, Chaofan Li, Kun Luo et al., 2026 — [arXiv:2607.21461](https://arxiv.org/abs/2607.21461)
