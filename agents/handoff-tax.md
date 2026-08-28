# The Handoff Tax
*Depth — the cost-quality penalty when a coding agent switches models mid-run.*

**TL;DR:** When an agent escalates to a stronger model (or downshifts to a cheaper one) partway through a run, the receiver has to continue a trajectory produced by another model. Ganz et al. (2026) measure the resulting penalty and find that full-trajectory *escalation* recovers less than half of the low-to-high capability gap while incurring a substantial cost premium — the handoff tax — while *downshifting* is cost-favorable and the preferred trajectory-transfer interface flips direction.

**Prereqs:** [agent-harness.md](agent-harness.md)
**Related:** [jit-harness-generation.md](jit-harness-generation.md), [../evaluation/swe-refactor-bench.md](../evaluation/swe-refactor-bench.md)

---

## What it is

Multi-model agent runtimes routinely swap the underlying LLM inside a single task: escalate to a stronger model when the cheaper one is stuck, downshift once the hard reasoning is done. Each swap forces the receiver to *continue a non-native trajectory* — a working state (chat history, edits, plans) authored by a different model. The paper introduces the "handoff tax" as the cost-quality gap this creates versus running the high-capability model end-to-end.

## How it works

The controlled experiment pairs low-cost/low-capability (LC) and high-cost/high-capability (HC) models from the Claude and GPT families and sweeps three axes:

1. **Direction** — LC→HC (escalate) vs HC→LC (downshift).
2. **Timing** — early, mid, or late in the run.
3. **Interface** — full-trajectory transfer, compacted trajectory, or trajectory removed (receiver sees only the repository/environment state).

Two headline results:

- **Escalation is expensive and half-effective.** Full-trajectory LC→HC recovers less than half of the LC-vs-HC quality gap while carrying a substantial cost premium. That gap is the handoff tax.
- **Interface asymmetry with direction.** *Reducing* LC-trajectory information given to the receiver **improves** escalation. *Removing* HC-trajectory information given to the receiver **degrades** downshift. Same operation, opposite sign.

Downshift itself lands at a favorable cost-quality point when done cleanly — a cheap way to finish once the hard step is over.

## Why it matters

The "escalate to Opus when stuck" heuristic is ubiquitous in production agent runtimes; this paper shows it comes with a large silent tax. The interface-asymmetry finding is actionable: escalation should compact or strip the LC trajectory (the weaker model's mistakes actively hurt the stronger receiver), while downshift should preserve the HC trajectory (the stronger model's reasoning helps the weaker receiver keep going).

## Gotchas & tricks

- **Timing matters.** Early handoffs behave differently from late handoffs; the paper varies this axis and the tax is not uniform.
- **Repository/environment state is doing work.** Even with the trajectory removed, the receiver sees the code/environment changes made so far — that shared substrate is why "no-trajectory" handoffs aren't catastrophic.
- **Applies within a family and across families.** The tax is measured in Claude→Claude and GPT→GPT pairs; cross-family handoffs would likely inherit the same interface asymmetry but the compaction recipe may need retuning.

## Sources

- Paper: *The Handoff Tax: Continuing Non-Native Trajectories in LLM Agents* — Ganz et al., 2026 — [arXiv:2608.24358](https://arxiv.org/abs/2608.24358)
