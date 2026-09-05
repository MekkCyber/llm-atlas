# RealSWE
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** SWE-bench tasks are curated GitHub issues — long, structured, information-rich. Real users don't file bug reports like that. **RealSWE** (Kim et al., 2026) systematically decomposes SWE-bench tasks into 381 variants that manipulate *which pieces of information* the request contains (behavior spec, motivation, reproduction steps) and *how* it's stylized. Coding agents drop **~6.4 points** on average under realistic inputs; explicit behavior and motivation matter, linguistic style barely does.

**Prereqs:** [../evaluation/README.md](README.md)
**Related:** [livecodebench](livecodebench.md), [humaneval](humaneval.md)

---

## What it is

A benchmark constructed *by ablation* of an existing benchmark. For each SWE-bench task, the authors identify the load-bearing pieces of the issue body:

- **Behavior specification** — what the code should do.
- **Motivation / user intent** — why the user wants it.
- **Reproduction steps** — how to trigger the current wrong behavior.
- **Expected vs. actual output** — the concrete comparison.

Each piece can be present or absent; each can be rewritten in a different style (terse chat message, verbose GitHub issue, bug report template). That defines a compositional space of task variants over one underlying repository fix; the paper samples **381 variants** covering realistic combinations.

## How it works

**Two ablation axes** run over the same underlying fix task:

1. **Information composition** — subsets of the four content pieces above. A variant that has only "expected output" is much closer to what a real user files than the full SWE-bench issue.
2. **Linguistic style** — rewrite the same content in different registers (formal issue, casual message, single-line request).

Each of the 381 variants is a distinct task from the agent's perspective; the ground-truth patch is the same. Measure resolution rate per variant, then aggregate by axis to isolate which axis causes the drop.

## Why it matters

**Benchmarks encode assumptions about the deployment distribution.** SWE-bench encodes "a well-written GitHub issue"; agents evaluated only on it inflate their apparent competence at *code repair*, when a chunk of the number really tracks *information availability in the input*. RealSWE lets you separate the two.

Key findings:

- Realistic inputs drop resolution rate by **~6.4 pts on average** across evaluated coding agents.
- **Explicit behavior and motivation statements substantially improve performance.** Removing them is the main driver of the drop.
- **Linguistic style produces minimal effects.** Terse vs. verbose framing barely matters; content matters.

Practical implication: when comparing coding-agent papers, ask whether the input matches the deployment setting. Two agents at the same SWE-bench score can differ substantially in a shorter-issue regime.

## Gotchas & tricks

- **Not a new task distribution — a slicing of the existing one.** The ceiling is still SWE-bench's underlying fix set; RealSWE reveals sensitivity along the request axis but not open-domain generalization.
- **The 6.4-pt drop is an average.** Per-agent, the gap ranges wider; some agents are much more information-fragile.
- **Ambiguous requests are handled by clarification in some agent stacks.** Agents that ask a follow-up question before attempting the fix win on RealSWE without changing the underlying coding ability — factor that into the comparison.
- **Style-invariance is an *average* result.** For requests near the informational floor, style choices can flip the interpretation.

## Sources

- Paper: *RealSWE: A Compositional Evaluation of Coding Agents under Realistic User Requests* — Kim, Gwon, Kim, Shim, Lee, Sungkyunkwan University, 2026 — [arXiv:2608.27831](https://arxiv.org/abs/2608.27831).
- Related benchmark: SWE-bench (Jimenez et al., 2023) — the underlying task set RealSWE re-slices.
