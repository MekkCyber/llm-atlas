# ASI-Bench
*Depth — one specific benchmark, grounded in its source paper(s).*

**TL;DR:** A 60-task, 11-domain benchmark for **open-ended scientific research** where the amount of human scaffolding is *itself* a variable. Each task is graded at multiple guidance levels, from "the method is specified" down to "you must choose the method". Yields a per-system **guidance-sensitivity curve** that separates agents that solve problems when told how from agents that self-direct. Frontier systems drop from **50.91 (full guidance) → 26.62 (no guidance)** — a ~50% relative collapse.

**Prereqs:** none
**Related:** [../post-training/reasoning/README.md](../post-training/reasoning/README.md)

---

## What it is

Most AI-scientist / deep-research benchmarks give the agent a fixed prompt and score its output. ASI-Bench sweeps the *prompt itself*: for each task it prepares multiple variants that progressively reduce the human's scaffolding — first the method is specified, then only the general approach, then only the goal. The score profile across guidance levels is the actual signal.

## How it works

### Task portfolio

- **60 project-level research tasks** across **11 scientific domains** (breadth deliberately favours generality over depth in any one field).
- Each task has multiple **guidance levels** — commonly full, partial, none.
- Grading is on the *final artifact* (a verifiable result, e.g. a numerical answer, a proof, a working codebase, an experimental report).

### The guidance-sensitivity curve

For a fixed system, plot score vs guidance level. A system that only executes when told how has a steep drop; a system with research initiative has a shallow drop. The curve, not the peak, is the primary metric.

### Headline numbers

Reported frontier scores:

| Guidance | Score |
| --- | --- |
| Full | **50.91** |
| No guidance (agent must select method) | **26.62** |

A ~50% relative collapse — the field is a long way from unaided ASI.

## Why it matters

- **Separates execution from initiative.** Two systems can look identical at the "full guidance" peak yet diverge sharply when scaffolding is removed. Prior benchmarks conflated these; the guidance curve exposes it.
- **Frames the ASI capability question quantitatively.** Rather than "did the AI do research?" it asks "how much human scaffolding did the AI still need?"
- **Portable across systems.** The guidance-sweep methodology can be applied to any task suite, not only ASI-Bench, and will likely be adopted by follow-up co-scientist / deep-research benchmarks.

## Gotchas & tricks

- **Guidance level ≠ prompt length.** A short prompt can carry heavy scaffolding (a method name), a long prompt can carry very little (a background dump). Grade by *information content* of the scaffolding, not tokens.
- **Full-guidance score alone is misleading.** Report the curve. A system with a high peak but a steep drop is not a research agent.
- **Domain coverage matters.** The 11-domain breadth surfaces initiative asymmetries that a single-domain benchmark (e.g. protein design only) would hide.
- **Grade on artifact, not process.** Trajectory-level grading rewards long chains of tool use even when the artifact is wrong. Artifact-level grading avoids that failure mode.

## Sources

- Paper: *ASI-Bench: At the Dawn of Artificial Superintelligence* — Zhou et al. (43 authors, large consortium), 2026 — https://arxiv.org/abs/2608.17271
- Related: Deep-Research benchmarks (OpenAI DR eval, Anthropic ARE), MLE-Bench, SciCode, DiscoveryWorld.
