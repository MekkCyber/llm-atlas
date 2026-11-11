# Longtail-SWE
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A repo-level software-engineering benchmark that extends SWE-bench-style issue resolution to 13 less-represented programming languages (Clojure, Dart, Elixir, Erlang, F#, Haskell, Julia, Kotlin, Lua, OCaml, R, Scala, Swift) — 202 tasks total. Constructed through a **solver → judge → repairer** loop so that ambiguous task descriptions and buggy tests are *fixed* instead of filtered out. All instances, environments, tooling, and trajectories released.

**Prereqs:** [./README.md](./README.md), [./livecodebench.md](./livecodebench.md)
**Related:** [./humaneval.md](./humaneval.md), [../data/quality-filtering.md](../data/quality-filtering.md)

---

## What it is

Agentic code benchmarks are now dominated by Python because Python has the richest test infrastructure and the biggest open repo corpus. SWE-bench, SWE-bench Verified, and clones concentrate scores there. Longtail-SWE is the opposite move: build a benchmark where *language coverage is the primary axis*, and keep task difficulty high.

The challenge is that the normal benchmark-construction pipeline (scrape issues → verify tests → filter) is brutal in niche ecosystems: too many issues are vague, and many tests either reject valid alternatives or accept wrong implementations. Simple filtering throws away precisely the hard-but-fixable tasks.

## How it works

**Construction pipeline:**

1. **Solver** attempts each candidate task (an issue + its repo state + a test suite) by running an agent and recording the trajectory.
2. **Judge** reviews the solver's trajectory, the task description, and the test outcomes. It flags three kinds of problems: (a) tests reject valid alternative implementations, (b) tests accept incorrect solutions, (c) the task description is ambiguous.
3. **Repairer** edits the tests or description to fix the flagged issues *while preserving intent*. The result is a modified task that is re-queued back to the solver.

The loop iterates until the judge is satisfied or the task is deemed unsalvageable (dropped). This lets the authors keep genuinely challenging tasks that naive filters would discard.

**Evaluation protocol.** Nine model configurations evaluated, including frontier models. Reports: pass rate, consistency across repeated attempts, and resource usage (tokens, wall-clock, tool calls). An "easier" subset is released for smaller models.

## Why it matters

- Fills a known gap: SWE-bench saturation is accelerating, and language-specific overfitting is a real concern. A 13-language benchmark with strong curation is a natural counterweight.
- The **solver-judge-repairer** recipe is reusable: any benchmark construction in a scarce-data ecosystem can borrow it to recover tasks that filters would reject.
- Releases executable environments and agent trajectories, which makes reproduction and ablation of agent recipes against the benchmark concrete.

## Gotchas & tricks

- **Repairer bias.** The judge and repairer are themselves LLMs; if they share failure modes with the solver, the final benchmark has correlated blind spots. Longtail-SWE mitigates by using strong diverse models for the pipeline.
- **"Consistency across attempts" is a first-class metric.** Agentic benchmarks that only report best-of-$N$ hide a lot of variance; Longtail-SWE reports it separately.
- **Resource usage matters.** Frontier-model pass rates may look high but at enormous token cost; the released usage logs let downstream papers normalize fairly.
- **Not a generic cross-language eval** — issues are drawn from real repos, so each language's tasks reflect that language's typical bug style (e.g. Haskell tasks skew type-system-heavy). Report per-language scores, not just averages.

## Sources

- Paper: *Longtail-SWE: High-Quality SWE Tasks for Underrepresented Languages* — 2026 — [arXiv:2611.10304](https://arxiv.org/abs/2611.10304)
