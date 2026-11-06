# FrontierPhysics
*Depth — the end-to-end physics-research benchmark for agents.*

**TL;DR:** 56 tasks across **9 physics domains** (theoretical, computational, experimental), each authored by one of 42 physicists from their **own published research**. Agents must review the literature, plan, run experiments, and produce structured outputs plus a scientific manuscript. Grading combines **deterministic verifiers** (did the code produce the expected number?) with **expert-authored rubrics** (is the reasoning rigorous enough for a referee?). Across **22 models in 26 configurations**, no configuration passes more than **46 of its 168 rollouts** (~27%). The main failure modes are not knowledge gaps — they are *insufficiently rigorous reasoning* and *limited expert intuition*.

**Prereqs:** *(none)*
**Related:** [livecodebench](./livecodebench.md) · [mmlu](./mmlu.md) · [../agents/README.md](../agents/README.md)

---

## What it is

A benchmark for agents doing **full research**, not isolated question answering. Each task is a slice of a real physicist's published work — hypothesis, data, analysis, plot, writeup — with the deliverable restated as an agent task. Three task families:

- **Theoretical** — derive, prove, or model.
- **Computational** — write and run simulations / numerics.
- **Experimental** — plan a procedure, analyze given data, interpret.

The benchmark covers 9 domains including condensed matter, high-energy, AMO, astrophysics, and biophysics. Every task has a (verifier, rubric) pair: the verifier is a programmatic check (numerical output, derivation against a sympy ground truth, plot matches an expected shape), the rubric scores the written manuscript on dimensions like clarity, rigor, acknowledgment of limitations.

## How it works

**Task authoring.** 42 physicists each authored ~1–2 tasks from their own papers, so the ground truth is a real research outcome. Tasks come with a deliverable spec: structured JSON output + a short manuscript section (methods + result + discussion).

**Agent execution.** The agent gets the task prompt and access to standard tools (code sandbox, file I/O, optionally web browsing and library documentation). It produces the structured output + manuscript.

**Grading.** Two-track:
- Deterministic verifier returns `pass/fail` (or a numeric distance from the reference).
- Rubric is graded by an LLM judge against expert-written criteria, sometimes spot-checked by a human. Each task has multiple rubric items with weights.

**Reporting.** Each configuration gets `k` rollouts per task (`k = 3` in the paper; `56 tasks × 3 = 168` rollouts per config). Scores are reported as pass-rate across those rollouts.

## Why it matters

Math and code benchmarks (AIME, LiveCodeBench, MATH-500) have saturated or are saturating at the frontier; "research-grade" benchmarks are the natural next test. FrontierPhysics does two things no earlier agent benchmark does at scale: (1) it scores **research quality** — not just correctness — via expert rubrics, and (2) it is grounded in *the authors' own published research*, which both guarantees ground truth and resists direct contamination because tasks are reformulations, not the papers themselves. The template (one task per physicist, from their own paper, verifier + rubric) transfers cleanly to chemistry, biology, economics, …

## Gotchas & tricks

- **Rubric judges drift.** Rubric scores depend on the LLM judge; the paper reports judge-ablation sensitivity but a different judge will shift the absolute numbers. Compare within a fixed judge.
- **56 tasks is small.** Even averaged over 3 rollouts, per-task variance is high — one well-matched / badly-matched task is ~2 pp.
- **Browsing creates contamination risk.** If browsing is enabled, agents can land on the source paper. The benchmark reports both configurations; always check which one is used.
- **Expert intuition is the stated bottleneck.** This suggests reasoning RL alone won't close the gap — domain-specific scaffolding or retrieval over a physicist's working notebook is likely where the next gains come from.

## Sources

- Paper: *FrontierPhysics: Evaluating Agents for End-to-End Physics Research* — You et al., 2026 — https://arxiv.org/abs/2611.06910
