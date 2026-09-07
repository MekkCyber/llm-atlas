# RealSWE
*Depth — coding-agent benchmark under short, casual, real-user-shaped prompts.*

**TL;DR:** SWE-bench tasks are long, structured, and information-rich; real user prompts are short and casual. RealSWE (2026) builds **381 multi-variant task families** from SWE-bench Verified and Pro whose variants share the same task and gold patch but vary the *information content* and *linguistic style* to mimic real requests. Realistic inputs cost **-6.4 pp resolution** on average across seven contemporary LLMs and can flip model rankings.

**Prereqs:** none
**Related:** [humaneval](humaneval.md), [livecodebench](livecodebench.md), [codeforces-benchmark](codeforces-benchmark.md)

---

## What it is

A controlled counterfactual benchmark over SWE-bench. Each of 381 task families anchors to one SWE-bench Verified or Pro task and contains multiple prompt variants; every variant of a family shares the underlying task and gold patch, and only the *prompt* varies along two axes:

- **Information content** (6-category taxonomy): Problem Statement, Desired Behavior, Motivation, Environment Info, Reproduction Steps, Additional Context.
- **Linguistic style** (4 dimensions): formality, verbosity, hedging, structure.

Real-user distribution (measured from SWE-chat prompts): 88% of real prompts include only Problem Statement (± limited context) vs. 7% of benchmark prompts; 87% of real prompts are casually written vs. 94% of benchmark prompts formal.

## How it works

Two evaluation modes:

1. **Realism drop.** Run models on the "closest-to-real" variant of every family and compare against the "closest-to-benchmark" variant.
2. **Controlled ablation.** Sweep one axis at a time (e.g. add or remove Desired Behavior) and measure the marginal effect on resolution.

Metrics: SWE-bench-style resolution rate (unchanged), plus per-axis marginal effects and rank-agreement across variants.

## Why it matters

- **Falsifies "SWE-bench score = real score."** Ranking shifts across variants — a model that leads in the formal-and-verbose regime can trail in the casual-and-terse regime.
- **Actionable user guidance.** Two axes matter (Desired Behavior, Motivation); the others (Environment, Reproduction) mostly add tokens. That is a concrete prompt-hygiene prescription for coding-agent users.
- **Actionable eval design.** Any coding-agent benchmark that reports one number under fixed prompts is under-specifying its result — RealSWE gives a template for variant-family evaluation.

## Gotchas & tricks

- **Gold-patch preservation is load-bearing.** Variants must not smuggle in extra information via casual phrasing; the paper enforces this per-family manually.
- **Casual ≠ ambiguous.** Casual variants stay unambiguous; the ambiguity axis is a separate line of work.
- **Rank flips are model-dependent.** A model whose training data leaned heavily on GitHub issues will drop more under casual variants than a model tuned for chat.
- **Small linguistic-style effects are real, not noise.** Report per-model deltas rather than a single average.
- **Extends but does not replace SWE-bench.** For head-to-head comparisons on structured issues, SWE-bench Verified/Pro remains the point of reference; RealSWE is complementary.

## Sources

- Paper: *A Compositional Evaluation of Coding Agents under Realistic User Requests* — Gwon et al., Sungkyunkwan University, 2026 — https://arxiv.org/abs/2608.27831
- Benchmark: SWE-bench Verified — Chowdhury et al., 2024 — the base task pool.
- Benchmark: SWE-bench Pro — expanded professional split.
