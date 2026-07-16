# SDABench
*Depth — capability-oriented benchmark for LLM-driven scientific data analysis, with a five-stage error taxonomy.*

**TL;DR:** Prior scientific-analysis benchmarks measure code execution or workflow completion — surface behavior. SDABench (Shi et al., 2026, HKUST, arXiv 2607.11079) instead measures **which type of scientific claim** an LLM can support: descriptive, exploratory, inferential, predictive, causal, or mechanistic. 527 real-data + 6,000 synthetic instances across Biology, Chemistry, Environment, Geography, Physics. Includes a five-stage error taxonomy that localizes failures — from "picked the wrong scope" to "drew an invalid conclusion."

**Prereqs:** *(none)*
**Related:** [mmlu](mmlu.md)

---

## What it is

A benchmark for "AI scientist" claims — the growing set of papers/products that use LLMs to do scientific data analysis end-to-end. SDABench reorganizes the eval question along two axes:

- **6 capabilities** — descriptive, exploratory, inferential, predictive, causal, mechanistic. Each has different assumptions and validity criteria.
- **5 domains** — Biology, Chemistry, Environment, Geography, Physics.

Cross-product yields the item space. Each item comes in both MC and open-ended forms so you can compare recognition vs. generation.

## How it works

### Data

- **SDA-Real** — 527 real-data instances built from actual scientific analyses.
- **SDA-Synth** — 6,000 synthetic instances generated through an automated pipeline for scale.
- Both cover all 30 (capability × domain) cells, though not uniformly.

### Five-stage error analysis

The benchmark ships with a taxonomy of *where* the model failed, not just *whether*:

1. **Scope identification.** Did the model correctly frame the question?
2. **Variable identification.** Did it pick the right variables?
3. **Procedure selection.** Did it choose an appropriate analytical procedure (test, model, method)?
4. **Model construction / execution.** Did it correctly model the relationship?
5. **Conclusion validity.** Did it draw a valid conclusion given the analysis?

The taxonomy lets papers report *per-stage* failure rates instead of a single top-line accuracy.

### Reported results

Across 15 evaluated LLMs:

- **Descriptive** analysis is nearly saturated.
- **Assumption selection** (choosing the right statistical test given data properties) is where most models degrade.
- **Latent-process modeling and mechanistic reasoning** are the hardest capabilities — even the strongest models struggle.
- More advanced models identify scope and variables correctly but still fail at procedure selection and conclusion drawing.

## Why it matters

- **Separates code execution from scientific validity.** An LLM that can run pandas is not the same as an LLM that can defend the inference the pandas run implies. SDABench measures the latter.
- **The error taxonomy is diagnostic.** "Model fails at mechanistic reasoning" is a target for training data curation. "Model fails at conclusion validity" is a target for reasoning post-training.
- **Grounds AI-scientist claims.** Provides a common yardstick for a class of papers that had been evaluating on ad-hoc rubrics.

## Gotchas & tricks

- **The five stages compound.** A failure at stage 1 propagates. Aggregating per-stage failure rates without accounting for cascading is misleading.
- **Synthetic vs. real gap.** Synthetic instances are cheap but distributionally narrower than real. Report both.
- **Domain coverage is uneven.** Not every capability × domain cell has enough instances for a robust per-cell score.
- **Depends on graders.** Open-ended items need a grader (LLM or human); grader quality bounds the reported numbers.

## Sources

- Paper: *Are LLMs Ready for Scientific Discovery? A Capability-Oriented Benchmark for AI Scientists* — Shi et al., HKUST, 2026 — arXiv 2607.11079.
