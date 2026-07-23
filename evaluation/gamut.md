# GAMUT
*Depth — a benchmark of factual completeness in long-form generation, grounded in wearable imagery.*

**TL;DR:** Long-form factuality evaluation has focused on precision (are the claims true?). GAMUT (Grounded Assessment of Multimodal factUality) is the missing half: it measures **completeness** (does the response contain all the facts it should?). 1,813 questions across 10 domains, each grounded in real wearable-camera imagery and paired with an evidence-backed rubric verified by expert annotators. Also released in a text-only variant.

**Prereqs:** *(none)*
**Related:** [meta-rubric.md](./meta-rubric.md), [ifeval.md](./ifeval.md)

---

## What it is

Existing factuality benchmarks (decompose-search-verify pipelines) catch *incorrect* claims well but say little about *missing* content. Completeness requires enumerating the full set of facts a good answer should contain — and those facts aren't a flat list: they form open-ended sets, ordered processes, and inter-fact relationships. GAMUT is the first benchmark to pose the completeness question with rubrics rich enough to answer it.

## How it works

- **1,813 questions** across 10 diverse domains, each grounded in a wearable-camera image (multimodal) or in text alone (text-only variant).
- **Two-level rubrics.** Each question is paired with a structured *meta-rubric* capturing the organization and importance of required content. This is mechanically compiled to a flat checklist of binary, machine-gradable rubrics that an LLM judge scores. See [meta-rubric.md](./meta-rubric.md) for the compilation methodology.
- **Expert verification.** Every rubric is evidence-backed and verified by human annotators.
- **Modality-agnostic framework.** The framework decouples the rubric from the grounding, so the same two-level structure applies to the text-only variant.

## Why it matters

- **Redefines the eval question.** Precision-only factuality is half the answer. GAMUT gives us a rigorous handle on the other half.
- **Discriminative on frontier models.** Best score is **58.7% (Gemini 3.1 Pro)** across 14 frontier and open-weight models — hard enough that headroom is visible.
- **Judge-robust.** Because the checklist compiles to binary questions, LLM-judge choice doesn't dominate scores. That's the recurring failure mode of open-ended judge benchmarks.
- **Multimodal grounding is real.** Wearable imagery forces the model to actually look at the scene rather than pattern-match on a text prompt.

## Gotchas & tricks

- "Completeness" isn't well defined without a rubric. GAMUT works precisely because it *defines* the target for each question, not because completeness has a universal metric.
- Meta-rubrics are expensive to build (expert time). The two-level structure is meant to amortize that cost per question across many judge runs.
- The wearable-image domain is a strong grounding signal but a specific one; text-only variant matters for portability.

## Sources

- Paper: *Two-Level Meta-Rubrics for Evaluating Open-Ended Generation: GAMUT, a Benchmark for Factual Completeness* — Chen et al. (AI at Meta), 2026 — [arXiv:2607.19322](https://arxiv.org/abs/2607.19322)
