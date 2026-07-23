# Two-level meta-rubrics
*Depth — a rubric-compilation methodology for open-ended-generation evaluation.*

**TL;DR:** A structured *meta-rubric* captures the organization and importance of the content a good answer should contain; a mechanical compiler flattens that meta-rubric into a checklist of binary, machine-gradable items an LLM judge scores reliably. Introduced with GAMUT to evaluate factual *completeness* — the framework is modality- and domain-agnostic.

**Prereqs:** *(none)*
**Related:** [gamut.md](./gamut.md), [ifeval.md](./ifeval.md)

---

## What it is

Rubric-based evaluation for long-form generation has been stuck between two failure modes: (1) a *flat checklist* of independent boolean questions misses ordering, coverage of open-ended sets, and relations between facts; (2) *free-form judge scoring* is subjective, unstable across judges, and hard to audit. Meta-rubrics are a bridge: structured author-time rubric, flat evaluation-time rubric, one compilation step in between.

## How it works

Two levels:

1. **Meta-rubric (structured).** Written by domain experts. Captures:
   - **Open-ended sets** where *coverage* matters — the answer should mention, say, at least K of N valid options, without enumerating them all.
   - **Ordered processes** where the sequence of facts matters.
   - **Relations** between facts (contingent, causal, part-of).
   - **Importance weighting** — which facts are load-bearing vs nice-to-have.

2. **Flat checklist (compiled).** A mechanical pass expands the meta-rubric into binary, machine-gradable items. Each item is a self-contained yes/no question a judge LLM can answer with high inter-run agreement.

At evaluation time, only the flat checklist reaches the judge. That's what makes scores robust to judge choice — the LLM never sees the structured intent, only the atomic binaries.

## Why it matters

- **Separates authoring from scoring.** Experts encode intent once (meta-rubric); the compiler produces stable evaluation artifacts.
- **Judge-robust.** Binary items have narrow variance across judges; the paper confirms this empirically on 14 models.
- **Generalizes.** Same framework works multimodal (GAMUT wearable-imagery) and text-only (released variant).

## Gotchas & tricks

- Compilation is where all the design decisions hide. "How coverage becomes binary items" and "how ordered processes become items" are the pieces to inspect when porting.
- Importance weighting is often naïvely averaged into the final score — the paper handles it more carefully; check before reusing.
- Handcrafted meta-rubrics are expensive. This is the throughput ceiling on the whole benchmark family.

## Sources

- Paper: *Two-Level Meta-Rubrics for Evaluating Open-Ended Generation: GAMUT, a Benchmark for Factual Completeness* — Chen et al. (AI at Meta), 2026 — [arXiv:2607.19322](https://arxiv.org/abs/2607.19322)
