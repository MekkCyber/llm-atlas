# Textbook-structured SFT
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A math-SFT corpus-building recipe that keeps a textbook's *structure* — chapter order, proposition↔proof links, cross-references, provenance — rather than flattening it to problem/solution pairs. The structural signals then drive curriculum learning and let scaling studies separate data volume from source diversity. TOME-Math (the paper's instantiation) lifts base-model math benchmarks by +16.7 points on average.

**Prereqs:** [_data-curation.md](_data-curation.md), [quality-filtering.md](quality-filtering.md)
**Related:** [decontamination.md](decontamination.md), [../evaluation/math500.md](../evaluation/math500.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)

---

## What it is

The dominant recipe for math SFT is "scrape the web and the exam sites for ⟨problem, solution⟩ pairs, filter for quality". That throws away the pedagogical scaffolding — which chapter the proposition sits in, which proofs it depends on, which variants are drilled first. Textbook-structured SFT keeps that scaffolding in the dataset and exposes it to training.

Concretely: start from a corpus of textbooks (TOME-Math uses 2,118 university-level books). Parse each book into a tree of chapters → sections → propositions → proofs, keep source IDs, and extract both free-form problems and the *dependency edges* between items (this proposition uses that lemma, this exercise follows that example).

## How it works

Three pieces together:

1. **Structural extraction.** Convert PDFs/HTML into a typed graph: nodes are propositions/examples/proofs/exercises; edges are "uses", "proves", "generalizes", "exercises". Keep chapter indices.
2. **Record formatting.** Each SFT record is a problem + its chapter context + the *in-book* proof. Dependencies are available as either visible prereqs or as a curriculum ordering constraint.
3. **Curriculum sampling.** Train in textbook order within a subject, mixing subjects on a schedule. Volume and source diversity are scaled independently so ablations can disentangle the two.

The result is ~380K structured proof records in the paper's instantiation.

## Why it matters

Suggests the ceiling on math SFT is set less by problem count than by how much *pedagogical structure* the dataset preserves. Three practical consequences:

- Curriculum ordering works again when the ordering is the author's, not a learned heuristic.
- Data-mix ablations gain a real axis (source diversity) to vary beyond token count.
- The recipe generalizes: any domain whose knowledge is already organized in books (physics, law, medicine) can plug in with minimal changes.

## Gotchas & tricks

- Structural extraction from PDFs is the bottleneck; expect more engineering than ML.
- Easy to leak eval content — textbook exercises overlap with public benchmarks, so decontamination is critical (the paper runs explicit decon against MATH/AIME).
- Curriculum only helps when the subject sequence is *actually* a prereq chain; mixing subjects too early washes the signal out.
- Volume scaling continues past 1K books in their runs — plateaus later than classic web-scale math data.

## Sources

- Paper: *Scaling Mathematical Proof Supervision from Structured Textbooks* — Fang, Hua, Tang, Gu, Kong, Liu, Zhang, Chen, Sun, Zhang (2026) — [arXiv:2610.40394](https://arxiv.org/abs/2610.40394)
