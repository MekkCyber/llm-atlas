# Knowledge Internalization (Inject → Align → Recover)
*Depth — three-stage post-training recipe for baking a bounded document collection into weights while preserving general capability.*

**TL;DR:** Fine-tuning a model on documents usually trades domain knowledge for general performance (catastrophic forgetting). *Inject → Align → Recover* (IAR) treats knowledge internalization as three staged sub-problems: (1) turn documents into structured knowledge objectives; (2) QA-supervise a fine-tune on those objectives; (3) merge the fine-tuned model back with the base to restore general capability. Reports **+3.6 pts domain QA** and **+12.1 pts general performance** vs plain fine-tuning across multiple corpora and model families.

**Prereqs:** [_post-training.md](_post-training.md), [../pre-training/model-souping.md](../pre-training/model-souping.md)
**Related:** [rejection-sampling.md](rejection-sampling.md), [../pre-training/mid-training.md](../pre-training/mid-training.md), [README.md](README.md)

---

## What it is

A pipeline for the "we own these documents, we want the model to *know* them without RAG at inference" problem. The naive approach — SFT on document text or on synthetic QA — has two known failure modes: (i) it doesn't internalize facts reliably (the model can echo passages without answering questions about them), and (ii) it degrades general capability.

IAR isolates the three problems into three stages, so each is easier to reason about.

## How it works

### Stage 1 — Inject: documents → structured knowledge objectives

Rather than pretraining on raw document text, extract structured "knowledge objectives": entity-attribute-value tuples, procedure descriptions, question-answer pairs derived from the docs. Objectives are the *targeted representations* the model should internalize.

The objective format matters: pretraining on raw text lets the model memorize surface form without extracting facts. Structured objectives force the model to represent the facts in a downstream-usable form.

### Stage 2 — Align: QA-supervised fine-tuning

SFT on QA pairs derived from the objectives. This is the phase where the domain-adapted checkpoint is produced. Standard SFT hyperparameters; the input is the objective-augmented QA corpus rather than raw documents.

### Stage 3 — Recover: merge with the base

Take the domain-adapted checkpoint and merge weights with the original base via model-souping (see [../pre-training/model-souping.md](../pre-training/model-souping.md)) — typically a uniform or task-weighted average. This is where general capability comes back.

The paper's ablations attribute the general-performance gain almost entirely to Stage 3: without the merge, domain QA rises but general benchmarks fall; with the merge, domain QA rises *and* general benchmarks recover.

## Why it matters

RAG is the default for "make a model know my documents," but is expensive at inference (retrieval + long context) and brittle at chunking boundaries. A weights-only alternative that avoids the usual forgetting cost is worth having when the corpus is bounded, the model is open-weight, and inference latency matters. IAR is also a case study in using *model merging* as a forgetting mitigator — a cheaper, sometimes-better alternative to KL penalties or replay.

## Gotchas & tricks

- **Objective extraction is the bottleneck.** Bad objectives make Stages 2 and 3 harder. Prefer LLM-assisted extraction with rule-based validation over pure LLM generation.
- **Merge weight is a knob.** 50/50 is a starting point; tune based on the domain-vs-general balance you want. Task-arithmetic-style weighted merges outperform uniform on some corpora.
- **Not a substitute for RAG when the corpus is unbounded.** IAR pays training cost per corpus; RAG amortizes over corpus updates. Break-even depends on query volume and corpus volatility.
- **Corpus contamination.** If the corpus overlaps with the base model's pretraining, "internalization" gains are inflated (the model already knew it). Evaluate on truly held-out facts.
- **Stage 2 alone is a trap.** The domain-adapted-only checkpoint scores well on domain QA and terribly on general benchmarks; shipping it is a common failure mode this pipeline is designed to prevent.
- **Merging can amplify hallucinations.** If the domain-adapted model learned wrong facts (bad objectives), the merge preserves them. Curate objectives before Stage 2.

## Sources

- Paper: *Inject, Align, Recover: Staged Post-Training for Retrieval-Free Document Knowledge Internalization* — Kou, Shi, Qiu, Zhou, 2026 — [arXiv:2608.20281](https://arxiv.org/abs/2608.20281).
- Related: *Model Soups* (Wortsman et al., 2022) — the merge mechanic used in Stage 3.
- Related: *Task Arithmetic* (Ilharco et al., 2023) — a weighted-merge variant for domain-general tradeoffs.
- Related: *TÜLU 3* — SFT-then-merge motif applied at post-training rather than for knowledge internalization.
