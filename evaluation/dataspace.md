# DataSpace
*Depth — benchmark for data agents producing verifiable tabular results over heterogeneous workspaces.*

**TL;DR:** DataSpace isolates the "return the exact table this question asks for" capability from open-ended analysis. 410 cross-language tasks × 7,439 artifacts (15 GB) across CSV, JSON, SQLite, Markdown, PDF, and video. A deterministic evaluator does header-invariant column alignment, type/precision-aware normalization, and order-aware row comparison — so scoring is reproducible across runs and grading is not LLM-mediated.

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** [../evaluation/README.md](../evaluation/README.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A benchmark and construction framework (DataSpace-Builder) for evaluating **data agents**: agents that answer natural-language analytics questions over a local, heterogeneous workspace and return a structured table. Served as the official evaluation benchmark for the KDD Cup 2026 Data Agents track.

Distinguishes itself from open-ended analytics benchmarks by requiring a *verifiable* tabular output — no LLM-as-judge in the grading loop.

## How it works

**Task shape.** Agent receives only `(question, workspace)`; returns the complete requested tabular result.

**Workspace composition.** Cross-modal artifacts: CSV, JSON, SQLite, Markdown, PDF, video. Cross-language: some artifacts are in non-English natural language.

**Deterministic evaluator.** Compares the agent's returned table to the ground-truth table under:
- **Header-invariant column alignment** — column names don't have to match verbatim.
- **Type- and precision-aware normalization** — `12.30` and `12.3` match; `"12"` and `12` match.
- **Order-aware row comparison** — row order matters when the question specifies ordering, is invariant otherwise.

**DataSpace-Builder.** The construction framework: cross-language transformation, constraint-aware relational sampling, modality routing and artifact rendering, human review and task repair by 11 domain experts.

## Why it matters

- **Verifiable + heterogeneous is rare.** Existing benchmarks either isolate structured querying (Spider), retrieval, or open-ended analysis. DataSpace unifies them with deterministic grading.
- **Harness-vs-backbone separation.** With 6 backbones × 5 harnesses, the best result reaches 66.34% accuracy — unsaturated. Harness choice creates a 15.36-point spread with the backbone fixed, giving teams direct evidence that "which agent scaffold?" is a first-class variable.
- **Modality integration is the bottleneck.** Multimodal-evidence integration and joins consistently reduce accuracy across all six backbones — pinpointing where next work should focus.

## Gotchas & tricks

- Deterministic grading eliminates one class of evaluation drift but is brittle if the ground-truth table has multiple valid formats — the header/order tolerances are load-bearing.
- Video artifacts add nontrivial cost; teams comparing backbones should equalize video-parsing tools rather than let them be a hidden variable.
- The KDD Cup 2026 setup used the benchmark as an evaluation partition, not for training — treat splits accordingly.

## Sources

- Paper: *DataSpace: Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces* — Li et al., 2026 — [arXiv:2608.03451](https://arxiv.org/abs/2608.03451)
