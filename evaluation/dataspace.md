# DataSpace
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** 410 data-analysis tasks where an agent must produce **verifiable tabular results** from workspaces mixing CSV, JSON, SQLite, Markdown, PDF, and video (15 GB total). Grading is deterministic — header-invariant column alignment, type- and precision-aware normalization, and order-aware row comparison — so equivalent-but-differently-formatted tables are correctly scored equal.

**Prereqs:** none.
**Related:** [../agents/README.md](../agents/README.md) · [../data/_data-curation.md](../data/_data-curation.md) · [README.md](./README.md) · [harnessopt-bench.md](./harnessopt-bench.md)

---

## What it is

Data-analysis agents are a canonical enterprise use case for LLMs, but existing benchmarks fall into two traps: (1) grade on text similarity, missing correct-but-differently-formatted answers, or (2) restrict to one file type, dodging the actual challenge of heterogeneous workspaces. DataSpace fixes both: multimodal source files, executable ground-truth answers, and a table-level grader.

## How it works

**DataSpace-Builder** (the construction pipeline):
- **Cross-language transformation**: workspaces mix formats; the builder ensures each task's ground truth can be computed via a well-formed query across all of them.
- **Constraint-aware sampling**: task instances are sampled to hit target difficulty and coverage of primitives (filter, join, aggregate, multimodal integration).
- **Human expert review** before inclusion.

**Grading protocol.** The agent produces a table (any format). The grader normalizes:
- **Headers**: alias resolution + column matching by semantic tag, not string identity.
- **Types and precision**: `"1.0"` == `1` == `1.00`; `"USD 1,000"` == `1000` when unit column agrees.
- **Row order**: order-sensitive if task specifies (Top-K), order-invariant otherwise.

Only after normalization does the grader compare cells. Correctness ↔ any tabular representation that captures the same underlying result.

## Why it matters

- **First deterministic multimodal analytics benchmark.** Grader is reusable for any tabular agent task.
- **Real workspace shapes.** Videos as evidence, PDFs as source docs, SQLite alongside CSVs — matches the actual mess of production data lakes.
- **Diagnostic bottlenecks.** Peak accuracy 66.34% across 6 multimodal models × 5 agent harnesses. Joins and multimodal evidence integration are the sharpest failure modes.

## Gotchas & tricks

- **Normalization has escape hatches.** Some tasks care about representation (financial reports); the protocol lets tasks opt into stricter comparison.
- **15 GB is large.** Loading full workspaces is expensive; harnesses that don't chunk or index will struggle with latency, not just accuracy.
- **Agent-harness quality dominates on some tasks.** Shared-model comparisons across harnesses show wide spread — factor this out when reporting model results.
- **Grader can be too permissive** on numerically noisy tasks (aggregations over long tails). Precision thresholds are per-column configurable.

## Sources

- Paper: *DataSpace: Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces* — Li et al., HKUST(GZ) / Tsinghua, 2026 — [arXiv:2608.03451](https://arxiv.org/abs/2608.03451).
