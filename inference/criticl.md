# CritICL — Critique-based inference-time weak-to-strong
*Depth — turning small-model failure modes into in-context critiques for the strong model.*

**TL;DR:** Standard inference-time scaling generates many samples and picks a winner (best-of-N, self-consistency). CritICL takes a cheaper path: mine the *failure modes* of a weaker sibling model, convert them to critiques, and pass those critiques to the strong model as in-context examples. One forward pass, no verifier, matching or beating test-time-scaling on cost.

**Prereqs:** [README.md](README.md)
**Related:** [_test-time-scaling.md](_test-time-scaling.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md), [../post-training/reasoning/ttpo.md](../post-training/reasoning/ttpo.md)

---

## What it is

An inference-time technique that uses a small model from the same family as a **failure-mode probe** for the strong model. The insight: within a model family, failure modes are structured — the small model makes systematic errors on classes of inputs, and the strong model, told what those errors look like, avoids them.

Two variants:

- **CritICL-dynamic** — per-input, retrieve or predict input-specific failure modes; retrieve matching critiques.
- **CritICL-static** — precompute a global failure-mode profile once; prepend it as static critique context to every query.

## How it works

**Offline (once per model family):**

1. Run the small model on a broad probe set.
2. Cluster its failures by error type.
3. For each cluster, generate a critique — an in-context example of the wrong-pattern paired with the correct reasoning.

**At inference time:**

- Dynamic: retrieve the top-`k` critiques relevant to the current input; prepend them.
- Static: prepend the fixed critique profile.

The strong model answers with the critiques in context — no repeated sampling, no verifier calls.

## Why it matters

Weak-to-strong has been mostly a *training-time* story (small models supervising fine-tuning of strong models). CritICL shows the same asymmetric-supervision idea works at inference too: exploit the failure structure that a small model reveals cheaply and inject it into the strong model's context. Reported to match or beat standard test-time scaling on reasoning benchmarks with **far fewer generations** and lower token cost.

## Gotchas & tricks

- Critique quality is bounded by the small model's failure diversity — a small model that fails on one narrow pattern gives narrow critiques.
- Dynamic retrieval helps most on heterogeneous benchmarks; static profile suffices on narrower ones.
- Failure-mode transfer across families is weaker — same-family (e.g. Llama-8B → Llama-70B) works best.
- Distinct from self-critique / self-refine: the critiques come from a *different* (weaker) model, not from the target's own outputs.

## Sources

- Paper: *CritICL: Inference-Time Weak-to-Strong Generalization from Small Language Model Failure Modes* — Wu, He, Hu, Wei, Li, Yang, Zhu, 2026 — [arXiv:2608.27455](https://arxiv.org/abs/2608.27455)
