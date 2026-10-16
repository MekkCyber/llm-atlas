# Memory Consolidation (Cortex)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Context in the prompt improves LLM performance but the gains vanish once the context is removed. Cortex consolidates those gains into **mergeable low-rank or structural FFN updates** so the model behaves as if the context were still present — while preserving non-target capabilities. On 50 sequential factual edits of a 14B model, Cortex-Structural recovers 88.9% of the context-induced gain while retaining 99.8% of non-target performance.

**Prereqs:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [README.md](./README.md)

---

## What it is

A unifying formulation that puts **context-based prompting**, **knowledge editing**, **skill learning**, and **behavioral steering** under the same objective:

- **Consolidation:** make the model behave on target queries as if the context were still in the prompt.
- **Retention:** keep every non-target capability intact.

The claim is that the Pareto frontier along these two axes is the right way to evaluate any method that moves information from context into weights (or back).

## How it works

Cortex has two moves.

### Prospective query generation

Rather than train on an arbitrary set of query/answer pairs about the context, Cortex **samples prospective queries** — questions that *the context would plausibly be asked about* if it were in the prompt. The model generates these queries from the context itself. The training set is `(query, response-with-context)` pairs where responses are produced by the base model running with the context in the prompt.

This matters because consolidation needs a distribution over *downstream uses* of the context; otherwise the delta over-fits to a narrow slice.

### Mergeable low-rank or structural updates

The consolidation delta is parameterized as either:
- **Cortex-LowRank:** LoRA-style low-rank updates on FFN layers.
- **Cortex-Structural:** structural (rank-selected or sparse) FFN deltas.

Both are **mergeable** — can be added to the base weights with no inference-time overhead and no routing. Multiple deltas can stack if structurally compatible.

The training objective minimizes a weighted sum of:
- **Target loss:** cross-entropy on `(query, response-with-context)` with context *removed* at train time — forces internalization.
- **Retention regularizer:** KL to the base model on a held-out non-target distribution — prevents collateral damage.

## Why it matters

- **Puts four research threads under one roof.** Prompting, model editing (ROME/MEMIT), skill learning (LoRA fine-tuning), and activation steering all become instances of MC with different parameterizations.
- **The retention number is step-change.** 99.8% non-target retention after 50 sequential edits beats prior sequential-editing methods, which typically degrade significantly past a handful of edits.
- **Cheap at inference.** Deltas merge into weights — no routing, no extra forward pass, no retrieval.
- **Scales to large / multimodal backbones.** Evaluated on 4B–120B dense and MoE backbones, text and multimodal.

## Gotchas & tricks

- **Prospective queries can be narrow.** Model-generated queries over-represent surface features of the context. Augmenting with queries from a different distribution (e.g. task-specific held-out) helps.
- **Structural deltas don't always compose.** Two deltas trained independently may conflict when merged; sequential consolidation with retention regularizer is the paper's recommended protocol.
- **Edit directionality.** Consolidation adds; the paper does not address *un*-consolidation (removing an edit).
- **Pareto position is backbone-dependent.** On smaller models (<4B) the frontier shifts — the retention gap under heavier edits widens.
- **Not an inference-time method.** Consolidation requires training. For one-shot runtime behavior, prompting remains the fallback.

## Sources

- Paper: *Cortex: Consolidating Context into Parametric Memory* — Zhang, He, Liang, Xu, Xu, Huang, Wu, Zhang (ByteDance Seed), 2026 — [arXiv:2610.20399](https://arxiv.org/abs/2610.20399). 50 sequential-edit result on a 14B model: 88.9% consolidation / 99.8% retention.
