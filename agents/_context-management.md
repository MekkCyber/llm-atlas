# Agent Context Management

*Taxonomy — the discipline of deciding what an agent holds in its reasoning context, over time.*

**TL;DR:** Production agents fail more often from what they *can't fit in context* than from what they can't reason about. The naive answer — retrieve into RAG — is a subset of the real problem: a full **lifecycle** covering architect / ingest / scope / anticipate / compact-and-consolidate. The taxonomy below organizes techniques by which lifecycle stage they target.

**Related taxonomies:** none yet (this is the entry taxonomy for `agents/`).
**Depth files covered here:** [context-compaction](./context-compaction.md)

---

## The problem

Long-running or high-tool-count agents accumulate context every turn — conversation history, tool schemas, tool outputs, retrieved documents. Naive accumulation is **quadratic in cost** in conversation length. Cutting via crude summarization is linear but loses fidelity (the *summarization cliff*). Framed as a storage-and-retrieval problem, RAG only touches ingest+scope and leaves the rest ad hoc. Framed as a lifecycle, every stage has a distinct technique family.

## The shared pattern

Every technique is a **decision about which tokens are alive right now**. They differ in *when* the decision is made (build time, ingest time, per turn, mid-generation) and *what evidence* it uses (schema, embedding similarity, LLM judgment, provenance metadata, engine metadata).

## Variants

| Stage | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Architecting | Route each data-type to a store designed for it (KV, vector, graph, doc DB). | Ops complexity | Multi-modal, multi-tenant agents. |
| Ingesting | Structured extraction of turn artifacts before they hit the store. | Extra LLM cost at write time | High-value, low-turn-volume conversations. |
| Scoping | Enforce org / user / session boundaries on retrieval. | Retrieval complexity | Enterprise / multi-tenant deployments. |
| Anticipating | Predict what the *next* turn will need; pre-fetch it. | Wasted work if wrong | Long horizons with predictable sub-tasks. |
| [context-compaction](./context-compaction.md) | Validated compaction: replace turns with a summary while measurably preserving recall. | Compaction LLM cost | The default fallback when the window is filling. |
| KV-cache retrieval (see [../inference/kv-cache-retrieval.md](../inference/kv-cache-retrieval.md)) | Bring past chunks back into the KV cache instead of re-prompting. | Requires trainable/known correspondence | Long-horizon autoregressive generation with revisit structure. |

## How to choose

- Start with **scoping + architecting** — pick the right store per data type, enforce boundaries.
- Add **validated compaction** as soon as the window is projected to overflow. Do *not* fall back to naive summarization.
- Reach for **anticipation** only when the horizon is predictable enough to justify pre-fetch cost.
- **KV-cache retrieval** is the systems-side counterpart when the model itself needs old context back mid-generation.

## Adjacent but distinct

- [../inference/README.md](../inference/README.md) — prefix-caching and KV eviction are systems-side controls on active tokens, not lifecycle decisions.
- Retrieval-augmented generation (classic RAG) is only the *ingest + scope* slice of this taxonomy.

## Sources

- Paper: *Agentic Context Management: Solving Agent Memory and Cost by Treating Them as Lifecycle and Architecture Problems* — Maximem, 2026 — [arXiv:2607.21503](https://arxiv.org/abs/2607.21503). Originates the five-primitive decomposition and the reference implementation (Maximem Synap).
