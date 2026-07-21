# Multi-Step GraphRAG
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** GraphRAG augments an LLM's retrieval with a **knowledge graph** built from the corpus (entities + typed relations). Naïve single-pass extraction produces noisy nodes and brittle retrieval; **multi-step GraphRAG** separates extraction from consolidation with a fixed pipeline — typed extraction → semantic dedup → LLM summarization → community detection — giving cleaner graphs and more stable retrieval. Introduced at engine scale by RAGU (2026).

**Prereqs:** [../data/deduplication.md](../data/deduplication.md), [../data/_data-curation.md](../data/_data-curation.md)
**Related:** [skill-libraries.md](skill-libraries.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

Standard RAG retrieves *chunks* by embedding similarity. GraphRAG retrieves via a graph whose nodes are entities and whose edges are typed relations, giving the LLM structure it can reason over rather than a bag of chunks. Multi-step GraphRAG is the engine pattern that turns a raw corpus into a *usable* graph: a five-stage pipeline where each stage's output is validated before the next runs.

## How it works

The RAGU pipeline (canonical multi-step):

1. **Typed two-stage extraction.** First pass extracts entity/relation candidates with types; second pass validates and normalizes. Two passes catch the class of errors that single-pass extraction silently commits (wrong-type entities, hallucinated relations).
2. **DBSCAN-backed dedup.** Cluster entity mentions in embedding space; collapse each cluster to a canonical entity. Robust to spelling variants, aliases, and near-duplicates in ways exact-string dedup isn't.
3. **LLM summarization.** For each canonical entity, produce a compact description synthesized from all mentions. This is the retrieval-time signal — richer than a raw span, cheaper than the full context.
4. **Leiden community detection.** Cluster the graph into topical communities; store per-community summaries. Retrieval can now return a community rather than an individual node, giving the LLM broader context.
5. **Query-time.** Given a user query, embed → hit relevant entities/communities → hand the LLM the community summaries plus the raw edges as context.

RAGU's second insight is architectural: **the extractor doesn't need world knowledge**, just language skills (comprehension, extraction, in-context reasoning). A compact 7B extractor specialized on those skills (Meno-Lite-0.1) matches Qwen2.5-32B on KG construction — 5× smaller, same quality.

## Why it matters

Single-pass GraphRAG is the default and produces silently-bad graphs — noisy nodes, half-typed edges, duplicated entities under different aliases. The failure mode is invisible at extraction time and only surfaces as poor retrieval. Multi-step separates concerns so each failure mode is caught in its own stage. Cheap-extractor result also means GraphRAG pipelines can run on consumer hardware, not a cluster.

## Gotchas & tricks

- **Community detection is O(V log V)** — fine on 100k-node graphs, painful on 10M-node ones. For very large graphs, run Leiden on a filtered subgraph or use hierarchical detection.
- **Dedup threshold matters more than dedup algorithm.** Too tight: aliases stay split. Too loose: distinct entities collapse. Tune per-domain.
- **The typed extractor is where quality is set.** A better dedup + summarization can't rescue a bad extraction pass. Invest here first.
- **Summaries drift.** Re-summarize when the underlying entity gets substantially updated; stale summaries poison retrieval quietly.
- **Query-side selection matters as much as ingestion.** Retrieving a whole community can flood the context; a routing step (entity vs community vs raw edges) is worth the extra LLM call.

## Sources

- Paper: *RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM* — Komarov et al., 2026 — [arXiv:2607.11683](https://arxiv.org/abs/2607.11683).
- Foundational: *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — Edge et al., Microsoft, 2024 — the original GraphRAG framing.
