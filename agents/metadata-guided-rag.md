# Metadata-Guided RAG

*Depth — using topic-level metadata as a navigational layer over paragraph chunks.*

**TL;DR:** Standard RAG forces a single granularity choice: small chunks are precise but blow up the search space; large chunks shrink the index but average multiple topics into one noisy embedding. **Metadata-guided RAG** keeps small (paragraph-level) chunks for precision but adds a *topic-metadata index* on top so the retriever first prunes the candidate pool to the right topic, then runs dense similarity only on the pruned subset. Introduced as MCompassRAG (Abaskohi et al., 2026), which calls the metadata layer a "semantic compass."

**Prereqs:** [_rag](_rag.md)
**Related:** [attention-tree-rag](attention-tree-rag.md), [_agent-memory](_agent-memory.md)

---

## What it is

A two-stage retrieval pipeline:

1. **Topic-level prune.** Each chunk in the corpus is labeled with one or more topic tags (or a topic embedding distinct from its content embedding). At query time, the query's topic intent is matched against this label index, yielding a much smaller candidate subset.
2. **Paragraph-level score.** Dense cosine similarity runs over the pruned subset only — same retriever architecture as flat RAG, just on a corpus 10–100× smaller in practice.

The topic labels can come from a topic classifier, an LLM at index time, or an extraction over document metadata (title, section headings, tags). Indexing cost is one-time; query-time cost is dominated by the small dense score pass.

## How it works

Indexing pipeline:

```
Document → paragraph chunks → (content embedding, topic label) per chunk → two indexes
                                       │
                                       ├─→ content index (HNSW / IVF over dense embeddings)
                                       └─→ topic index (inverted index on labels)
```

Query pipeline:

```
Query → (content embedding, topic intent)
                │                    │
                │                    └─→ topic index → candidate IDs (~10² of ~10⁶)
                │
                └─→ score content embedding only against candidate IDs → top-K
```

The "topic intent" can be extracted by the same model that produces the content embedding (joint training), or by a separate light-weight classifier on the query text.

## Why it matters

The granularity tradeoff has been the dominant cost driver in production RAG: precision needs small chunks but small chunks need lots of storage and lots of similarity scoring per query. Metadata pruning *separates* the precision concern (still at paragraph granularity) from the search-space concern (now at topic granularity), and you only pay the dense-similarity cost on a small filtered set. Reported wins on deep-research benchmarks across heterogeneous corpora (mixing scientific, legal, and open-domain text).

## Gotchas & tricks

- **Topic-label quality is the bottleneck.** A noisy topic classifier produces a noisy prune; you lose recall before dense similarity even sees the right chunk. Audit prune coverage on held-out queries.
- **Multi-topic chunks.** A chunk that's genuinely about two topics needs to be indexed under both, or the wrong-topic query loses it. Multi-label > single-label.
- **Topic drift over time.** The label taxonomy needs maintenance as the corpus grows; otherwise new content gets assigned to coarse legacy topics and pruning gets less effective.
- **Combine with hybrid retrieval.** The topic prune composes cleanly with BM25 reranking on the pruned set, since the search space is now small enough that BM25 cost is negligible.

## Sources

- Paper: *MCompassRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieval* — Abaskohi, Li, Cimino, West, Carenini, Laradji, 2026 — https://arxiv.org/abs/2606.18508
