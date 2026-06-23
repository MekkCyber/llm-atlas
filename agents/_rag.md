# Retrieval-Augmented Generation

*Taxonomy — how an agent decides which slice of an external corpus to ground a generation on.*

**TL;DR:** RAG bolts a retriever in front of an LM so generations can be grounded in a corpus that's larger than the context window or fresher than the training cutoff. The whole design space is structured by one tradeoff: **chunk granularity vs. search-space size**. Smaller chunks → more precise dense similarity but a much larger index to score; larger chunks → smaller index but noisier embeddings that mix multiple topics. The modern variants all attack this tradeoff from different angles (learn the structure, add metadata, add hierarchy, call an LLM as part of the indexer).

**Related taxonomies:** [_agent-memory](_agent-memory.md)
**Depth files covered here:** [metadata-guided-rag](metadata-guided-rag.md) · [attention-tree-rag](attention-tree-rag.md)

---

## The problem

A naïve "retrieve top-K cosine-similar chunks, dump into the prompt" pipeline breaks in two complementary ways: with fine chunks, the retriever has to score a huge candidate pool per query (latency and noise); with coarse chunks, each chunk embedding averages many topics and the similarity signal gets washed out. Either way, deep-research queries over heterogeneous corpora hit a wall.

## The shared pattern

Every RAG variant is some combination of:

1. **An indexing structure** over the corpus — flat, hierarchical, tree, graph, metadata-labeled.
2. **A retriever** — typically dense (bi-encoder), sometimes hybrid with BM25, sometimes a learned reranker on top.
3. **A retrieval policy** — top-K cosine, beam search over a tree, two-stage (filter → score), LLM-guided expansion.
4. **A generator** — the LM that conditions on the retrieved spans.

The interesting variants change (1) or (3); changing (2) is mostly an embedding-model question.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Flat dense RAG (baseline) | Fixed-size chunks, top-K cosine | Granularity tradeoff is hardcoded | Small corpus, homogeneous content |
| LLM-guided chunking | An LLM splits documents at semantic boundaries | Pays LLM cost at index time | When chunk quality matters more than index cost |
| Hierarchical summarization (RAPTOR-style) | Build a tree of LLM-written summaries | Information loss in summaries | Long documents with clear hierarchy |
| [metadata-guided-rag](metadata-guided-rag.md) | Topic metadata prunes the candidate set before dense scoring | Needs reliable topic labels | Large heterogeneous corpora ("semantic compass") |
| [attention-tree-rag](attention-tree-rag.md) | Learn a binary chunking tree from attention patterns; hierarchical beam search | More complex training | Long single documents; multi-granularity queries |
| GraphRAG | Index entities and relations as a graph | Heavy index build | Knowledge bases with clear entity structure |

## How to choose

Default to flat dense RAG with reasonable chunks (1–2 paragraphs) until you hit a concrete failure. If retrieval is *slow*, add metadata pruning (#5). If retrieval is *imprecise* on long documents, switch to a hierarchical variant (#6 or summarization). If you're indexing structured knowledge (people, places, events with relations), consider GraphRAG. Don't pay LLM-at-index-time costs until you've ruled out the cheaper variants.

## Adjacent but distinct

- **[_agent-memory](_agent-memory.md)** — agent memory often *uses* RAG as its retrieval substrate, but adds writes, governance, and lifecycle. RAG by itself is read-only over a static corpus.
- **Long-context LMs** — an alternative to retrieval at small enough scale. Becomes uncompetitive on corpus sizes > context window.

## Sources

- *MCompassRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieval* — Abaskohi et al., 2026 — https://arxiv.org/abs/2606.18508
- *SproutRAG: Attention-Guided Tree Search with Progressive Embeddings for Long-Document RAG* — Abaskohi et al., 2026 — https://arxiv.org/abs/2606.18381
- *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* — Sarthi et al., 2024 — the hierarchical-summarization baseline.
