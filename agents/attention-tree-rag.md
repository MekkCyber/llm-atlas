# Attention-Tree RAG

*Depth — building a hierarchical chunking tree from learned inter-sentence attention, retrieved via beam search.*

**TL;DR:** A RAG approach that learns a *binary chunking tree* over sentences from the retrieval model's own attention patterns, then runs a hierarchical beam search at retrieval time to return matches at the granularity (sentence, sub-tree, branch) the query actually needs. Trained end-to-end with a joint embedding + tree-structure objective. Introduced as SproutRAG (Abaskohi et al., 2026). Avoids both LLM-guided chunking (expensive) and hierarchical summarization (lossy).

**Prereqs:** [_rag](_rag.md)
**Related:** [metadata-guided-rag](metadata-guided-rag.md)

---

## What it is

Three components:

1. **A binary chunking tree** over the document's sentences. Leaves are individual sentences; internal nodes are spans formed by merging children. Each node carries a learned embedding.
2. **A learned construction signal.** Specific attention heads / layers of the retrieval encoder are designated as "structure heads"; the model learns *which* heads best capture semantic-coherence between adjacent sentences and uses those scores to decide the merge order.
3. **A hierarchical beam search** at retrieval time that traverses the tree, returning whichever node (leaf, sub-tree, branch) has the strongest match — so the retriever returns the right *granularity* per query, not just the right location.

Training is end-to-end with one joint loss: the embeddings have to be discriminative *and* the induced tree has to align with the supervised retrieval targets.

## How it works

Indexing:

```
Document → sentences → encoder forward pass → attention from designated structure heads
                                                      │
                                                      └─→ greedy / DP merge over highest-attention adjacent pairs
                                                              │
                                                              └─→ binary tree with node embeddings = (mean of leaves) or learned aggregator
```

Retrieval:

```
Query embedding → beam search over the tree:
                  visit root → choose between {root, left child, right child} based on similarity
                  expand the best partial paths up to beam width B
                  return top-K nodes (mixed-granularity)
```

No LLM calls at index time or retrieval time — pure encoder + structured search.

## Why it matters

The granularity tradeoff (small precise chunks vs. large coherent chunks) becomes a non-tradeoff if the index can return matches at *any* level: short queries land on sentences, scoped queries land on sub-trees, document-level queries land near the root. Reported result: **+6.1% average information-efficiency** over the strongest baseline across four benchmarks (scientific, legal, open-domain), with no per-query LLM cost.

## Gotchas & tricks

- **Structure-head selection is a hyperparameter.** Different domains land on different heads; the model has to be trained per-domain or with enough domain diversity to learn robust structure detection.
- **Tree depth scales with document length.** For very long documents (book-length), beam width has to grow or the search misses deep matches.
- **End-to-end training is non-trivial.** The tree-structure loss has to be differentiable (or approximated, e.g. soft trees / Gumbel) to backprop through the merge decisions; details matter.
- **Composes with metadata pruning.** Attention-tree RAG handles intra-document structure; metadata-guided RAG ([metadata-guided-rag](metadata-guided-rag.md)) handles inter-document pruning. They're complementary.

## Sources

- Paper: *SproutRAG: Attention-Guided Tree Search with Progressive Embeddings for Long-Document RAG* — Abaskohi, Laradji, West, Carenini, 2026 — https://arxiv.org/abs/2606.18381
