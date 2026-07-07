# GraphRAG
*Depth — retrieval-augmented generation over graph-structured knowledge for frozen LLMs.*

**TL;DR:** GraphRAG feeds graph-structured knowledge — property graphs, knowledge graphs, code graphs — to a frozen LLM so the model can answer over relations and multi-hop paths that plain text embeddings destroy. The core challenge is the *representation gap*: the graph encoder's latent space and the LLM's text embedding space don't align, and sparse graphs break BERT-style masked SSL because a handful of "key" nodes carry most of the signal. AGE (Hashimoto & Nguyen, 2026) fixes both with a Transformer graph encoder trained via masked SSL plus a **learnable node sampler** that avoids masking the dominant nodes.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../multimodal/README.md](../multimodal/README.md), [README](README.md)

---

## What it is

Standard RAG maps documents into a dense vector space; retrieval picks the top-k. It works well on unstructured text but flattens *structure* — relational schemas, entity-relation triples, code call graphs — that makes some knowledge bases useful in the first place. GraphRAG keeps the graph structure explicit: the retriever pulls subgraphs (nodes + edges) matching the query, and those subgraphs are serialized as context for the LLM.

The frozen-LLM constraint matters. Fine-tuning the LLM to consume graphs would work but requires per-domain training. GraphRAG-for-frozen-LLMs asks: given a fixed LLM whose text-side embeddings we cannot touch, how do we build a graph encoder whose outputs the LLM can use as context?

## How it works

**Two-part pipeline.**

1. **Graph encoder.** Maps nodes (and optionally edges / paths) to embeddings. The embeddings must sit in a space compatible with what the LLM expects to see — either literally aligned to text embeddings or serialized into a text form the LLM can prefix-attend over.
2. **Retriever + serializer.** At query time, embed the query, retrieve top-k subgraphs, serialize them (nodes / edges as text, or embedding tokens injected via cross-attention), and prefix the LLM's context.

**Training the graph encoder — the masked-SSL twist.** AGE's contribution is a training recipe for step 1. It uses a BERT-style masked-modeling objective — mask a subset of nodes, predict them from context. The twist: naïve node masking hits a wall on sparse graphs because a small number of *key* nodes hold most of the contextual weight (think: hub nodes in a knowledge graph, `main` entry-points in a code graph). Predicting a key node from the tail is essentially impossible; wasted training signal.

AGE learns a **node sampler** that concentrates the mask distribution on *non-key* nodes — ones whose surroundings actually predict them. Formally, the sampler is a small learned network that scores each node for "how predictable from context I am"; the mask is drawn from this distribution. Reconstruction loss is well-defined, gradient signal is dense, and the encoder converges to embeddings that align better with the text-side embedding structure.

**Downstream integration.** Once trained, the graph encoder can plug into any GraphQA pipeline. AGE reports gains on non-parametric-search GraphQA methods across four benchmarks with distinct graph characteristics.

## Why it matters

- **Frozen-LLM story is where production sits.** Most enterprise deployments are on top of a closed frontier model or a fine-tuned but stable base. A graph encoder they can train without touching the LLM removes the biggest deployment blocker.
- **Adaptive masking generalizes.** Any SSL objective on a sparse or heavy-tailed structure hits the "dominant items are impossible to predict" wall. Learned samplers are the general answer — applicable to code-graph pretraining, entity-linking, structured document retrieval.
- **Grounds an active research area.** GraphRAG has been fragmented across GNN-based and LLM-driven approaches; AGE is a clean statement that the encoder is the bottleneck and text-aligned representation is the objective.
- **A real alternative to schema flattening.** Many enterprise RAG stacks flatten SQL / KG data into dense embeddings, which loses composition. GraphRAG preserves schema; AGE makes it competitive.

## Gotchas & tricks

- **The sampler is only as good as its "key-node" signal.** If the graph is homogeneous (no hubs), adaptive masking degenerates to uniform. In practice most real graphs are heavy-tailed enough that the sampler adds signal.
- **Alignment vs. serialization.** Two integration modes exist: *align* the graph encoder's output to the text embedding space (cross-attention or projection), or *serialize* the retrieved subgraph as text and let the LLM handle it. Alignment saves tokens but requires access to internal embeddings; serialization works against any API-only LLM.
- **Retrieval budget matters.** GraphRAG subgraphs can blow context — a subgraph of 100 nodes with 3-hop neighborhoods serializes to thousands of tokens. Truncation / summarization at retrieval time is essential.
- **Graph freshness.** Property graphs mutate. The encoder must be retrained (or continually trained) as the underlying graph evolves; adaptive masking helps stability across retrainings.
- **Not a substitute for query planning.** GraphRAG improves what the LLM sees, but multi-hop queries still benefit from explicit query-plan generation (dispatch to Cypher/SQL) — the two are complements, not substitutes.

## Sources

- Paper: *AGE: Adaptive-masking for Graph Embedding in Graph Retrieval-Augmented Generation* — Hashimoto & Nguyen, OMRON SINIC X, 2026 — [arXiv:2607.00052](https://arxiv.org/abs/2607.00052)
- Related: *GraphRAG* (Microsoft, 2024), *G-Retriever* (He et al., 2024), *KG-RAG* survey (2025) — landscape of graph-augmented retrieval.
