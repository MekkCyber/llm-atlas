# Agent Memory
*Taxonomy — how LLM agents remember across long interactions and sessions.*

**TL;DR:** Agents need memory to act consistently over long horizons, but "memory" spans multiple mechanisms with very different cost/quality tradeoffs. The main choice: whether memory operations invoke an LLM (summarisation, decide-what-to-store, decide-what-to-retrieve) or run without LLM calls (structured indices over raw traces). The frontier is moving toward hybrids — LLM-mediated for high-value writes, deterministic for reads.

**Related taxonomies:** none in this folder yet
**Depth files covered here:** [zero-token-memory](zero-token-memory.md)

---

## The problem

Agents run tasks that require context spanning many turns, days, or sessions. The naïve solution — stuff everything into one growing context — hits three walls:

1. **Context budget.** Even 200k-token windows run out on multi-day work.
2. **Attention degrades with length.** "Lost in the middle" effects mean information deep in a long context is effectively invisible.
3. **Cost.** LLM tokens are the dominant cost of any nontrivial agent stack; every retained turn is billed again on every subsequent step.

Memory systems compress or externalise state to escape all three.

## The shared pattern

```
raw interaction log ──▶ [WRITE-PATH: what to store, how to index]
                                                   │
user query ──▶ [READ-PATH: what to retrieve, how to score]
                                                   │
                                              ▼
                          reader LLM → answer
```

Every variant has:
- A **write path** — from raw turns to storable representation.
- A **read path** — from a query to retrieved evidence.
- A **reader** — the LLM that produces the final answer.

Variants differ mainly in *whether the write and read paths invoke LLMs at all*, and *what representation the store uses*.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| **Full-context replay** (baseline) | Just keep every turn in the LM context | Cost and attention degrade with length | Very short-horizon agents |
| **Sliding window + summary** (ChatGPT-style) | Truncate old turns, replace with an LLM-written summary | Summarisation loses detail; every write costs an LLM call | General chat over medium horizons |
| **MemGPT-style OS memory** (Packer 2023) — *no depth file yet* | Model actively pages memory in/out via tool calls | LLM in the loop for both read and write | Long-horizon single-agent tasks with dense recall |
| **Vector-store RAG memory** (LangChain-style) — *no depth file yet* | Chunk turns, embed, retrieve by cosine similarity | Loses structure; over-fetches | Semantic recall over document-like traces |
| **Graph-based memory** (e.g. Graphiti, Zep) — *no depth file yet* | Entity–relation graph over interaction facts | Extraction step is LLM-driven | When entity linking across sessions matters |
| [**Zero-token memory**](zero-token-memory.md) (Zero-Mem, 2026) | Structured indices over raw traces, no LLM calls in memory ops | Loses LLM-driven abstraction | Cost-sensitive long-memory QA |

## How to choose

**Default for a new agent (2026):** entity-graph + temporal-hierarchy indices over the raw log, LLM-mediated only for the final answer. This is the [Zero-Mem](zero-token-memory.md) shape and it dominates on cost while matching heavier baselines on quality.

**When to add LLM-mediated writes:** if your traces contain long unstructured prose (interviews, meeting transcripts) that benefits from summarisation before retrieval.

**When to keep it simple:** for short-lived agents (single-task, single-session), plain sliding-window + summary is fine — the fixed overhead of a memory system isn't worth it.

**Compose, don't replace.** A frontier stack often runs vector retrieval, graph retrieval, and hierarchical time-slicing in parallel and fuses the results. Different retrieval angles catch different queries.

## Adjacent but distinct

- **KV-cache management** (paged attention, prefix caching) — memory of the *transformer* over a single generation, not of the agent across turns. See `../inference/`.
- **Fine-tuning as memory** — bakes information into weights. Complementary but different lifecycle (offline, not per-session).
- **Retrieval-augmented generation over external corpora** — RAG memory over agent-conversation logs is a subset of RAG generally, but the queries and access patterns differ enough to treat separately.

## Sources

- Paper: *MemGPT: Towards LLMs as Operating Systems* — Packer et al., 2023 — LLM-mediated memory paging.
- Paper: *Zero-Mem: Zero-Token Memory Operations for LLM Agents* — arXiv:2607.29377, 2026 — no-LLM-call memory ops.
- Blog: *Zep Graphiti* — 2024 — graph-based agent memory.
- Paper: *Lost in the Middle* — Liu et al., 2023 — the attention-degradation motivation.
