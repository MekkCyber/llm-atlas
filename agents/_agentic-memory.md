# Agentic Memory

*Taxonomy — long-horizon-agent memory systems: how an agent stores, organizes, and retrieves state across turns / sessions / tasks.*

**TL;DR:** Long-horizon agents outgrow the context window: they need a store of past observations, decisions, and derived facts they can query later. Every variant makes three choices — **what to write**, **how to organize what's written**, and **what to read at query time** — and increasingly, *who* makes those choices (a full LLM vs. a lightweight controller). The modern default is a graph- or entity-indexed store with an LLM-controlled write path; the emerging frontier is offloading the read/route logic to a small "System-One" controller so the LLM only fires for genuine reasoning.

**Related taxonomies:** [../post-training/_post-training.md](../post-training/_post-training.md) *(training an agent to use memory well)*
**Depth files covered here:** *(none yet — this taxonomy predates its depth files; slots reserved for `memgpt.md`, `zep.md`, `jev-mem.md` as they land)*

---

## The problem

A conversation, a task, a codebase, a persistent user relationship — all outrun a fixed context window. Naive strategies fail predictably:

- **Concatenate everything** → hits the context limit; irrelevant history distracts.
- **Fixed-size sliding window** → loses long-range facts.
- **Summarize on every turn** → lossy; the summary drifts; expensive.

The class of *agentic memory systems* is anything that separates "what the agent knows" from "what fits in the current prompt", plus a policy for moving between the two.

## The shared pattern

Every variant has the same three moving parts:

1. **Write path** — deciding *what* to store (raw turns, extracted entities, distilled facts, plans, …) and *how* (chunks, entities, edges, embeddings, symbolic records).
2. **Store organization** — flat log, vector index, KV store, entity/relation graph, layered hierarchy.
3. **Read path** — how a query at turn $t$ pulls the right subset back: top-K vector search, graph traversal, template retrieval, hybrid.

An orthogonal axis, increasingly central: **who** runs each path.

- **LLM-in-the-loop.** Each write and each retrieval decision is an LLM generation call. Rich behavior, high latency, high cost.
- **Controller-in-the-loop (System-One / System-Two split).** A small learned or rule-based controller handles routing, budgeting, and traversal; the LLM is invoked only for reasoning and answer synthesis.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| MemGPT-style *(no depth file yet)* | LLM decides what to page in/out of a virtual context | Rich, general | Long, unstructured chat with a single agent |
| Zep / graph-of-facts *(no depth file yet)* | Extract entities + relations into a temporal knowledge graph; retrieve by graph query | Structured recall, cheap read | Repeated queries about people/entities/events |
| Vector store + RAG *(no depth file yet)* | Chunk, embed, top-K retrieve at query time | Simplest to implement | Documents, FAQs, one-shot factual lookup |
| LettaMem / hierarchical summaries *(no depth file yet)* | Multi-scale summarization: recent verbatim, older summarized | Handles very long sessions | Long single-session chats |
| Jev-Mem *(no depth file yet)* | System-One controller for typing / routing / budgeting; System-Two only for answers | Fast read+write; less flexible | High-throughput agents where LLM calls dominate cost |

## How to choose

Start with **vector-store RAG** for anything with a stable corpus and single-shot queries — it's the cheapest thing that works.

Move to a **graph/entity store** (Zep-style) once the agent's job is to remember relations *between* things (people mentioned, events, decisions) and answer questions across them.

Move to a **paged / virtual-context** design (MemGPT-style) once the workload is long, unstructured conversation with a single agent and the retrieval decisions are themselves task-relevant.

Move to a **controller-in-the-loop** design (Jev-Mem-style) once LLM invocations on the memory hot path are the dominant cost — every read or write invocation you can push to a small controller is money and latency saved without much quality loss.

Combine liberally: production systems often stack graph + vector + working-buffer, with a controller deciding which store to hit.

## Adjacent but distinct

- **Retrieval-augmented generation (RAG)** — a static corpus, read-only for the model. Agentic memory writes back into itself over time.
- **KV cache** — architectural, per-request, invisible to the agent's logic. Agentic memory is application-level.
- **Prompt compression / context distillation** — compresses *within* a single context; agentic memory persists *across* contexts.
- **Fine-tuning / continual learning** — bakes information into weights instead of writing into a store. Complementary, much slower loop.

## Sources

- Paper: *Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents* — Jiang, Li, Li — 2026 — the controller-driven design pattern. [arXiv:2609.23986](https://arxiv.org/abs/2609.23986)
- Paper: *MemGPT: Towards LLMs as Operating Systems* — Packer et al., 2023 — the LLM-as-memory-manager pattern.
- Paper: *Zep: A Temporal Knowledge Graph Architecture for Agent Memory* — Rasmussen et al., 2024 — the graph-of-facts pattern.
- Benchmark: *LoCoMo* — the long-conversation memory eval most variants report against.
