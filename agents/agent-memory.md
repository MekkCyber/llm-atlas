# Agent memory (typed records + provenance hypergraph)
*Depth — long-running LLM-agent memory as a typed record store indexed by a provenance hypergraph, read under explicit retrieval budgets.*

**TL;DR:** Instead of summarizing observations at write time (lossy) or dumping a growing context window (slow and noisy), store every observation as a typed record and index it with a hypergraph that captures how facts relate. Reads happen under a retrieval *budget* that bounds latency and token cost. An offline consolidator (AutoDream) compresses and re-indexes outside the write path. MIRIX-2.0 reports 91.4% on LoCoMo and 72.6% on LongMemEval-S with no fine-tuning.

**Prereqs:** [./README.md](./README.md)
**Related:** [./README.md](./README.md)

---

## What it is

An agent-memory architecture for agents that live for days, months, or longer. Three design choices separate it from simpler scratchpad or summary-buffer memories:
1. **Typed records** (observation, decision, user-turn, tool-result) as primary store.
2. **Provenance hypergraph** linking records that share entities, causes, or user intents.
3. **Budgeted retrieval** at read time — fixed token/latency cap, enforced at the query layer.

## How it works

**Write path (hot).** Every event is appended to the typed-record store with its origin, timestamp, and type. No summarization, no re-ranking — latency-cheap. Entities and relations detected on the event create hyperedges in the provenance graph.

**Read path (hot).** A retrieval query specifies a budget `B` (tokens or ms). The hypergraph is traversed from entity anchors in the current query, scoring candidate records by relation weight and recency; the top records are returned until `B` is exhausted. Deterministic, bounded latency.

**Consolidation (cold, async).** AutoDream runs outside the critical path: it re-ranks the hypergraph, compresses long chains into derived records, prunes obsolete contradictions, and promotes high-utility patterns to a fast-access tier. Analogous to sleep-time replay in biological memory.

## Why it matters

Agent-memory designs fall apart on two failure modes: (1) summaries that drop the detail a later task needs, and (2) context dumps that overwhelm the model and inflate cost. Splitting the system across a complete-record store, a hypergraph index, and an offline consolidator decouples the three concerns and gives explicit cost knobs. Reported SOTA on four of five long-memory QA benchmarks at inference time, no fine-tuning.

## Gotchas & tricks

- **Budget tuning dominates quality.** Too small a `B` misses relevant history; too large wastes tokens and confuses the model. Report tunes `B` per task class.
- **Hypergraph construction** is the main engineering cost — entity extraction, coreference, and relation typing on every event.
- **Consolidation lag** matters: if AutoDream is too slow, the hot path still reads stale compressions. Report runs it continuously with bounded concurrency.
- Works with open and closed models — the memory is external and the LLM is called through its normal interface.

## Sources

- Paper: *MIRIX-2.0: Multi-Agent Memory System* — Zhang, Nan, Xia, Wang (MIRIX AI), 2026 — [arXiv:2611.12068](https://arxiv.org/abs/2611.12068)
