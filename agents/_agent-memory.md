# Agent Memory

*Taxonomy — how long-horizon LLM agents store, retrieve, and update facts across many turns.*

**TL;DR:** "Memory" for an LLM agent is not the context window — it's a *data management system* that persists across runs, with four modules: **representation/storage** (chunks vs triples vs graphs), **extraction** (rollout → fact), **retrieval/routing** (query → relevant fact), and **maintenance** (update, merge, evict). No single design wins across workloads — match the structure to the bottleneck. Empirically, **localized maintenance is consistently cheaper than periodic global reorganization** (Are We Ready For An Agent-Native Memory System?, 2026).

**Related taxonomies:** [_post-training](../post-training/_post-training.md) (for memory written via SFT/RL), [_rewards](../post-training/_rewards.md) (memory-utility rewards)
**Depth files covered here:** *none yet — depth files for individual systems to be added as the field consolidates.*

---

## The problem

A vanilla LLM agent has only the context window. As tasks grow longer, the window gets evicted and the agent loses information it relied on — sometimes the plan itself (see *Plans Don't Persist*, 2026). Worse, the same fact ("user's home address") needs to survive across *runs*, not just within one. Caching the entire history is infeasible; chronological summaries lose detail; ad-hoc RAG over chat logs returns the wrong chunks.

Agent memory is the answer: a side-channel store that the agent reads from and writes to like a database, distinct from the model's parameters and from the live context window.

---

## The shared pattern — four modules

Every agent-memory system can be decomposed along four axes:

1. **Representation & storage.** What is a memory entry? Options: raw text chunk, summary, key-value pair, $\langle$subject, relation, object$\rangle$ triple, typed graph node.
2. **Extraction.** When a turn completes, what gets written? Options: append-everything, LLM-summarized, fact-extracted via prompting, structured slot-filling.
3. **Retrieval & routing.** Given the current query, which entries come back? Options: dense semantic search, BM25, symbolic graph walk, learned router that picks among them.
4. **Maintenance.** As new memories arrive, how is the store kept consistent? Options: no maintenance, periodic global rebuild, *localized maintenance* (touch only entries adjacent to the new one), TTL-based eviction.

The Are We Ready For An Agent-Native Memory System? (2026) study evaluates 12 systems on 11 datasets and shows that workload bottleneck dictates which module dominates — there is no single "best" architecture.

---

## Variants

| System / approach | Representation | Extraction | Retrieval | Maintenance | When it wins |
| --- | --- | --- | --- | --- | --- |
| RAG-over-history | Raw text chunks | Append turn-as-chunk | Dense embedding | None | One-shot QA over chat logs |
| Summary buffer | LLM-rolled summary | Periodic summarization | Always-in-context | Overwrite on re-summarize | Short tasks, fixed budget |
| MemGPT-style hierarchical | Working set + archival | LLM decides what to archive | Self-issued queries | LLM-triggered re-organization | Open-ended assistants |
| Graph memory (e.g. Mem0, Zep) | Typed nodes + edges | NER + relation extraction | Multi-hop graph walk | Localized edge updates | Multi-hop reasoning over evolving facts |
| Hybrid (graph + dense) | Graph w/ dense node embs | Mixed extraction | Router picks per query | Localized | Production agents over heterogeneous queries |
| Learned memory ops | Latent vectors | Differentiable write | Differentiable read | Trained eviction | Research; not yet practical at scale |

(*Depth files not yet written — added as systems consolidate.*)

---

## How to choose

- **Default for small assistants.** Summary buffer or RAG-over-history. Cheap, works, no infra debt.
- **Multi-hop reasoning matters.** Graph memory wins because the dense store can't follow A → B → C.
- **Facts mutate frequently (user prefs, schedules).** Graph + localized maintenance. Periodic global rebuilds are 5–10× more expensive at matched quality.
- **Long-horizon agents with planning.** Add a plan-protection layer on top — naive eviction collapses ALFWorld success by 34.7 pp (*Plans Don't Persist*, 2026).
- **Production deployment.** Hybrid retrieval; instrument cost per turn; localized maintenance.

---

## Adjacent but distinct

- **Context-window compression.** Tokenization-level tricks (LongLoRA, RoPE scaling) and KV-cache eviction operate inside one run; agent memory persists across runs.
- **RAG (retrieval-augmented generation) over fixed corpora.** Same retrieval mechanics, different content: RAG retrieves *documents*, agent memory retrieves *facts the agent itself has written*.
- **Fine-tuned facts.** Memory written into parameters via SFT/RLHF. Persistent but slow to update; agent memory is the fast-write tier.
- **Tool-call history.** The trace of past tool invocations. Often treated as memory but doesn't need a separate store — re-running the tool is usually fine.

---

## Sources

- Paper: *Are We Ready For An Agent-Native Memory System?* — OpenDataBox, 2026 — [arXiv 2606.24775](https://arxiv.org/abs/2606.24775). The four-module decomposition and 12-system benchmark.
- Paper: *Plans Don't Persist: Why Context Management Is Load Bearing for LLM Agents* — Mehta & Datta (Snowflake AI Research), 2026 — [arXiv 2606.22953](https://arxiv.org/abs/2606.22953). Shows that compression of plan state alone is dangerous; supports the case for explicit memory.
- Paper: *MemGPT: Towards LLMs as Operating Systems* — Packer et al., 2023 — [arXiv 2310.08560](https://arxiv.org/abs/2310.08560).

---

## Conventions

- **Filename:** `_agent-memory.md` (leading underscore — taxonomy).
- **Folder placement:** `agents/`, alongside future depth files for individual memory systems.
- **Scope:** persistent memory across agent runs; in-context compression belongs under inference/context management.
