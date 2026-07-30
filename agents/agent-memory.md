# Agent Memory
*Depth — long-term memory subsystems for LLM agents and their retrieval failure modes.*

**TL;DR:** An "agent with memory" stores information about the user or environment in an external store and retrieves it when a related query arrives. The interface hides a strong assumption: *the memory needed will look like the query that needs it*. This works for direct-cue recall ("what's my dog's name?") and breaks for **implicit associations** ("can I eat this macaron?" needing "I'm allergic to tree nuts" via the almond-flour inference).

**Prereqs:** [README](README.md)
**Related:** [agentic-search](agentic-search.md)

---

## What it is

A separable subsystem attached to an LLM agent that (1) writes salient facts to a persistent store as the agent runs, and (2) reads relevant facts back on demand. Common designs:

- **Raw store + dense retriever.** Log each user utterance verbatim; embed and retrieve by cosine similarity to the query.
- **Summarized store.** LLM condenses past interactions into short "memories"; retriever queries those.
- **Structured store.** Extract entities/attributes to a graph or DB; query by symbolic match.
- **Hybrid.** All of the above with a rerank / fusion layer.

Memory is orthogonal to the agent's *tools* and *planner* — it feeds the context window, not the tool loop.

## How it works

```
on user message m:
    facts_new = extract(m)                       # what to remember
    store.append(facts_new)

on query q:
    candidates = store.retrieve(q, k)            # similarity to q
    context = rank_and_pack(candidates, budget)
    response = llm.answer(q, context=context)
```

The retriever is the choke point. Every popular retriever — BM25, dense, hybrid, LLM-summary — scores memories by their **query-visible similarity** to the incoming query.

## Why it matters

- Agents that "remember the user" are one of the most product-visible LLM features.
- Memory failures cause the model to *contradict the user* on facts it was explicitly told — a trust-destroying error class.
- The right memory design determines whether an agent can carry a preference across weeks or resets it on every session.

## Gotchas & tricks

- **The implicit-association blind spot.** Retrievers work by textual/semantic similarity to the query. A stored memory that requires world-knowledge inference to *connect* to the query is invisible to the retriever. Example: "tree-nut allergy" and "macaron" share no cue; almond-flour is the bridge. See the [InMind benchmark](https://arxiv.org/abs/2607.24368) for a 125-task audit — current systems collapse on this class.
- **Write-time vs. read-time cost.** Aggressive extraction at write time gives a smaller, cleaner store but costs LLM calls on every user message. Deferred extraction is cheap but leaves noise in the store.
- **Recency vs. relevance.** Pure similarity retrieval surfaces a highly relevant memory from a year ago over a moderately relevant one from yesterday. Most systems inject a recency prior.
- **Contradiction handling.** New facts can contradict stored ones ("my favorite color is blue" → later "my favorite color is red"). Naive append leaves both in the store; the retriever surfaces both; the LLM has to choose. Explicit versioning or supersede rules help.
- **Privacy failure surface.** A memory store is a durable record of user disclosures; retrieval brings them into completions where they may leak. Treat memory as sensitive data by default.
- **Test with hostile queries.** Evaluate on queries that require memories the retriever *can't* see are relevant. Retrieval-only ablations give inflated scores.

## Sources

- Paper: *Keep It InMind: Benchmarking the Implicit-Association Blind Spot in Agent Memory* — 2026 — [arXiv:2607.24368](https://arxiv.org/abs/2607.24368) — names the failure mode and provides a 125-task expert-verified benchmark.
- Related: MemGPT, MemoryBank, LangGraph memory — production-oriented memory designs (see linked papers/repos).
