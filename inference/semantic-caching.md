# Semantic Caching for LLM Serving

*Depth — cache LLM responses and computed embeddings by *semantic equivalence* (embedding similarity), not by exact-string match, to collapse redundant work across queries and sessions.*

**TL;DR:** LLM-search and LLM-agent products issue many *semantically equivalent* queries — user rephrasings, tool retries, sibling agent calls — that a lexical cache misses entirely. **Semantic caching** stores past inputs as embedding vectors and returns a cached response whenever a new input's embedding is within a cosine-similarity threshold of an existing one. Extend the same idea to URL-derived embeddings (dedupe web-content processing across sessions) and session-context storage (rolling Redis window with disk overflow) and one commodity CPU server can serve a real LLM-search workload at 89.3% cache hit rate and 0.1 ms read latency.

**Prereqs:** [README](README.md)
**Related:** [../data/deduplication.md](../data/deduplication.md), [../systems/ray.md](../systems/ray.md)

---

## What it is

Three cache layers work together in a modern LLM-search / LLM-agent serving stack:

1. **Session context cache.** A rolling window of recent conversation messages in Redis, with automatic overflow to compressed disk archives. Enables multi-day session resumability.
2. **Semantic query cache.** Cache LLM responses keyed by the query's embedding; a new query with cosine similarity above a threshold to an existing key returns the cached response instead of invoking the LLM.
3. **URL / document embedding cache.** Dedupe embedding computation for web pages across sessions — the same URL fetched by two users produces the same document embedding, which is a lot of GPU/CPU work to redo.

Together they cover three redundancy classes: same conversation (context cache), semantically equivalent queries (query cache), and same source document (URL embedding cache).

## How it works

**Semantic query cache lookup:**

```
q_emb   = embed(query)
match   = ann_index.search(q_emb, top_k=1)
if match.similarity > τ:
    return cached_response[match.key]
else:
    response = llm.generate(query)
    ann_index.insert(q_emb, key=fresh_key)
    cached_response[fresh_key] = response
    return response
```

`τ` is the similarity threshold — too low returns wrong answers; too high misses. The paper uses cosine similarity on a small embedding model.

**Session context storage:** rolling window in Redis with LRU-based migration to Huffman-compressed disk archives when idle; re-hydrate on demand.

**URL embedding cache:** hash the URL, embed once, share across all sessions that fetch it.

## Why it matters

- **Real workload numbers.** On a single 8-vCPU box (30 workers × 3 replicas), the paper reports 89.3% aggregate Redis hit rate, 0.1 ms read latency, and 1.38 MB of memory overhead.
- **CPU-only is enough for the caching tier.** Embedding + lookup can run on commodity CPUs; only generation touches the GPU.
- **Compounds across the stack.** Query cache lowers LLM invocations; URL cache lowers embedding compute; context cache lowers reprocessing per turn.

## Gotchas & tricks

- **Threshold picking is the ML problem.** Wrong-answer rate rises fast below the optimum τ. Instrument false-hit rate on a held-out set to pick.
- **Embedding drift.** If the embedding model is updated, keys and cached responses no longer match — plan a re-embedding pass or a versioned cache.
- **Stale-answer risk on volatile queries.** "What is the current price of X?" is semantically stable but the answer isn't. Tag time-sensitive query classes to bypass the semantic cache.
- **Privacy.** Semantic caches share responses across users by similarity — for anything user-specific, key by (user_id, query_embedding), not just embedding.
- **Compression is cheap on session data.** Huffman on chat text is nearly free and cuts idle-session storage by ~40%.

## Sources

- Paper: *A Three-Layer Caching Architecture for Low-Latency LLM Web Search on Commodity CPU Hardware* — Nihal Gazi, pollinations.ai, 2026 — [arXiv:2609.05463](https://arxiv.org/abs/2609.05463).
