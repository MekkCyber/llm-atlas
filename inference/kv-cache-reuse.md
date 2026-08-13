# KV Cache Reuse (Chunk and Nugget)
*Depth — the family of KV-cache reuse techniques for RAG-style workloads.*

**TL;DR:** In long-context RAG, prefill dominates cost. **KV cache reuse** ships precomputed key/value tensors for the retrieved documents instead of re-encoding them per query. The naive granularity is **chunk-level** — cache the full retrieved chunk. **CoinRAG (2026)** pushes to **nugget-level** — precompute caches for *semantic units* inside chunks, retrieve which nuggets to reuse with a two-stage retrieval, and reassemble their sliced KV representations at query time. Cuts prefill while improving answer quality on multi-hop QA.

**Prereqs:** [README.md](README.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

A vanilla LLM prefill re-encodes every token in the context on every request. For RAG deployments where the same documents are retrieved often, the same tokens get re-encoded over and over.

**KV cache reuse** family:

1. **Prefix caching.** Cache the KV of a shared prompt prefix. Widely deployed (vLLM, SGLang defaults).
2. **Chunk-level RAG cache reuse.** Precompute KV for each retrieved chunk offline; concatenate the cached chunks + fresh query at request time. Cuts prefill for the reused portion.
3. **Nugget-level reuse (CoinRAG).** Chunks contain redundant / noisy tokens. Precompute KV for smaller *semantic nuggets* (sub-chunk units); retrieve which nuggets a query actually needs and stitch their KV slices together.

Each step trades off cache granularity, retrieval complexity, and quality. Nugget-level is the newest step and the current Pareto frontier for multi-hop long-context RAG.

## How it works

**Chunk-level (baseline).**
- Offline: for each chunk, run the model over the chunk and save its KV.
- Online: retrieve top-$k$ chunks, concatenate their cached KV in the position layout the model expects, prefill only the query, generate.

**Nugget-level (CoinRAG).**
- Offline: **decompose each chunk into semantic nuggets** (query-typed spans). For each nugget, cache its KV *with* the surrounding chunk-level context that gives it meaning.
- Online **two-stage retrieval**:
  1. Retrieve top chunks for the query (coarse recall).
  2. Within selected chunks, retrieve the specific nuggets that answer the query (fine precision).
- **Assembly:** take the sliced KV of each retrieved nugget, place them in the model's position layout together with a lightweight chunk-level context, prefill query, generate.

**Key subtlety.** A nugget's KV depends on its surrounding context (attention isn't local). Reusing a raw isolated-nugget KV would degrade quality. CoinRAG's cache is **contextualized** — the offline pass encodes each nugget with its chunk context so the reused KV is a valid slice of a real long-context prefill.

## Why it matters

- **Prefill dominates RAG cost.** Reusing precomputed KV directly attacks the biggest line item.
- **Nugget-level improves quality *and* efficiency.** On LongBench multi-hop QA, CoinRAG delivers **+5.3% average relative F1** while cutting prefill — a new Pareto frontier vs chunk-level caching.
- **Composes with other inference optimizations.** KV reuse is orthogonal to speculative decoding, paged attention, and continuous batching.

## Gotchas & tricks

- **Position ID handling.** Reused KV was computed at some original position; you must map it to the query-time position layout consistently, or attention gets confused.
- **RoPE and position-dependent encodings.** RoPE-encoded KV embeds position; reusing at a different position needs a re-rotation. Ignore this and quality collapses.
- **Cache invalidation.** When documents change, the cache is stale. Invalidate on write and eat the recompute cost.
- **Nugget decomposition is the fragile part.** Bad nuggets (too small, over-fragmented) hurt both retrieval quality and KV quality.
- **Memory pressure.** Cached KV is large — orders of magnitude bigger than the token text. Budget disk / GPU-memory accordingly.
- **Model-specific.** KV formats differ across model families; caches are not portable across architectures.

## Sources

- Paper: *CoinRAG: Contextualized Information Nugget KV Cache Reuse for Long-Context RAG* — Kim, Park, Yang (UC Santa Barbara), arXiv 2608.07458, 2026.
- Related lines: prefix caching (vLLM / SGLang), chunk-level RAG caching (multiple prior systems). See the paper's related work section.
