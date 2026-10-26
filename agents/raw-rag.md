# Raw-RAG

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The simplest possible agentic-memory baseline: **dense retrieval + reranking over the raw conversation history**, with no extraction, summarization, mutation, or graph. On a controlled study across LoCoMo, LongMemEval, MEME, and MemoryAgentBench, Raw-RAG beats more complex "agentic memory" systems in 6 of 7 benchmark-model settings — primarily because it preserves evidence coverage.

**Prereqs:** [README.md](README.md)
**Related:** [_agent-memory](_agent-memory.md)

---

## What it is

The agentic-memory literature has stacked more and more layers between the raw conversation history and the downstream LM: entity extractors, summarizers, memory updaters, graph structures, episodic buffers, forgetting rules. Each layer *compresses* history — and compression trades coverage for compactness.

Raw-RAG is the deliberately uncompressed baseline. Build a dense index over every utterance of the conversation. At each turn, retrieve top-$k$ passages with a dense retriever, rerank them with a cross-encoder, and prepend the reranked top-$m$ to the current turn as context. Nothing is extracted, summarized, or edited.

---

## How it works

### Pipeline

```
          raw conversation history  (every utterance, unchanged)
                        │
                        ▼
             dense retriever  → top-k candidate utterances
                        │
                        ▼
            cross-encoder reranker  → top-m reranked passages
                        │
                        ▼
            prepend to current-turn prompt → LM response
```

- **Retriever:** a standard dense embedding model (e.g. BGE, E5-class). Each utterance is embedded independently and stored in a flat index.
- **Reranker:** a cross-encoder that scores (query, passage) pairs. Operates on top-$k$ candidates only, so the cost is bounded.
- **No mutation.** The index is append-only. Nothing is rewritten, merged, or deleted.

### What it deliberately omits

- No entity graph of people/places/events mentioned.
- No per-session summary buffer.
- No memory-updating LM pass.
- No forgetting policy.
- No chunking (operates at utterance granularity).

### Why it wins the coverage game

Agentic-memory systems that extract/summarize lose information at every layer. Downstream LMs can recover from retrieving *too much* information (they'll ignore irrelevant passages) but not from retrieving *not enough* (they can't synthesize what was discarded upstream). Raw-RAG keeps the ceiling at 100% coverage; everything else lowers it.

---

## Why it matters

- **Hard baseline to beat.** 6/7 benchmark-model settings — before shipping a complex agent memory, prove it beats a reranker over raw history on the same data.
- **Realigns the field.** The study is a rebuttal to the "extract, graph, mutate" direction of agentic-memory work. The paper's framing — measure *coverage*, not sophistication — is a reusable methodology point.
- **Easy to deploy.** Needs only a retriever + reranker; no additional LM passes to maintain memory state. Lower operational cost than any extraction-based system.
- **Clear failure predictions.** Where Raw-RAG loses: tasks requiring explicit cross-turn aggregation of many weakly-related mentions. These are a smaller fraction of benchmarks than the field assumed.

---

## Gotchas & tricks

- **Reranker is doing most of the work.** The dense retriever provides recall, but ablations in the paper show the cross-encoder reranker is responsible for most of the quality — a weak reranker collapses Raw-RAG below extraction-based systems.
- **Index growth.** Append-only indices grow unboundedly over very long conversations. For true long-horizon deployments, a time-decayed or sliding-window index becomes necessary — this *is* compression and lowers Raw-RAG back toward the extraction baselines.
- **Context budget still bites.** Top-$m$ passages must fit in the current-turn context. For very long passages, Raw-RAG must either chunk (reintroducing a form of compression) or drop candidates.
- **One benchmark setting bucks the trend.** The 1/7 setting where extraction-based memory wins involves tasks where many weak signals must be fused — Raw-RAG's reranker can only promote $m$ passages, so it misses the signal distributed across more.

---

## Sources

- Paper: *Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought* — 2026 — [arXiv:2610.30755](https://arxiv.org/abs/2610.30755).
- Benchmarks referenced: LoCoMo, LongMemEval, MEME, MemoryAgentBench — the four long-horizon memory evaluations used in the paper.
