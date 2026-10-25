# Raw-RAG
*Depth — dense retrieval + reranking over raw conversation history as an agent-memory baseline.*

**TL;DR:** Raw-RAG is the deflationary baseline for agentic long-term memory: a dense retriever + reranker runs over the *untransformed* conversation history and feeds the top-k chunks into the agent as context. Across four long-conversation benchmarks it ranks first in six of seven settings, beating structured memory systems (summarization, episodic/semantic splits, graph memory, parametric memory). The reason it wins is evidence coverage: every memory transformation collapses relational and dependency paths that raw text keeps intact.

**Prereqs:** [attention.md](../fundamentals/attention.md)
**Related:** [cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

A memory architecture for long-horizon conversational agents that *refuses* to transform history. The agent's "memory store" is just the chat log, chunked. At each turn, a dense retriever (encoder + ANN index) finds the top-N relevant chunks and a cross-encoder reranks them. The top-k reranked chunks are concatenated into the prompt alongside the current user turn.

## How it works

1. **Chunking.** Fixed-size (or utterance-boundary) chunks over the raw history. No summaries, no segmentation into episodic/semantic, no graph extraction.
2. **Dense retrieval.** Encode chunks with a passage encoder; store in a vector index. At query time, encode the current user turn and retrieve top-N.
3. **Reranking.** A cross-encoder scores (query, chunk) pairs; keep top-k.
4. **Fuse.** Prepend the top-k chunks to the generation prompt.

No transformation step and no learned memory controller: the retriever/reranker are off-the-shelf; the agent is unchanged.

## Why it matters

- **Deflationary.** A ladder of structured memory systems (MemGPT, Letta, hippocampal / graph / parametric variants) is beaten by a baseline most of them use as a strawman. Suggests much of the agent-memory literature was premature optimization.
- **Evidence coverage ≠ compactness.** Transformations trade recall for compactness; for the tasks measured, the recall loss dominates.
- **First on 6 of 7** benchmark–model settings across LoCoMo, LongMemEval, MEME, and MemoryAgentBench.
- The honest research direction becomes *better retrievers* (coverage, calibration, deduplication) rather than more memory tiers.

## Gotchas & tricks

- Context budget: raw chunks are long; aggressive top-k keeps latency reasonable but can push out current-turn context. A hard token budget enforced per retrieval is important.
- Freshness: pure cosine-sim retrieval over long logs favors older, repeated content. Recency boosts or decay functions help.
- Hybrid works too: Raw-RAG + a thin summary layer for very old history retains most of the win and shrinks the index.
- Benchmarks measured here lean QA-style; truly open-ended "remember my preferences" dialogues weren't the central eval.

## Sources

- Paper: *Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought* — 2026 — [arXiv:2610.30755](https://arxiv.org/abs/2610.30755)
