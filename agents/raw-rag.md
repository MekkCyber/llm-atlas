# Raw-RAG
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A deliberately boring baseline — dense retrieval + reranking over the **raw** conversation history — that beats most agentic memory systems on four long-term-memory benchmarks (LoCoMo, LongMemEval, MEME, MemoryAgentBench). The paper's point is that every transformation an agentic memory system does to raw turns (summarise, cluster, restructure) sheds signal, and that higher evidence coverage from raw text beats structured memory stores (2026).

**Prereqs:** [README.md](README.md)
**Related:** [../evaluation/README.md](../evaluation/README.md)

---

## What it is

A "naive" retrieval-augmented generation baseline for long-conversation agents. The agent's memory *is* its full conversation log, verbatim. On each query it runs dense retrieval (plus a reranker) over all prior turns, concatenates the top-K into context, and answers. No summarisation, no clustering, no graph, no episodic store.

## How it works

- **Index.** Every turn (user and assistant) is embedded and stored. No chunking magic; one turn per record is typical.
- **Query time.** Embed the current query, retrieve top-K by cosine, rerank with a cross-encoder, pass to the LLM.
- **No write path beyond logging.** Memory grows monotonically with the conversation; there is no selective-write or forgetting policy.
- **Benchmark setup.** Compared against agentic memory stacks across four benchmarks; both systems get the same LLM backbone and the same evaluation harness.

## Why it matters

- **Resets the sophistication baseline.** Agent-memory research has been climbing toward structured stores (knowledge graphs, episodic/semantic splits, learned memory controllers). Raw-RAG shows that most of these transforms are lossy, and that retrieval alone — if done well — covers the evidence need.
- **Diagnostic for new systems.** Any new agentic memory proposal that doesn't beat Raw-RAG on these four benchmarks should justify its complexity.
- **Attribution of wins.** Even granting agentic systems *full* access to their transformed stores does not close the gap to raw evidence — the loss is at the memory-construction step, not retrieval.

## Gotchas & tricks

- **Doesn't scale to very long logs.** Raw retrieval over 10⁵+ turns gets expensive; the comparison benchmarks have bounded history.
- **No privacy / forgetting story.** Agentic memory stacks often exist for compliance reasons (forget on request, forget expired context). Raw-RAG has no answer there.
- **Reranker matters.** Dense retrieval alone is weaker; the cross-encoder rerank is doing real work in the paper's wins.
- **Not a claim that memory structure is useless.** The claim is that structured memory as currently practiced *loses* to raw evidence on these evals. A system that genuinely adds information (reasoning over turns, resolving contradictions) could still win.

## Sources

- Paper: *Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought* — 2026 — [arXiv:2610.30755](https://arxiv.org/abs/2610.30755).
