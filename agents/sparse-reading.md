# Sparse reading
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A model-agnostic control layer that sits between an agent's "read external content" tool calls (web pages, documents, long tool outputs) and the agent's context window, scoring spans against the running task state and streaming only the useful ones back. *SparseRead* (2608.22237) reports up to 92.9% fewer input tokens and 89.0% lower wall time at preserved-or-better quality in 97% of tested settings, across six models and five task scenarios.

**Prereqs:** [README.md](README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

Agents today use long-context models as an excuse to ingest whole pages verbatim. Most pages are mostly filler — navigation, boilerplate, legal, unrelated sections. Sparse reading separates *what the agent asks for* (a URL / document / tool output) from *what the context window receives* (only the spans that look useful right now).

It is middleware, not a model change: no weights touched, drop in between the tool call and the model.

## How it works

Three moving parts:

1. **Span extraction.** The fetched content is split into candidate spans (paragraphs, bullets, cells, code blocks).
2. **Relevance scorer.** A small policy scores each span against the current task state (the user goal, the recent CoT, prior reads). Scoring can use cheap embeddings plus task-conditioned features; a tiny ranker is sufficient.
3. **Streaming controller.** A streaming policy decides whether to feed only high-scoring spans, expand on request, re-read on confusion, or drop content entirely. The agent never sees the pruned tokens, and can issue a "read more" tool call if its reasoning stalls.

The agent above this layer is unchanged — it still issues normal fetch/read tool calls.

## Why it matters

- Collapses the cost of "stuff the page in and let the model figure it out" without touching weights, making the technique portable across model providers.
- Decouples long-context *capability* (model) from long-context *economics* (middleware). You can ship a 128K agent on a 32K model.
- Shifts latency hot spots: prefill shrinks, decode dominates, which composes well with [phase-reconfigurable serving](../inference/phase-reconfigurable-serving.md).

## Gotchas & tricks

- Overpruning silently degrades reasoning on tasks that need obscure page detail — budget a fallback "read more" affordance.
- Span boundaries matter: scoring whole tables as one span, or splitting mid-sentence, both hurt.
- Scorer needs the agent's current task state, not just the raw prompt; reconstructing that from the trajectory is the fiddly bit.
- Cache the scored span lists per (url, task-state hash) to avoid rescoring on re-reads.

## Sources

- Paper: *Read Less, Solve More: Token-Efficient Sparse Reading of External Content for AI Agents* — Wu, Ma, Xu, Wang, Hu, Tao, Tan (2026) — [arXiv:2608.22237](https://arxiv.org/abs/2608.22237)
