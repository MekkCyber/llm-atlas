# LongMemEval
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for **long-horizon agent memory** — can an agent answer questions that require information from turns far back in a multi-session conversation? Distinct from long-context QA in that the model can't fit the entire history in-context; success depends on the agent's memory system (retrieval, summarization, compaction), not the base LM's window size. Cited widely by memory-management papers; e.g., Maximem Synap reports 92% on it.

**Prereqs:** [../agents/_context-management.md](../agents/_context-management.md)
**Related:** [locomo.md](./locomo.md), [../agents/context-compaction.md](../agents/context-compaction.md)

---

## What it is

An evaluation suite of long, multi-session dialogues where downstream queries deliberately reference facts introduced many turns (or sessions) earlier. Passing requires the agent under test to (a) *store* the earlier fact reliably, and (b) *retrieve* it when the later query is issued. The benchmark exercises the agent's memory *architecture* end-to-end rather than isolating any single component.

## How it works

- **Dialogue construction.** Long conversations are constructed so that many query-answering facts appear well before the query — beyond the reasonable in-context budget of the base LM.
- **Evaluation loop.** The agent processes turns one at a time; at query turns, its answer is scored against ground truth. Sessions can be interleaved (multi-thread) to stress scoping.
- **What varies.** Systems can plug in any memory stack — raw prompt-only baseline, vector-store RAG, structured extraction, compaction pipelines. Score differences isolate the memory system's contribution.

## Why it matters

Prior long-context benchmarks confounded model window size with memory quality — a model with a larger window "passed" without doing any memory work. LongMemEval is designed so the base LM cannot brute-force it, which forces the agent's memory system to do real work. This is what makes it a load-bearing signal for context-management research.

## Gotchas & tricks

- **The base-LM window is a confound.** As frontier models push to 1M+ token contexts, some LongMemEval subsets become brute-forceable; report scores alongside effective context used.
- **Retrieval hyperparameters matter as much as the model.** k, chunking, hybrid dense/sparse — swapping these often changes scores more than swapping the base LM.
- **Scoping subsets are the interesting ones.** Multi-user / multi-session subsets separate systems with real boundary logic from those that only handle single-session accumulation.

## Sources

- Referenced by: *Agentic Context Management* — Maximem, 2026 — [arXiv:2607.21503](https://arxiv.org/abs/2607.21503) (reports 92% under the Maximem Synap configuration).
- Primary benchmark citation: LongMemEval — see arXiv listing for the original benchmark paper for setup details.
