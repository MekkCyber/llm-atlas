# LoCoMo
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for **long-conversation memory** in dialog agents. Sessions run over weeks of simulated interaction; queries probe whether the agent can recall facts, events, and preferences established across many prior sessions. A companion to [longmemeval.md](./longmemeval.md), stressing longitudinal (many-session) rather than dense within-session recall.

**Prereqs:** [../agents/_context-management.md](../agents/_context-management.md)
**Related:** [longmemeval.md](./longmemeval.md), [../agents/context-compaction.md](../agents/context-compaction.md)

---

## What it is

A multi-session conversational benchmark: an agent has many prior sessions with a user; a new query requires it to pull the right fact out of that longitudinal history. Where LongMemEval targets *depth-within-session* recall, LoCoMo targets *breadth-across-sessions* recall — the two together stress the two dominant failure modes of production agent memory.

## How it works

- **Long timelines.** The synthetic user-agent history is deliberately much longer than any base LM's context window.
- **Cross-session queries.** Questions require pulling from a specific past session, not the current one. Session identifiers and timestamps are part of the evidence.
- **Metric.** Answer accuracy against ground truth; some variants also score latency and token efficiency.

## Why it matters

Real production assistants — customer support, personal copilots — accumulate months of history per user. The token bill of stuffing that history in-context is impossible; the fidelity loss of naive summarization is unacceptable. LoCoMo makes the trade-off measurable: any memory pipeline that isn't scoring well here is unlikely to hold up under real longitudinal load.

## Gotchas & tricks

- **Latency and token efficiency are second-order but real.** A system that scores well on accuracy while paying quadratic cost per turn isn't shippable; check secondary metrics.
- **Multi-user boundary tests separate real systems from toys.** Cross-contamination between users often only appears in the harder subsets.
- **Score is a function of the whole pipeline.** Chunking, embedding model, retriever, and base LM all move the number; report configurations exhaustively.

## Sources

- Referenced by: *Agentic Context Management* — Maximem, 2026 — [arXiv:2607.21503](https://arxiv.org/abs/2607.21503) (reports 93.2% under the Maximem Synap configuration).
- Primary benchmark citation: LoCoMo — see arXiv for the original benchmark paper for setup details.
