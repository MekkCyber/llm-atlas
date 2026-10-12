# Context Compaction with Residual Soft Memory (REMORY)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Long-horizon agents either summarise history (lossy) or carry everything (expensive). REMORY sits in between: alongside a textual summary, it appends a short, bounded sequence of **learned soft memory tokens** that make a *frozen* LLM behave as if it saw the full un-summarised history. Improves source attribution without hurting insight coverage, cuts repeated tool outputs and errors on long-horizon benchmarks.

**Prereqs:** [README.md](./README.md), [../inference/README.md](../inference/README.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

A context-compaction scheme that augments a textual summary with a bounded continuous-embedding "memory residual." The LLM backbone stays frozen; only the memory tokens are trained. At inference time the agent carries only the summary + residual tokens, not the full history.

## How it works

- **Training.** On trajectories with full history available, learn a short sequence of continuous tokens (soft memory) that, when prepended/appended to the textual summary, minimises the KL between the frozen LLM's next-token distribution and its distribution *under the full history*. The textual summary is produced by the same summariser the agent would use at inference.
- **Inference.** The agent summarises as usual, then attaches the trained residual tokens. The LLM reads summary+residual; the residual carries the detail the summariser dropped.
- The number of memory tokens is bounded (dozens to a few hundred), so the context budget is predictable.

## Why it matters

Agent context compaction has lived at two extremes: free-text summary (loses specifics the agent later needs) and raw history (blows the context window). REMORY gives a third option that is cheap at serving time (no backbone retraining), bounded in context cost, and empirically recovers the specific thing summaries hurt most — *source attribution*. Reports fewer repeated tool outputs and errors on BrowseComp and Terminal-Bench 2.1, both pathologies caused by summarising away the fact "we already tried this."

## Gotchas & tricks

- The residual is specific to the summariser it was trained with. Change the summariser and the residual drifts.
- Soft memory tokens are opaque — unlike a textual summary, you can't inspect them. Debugging requires evaluating the composite (summary + residual) on probe queries.
- The frozen-backbone constraint matters: the method does not alter the base model, so it stacks with any post-training the base already had.
- The residual helps most when the summariser compressed something load-bearing; on tasks where the summary was already lossless, gains are small.

## Sources

- Paper: *REMORY: Learning Residual Memory for Context Compaction* — Xia et al., Fudan / SII, 2026 — [arXiv:2610.11287](https://arxiv.org/abs/2610.11287)
