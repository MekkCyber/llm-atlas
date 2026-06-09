# SubtleMemory — relational-memory discrimination benchmark
*Depth — evaluate long-horizon agents on memories that complement, nuance, or contradict each other, not just isolated recall.*

**TL;DR:** Long-running AI assistants accumulate memories that *interact*. Existing benchmarks test "did the agent remember X?". SubtleMemory tests "did the agent correctly combine X, Y, and Z given that Y nuances X and Z contradicts X?". 1,522 evaluation instances spanning complementary / nuanced / contradictory memory relations across ten long interaction histories.

**Prereqs:** [rlvr](../post-training/rlvr.md)
**Related:** [toolmaze](toolmaze.md), [ifeval](ifeval.md), [unpredictabench](unpredictabench.md)

---

## What it is

A benchmark for *fine-grained relational memory discrimination*: when an assistant has stored many related memories, the correct answer depends on how those memories *relate*, not just on retrieving them. Three relation types:

- **Complementary** — memories together provide more than either alone (different facts about the same entity).
- **Nuanced** — one memory qualifies or partially overrides another (a previously stated preference is later refined).
- **Contradictory** — memories directly conflict (an earlier statement was wrong or has been retracted).

Each test instance is embedded in one of ten long interaction histories.

## How it works

- Each instance specifies: the interaction history, a downstream task that requires consulting multiple memories, and the gold relation type.
- Scoring is task-success per instance, plus per-relation-type breakdowns so that the contradiction-handling weak spot is visible.
- The benchmark is intentionally history-grounded — memories live inside a realistic dialogue trace, not a hand-curated key-value store, so retrieval and relation reasoning are jointly tested.

## Why it matters

- Targets a known production failure: persistent assistants that remember everything but believe all of it.
- Exposes the gap between retrieval (mostly solved) and relational reasoning (mostly not) as a distinct capability axis.
- Becomes a natural training target: any agent post-training stack that does memory edits can be graded directly on contradiction handling.

## Gotchas & tricks

- **History length is a confound.** Strong retrievers can paper over weak relational reasoning if histories are short; the benchmark uses long traces to neutralise this.
- **Annotation cost.** Relation labels need human review for plausibility — distinguishing "nuanced" from "contradictory" is itself a fine judgement.
- **Contamination risk.** Synthetic-dialogue construction must avoid leaking the gold relation type into the trace.

## Sources

- Paper: *SubtleMemory: A Benchmark for Fine-Grained Relational Memory Discrimination in Long-Horizon AI Agents* — Wang, Sun, Hou, Song, Zhang, Cheng, Yang — 2026 — [arXiv:2606.05761](https://arxiv.org/abs/2606.05761)
