# Memory Tax
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A measurement framework that evaluates agent-memory systems against the right baseline — **RAG over raw conversation history**, not "no memory." The headline finding: many popular memory systems have *net negative* value because the distillation/compression they perform strips evidence a later task needed. The paper names this cost the **memory tax** and the complementary gain the **memory benefit**, and attributes most of the tax to design choices like summarization, graph consolidation, and forgetting.

**Prereqs:** [_memory.md](_memory.md)
**Related:** [_memory.md](_memory.md) · [../evaluation/habit-bench.md](../evaluation/habit-bench.md)

---

## What it is

Agent memory systems are usually evaluated against a trivial baseline: *no memory*. Any system that remembers anything looks good against that. He et al. (2026) argue the right baseline is **RAG over the full raw history**: a dense retriever over every past turn, with no distillation. Any memory layer added on top only earns its keep by exceeding this baseline on the *same* evaluations.

The paper defines two quantities per task:

- **Memory benefit** $B = \max(0, \text{Acc}_{\text{mem}} - \text{Acc}_{\text{RAG}})$
- **Memory tax** $T = \max(0, \text{Acc}_{\text{RAG}} - \text{Acc}_{\text{mem}})$

Net value = $B - T$. Many popular memory systems have net value $\leq 0$ on realistic workloads.

## How it works

The measurement protocol:

1. Build an evaluation set with queries that depend on specific past-turn details (fact recall, preference, cross-session reasoning).
2. For each system under test, run (a) RAG-over-history baseline and (b) the memory system.
3. Score accuracy per task and bin by task type.
4. Decompose aggregate accuracy change into $B - T$; attribute $T$ to specific memory operations by ablation.

Common causes of memory tax identified:

- **Compression / summarization** replaces specific tokens with generic paraphrases; the specific tokens were what later retrieval needed.
- **Graph consolidation** merges mentions into one node, losing the context in which each mention occurred.
- **Forgetting / decay** removes material whose relevance only becomes clear later.
- **Reorganization** moves facts away from their disambiguating neighbors.

Systems with high memory benefit tend to *add* lightweight indices without removing material. Systems with high tax do heavy transformations.

## Why it matters

- **Reframes the subfield.** Memory-system wins over "no memory" were essentially free. Wins over RAG-over-history are the real measure.
- **Explains why RAG keeps winning.** Teams that benchmarked memory against RAG saw regressions and blamed implementation; the paper argues the regressions are often structural.
- **Design implication: index, don't rewrite.** Memory layers that preserve the raw text and add retrieval affordances outperform layers that compress or restructure.

## Gotchas & tricks

- **The tax is workload-dependent.** On tasks where only coarse topic matters, compression loses little. On fact-recall and habit tasks, it loses a lot.
- **"No memory" baselines are still common.** When reading a memory-systems paper, check what it beats and whether a RAG baseline is reported.
- **Mitigation: hybrid stores.** Keep raw history for RAG, add structured indexes alongside. The structured side pays only its own cost, not the raw history's.
- **Not a verdict against all memory.** Entity graphs and summaries have real uses (role-consistency, long-horizon planning); the point is to measure net value, not avoid them.

## Sources

- Paper: *Memory Tax: Dissecting the Net Value of Agent Memory* — He, Zhoubian, Yang, Wen, Wang, Hu, Yang, Zhang, Nie, Han, Zhai, 2026 — [arXiv:2610.34073](https://arxiv.org/abs/2610.34073)
