# MemTrapBench
*Depth — benchmark that measures how faithful, semantically relevant memory can still degrade reasoning through Reasoning Fixation and Belief Distortion.*

**TL;DR:** Long-lived LLM agents ship memory systems, and the default assumption is that better recall means better help. MemTrapBench constructs tasks where the *right* answer requires **ignoring or contradicting** a memory that is both accurate and topically relevant. All tested memory strategies (naive, hierarchical, retrieval-augmented) underperform the no-memory baseline; the authors' *AdaptiveMem* mitigation, which gates memory use per-query, partially closes the gap.

**Prereqs:** [README.md](README.md)
**Related:** [../agents/README.md](../agents/README.md), [../safety/distractor-attack.md](../safety/distractor-attack.md), [mmlu.md](mmlu.md)

---

## What it is

A benchmark suite explicitly targeting two agent-memory failure modes:

- **Reasoning Fixation.** The model over-anchors on prior reasoning traces stored in memory, importing conclusions or intermediate steps that no longer apply to the current problem. The prior reasoning is *correct*; its transfer is *wrong*.
- **Belief Distortion.** The model updates its beliefs in the wrong direction after seeing memory that superficially matches the current context. The memory is *truthful*; the resulting belief is *distorted*.

Task design: each item ships with a curated memory bank (retrieved items that look relevant) and a question whose correct answer requires either dismissing the memory or reasoning in tension with it. Scoring compares against a matched no-memory baseline on the same items.

## How it works

The scoring protocol:

1. For each item, evaluate the model under N conditions: no memory; naive memory injection; retrieval-augmented memory; hierarchical/summarized memory; the paper's *AdaptiveMem*.
2. For each condition, record accuracy on the correct answer.
3. Report per-condition score and delta vs no-memory. A negative delta means the memory strategy *hurts*.

The finding: every non-adaptive strategy has negative delta on trap items. The trap items themselves come in two subsets — Reasoning Fixation and Belief Distortion — with distinct construction rules but similar effect sizes.

The proposed mitigation, AdaptiveMem, gates memory use per-query: a small classifier decides whether the retrieved memory is likely to help *this* query, and skips injection when the estimated risk of a trap outweighs the benefit. Not a solution — a partial recovery.

## Why it matters

Refutes the folk theorem that faithful, relevant memory can only help. Reframes agent memory as a **retrieval-decision** problem, not just a retrieval problem: the hard question is not "which items do I retrieve" but "given retrieved items, should I use them?" Every long-lived agent product ships some memory story; this benchmark gives them a way to detect when their memory is a net negative.

Adjacent to distractor-attack literature — trap memories are a benign-labeled cousin of adversarial distractors, sharing the "plausible-but-misleading context" mechanism.

## Gotchas & tricks

- **Trap items are minority.** In real workloads most retrieved memory is helpful; MemTrapBench isolates the tail. Report both trap and non-trap performance to avoid pessimistic mis-tuning.
- **Adaptive gating must be cheap.** A gate that costs an extra LLM call per query dominates savings from the whole system. Distill it into a small classifier or a heuristic on retrieval scores.
- **Item construction matters.** Reasoning Fixation items require careful anti-inheritance design; sloppy items are trivially solvable by any calibrated model.
- **Prior-context vs memory.** MemTrapBench isolates external memory. In-context reasoning fixation (from earlier in the same conversation) is a related but distinct failure mode.
- **Benchmark contamination.** As with any benchmark that publishes the trap patterns, training on the paper directly will inflate scores. Watch training data.

## Sources

- Paper: *MemTrapBench: Benchmarking Cognitive Traps in LLM Memory Use* — Wang, Luo, Xu et al., ZJU/USTC/HKU, 2026 — [arXiv:2608.20202](https://arxiv.org/abs/2608.20202).
- Related: *Distractor-Attack* — [../safety/distractor-attack.md](../safety/distractor-attack.md) — adversarial version of the same "plausible misleading context" mechanism.
