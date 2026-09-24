# Recursive Self-Improvement (of research agents)
*Depth — AI research agents that rewrite their own scaffolding, selected by held-out task performance.*

**TL;DR:** Recursive self-improvement (RSI) for LLM agents is the loop where the agent proposes edits to its **own code** (search policy, memory, tool selection, orchestration), benchmarks candidate variants of itself on an ML R&D task suite, and keeps the best-scoring variants for the next round. Unlike weight-level self-training, the object of optimization is the agent's scaffolding, not the model. AIDE² (2026) is the first published system to run this loop autonomously at scale with measurable transfer to held-out tasks.

**Prereqs:** [README.md](./README.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../safety/scheming.md](../safety/scheming.md), [../safety/README.md](../safety/README.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

An RSI loop where the *code of an AI research agent* is the search space, not the weights of the underlying model. Each accepted rewrite becomes the base agent for the next iteration. The signal comes from **out-of-fold benchmark scores on a suite of ML R&D tasks** the agent runs against — held-out variants of the same task family plus hidden evaluations to catch over-fitting to the selection tasks.

Sits at the intersection of agent design and long-horizon self-improvement, with obvious safety-relevant properties (the loop can amplify capability gains without any explicit reward-model / preference data on the underlying LM).

## How it works

Loop, per AIDE² (2026):

1. **Propose.** Current agent variant reads its own code and proposes edits — new search strategies, memory / context-compression mechanisms, tool-call heuristics, orchestration policies.
2. **Benchmark.** Each proposed rewrite runs the ML R&D task suite (e.g. FML-Bench-style tasks). Score = task success rate; multiple tasks per proposal to reduce noise.
3. **Select.** Keep the highest-scoring proposals under a held-out evaluation (variants of the same tasks not used in the proposal phase) to guard against selection-set overfitting.
4. **Iterate.** Selected variants become the base for the next round; discovered improvements compound.

AIDE²'s 8-day autonomous run yielded seven sequential improvements — a new search policy, memory-compression, context-management. The strongest discovered agent matches or exceeds a human-engineered production research agent on four held-out benchmarks including out-of-distribution weather forecasting. Reward-hacking rate on a separate held-out family fell from 55% → 32%, an emergent property the loop never explicitly optimized for.

## Why it matters

- **New scaling axis.** Cumulative R&D spend traditionally shows diminishing returns; RSI is a candidate for pushing that curve, since it re-invests agent improvements into finding further improvements.
- **Empirical existence proof.** Transferable, non-trivial improvements *do* exist on the "code of the agent" axis alone — no LM weight updates required.
- **Alignment-relevant surface.** Self-modifying agents are exactly the class of systems whose safety properties (scheming, reward-hacking, capability elicitation) need scrutiny. Reduced reward hacking as an emergent property is a data point to interrogate, not accept at face.

## Gotchas & tricks

- **Selection-set overfitting is the failure mode.** Any RSI loop where selection and proposal see the same tasks will hill-climb the metric. AIDE² uses held-out variants + hidden evals; borrow this or the loop will just find prompt-engineering exploits.
- **Compound gains break under drift.** Improvements chained across 7 rounds only transfer if selection tasks are diverse enough. Narrow selection → discovered agent may collapse on tasks outside the loop's diet.
- **Emergent-property claims (e.g. reduced reward hacking) deserve independent replication.** They arise as byproducts of the selection signal; the causal path from selection → property is not tight.
- **Compute cost is high.** Each round runs the full task suite $N$ times. For weeks-long runs, task-suite reliability (deterministic scoring, low variance) is essential — a noisy benchmark drives spurious accepted rewrites.

## Sources

- Paper: *Recursive self-improvement of AI research agents* — Srikanth, Zhao, Xu, Wu, Jiang (Weco AI / Edinburgh), 2026 — [arXiv:2609.26457](https://arxiv.org/abs/2609.26457) — introduces AIDE².
