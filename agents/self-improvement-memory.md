# Self-Improvement Memory
*Depth — a persistent memory of prior edits + predictions + observed effects that keeps optimizer agents improving across changing benchmarks.*

**TL;DR:** Agent self-improvement loops (think DSPy / TextGrad) plateau because each round forgets why the last round changed what it did. A **self-improvement memory** stores `(edit, predicted effect, observed effect, context)` tuples; retrieval over that memory conditions the next edit proposal. Lets a frozen optimizer keep improving over benchmark distributions that shift between rounds.

**Prereqs:** (agents/ is still bare — this seeds it)
**Related:** [environment-evolution-loop](environment-evolution-loop.md)

---

## What it is

A **harness-optimizer loop** component. Normal setup: an **optimizer agent** proposes edits to a **target** (a prompt, a policy, a program); the target is evaluated on a benchmark; the score goes back to the optimizer; repeat.

This paper's contribution is the **memory layer** between rounds. Instead of the optimizer seeing only the current target and score, it also retrieves from a persistent store of its own past proposals: what it changed, what it predicted would happen, what actually happened, and what evaluation context prevailed at that time.

## How it works

Each optimizer round produces a memory entry:

```json
{
  "edit": "replaced FewShot(k=3) with FewShot(k=5)",
  "rationale": "harder questions need more exemplars",
  "predicted_effect": "+3 accuracy on OOD examples",
  "observed_effect": "+4.1 on OOD, -0.5 on IID",
  "context": {"benchmark": "GPQA-v2", "round": 7, "target_version": "v34"}
}
```

Before proposing the next edit, the optimizer **retrieves top-k relevant entries** from memory, filtered by context similarity (e.g., same benchmark family, same target region being edited). The prompt that proposes the next edit therefore sees not only "here is the current target and score" but "here is what you tried in similar situations and what actually happened."

**Distribution shift handling** is the headline use case. When the benchmark changes between rounds (new topics, new difficulty, new format), the memory entries with the *closest* new-benchmark context get retrieved first — old edits transfer *because* their observed effects were conditioned on explicit context.

## Why it matters

Recursive self-improvement is a load-bearing assumption for a lot of agent roadmaps. Empirically, closed-loop self-improvement tends to plateau after a few rounds: the optimizer runs out of novel ideas, or it starts proposing edits whose effects it already knows won't work.

Self-improvement memory **shifts the plateau later**. Measured gains: continual self-improvement over changing benchmarks beats fresh-start continuation baselines on final scores, and crucially the gain *grows with the number of rounds* — the opposite of the usual flattening curve.

The result isolates where the win comes from: it's the memory of prior edits, not the edit mechanism itself. Multiple optimizer architectures (textual-gradient, instruction-mutation, program-synthesis) all benefit when you bolt on the memory.

## Gotchas & tricks

- **Memory needs a context schema.** Free-form memory entries retrieve poorly. The paper standardizes the schema (edit / rationale / predicted / observed / context fields) so similarity search is meaningful.
- **Retrieval filter matters more than retrieval model.** A simple benchmark-tag filter outperforms generic dense retrieval when the memory is small.
- **Memory pollution risk.** If an early round was pathologically bad, those entries keep getting retrieved and dragging proposals toward their failure mode. The paper uses observed-effect reweighting to dampen bad entries.
- **Not RL.** The optimizer is frozen; only its *context* changes. Think of it as test-time retrieval, not gradient descent.

## Sources

- Paper: *Self-Improvement Memory Enables Continual Recursive Self-Improvement* — Yu, Xia, Pan, Tang, Liu, Yu, Jiang, Chen, Xue, Yu (Google Cloud AI / Google DeepMind / Penn State), 2026 — [arXiv:2611.18856](https://arxiv.org/abs/2611.18856)
