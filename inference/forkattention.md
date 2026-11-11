# ForkAttention (Flide)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A KV-cache layout for hybrid architectures (softmax attention + Gated DeltaNet / linear attention) that makes forked-prefix inference **exact** — the same output distribution as running each fork independently in float32. Used by the Flide inference engine to serve tree-of-thought, MCTS, best-of-$N$, and agentic-routing workloads up to 9.5× faster than vLLM/SGLang on comparable workloads.

**Prereqs:** [./README.md](./README.md), [../architectures/mla.md](../architectures/mla.md)
**Related:** [../systems/partial-rollouts.md](../systems/partial-rollouts.md), [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md)

---

## What it is

Modern "decision inference" workloads — ToT, MCTS, best-of-$N$, agentic tool routing — share a common shape: a long shared prefix branches into many short continuations. vLLM's radix tree and SGLang's RadixAttention exploit this for *softmax attention* layers by sharing their KV cache across forks. But recent architectures mix in **recurrent / linear-attention** layers (e.g. Gated DeltaNet), whose state is a running summary that cannot be concatenated like a KV cache. Sharing across forks in those layers was either approximate (quality loss) or not done (speed loss).

ForkAttention introduces a fork primitive that is **exact across both layer types**: it defines how to snapshot and restore the per-layer state at a fork point so that resuming from the snapshot produces bit-identical distributions to a from-scratch run.

## How it works

At a fork point in the token sequence:

1. **Attention layers.** Cache the KV tensors up to the fork point. Each fork shares the same KV prefix tensor (pointer, not copy) and appends its own tail.
2. **Gated DeltaNet / linear-attention layers.** The per-layer state $S_\text{prefix}$ at the fork is a dense matrix of shape $(d_k, d_v)$. ForkAttention *snapshots* $S_\text{prefix}$ once and stores it alongside the KV prefix. Each fork then evolves its own copy forward from $S_\text{prefix}$ using the per-layer update rule.

Crucially, both representations are the exact recurrence state (not an approximation), so decoding from the snapshot is identical to decoding from a fresh run. The engine's scheduler treats the fork tree as the primary object: tokens from different forks are batched together only where they share the same *physical* state, so there is no wasted work.

## Why it matters

- **Exact.** Flide reports output distributions closest-to-fp32 in total variation distance, matching the reference while being ~9.5× faster than vLLM/SGLang on decision-tree workloads.
- Makes hybrid architectures (MLA + Gated DeltaNet) viable for search-heavy serving, which they weren't before: the recurrent state's non-concatenative structure was the blocker.
- Extends the "prefix sharing" idea beyond attention — recurrent-state snapshots are the right generalization for any layer whose forward depends only on a fixed-size summary of history.

## Gotchas & tricks

- **Snapshot memory.** Each unique fork point costs one state snapshot per Gated-DeltaNet layer. For deep trees, this dominates memory — budget by tree depth, not just total forks.
- **Exactness assumes deterministic kernels.** FP8 / FP16 reductions with nondeterministic reduction orders break bit-identity (though distribution is still close). Use deterministic kernels in the recurrent layers.
- **Not a decode speedup per se.** The win comes entirely from shared prefix compute at scale; a single-fork workload sees no benefit over vLLM/SGLang.
- **MLA integration.** Works naturally with MLA (which caches latent $c_t$) because the latent is itself a fixed-size summary — the same snapshot primitive applies.

## Sources

- Paper: *Flide: Accelerating Decision Inference with Exact State Forking* — Song, Fei, Hsu, Lee (NUS / Oxford), 2026 — [arXiv:2610.42148](https://arxiv.org/abs/2610.42148)
