# Memory Attention (Native Memory State)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **native memory state** carried inside the model itself — persistent across turns, compressed as it grows, read via a dedicated **memory-attention** pathway alongside standard self-attention. Reframes agent memory from an external plumbing problem (RAG, vector stores, scratchpads) into an *architecture* choice that gets trained end-to-end.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md)

---

## What it is

Standard transformer attention is *stateless* between requests — the KV cache lives for one context and is thrown away. Longer-horizon "memory" has historically been bolted on:

- **External stores** (vector DBs, RAG, summarization-into-prompt).
- **Ephemeral scratchpads** (context window as memory).
- **Fine-tuning** for per-user knowledge.

Native memory attention pushes the memory into the *model* as a first-class state: a set of memory slots (or a compressed KV bank) that the model updates during interaction and reads via a dedicated attention head at each generation step. Because the update rule and read pathway are trained jointly with the base LM, memory access is learned end-to-end rather than approximated by prompt engineering.

## How it works

Metis (Zhang et al., 2026) instantiates the pattern:

1. **State.** Alongside the transient KV cache, the model carries a persistent, size-bounded memory tensor $M$. Historical information from prior contexts is compressed into $M$ (learned compression, not a summarizer).
2. **Write path.** As new context arrives, an update operator (a small module trained jointly with the backbone) folds fresh content into $M$ — evicting or merging older entries.
3. **Read path — memory attention.** At each transformer layer (or a dedicated subset), an extra attention branch queries $M$ in addition to the current KV cache. The two branches' outputs are combined (concatenated, added, or gated) before feeding the residual stream.

This makes memory a **training axis**: how much memory to carry, how to compress it, and how to read it are all learned choices, not runtime heuristics.

## Why it matters

- **End-to-end training.** Retrieval-augmented systems train the retriever separately, if at all. Native memory learns the whole loop under the same loss.
- **Long-horizon consistency.** Agents that need coherent behavior across many turns (multi-day sessions, persistent tools) currently rely on brittle summarization pipelines. Native memory replaces those with a learned state.
- **Compute story.** Growing the memory tensor is cheaper than growing the context window — memory attention costs $O(N_M)$ per token vs full attention's $O(N_{\mathrm{ctx}})$ over the entire past.

## Gotchas & tricks

- **Compression is the whole game.** Naive slot-writing loses information as fast as it arrives; the compressor's inductive bias determines what the model remembers.
- **Interference with KV cache.** Two attention branches with different memory horizons can compete for the residual stream; gating between them is usually needed.
- **Evaluation is hard.** Long-horizon memory benefits don't show up on standard short-context benchmarks — needs multi-session and multi-turn evals to measure.
- **Not RAG.** External retrieval hits *documents*; memory attention updates a *learned latent state*. The two compose but are not substitutes.

## Sources

- Paper: *Metis: Memory Foundation Model* — Zhang, Guo, Sun et al. (MemTensor / RUC / NUS / SJTU / Tongji), 2026 — [arXiv:2607.26760](https://arxiv.org/abs/2607.26760).
- Related: [mla](mla.md) — MLA compresses the KV cache within a single context; memory attention compresses across contexts.
