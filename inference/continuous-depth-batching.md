# Continuous Depth Batching (CDB)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Looped language models (Ouro, Huginn) run a shared block for a variable number of loops per token — but standard serving stacks like vLLM assume every token in a batch takes the same forward pass. CDB re-batches tokens **between loop steps**, so tokens that need $k$ loops and tokens that need $k+1$ can coexist without wasting compute. It also predicts loop-exit probabilities ahead of time so the next batch can be prepared asynchronously.

**Prereqs:** [looped-transformers](../architectures/looped-transformers.md), [attention](../fundamentals/attention.md)
**Related:** [prefill-decode-disaggregation](./prefill-decode-disaggregation.md)

---

## What it is

A looped LM lets each token exit its loop when a halting signal fires. This is the whole point of depth-adaptive inference. But once one token in a batch is "done" and another still needs three more loops, standard batching either:

- Pads by running max-$k$ for the whole batch (wasted compute for early-exiters), or
- Falls back to sequential per-token forwards (loses batch efficiency).

CDB is the third option: **form new batches at each loop boundary**, grouping tokens by "still-alive at loop step $s$."

## How it works

Between every loop step $s$:

1. **Read halting signals.** For each token in the current batch, evaluate the halting probability at step $s$ (or the model's exit predictor).
2. **Partition.** Split tokens into "exit this step" (write to KV-cache and hand back to the decoder) and "continue to step $s+1$" (form the new batch).
3. **Rebatch.** Merge continuing tokens with newly-arriving tokens (from freshly-started sequences) that are also at loop step $s+1$. Because the model at each loop step is the *same block*, tokens from different sequences at the same step can share a forward pass.
4. **Predict ahead.** A small predictor forecasts which tokens will exit at the next step, so the router can prepare the next batch *before* the current forward finishes — hiding scheduling latency behind compute.
5. **KV-cache management.** Each token owns a per-loop-step KV entry. The scheduler tracks per-token loop count so the correct KV state feeds into attention on subsequent decodes.
6. **Non-looped layers scheduled separately.** Embeddings, LM head, and any unshared outer layers run in their own passes with standard batching.

## Why it matters

- **Realizes the promise of looped LMs.** Without CDB, depth-adaptive inference is a research curiosity because it can't be served efficiently. With CDB, up to **99%** of the estimated theoretical maximum speedup from depth adaptivity is realized in practice on Ouro-1.4B and Huginn-3.5B.
- **Enables per-token compute allocation** as a serving primitive, not just a research setting.
- **Composes with continuous batching** (the vLLM-style token-level scheduling) — CDB adds the loop-step dimension on top.

## Gotchas & tricks

- **Fully-looped architectures serve best.** Large non-looped layers (embeddings, LM head, side blocks) drag down speedup because they don't participate in the loop-step rebatching.
- **Halting-predictor accuracy is the bottleneck.** A weak predictor means many mispredicted exits (wasted continue-forwards) or many surprise-exits (idle GPU while the scheduler catches up).
- **KV cache is larger per token.** $k$ separate per-layer entries. Worth it when saved compute > extra memory bandwidth for the average token.
- **Not compatible with lockstep speculative decoding.** Draft-verify assumes matching depths; either loop the draft to the verify depth or fall back to non-CDB batching for spec-dec-heavy workloads.
- **First implementation is model-specific.** Kernels currently tuned for Ouro/Huginn; a general Triton implementation is possible but not yet released.

## Sources

- Paper: *Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching* — Schwethelm, Rueckert, Kaissis — TUM + Helmholtz Munich, 2026 — https://arxiv.org/abs/2608.09444
- Model: *Ouro-1.4B* — fully-looped open LM (CDB reports up to 99% of theoretical max speedup).
- Model: *Huginn-3.5B* — partially-looped open LM.
- Related: continuous batching in *vLLM: Efficient Memory Management for LLM Serving* — Kwon et al., 2023.
