# Chunk-level parametric memory
*Depth — Deep Fusion Memory (DFM): learned chunk-granularity memory for LLMs with no retrieval at inference.*

**TL;DR:** Rather than retrieving tokens or compressing a KV cache, DFM trains a causal predictor to emit a *chunk-level* memory vector from preceding context, supervised during training against retrieved chunk embeddings. At inference the chunk memory fuses into hidden states via a gated cross-attention side channel, and no external retrieval is run. The backbone stays frozen; only the memory module trains.

**Prereqs:** [attention.md](../fundamentals/attention.md)
**Related:** [mla.md](../architectures/mla.md)

---

## What it is

A middle path between RAG (external retrieval at inference) and long-context attention (expensive). DFM internalizes a learned, chunk-granularity memory signal into the model itself. The chunk is the training unit: supervision happens on chunk-level targets, retrieval cost at training is amortized across many tokens per chunk, and at inference the model operates self-contained.

## How it works

1. **Chunk the context.** Preceding context is split into non-overlapping chunks (fixed length or boundary-aware).
2. **Causal predictor.** A small learned module takes the stream of chunk embeddings so far and produces a chunk-memory vector for the current position. It's causal — only prior chunks are visible.
3. **Supervision.** During training, each predicted chunk memory is pulled toward the embedding of a *retrieved* reference chunk. Retrieval during training is per chunk, not per token — 32× fewer queries and 2048× fewer items than token-level parametric memory baselines.
4. **Fusion.** Chunk memory feeds into the backbone via a gated cross-attention side channel. The main attention stack is unchanged; the gate controls how much memory is admitted into each hidden state.
5. **Inference.** Chunk memory is produced from past context online by the causal predictor. No retrieval system is active.

## Why it matters

- **Collapses retrieval cost at training** by 3+ orders of magnitude vs token-level parametric memory.
- **No retrieval at inference** — serving is just the base model plus a thin memory head. Latency and operational complexity are much lower than RAG.
- **Frozen backbone** keeps the base model's quality intact; the memory module is a bolt-on.
- Reports QA and domain-adaptation gains over both long-context and RAG baselines, though the ceiling on absolute recall is still below exhaustive retrieval.

## Gotchas & tricks

- Chunk size trades specificity vs capacity: too small and the module is just a token-level autoencoder; too large and semantics blur.
- Gated cross-attention gate initialization matters — initializing closed and ramping up during training avoids destabilizing the frozen backbone.
- Reference-chunk retrieval quality during training is a direct cap on memory quality; a weak training retriever bakes weakness into the module.
- Not a replacement for RAG when freshness or source attribution matters — parametric memory can't cite.

## Sources

- Paper: *Deep Fusion Memory: A Chunk-Level Parametric Memory for Large Language Models* — Wei et al., 2026 — [arXiv:2610.25100](https://arxiv.org/abs/2610.25100)
