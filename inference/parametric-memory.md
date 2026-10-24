# Deep Fusion Memory (chunk-level parametric memory)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **parametric** long-context memory for frozen LLMs: a small module predicts chunk-level memory tokens from preceding context, and a **gated cross-attention** layer fuses them into the decoder's hidden states. No retrieval index is called at inference. Compared to token-level parametric-memory methods it uses **32× fewer retrieval queries** and **2048× fewer retrieved items** for training-data construction, and still beats the strongest baseline on QA with Mistral-7B-v0.3 and Qwen3-8B-Base (SJTU LUMIA Lab · Shanghai AI Lab, 2026).

**Prereqs:** [README.md](README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

Memory implemented as a trained module rather than an external vector store. The backbone LLM is kept frozen; a lightweight memory predictor and a gated cross-attention channel are added. At inference, prior context is turned into chunk-level memory tokens *parametrically* (no index, no top-K), and those tokens are fused into the backbone through gated cross-attention. The retrieval tax of RAG — index, store, top-K, re-rank — is skipped.

## How it works

- **Chunk-level memory tokens.** Preceding context is split into chunks (one memory slot per chunk, not per token). A small predictor consumes each chunk and emits a fixed-size bundle of memory tokens.
- **Gated cross-attention.** Each decoder block gains a cross-attention channel over the memory tokens, with a learned gate controlling how much to let in. Backbone attention is unchanged.
- **Frozen backbone.** The pretrained LLM's weights are not updated. Only the memory predictor and the gated cross-attention adapters are trained.
- **Training-data cost savings.** Because memory is chunk-level, constructing the training supervision needs 32× fewer retrieval queries and 2048× fewer retrieved items than token-level parametric-memory baselines.

## Why it matters

- **No retriever at inference.** Removes the latency, cost, and operational complexity of a vector store from the serving path.
- **Backbone-agnostic adapter.** Works on top of frozen open models (Mistral-7B-v0.3, Qwen3-8B-Base); you don't need to retrain the base.
- **Beats token-level parametric memory where it matters.** Chunk-level granularity turns out to be enough for QA; the overhead savings during training-data construction are large.
- **Domain transfer.** QA gains plus domain adaptation gains in code, math, and law — i.e. not just one benchmark.

## Gotchas & tricks

- **Memory predictor is a bottleneck.** It must learn to produce useful chunk tokens; a weak predictor caps downstream quality regardless of fusion.
- **Chunk boundary choice matters.** Sentence-level chunks vs document-level chunks give different signal; the paper's results assume a specific chunking policy.
- **Gated cross-attention adds parameters.** Small, but non-zero — the "free" part is only the retrieval infra, not the model.
- **Not a RAG replacement for every task.** Tasks that need up-to-the-minute external knowledge (news, prices) still need a retriever; parametric memory is for *contextual* memory, not external world state.

## Sources

- Paper: *Deep Fusion Memory: A Chunk-Level Parametric Memory for Large Language Models* — Wei, Sun, Sun, Liu, Shi, Wang, Zhang, Lin, SJTU LUMIA Lab · Shanghai AI Lab, 2026 — [arXiv:2610.25100](https://arxiv.org/abs/2610.25100).
