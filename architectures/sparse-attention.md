# Sparse Attention (Blockwise, GQA-compatible)
*Depth — blockwise sparse attention as a drop-in replacement for dense GQA at million-token contexts.*

**TL;DR:** Dense softmax attention is quadratic in sequence length; at 1M tokens, attention dominates both prefill and decode cost. Blockwise sparse attention reduces per-token compute by selecting a small fraction of *KV blocks* to attend to instead of the full sequence — and crucially does so in a way that respects **Grouped-Query Attention's** shared-KV layout, so the memory bandwidth wins survive on real hardware. Introduced as MSA (MiniMax Sparse Attention, 2026): 28.4× per-token compute reduction at 1M context, 7.6× decode speedup on H800.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [mla.md](mla.md) · [transformer-block.md](transformer-block.md)

---

## What it is

A replacement for dense self-attention in transformer blocks where each query attends to a *selected subset of KV blocks* rather than the full sequence. Two design constraints distinguish production-ready blockwise sparsity from earlier attempts:

1. **Block-aligned with GQA.** The KV blocks shared across query heads in a GQA group must be selected coherently, otherwise the per-head selection thrashes the KV cache.
2. **Bolt-on training.** A scheme that requires retraining the whole stack from scratch is uninteresting; the win has to compose with an existing pretrained GQA model and a short adaptation phase.

---

## How it works

### Block selection

The sequence is partitioned into fixed-size KV blocks (~128–512 tokens). For each query position, a lightweight scorer (typically a low-rank projection on top of the query) ranks blocks and selects the top-K. The selected K is a small fraction of total blocks — at 1M context with 256-token blocks and K=64, attention is over ~16K KV tokens instead of 1M.

### GQA compatibility

In GQA, multiple query heads share one KV head. MSA selects blocks **per GQA group**, not per query head, so all heads in a group attend to the same KV blocks. This means:
- KV cache reads are coalesced across heads (high bandwidth utilization).
- Decode-time block selection runs once per group, not per head.

### Training

The model is adapted from a dense-attention checkpoint with a short fine-tune that teaches the block scorer. The downstream model parameters are largely preserved.

### Inference

- **Prefill** — block selection is computed in parallel across all positions; effective attention cost scales as $O(LK)$ instead of $O(L^2)$.
- **Decode** — per-step attention reads only the selected K blocks of the cache.

---

## Why it matters

- **Million-token contexts become economical.** 7.6× decode speedup at 1M shifts the cost-per-token curve enough that agentic, repo-scale, and persistent-memory workloads stop being prohibitive.
- **Composes with the rest of the stack.** Because the scheme respects GQA, it slots into existing inference engines (vLLM, SGLang) with cache-management changes rather than a kernel rewrite.
- **Adaptation, not retraining.** A short fine-tune from a dense GQA checkpoint is much cheaper than pretraining a sparse-from-scratch model.

---

## Gotchas & tricks

- **Block size is a tradeoff.** Small blocks give finer-grained selection but more selection overhead and worse memory coalescing. Large blocks coalesce well but waste compute on irrelevant tokens inside the block.
- **Selector quality limits the ceiling.** If the scorer misses the *one* block containing the answer, the model can't recover. Train the selector with a quality loss aligned to the downstream task, not just reconstruction.
- **Decode-time selection latency.** The selection step itself runs every decode step; if implemented naively it can eat the speedup. Fuse it into the attention kernel.
- **Long-context retrieval evals still needed.** Needle-in-haystack and multi-hop evals at 1M context are the honest test — perplexity holds up long before retrieval breaks.

---

## Sources

- Paper: *MiniMax Sparse Attention* — Xu et al., MiniMax, 2026 — [arXiv:2606.13392](https://arxiv.org/abs/2606.13392) — introduces MSA, the blockwise GQA-compatible sparse attention scheme.
