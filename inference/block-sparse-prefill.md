# Block-Sparse Prefill Attention (FlashPrefill V2)
*Depth — a memory-optimized block-sparse prefill kernel with a mean-correction term that closes most of the approximation gap to dense attention.*

**TL;DR:** Prefill on long-context LLMs is quadratic in sequence length and dominates inference cost for RAG / code / agentic workloads. **FlashPrefill V2** is a block-sparse prefill attention kernel that (a) adds an explicit **mean correction** term to compensate for the mass dropped by block sparsity, (b) redesigns the operator for memory efficiency, and (c) plugs natively into modern inference frameworks. On NVIDIA H20 at 128K context: **up to 47.26×** speedup over FlashAttention-2 in FP8 and **27.19×** in BF16.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../quantization/fp8.md](../quantization/fp8.md)
**Related:** [../architectures/mla.md](../architectures/mla.md)

---

## What it is

A drop-in prefill attention kernel with three properties:

- **Block-sparse.** Attention is computed over a selected set of (query-block, key-block) pairs rather than the full N×N matrix.
- **Mean-corrected.** For blocks that are *dropped* rather than computed, a mean correction estimates their contribution to the softmax denominator and output, keeping approximation error bounded.
- **Serving-native.** Ships as an operator wired into modern serving frameworks (vLLM / SGLang-style stacks), not just a research kernel.

## How it works

**Selection.** For each query block, a small selection step chooses which key blocks to compute exactly. The selection can use structured patterns (windowed, strided) or content-derived scores; the paper's operator is agnostic to how the selection is produced.

**Mean correction.** Naive block-sparse attention drops the softmax mass and output contribution of un-selected blocks — this compounds as sequence length grows. FlashPrefill V2 estimates a **mean over the dropped block group** (single scalar contribution to the denominator + a mean value vector contribution to the numerator), then adjusts the per-query softmax accordingly. This closes most of the accuracy gap to dense attention without recomputing anything.

**Memory optimization.** The operator is written to minimize on-chip memory traffic under the block-sparse access pattern (tiling that respects the selected-block layout, avoiding gather overhead that has historically killed sparse-attention wall-clock).

## Why it matters

Long-context prefill is *the* bottleneck for RAG, code understanding, and agentic serving. FlashAttention itself moved from research artifact to serving-stack standard once its kernel-integration story landed; a block-sparse variant that (a) preserves quality via mean correction and (b) ships operator-first has the same trajectory available. 27–47× at 128K on H20 is production territory.

## Gotchas & tricks

- Mean correction assumes the dropped blocks are approximately zero-centered in value space; heavy-tailed value distributions (or highly-attended outlier tokens in the dropped set) blow the approximation. Selection must protect outliers.
- FP8 gains (47.26×) exceed BF16 (27.19×) because the sparse compute is memory-bound and FP8 halves the memory footprint of both keys and values — the block-sparse and FP8 stories multiply.
- H20 has specific memory-bandwidth characteristics; speedups on H100 / MI300X will differ and should be re-measured before promising them in a serving contract.

## Sources

- Paper: *FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM Serving* — Fan, Huang, Wu, Wang, He, 2026 — [arXiv:2608.19758](https://arxiv.org/abs/2608.19758)
