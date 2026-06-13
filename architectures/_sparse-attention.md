# Sparse Attention

*Taxonomy — attention variants that compute only a subset of the QK² interactions to control long-context cost.*

**TL;DR:** Softmax attention is quadratic in sequence length, and at hundreds of thousands to millions of tokens this becomes the dominant inference cost. Sparse-attention variants drop most QK pairs entirely, attending to a *structured subset* of keys per query. They differ on **which subset** is chosen (fixed pattern, learned mask, block-clustered, content-based) and **how the kernel exploits it** (block-sparse GEMM, gather-based, hashing). The modern direction is block-sparse + learned selection co-designed with a custom kernel so the FLOP savings turn into wall-clock speedup.

**Related taxonomies:** [_moe.md](_moe.md) (sparse along the FFN axis instead of attention), [_normalization.md](_normalization.md)
**Depth files covered here:** [multi-head-attention](multi-head-attention.md) (dense baseline) · [mla](mla.md) (compresses KV cache, complementary) · [minimax-sparse-attention](minimax-sparse-attention.md)

---

## The problem

For sequence length $N$, vanilla attention computes $O(N^2)$ QK dot products and stores $O(N)$ KV per layer per head. At $N = 10^6$, both compute and KV memory go from "manageable" to "infeasible at deployment scale". Long-context inference for agents, repo-scale code, and persistent-memory systems sits firmly on this side of the cliff.

## The shared pattern

Each variant defines, for every query token, a **sparse key set** $S(q) \subset \{1, \dots, N\}$ that it attends to. Attention is then computed only over $S(q)$, and the rest of the softmax is implicitly zeroed. The variants differ on:

- **Pattern selection** — fixed (sliding window, dilated, strided), data-independent learned (LSH, clustered), or content-dependent (per-query top-k key blocks).
- **Granularity** — token-level masks, block-level masks (more kernel-friendly), or head-level patterns.
- **Kernel co-design** — naive masking still pays $O(N^2)$ on GPU; the wins materialize only when the kernel iterates over blocks of $S(q)$.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Sliding window | Each query attends to a fixed window around itself | Loses long-range info | Strong locality (audio, code lines) |
| Dilated / strided | Sparse fixed pattern (Longformer / BigBird) | Hand-designed pattern may miss task structure | General long-context with global anchors |
| LSH attention (Reformer) | Hash queries to buckets, attend within | Bucket-boundary errors, irregular memory | Variable-pattern long sequences |
| Block-clustered (Routing Transformer) | Cluster tokens, attend within cluster | Cluster quality drives quality | When clustering is cheap and meaningful |
| [minimax-sparse-attention](minimax-sparse-attention.md) | Blockwise sparse on top of GQA, learned mask, custom kernel | Pattern training adds complexity | 1M-context frontier serving (matches GQA, 28× cheaper) |
| Native sparse (NSA, DeepSeek) | Hierarchical compression + selection + sliding window | Pattern combines three sources | Production long-context decoding |

Link variants with a depth file; leave others as plain text until a depth file lands.

## How to choose

The modern default for million-token serving is **block-sparse + GQA + a content-dependent selector** with a co-designed kernel (MSA, NSA-style). Sliding-window-only is the fallback when training compute is tight or the task is strongly local. LSH/clustering variants are rare in 2026 frontier deployments — block-sparse selection has caught up on quality and is much friendlier to GPU kernels. Sparse attention combines cleanly with **MLA** (compress the KV cache) and **GQA** (share KV across heads); they attack different axes of the long-context cost.

## Adjacent but distinct

- [mla](mla.md) — KV cache compression. Reduces *memory*, not attention FLOPs. Often stacked with sparse attention.
- [_moe](_moe.md) — sparsity along the FFN axis (expert routing), not the attention axis. Orthogonal.
- Linear attention / SSM (Mamba) — drops softmax entirely for a recurrent state. A different design family than "softmax + sparse mask".

## Sources

- Block-Sparse Attention (Child et al., 2019) — fixed block patterns.
- Longformer / BigBird — dilated + global tokens.
- Reformer — LSH attention.
- Native Sparse Attention (DeepSeek, 2025).
- MiniMax Sparse Attention (2026) — see depth file.
