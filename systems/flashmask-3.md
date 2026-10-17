# FlashMask-3
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** FlashMask's O(n) column-wise sparse mask (four query-axis boundary vectors per key column) was Ampere-only and single-GPU. FlashMask-3 does two things: (i) ports it to a **Hopper-optimized FlashAttention-3 kernel** with warp-specialized TMA/WGMMA pipelines, and (ii) adds a **distributed context-parallel extension** whose key move is proving that per-rank mask localization reduces to a **zero-communication clip–shift–max** on the boundary vectors.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [dualpipe.md](./dualpipe.md) · [../architectures/_moe.md](../architectures/_moe.md)

---

## What it is

Long-context training (document packing, prefix-LM, shared-question, sliding-window, block-sparse) uses structured attention masks. Representing the mask naively costs O(n²), which eats memory and bandwidth. FlashMask (Ampere) compressed it to O(n) via **four query-axis boundary vectors per key column**: lower-left, upper-left, lower-right, upper-right intervals of query positions that *attend to* each key.

Two things stopped it from being the default:

1. **Hopper's warp-specialization pipeline** (TMA async loads, WGMMA async matmuls, three roles: producer / MMA / correction) requires rewriting the kernel. FlashMask's Ampere kernel didn't expose the mask metadata to those warp roles correctly.
2. **Context parallelism (CP)** — partitioning the sequence across ranks — needs each rank to know what mask tiles it owns. Prior CP attention implementations either re-materialized masks on every rank (defeats the sparsity) or exchanged mask indices via communication (defeats the point of boundary-vector representation).

FlashMask-3 solves both.

## How it works

### Hopper mask-aware kernel

Three-role warp specialization atop FA-3:
- **Producer warps** TMA-load Q/K/V tiles and the mask's boundary vectors.
- **MMA warps** run WGMMA on tiles the mask says are non-empty; skip entirely for empty tiles. The boundary vectors tell them which (q, k) tile pairs to touch.
- **Correction warps** handle the softmax/online-softmax corrections with mask-aware normalization constants.

Mask is checked at **tile granularity**, so sparse tiles cost zero WGMMA — same spirit as block-sparse FA but across arbitrary structured masks.

### Zero-communication mask localization under CP

Key result: under any multi-chunk CP assignment where rank $r$ owns query chunk $[a_r, b_r]$ and key chunk $[c_r, d_r]$, the per-rank mask is obtained by a **clip–shift–max** on the global boundary vectors — a strictly local function of the global metadata. No cross-rank communication is required to re-localize masks.

Proof sketch (paper): because boundary vectors store *query-axis* intervals indexed by key column, intersecting a global interval with a rank's local query range is interval-arithmetic; the resulting local interval is still representable in four boundary scalars. Composition over chunks is max/min on endpoints.

### Sparsity-aware load balancer (HCC-IPO)

Even with correct localization, different chunks of a structured mask have wildly different compute. HCC-IPO (hierarchical compute-cost integer programming optimizer) assigns chunks to ranks to equalize *actual* FA FLOPs under the mask, not raw sequence length.

### Mask-aware compute–communication overlap

CP needs all-gather on K/V (forward) and reduce-scatter on dK/dV (backward). FlashMask-3 **prunes communication** for key tiles that no local query will ever touch under the mask — mask-driven comm elimination, not just overlap. The forward all-gather uses topology-aware hierarchical routing (intra-node NVLink + inter-node network).

## Why it matters

- **Removes mask communication cost from CP.** Before FM-3, mask metadata handling was a non-negligible fraction of CP overhead; now it's zero.
- **Lifts long-context training throughput at Hopper.** +45–141% TFLOPs vs FlashMask on single H100; **1.80×–3.71×** throughput over Megatron-Core CP ring at CP degrees 4–32.
- **End-to-end gains on a real MoE.** +7.3–11.0% end-to-end throughput on a 118B-A14B MoE at 128K context on 128 H100 GPUs. Mask handling genuinely was the bottleneck.

## Gotchas & tricks

- **Mask structure must fit four boundary vectors.** Covers document, prefix-LM, shared-question, sliding-window, block-sparse. Arbitrary learned sparsity patterns don't fit this form and keep paying O(n²) metadata.
- **HCC-IPO assumes mask-FLOPs are predictable.** Dynamic masks (learned sparsity, routing-based) break the balancer's assumption; use with static structured masks.
- **Hopper-specific numbers.** The reported kernel speedups are on H100 TMA/WGMMA; Ampere (A100) still runs FlashMask-2.
- **Compatible with [dualpipe.md](./dualpipe.md).** CP and PP are orthogonal; FM-3 doesn't change pipeline-parallel communication, so DualPipe-style overlap at the pipeline axis stacks with FM-3's at the context axis.
- **Open at [PaddlePaddle/FlashMask](https://github.com/PaddlePaddle/FlashMask).** Reproducing requires PaddlePaddle, which limits ecosystem adoption; PyTorch port is the obvious follow-up.

## Sources

- Paper: *FlashMask-3: Efficient and Expressive Mask-Aware Distributed Attention* — He, Wu, Xie, Bao, Zeng, Yang, Wu, Yu, Wang (Baidu Inc.), 2026 — [arXiv:2610.17901](https://arxiv.org/abs/2610.17901).
- Code: https://github.com/PaddlePaddle/FlashMask
- Precursors: FlashAttention-2 (Dao, 2023); FlashAttention-3 (Shah et al., 2024); FlashMask (Wang et al., 2024).
