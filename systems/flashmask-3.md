# FlashMask-3
*Depth — mask-aware distributed attention with Hopper-optimized FA3 kernel and localized mask shards.*

**TL;DR:** Structured attention masks (document, prefix-LM, sliding-window) become a bottleneck in multi-node context parallelism — mask broadcast and uneven mask sparsity across shards cost real time. FlashMask-3 extends FlashMask's compact column-wise mask representation with (1) a Hopper-optimized kernel that pipelines mask processing inside FlashAttention-3, (2) a distributed extension that keeps masks local (no cross-rank mask communication), and (3) a sparsity-aware load balancer with communication overlap. Reports 7.3–11.0% end-to-end training throughput gain for a 118B-A14B MoE on 128× H100.

**Prereqs:** [attention.md](../fundamentals/attention.md), [dualpipe.md](dualpipe.md)
**Related:** [mla.md](../architectures/mla.md), [fp8-training.md](../pre-training/fp8-training.md)

---

## What it is

A training-systems contribution aimed at long-context MoE pretraining, where attention masking becomes the quiet long pole. Three bottlenecks: kernel-level (mask operations stall FA3's pipeline), communication-level (masks broadcast across CP ranks), and balance-level (sparse-mask shards finish early, dense ones straggle).

## How it works

- **Compact column-wise masks.** Inherited from FlashMask: each attention mask stored as a per-column range `[lo, hi]`, dense data types avoided.
- **FA3-integrated kernel.** The Hopper-tuned kernel folds mask-range lookup into FlashAttention-3's prefill pipeline so no extra memory round trip.
- **Localized masks for context parallelism.** Each rank reads only its shard's slice of the column-wise mask; no all-to-all for mask metadata.
- **Sparsity-aware load balancer.** Pre-scans the mask to estimate per-shard compute and reshuffles sequence tokens across ranks so wall-clock per shard is roughly equal; also schedules communication to overlap with the mask-aware compute.
- 12 mask patterns benchmarked (document, prefix-LM, sliding, mixed).

## Why it matters

Long-context MoE training is the hot path for 2026 frontier models. ~10% end-to-end throughput at 128× H100 is tens of thousands of GPU-hours per training run — real money, visible on the balance sheet.

## Gotchas & tricks

- Column-wise mask representation requires masks whose non-zero pattern is per-column contiguous (document/prefix/sliding fit; arbitrary graph masks don't).
- Load balancer's shuffle changes which token goes to which rank; downstream code that assumes contiguous sequence-rank mapping may break.
- Hopper-specific kernel; non-Hopper training needs a fallback path.

## Sources

- Paper: *FlashMask-3: Efficient and Expressive Mask-Aware Distributed Attention* — He, Wu, Xie et al., Baidu Inc., 2026 — [arXiv:2610.17901](https://arxiv.org/abs/2610.17901)
