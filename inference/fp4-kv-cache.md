# FP4 KV Cache
*Depth — 4-bit floating-point quantization of the KV cache for long-context serving.*

**TL;DR:** After MLA / CSA2 have compressed the *shape* of the KV cache, the remaining cached tensors are still stored at BF16 or FP8. **FP4 KV caching** takes them to 4-bit floating-point (E2M1 or block-scaled variant, in DeepSeek's precision-frontier line), roughly halving cache footprint again vs. FP8 and quartering it vs. BF16. Deployed in DeepSeek-V4.1-Flash (2026) as one of four stacked KV-compression tricks — combined with CSA2 and CED it brings the global HBM cache to ~890 bytes/token, ~4× denser than V4-Flash.

**Prereqs:** [../quantization/fp8.md](../quantization/fp8.md), [../quantization/_number-formats.md](../quantization/_number-formats.md), [../architectures/mla.md](../architectures/mla.md)
**Related:** [../architectures/csa2.md](../architectures/csa2.md), [../pre-training/fp8-training.md](../pre-training/fp8-training.md), [../case-studies/deepseek-v4-1-flash.md](../case-studies/deepseek-v4-1-flash.md)

---

## What it is

A 4-bit floating-point representation applied to KV-cache values at inference time. Concrete formats in use:

- **E2M1** — 2-bit exponent, 1-bit mantissa, sign. Standard hardware-friendly FP4.
- **MXFP4** — E2M1 with a per-block scale factor stored separately (OCP microscaling format).
- **DeepSeek block-scaled variants** — the V3 FP8 recipe used fine-grained per-tile scaling (1×128 activation tiles, 128×128 weight blocks); expect a similar per-tile FP4 recipe.

The KV cache is a natural target: it's stored once and read many times, so decompression cost is amortized, while storage/bandwidth savings apply to every attention step.

---

## How it works

At every attention step during decode:

```
compressed latent (from CSA2)
        │
        ▼
   FP4 (E2M1) with per-block scale
        │
    dequantize on read into attention kernel
        │
        ▼
   Softmax attention over reconstructed KV
```

The FP4 tensor lives in HBM (or SSD, via SWA Bounded Replay). The attention kernel dequantizes on read into a higher-precision compute type (BF16 or FP8 accumulate). Because the reconstruction and the attention math fuse into one kernel, extra memory traffic is avoided.

Per-block scaling is what makes FP4 usable for cache values with wide dynamic range: without a scale factor, E2M1's ~4-value dynamic range would saturate on any tensor with outliers. Blocks are typically 32–128 elements; smaller blocks give better accuracy at higher metadata cost.

---

## Why it matters

- **Compounds with cache-shape compression.** MLA / CSA2 shrink the *shape*; FP4 shrinks the *precision*. Together they multiply.
- **HBM bandwidth wins.** For long-context inference, KV-cache reads are the memory-bandwidth bottleneck. FP4 halves the bytes per read vs. FP8.
- **Enables million-token contexts.** DeepSeek-V4.1-Flash serves 1M-token contexts in HBM budgets that would have needed SSD offloading a generation earlier.
- **Follows the DeepSeek precision-frontier line.** V3 established FP8 training; V4.1-Flash establishes FP4 caching. Serving-side low precision is now a first-class design axis.

---

## Gotchas & tricks

- **Block size matters.** Too large → saturation on outliers; too small → metadata overhead. 32-element blocks are a common starting point.
- **Not all layers tolerate FP4 equally.** Some attention heads have larger dynamic range and need FP8 or BF16. Mixed-precision KV cache (per-head or per-layer FP4 vs. FP8) is a natural extension.
- **Rotary embeddings before quantization.** Apply RoPE (or MLA's decoupled RoPE slice) to keys *before* quantization — quantizing then rotating drifts.
- **Kernel fusion is non-negotiable.** Non-fused dequant-then-attend defeats the bandwidth win. Requires custom attention kernels.
- **Accuracy validation.** FP4 caching is *not* free at model quality; the paper's headline "quality goes up despite compression" is at the model level (the model was co-designed), not the isolated FP4 tradeoff. When retrofitting FP4 to an existing model, expect some quality drop.
- **SSD-resident FP4 caches** need decompression pipelining with SSD reads to hide latency.

---

## Sources

- Paper: *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression* — DeepSeek-AI, 2026 — [arXiv:2609.19969](https://arxiv.org/abs/2609.19969).
- Precedent: FP8 training in *DeepSeek-V3* — DeepSeek-AI, 2024 — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).
- Format reference: OCP microscaling FP4 (MXFP4) spec.
