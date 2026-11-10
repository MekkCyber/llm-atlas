# KV Cache Compression
*Depth — shrinking the KV cache at serve time, composably with quantization and eviction.*

**TL;DR:** The KV cache grows linearly with context length and is the dominant memory cost of long-context serving. Compression lives *on top of* the cache's existing optimizations (paged attention, quantization, eviction) rather than replacing them. LowRAM is the clean example: fit a tiny per-head predictor at test time that reconstructs values from keys, keep only hard-to-reconstruct residuals, and pair it with cross-layer low-rank key compression — reports up to 5× peak decode throughput at ~4× compression and 26.9× total compression at 94.5% of baseline when composed with quantization and eviction.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [mla](../architectures/mla.md), [fp8](../quantization/fp8.md), [_number-formats](../quantization/_number-formats.md)

---

## What it is

During autoregressive decoding, each attention layer caches the keys `K ∈ R^(T × d_h)` and values `V ∈ R^(T × d_h)` for every token already processed. For a 70B-class model at 128K context with GQA, this is tens to hundreds of gigabytes of cache per concurrent request — the bottleneck is almost never compute, almost always memory.

KV-cache *compression* is a category of techniques that reduce the stored footprint of `K` and `V`. It splits into:

- **Lossless/near-lossless compression** — low-rank factorization, learned predictors, structured sparsity on `K` or `V`.
- **Lossy with retention policy** — token eviction, quantization, approximate attention.

The modern design discipline is that these should **compose**, not replace. A good compression layer survives the cache being paged, quantized, and selectively evicted underneath it.

## How it works

### Low-rank key compression (cross-layer)

Keys across adjacent layers are correlated. Factorize the stack of per-layer key projections into a shared low-rank basis plus small per-layer residuals:

```
K_l ≈ U · V_l                # U shared, V_l per-layer, rank r ≪ d_h
```

Storage drops from `L · T · d_h` to `T · r + L · T · r · (d_h / r')` for the residuals. Reconstruction at attention time is a `r → d_h` upscale per layer.

### Value prediction with amortized residuals

Within an attention head, `V_t` is largely a function of `K_t` (both come from the same token's projection). Fit a tiny per-head predictor `f_h(k) → v̂` that reconstructs `V_t` from `K_t`. Store only the tokens for which `‖V_t − f_h(K_t)‖` exceeds a threshold — the rest live implicitly in the predictor.

The predictor can be:
- **Meta-learned** across tasks for a good initialization, so prefill is cheap.
- **Fitted at test time** per request if a one-off cost is acceptable.

Either way the predictor is tiny (sub-MB) compared to the values it replaces.

### The composable pipeline

```
raw KV        →  low-rank key compression   →   value prediction + residuals
                                         ↓
                             INT4/FP4 quantization of residuals
                                         ↓
                             attention-sink / eviction for cold tokens
                                         ↓
                             paged KV storage
```

Each layer is near-lossless at its own stage; errors do not compound catastrophically because downstream layers operate on *reconstructions* that are already within tolerance.

### Decode throughput

Compression helps decode throughput by two mechanisms:

1. **More requests per GPU.** Smaller cache → more concurrent sequences → higher aggregate tokens/sec.
2. **Lower cache-read bandwidth.** The attention step's memory-bound operation moves less data per token.

## Why it matters

- **Long context is memory-bound, not compute-bound.** Serving 128K+ context at reasonable throughput requires aggressive cache management. Compression is a bigger lever than raw compute optimization.
- **Composable with the existing stack.** Doesn't fight paged attention, doesn't exclude quantization, doesn't preempt eviction. Deployment teams don't have to rip out their current cache infrastructure.
- **Orthogonal to architectural fixes.** MLA, GQA, and sliding-window attention reduce cache growth by architecture; test-time compression reduces what's already been stored. Both can apply at once.
- **Peak throughput wins at serving time.** LowRAM's 5× peak decode throughput at 4× compression is the kind of number that moves a production economics equation.

## Gotchas & tricks

- **Predictor fit is request-sensitive.** A per-head predictor trained on one domain may underperform on another. Meta-learning helps; falling back to dense residuals on hard tokens is cheap insurance.
- **Error accumulation across layers.** Per-layer reconstruction error can compound through the stack. Validate end-to-end (perplexity, downstream task) on long contexts, not just per-layer MSE.
- **Interacts with attention-sink and streaming eviction.** Compressed caches play well with eviction because eviction operates on token granularity and compression operates on `(key, value)` representation. Don't evict the predictor's calibration set.
- **Quantization order matters.** Compress first, then quantize the residuals — quantizing raw `V` and then trying to predict quantized values is noisier.
- **Benchmarks must span context lengths.** Compression gains concentrate at long context; short-context evals under-estimate the benefit and over-estimate the quality cost.
- **Not useful for prefill.** These techniques are decode-time. Prefill is compute-bound and compression there is a different problem.

## Sources

- Paper: *LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at Test-Time* — Delgado, Hazard, Benfeghoul, Wang, Fountas, 2026 — arXiv 2611.07025 — the primary source for value amortisation and the composable pipeline described above.
- Paper: *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* — DeepSeek, 2024 — architectural KV compression via MLA (see [mla](../architectures/mla.md)).
- Paper: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* — Zhang et al., 2023 — token-eviction cache compression.
- Paper: *StreamingLLM: Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2024 — eviction pattern that composes with compression.
