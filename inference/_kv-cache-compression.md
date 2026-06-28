# KV-Cache Compression

*Taxonomy — bound the memory footprint of the attention KV cache during long prefilling and decoding.*

**TL;DR:** Attention's KV cache grows linearly with context length; for long-CoT reasoning models it dominates serving memory. KV compression evicts or merges past tokens' K/V tensors so the cache stays bounded. Variants differ in *what signal* they use to rank tokens for eviction (attention scores, hidden-state similarity, predictive uncertainty) and *when* they evict (prefill-time vs streaming-decode).

**Related taxonomies:** [_speculative-decoding.md](_speculative-decoding.md)
**Depth files covered here:** [infokv.md](infokv.md)

---

## The problem

The KV cache stores every past token's K and V projections at every attention layer. Memory cost is `O(2 · L · H · d · seq_len)` per request. With reasoning models routinely emitting 10k+ token CoTs, KV-cache is the dominant per-request memory cost — bigger than the model weights *amortized per request*. Either you cap context (hurts long-CoT), shrink batch size (hurts throughput), or compress the cache.

## The shared pattern

Every KV-compression scheme has the same anatomy:

1. **Score** past tokens by some importance signal.
2. **Select** a budget of tokens (top-N, or a streaming policy).
3. **Evict** the rest, or merge them into representatives.

Schemes preserve a subset of "important" K/V tensors and rely on attention's tolerance to missing distant context. The key design choice is the *importance signal*.

## Variants

| Technique | Importance signal | Eviction policy | When it wins |
| --- | --- | --- | --- |
| StreamingLLM | First few + sliding recent window | Fixed window | Pure streaming; long inputs with no need to recall mid-context |
| H2O | Cumulative attention score | Top-K per layer | Long contexts where past tokens have stable attention "heaviness" |
| SnapKV | Recent window's attention to past | Cluster-aware top-K | Long prompt prefilling with focused attention patterns |
| Scissorhands | Persistence under multiple queries | Multi-query test | Robustness under variable query patterns |
| FastGen / adaptive | Per-head sparsity pattern detection | Pattern-specific policy | Heterogeneous attention heads (local vs strided vs heavy-hitter) |
| [infokv](infokv.md) | Attention + per-layer predictive uncertainty (entropy) | Combined score | Long-CoT reasoning, where high-entropy "pivot" tokens influence the distant future |
| KV quantization (FP8 / INT4) | n/a (lossy compression) | Quantize, don't evict | Stack with eviction; saves 2–4× memory at small quality cost |

## How to choose

- **Default for long-CoT reasoning serving: an attention + uncertainty hybrid** ([InfoKV](infokv.md)). Attention-only schemes miss long-range pivot tokens that reasoning depends on.
- **Pure streaming, short attention windows: StreamingLLM-style sinks + sliding window.** Cheapest, no extra signal needed.
- **Heavy-hitter dominated workloads: H2O / SnapKV.** Strongest with stable attention "heaviness" patterns.
- **Memory-bound but quality-sensitive: stack KV quantization** (FP8 KV / INT4 KV) on top of any of the above. Eviction and quantization are orthogonal.
- Practical rule: pick the scheme that *adds the least signal to compute*. Entropy is already computed during prefilling, attention is already collected — both are nearly free. Cluster-based or pattern-detection methods add latency that can erase the memory savings.

## Adjacent but distinct

- **Sliding-window attention** — architectural, not a cache compression. Bounds *what's attended to* not *what's stored*.
- **DCA / dynamic chunk attention** — see [dca.md](../fundamentals/dca.md). Reuses KV by chunk reorganization, not by eviction.
- **Paged attention** — manages KV-cache *allocation* (no fragmentation) but doesn't compress it.

## Sources

- Paper: *Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2023 — StreamingLLM.
- Paper: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* — Zhang et al., 2023.
- Paper: *SnapKV: LLM Knows What You are Looking for Before Generation* — Li et al., 2024.
- Paper: *Scissorhands* — Liu et al., 2023.
- Paper: *InfoKV: Information-Aware KV Cache Compression for Long Reasoning* — Xiao, Birch, Lin, 2026 — entropy + attention signal.
