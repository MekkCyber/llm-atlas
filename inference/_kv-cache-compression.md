# KV-Cache Compression

*Taxonomy — methods for shrinking the transformer KV cache to fit long contexts in memory.*

**TL;DR:** The KV cache dominates memory for long-context inference — quadratic in context length before compression, linear after. This class of techniques trims the cache along four axes (token, layer, head, precision) to fit longer contexts on the same hardware. The modern default for serving is a *fixed-budget* pruner (H2O, SnapKV, PyramidKV) tuned per workload; *threshold-free* methods (ReFreeKV) and *learned* eviction are the emerging alternatives.

**Related taxonomies:** [../architectures/_moe.md](../architectures/_moe.md), [../quantization/_number-formats.md](../quantization/_number-formats.md)
**Depth files covered here:** [refreekv](refreekv.md)

---

## The problem

The KV cache stores keys and values for every past token, at every layer, for every head. Memory footprint is:

```
KV_bytes = 2 · L · H · d_head · seq_len · precision
```

For a 70B model with 80 layers, 8 KV heads (GQA), d_head = 128, at fp16, 100k tokens of context: ~250 GB. That doesn't fit on a single GPU. The cache must be compressed — either by dropping entries, quantising them, sharing them, or restructuring attention so fewer are needed.

## The shared pattern

Every variant answers the same three questions:
1. **What to keep** — an importance score per (token, layer, head).
2. **How much to keep** — a budget or criterion.
3. **When to decide** — at prefill only, or per decode step.

The design axis usually cited is *budget-based vs threshold-free*: does the method require an operator-tuned target size, or does it derive its own?

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| StreamingLLM | Keep attention sinks + local window | Loses mid-context info | Streaming chat, long single-turn |
| H2O | Score by cumulative attention; keep top-k | Fixed budget must be tuned | General long-context serving |
| SnapKV | Score using a lookahead window at prefill | Prefill-only decision | Prompt-heavy workloads |
| PyramidKV | Deeper layers get smaller budget | Fixed shape | Long prompts, moderate generation |
| TOVA | One-attention-step scoring; discard rest | Aggressive; quality risk | Extreme compression |
| [ReFreeKV](refreekv.md) | Adaptive per-input budget, no threshold | Variable cache size at runtime | Open-domain traffic, unknown distributions |
| KV quantisation (KIVI, KVQuant) | Store K/V in INT4/INT2 | Precision loss on outliers | Compose with pruning |
| MLA (architectural) | Latent-space KV compression at train time | Requires new architecture | New models, not retrofits |

## How to choose

- **Fixed workload with known distribution → H2O / SnapKV / PyramidKV**, tuned to your latency/quality target.
- **Open-domain traffic where inputs span many distributions → ReFreeKV** or another threshold-free method — avoids silent quality collapse on the wrong domain.
- **Extreme memory pressure → compose pruning with KV quantisation.** Prune to a moderate budget, then quantise what remains.
- **Training a new model → MLA** (architectural compression at pretraining), then apply serving-time compression on top.

## Adjacent but distinct

- **Speculative decoding** — shortens *decode* by drafting, doesn't shrink cache. Compose.
- **Paged attention** — memory *management*, not compression. Solves fragmentation, not footprint.
- **Continuous batching** — improves throughput; orthogonal to per-request cache size.

## Sources

- Paper: *ReFreeKV: Towards Threshold-Free KV Cache Compression* — Ni et al., 2026 — introduces the threshold-free objective.
- Paper: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference* — Zhang et al., 2023.
- Paper: *StreamingLLM* — Xiao et al., 2023.
- Paper: *SnapKV* — Li et al., 2024.
- Paper: *PyramidKV* — Cai et al., 2024.
