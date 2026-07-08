# KV-Cache Compression

*Depth — reduce the memory footprint of the transformer KV cache at inference time, without architectural changes.*

**TL;DR:** Transformer decoding stores the K and V tensors for every past token. At long context, the KV cache dominates GPU memory — often larger than the model weights. Compression methods evict, quantize, share, or *predictively prune* KV entries to hold the working set within a memory budget while keeping quality close to the uncompressed baseline. KVpop (2026) is the online-predictive point in this design space: a small learned predictor decides one step ahead which entries won't matter, evicting them before they blow up memory.

**Prereqs:** [README.md](README.md), [../architectures/mla.md](../architectures/mla.md)
**Related:** [../architectures/mla.md](../architectures/mla.md) · [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

The KV cache is a per-layer, per-head tensor storing K and V for every past token. Size per sequence:

$$
\text{bytes} = 2 \cdot L \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot T \cdot \text{bytes/elem}
$$

For a 70B model with 80 layers, 8 KV heads, 128 head dim, 32k context, bf16: ~10 GB per sequence. At batch size 32: ~320 GB — well past a single H100. Compression is one of a small set of levers (paged attention, MLA/GQA architectures, quantization, offload) that keep long-context serving feasible.

## How it works

Four axes on which methods vary:

- **When to compress.** Offline (post-prefill), online-reactive (after each decode step, based on attention weights), or online-predictive (before the next step, based on a learned model of relevance).
- **Which entries to drop.** Attention-score based (H2O, SnapKV), position-based (StreamingLLM keeps sinks + a sliding window), or importance-predicted (KVpop).
- **Whether to drop, quantize, or share.** Eviction removes entries; quantization keeps them at lower precision (KIVI, KVQuant); sharing merges similar entries (SVD-style low-rank compression).
- **Granularity.** Per-token, per-block, per-head, or per-layer.

KVpop's contribution is the *predictive* + *online* combination: a lightweight predictor over recent attention statistics estimates each cached token's future relevance and pops the lowest-scoring entries *before* the next decode step, keeping the working set bounded from above rather than reactively shrinking after it grows. The predictor is cheap enough to run inside the decode loop and trainable with a small amount of teacher-attention data.

## Why it matters

- **Working-set bound is the memory story.** Paged attention and MLA reduce layout overhead and per-head cost; only cache compression reduces the raw entry count. For very long context (100k+), compression is unavoidable.
- **Composable.** Cache compression stacks with paged attention (page size just gets smaller in tokens), quantization (compress + then quantize survivors), and speculative decoding (both draft and target caches shrink).
- **Enables long-context batching.** More sequences per GPU because each takes fewer bytes — often the metric that matters for serving economics.
- **Query-independent policies age poorly.** Recency- or sink-based policies (StreamingLLM) are fast but oblivious to query content. Predictive policies (KVpop) adapt to the sequence but pay a predictor overhead.

## Gotchas & tricks

- **Long-range dependencies.** Aggressive compression can silently drop the token that carries the answer. Evaluate on needle-in-a-haystack + long-form reasoning, not perplexity alone.
- **Attention sinks.** Always retain the first few tokens (StreamingLLM's finding). Dropping sinks tanks quality regardless of what else you keep.
- **Head heterogeneity.** Some heads are retrieval-heavy (need long context), others are local. Per-head compression policies beat uniform ones.
- **Prefix caching.** Compression complicates prefix reuse across requests — a compressed cache is a function of the query, not just the prefix.
- **Predictor training data.** For KVpop-style methods, the predictor is trained on teacher-attention traces from the same model family. Cross-family transfer is untested; expect calibration mismatch.
- **Interaction with speculative decoding.** Compressing the draft cache differently from the target cache can invalidate the speculative match; keep policies consistent or coordinate the drops.

## Sources

- Paper: *KVpop — Key-Value Cache Compression with Predictive Online Pruning* — Schmidinger, Hartl, Stap, Schmied, Böck, Klambauer, Hochreiter, 2026 — online-predictive KV pruning.
- Paper: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* — Zhang et al., 2023 — attention-score based online eviction.
- Paper: *SnapKV: LLM Knows What You are Looking for Before Generation* — Li et al., 2024 — post-prefill compression using a small observation window.
- Paper: *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)* — Xiao et al., 2023 — sink-based sliding window policy.
- Paper: *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* — Liu et al., 2024 — quantization axis of the same problem.
