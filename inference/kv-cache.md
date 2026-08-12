# KV Cache
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** During autoregressive decoding, each new token attends to every previous token's key and value vectors. The **KV cache** stores those K/V tensors so they aren't recomputed on every step — the difference between O(seq²) and O(seq) per generated token. In modern long-context serving, KV cache memory (not weights, not activations) is usually the binding resource.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [../architectures/mla.md](../architectures/mla.md), [speculative-decoding.md](speculative-decoding.md), [sparse-kv-prefetch.md](sparse-kv-prefetch.md)

---

## What it is

Transformer decoding proceeds one token at a time. At step `t`, the model computes a new query `q_t` and attends to the keys/values of tokens `1..t`. If K and V of tokens `1..t-1` were recomputed each step, decoding one token would cost O(t·d) FLOPs *per layer* — quadratic in output length overall. The KV cache is the buffer that keeps `K_{1..t-1}`, `V_{1..t-1}` around so only the new `K_t`, `V_t` need to be produced each step.

Per-token cache size for standard multi-head attention is `2·L·H·d_h` scalars (K and V, over L layers, H heads, head dim `d_h`). At long context this dominates GPU memory — for a 70B model at 128K context this can be 30+ GB per sequence, more than the weights themselves.

## How it works

- **Layout.** Two `[batch, heads, seq_len, head_dim]` tensors per layer, one for K and one for V, allocated up to a maximum context length. New rows are appended at each decode step.
- **Paging.** Naïve contiguous allocation wastes memory on padding and fragmentation. **PagedAttention** (vLLM) splits the cache into fixed-size blocks tracked by a page table, so per-sequence memory is packed and shared across requests where possible.
- **Reuse.** Prompt tokens' K/V are computed once during **prefill** and reused across all decode steps; the same cache is often shared between many completions off the same prompt (prefix caching).
- **Quantization / compression.** FP8 or INT4 KV cache, MLA-style latent compression, or grouped attention (GQA/MQA) all trade a fixed accuracy hit for a proportional memory / bandwidth win.

## Why it matters

- **Sets the throughput ceiling.** More KV budget = larger batch size = higher tokens/sec. Every KV-cache reduction technique translates almost linearly into serving throughput on long-context workloads.
- **Sets the max context.** For a fixed GPU, doubling the per-token cache size halves the maximum context you can serve without spilling.
- **Determines the design of every recent attention variant.** MLA, GQA, MQA, sliding window, sink attention — all are answers to "the KV cache is too big."
- **Enables new serving architectures.** Prefill-decode disaggregation, KV offloading to host/remote memory, sparse-KV prefetching (OasisKV) all exist because the KV cache is the resource worth optimizing.

## Gotchas & tricks

- **Cache lives on device; weights are also on device.** The two compete for HBM. A model that fits at 4K context may OOM at 32K purely from KV growth.
- **Padding-aware batching is required.** Sequences in the same batch grow at different rates; naïve batching wastes cache slots. PagedAttention + continuous batching solve this.
- **Prefill and decode have different cache access patterns.** Prefill writes a whole prefix in one shot (compute-bound); decode does one row at a time (memory-bound). Disaggregating them lets each phase run on hardware tuned for it.
- **Prefix caching's blast radius.** Sharing prefix cache across requests is a huge throughput win but leaks structure — safe to enable within one tenant, risky across tenants.
- **Quantized KV = free memory, but downstream stability matters.** FP8 KV cache is nearly free at inference; INT4 needs careful calibration for long-context recall.

## Sources

- Paper: *Efficient Memory Management for Large Language Model Serving with PagedAttention* — Kwon et al., 2023 — the vLLM paper, canonical paged-KV reference.
- Paper: *DeepSeek-V2* — DeepSeek, 2024 — introduces MLA specifically to shrink KV cache.
- Blog: *How Continuous Batching Enables 23x Throughput in LLM Inference* — Anyscale, 2023 — motivates paging and continuous batching in plain terms.
