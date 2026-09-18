# Sparse KV Decoding — Variable-Precision Key Scans over Offloaded Caches
*Depth — top-k / variable-precision key selection for offloaded KV caches, exemplified by Fathom.*

**TL;DR:** At million-token agent contexts with many resident sessions, the KV cache lives in host memory and the **top-$k$ key scan** (used to pick which cached keys the current query attends to) becomes the bandwidth-bounded step of decoding. Prior sparse decoders (SparQ, Loki, Double Sparsity) read a fixed number of bits per key across the whole cache. **Fathom** (Kalyanarangan 2026) makes the read *per-query, per-channel*: store the 4-bit K cache **channel-major as bit planes**, and let each query spend a byte budget by **reverse water-filling** over variance-weighted channel importance.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/mla.md](../architectures/mla.md) · [../quantization/_number-formats.md](../quantization/_number-formats.md) · [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

A family of decoding techniques that avoid reading the entire KV cache for each attention step:

- **Full attention** reads all $n$ keys at full precision — accurate, bandwidth-bound at long context.
- **Fixed-bit sparse scans** (SparQ / Loki / Double Sparsity) read all $n$ keys at reduced precision (e.g. 68 or 136 bits per key) and use the approximate scores to pick top-$k$.
- **Variable-precision sparse scans** (Fathom) read *fewer* bits from *fewer-important* channels *per query*, adaptively.

Fathom's specific trick — bit-plane storage — is a fundamentally new capability: at read time, the cache's precision is a *choice*, not a stored constant.

## How it works

### Bit-plane storage

The 4-bit K cache is laid out **channel-major**, and each channel is further **transposed to bit planes**:
- Read the first plane → 1-bit quantizer of that channel.
- Read the first two planes → 2-bit quantizer.
- ...up to 4 planes → the full 4-bit K value.

So a prefix of $t$ planes is *exactly* the channel's $t$-bit quantizer, with no re-decoding overhead.

### Reverse water-filling over channel importance

For a given query $q$, each channel $c$ has an *importance* proportional to $q_c^2 \cdot \mathrm{Var}(K_c)$ (variance of the cached values on that channel). Given a byte budget $B$:
1. Sort channels by importance.
2. Fill precision from most-important down until $B$ is exhausted.
3. Read the corresponding bit-plane prefixes for each channel.

Different queries get different bit budgets on different channels; the top-$k$ scan uses the reconstructed keys.

## Why it matters

- **Faster and more accurate than the incumbents** at the same setting. On Qwen3-8B at 1M tokens: decode step **1.67× faster** in GPU time than Double Sparsity / Loki / SparQ (r=32) at 136-bit scans. At *matched GPU time* as SparQ-68-bit (r=16), Fathom reads **18% fewer bytes with lower attention error on 6/7 model+context settings**.
- **Exact top-$k$ on RULER.** For synthetic long-context recall, per-token scans match full top-$k$ decoding.
- **Real-agent robustness.** On coding-agent sessions, Fathom reaches the step-agreement of the best 136-bit scan at only **92 bits**.
- **Reuses existing storage.** The 4-bit K copy is the one a quantized serving stack already holds — no extra storage cost. Only the layout changes.

## Gotchas & tricks

- **Not faster when K is HBM-resident.** Fathom's win is bandwidth; if the K cache fits in GPU memory, its scan is not the bottleneck and the extra bookkeeping is loss.
- **Per-channel variance is a static estimate.** Recomputing variance during long agent sessions is possible but adds cost; the paper uses a fixed estimate.
- **Water-filling is per-token overhead.** At very small batch sizes, the water-filling arithmetic can dominate the read it schedules. Paper measures this — it's a non-issue at production batch sizes but a footgun in microbenchmarks.
- **Bit-plane layout is compatible with 4-bit K only.** Extension to 8-bit K or to V would require re-designing the layout.

## Sources

- Paper: *Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches* — Kalyanarangan, 2026, arXiv 2609.17652.
- Baseline: *SparQ Attention: Bandwidth-Efficient LLM Inference* — Ribar et al., 2023.
- Baseline: *Loki: Low-Rank Keys for Efficient Sparse Attention* — Singhania et al., 2024.
- Baseline: *Double Sparsity: Accelerating LLM Inference with Token and Channel Sparsity* — Yang et al., 2024.
