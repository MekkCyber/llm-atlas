# KV-Cache Quantization
*Depth — quantize the cached keys and values to low precision (INT4/INT3/INT2), the dominant lever for long-context serving cost.*

**TL;DR:** During decoding, the per-layer KV cache dominates memory (≈ tens of GB for long-context serving). Quantizing K and V to 2–4 bits per element shrinks the cache 4–8× and accelerates attention by reducing memory traffic. The naïve recipe — uniform quantization treating each key as a flat vector — collapses at low bits. *RoPE-aware* allocation (Block-GTQ, 2026) gives more bits to the rotational-frequency blocks that carry the most attention-logit energy and recovers near-fp16 quality at K3V2.

**Prereqs:** [_number-formats](_number-formats.md), [fundamentals/rope](../fundamentals/rope.md), [architectures/multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [fp8](fp8.md), [architectures/mla](../architectures/mla.md)

---

## What it is

The KV cache stores, for each layer × head, the projected keys $K \in \mathbb{R}^{T \times d_{\text{head}}}$ and values $V \in \mathbb{R}^{T \times d_{\text{head}}}$ for every previously decoded token. At long context ($T \in [32k, 1M]$) the cache is the memory bottleneck.

KV-cache quantization replaces the fp16 entries with low-precision ones — typically per-channel (per feature dim) INT4 or below — and dequantizes on the fly when attention is computed. Distinct from weight quantization (which is one-time, off-policy) because KV is *written during inference*.

## How it works

Two failure modes that distinguish KV-cache quantization from weight quantization:

1. **Per-token dynamic range.** Different tokens at different positions span very different ranges. A fixed grid that fits the average token over-quantizes the outliers.
2. **Position-aware structure.** Under RoPE, a key's contribution to the attention logit $q^\top k$ at relative position $\Delta$ decomposes into a position-weighted sum over **2-D rotational-frequency blocks**:

    $$
    q^\top k = \sum_{b=1}^{d/2} q_b \cdot R(\theta_b \cdot \Delta) \cdot k_b
    $$

    Some blocks carry far more of the attention-logit variance than others. Uniform per-channel quantization wastes bits on low-energy blocks and starves high-energy ones.

### The Block-GTQ recipe (Liang, Zhang, Jia, 2026)

For each layer × KV head:

1. Compute a **label-free energy score** per RoPE block (e.g. $\mathbb{E}[k_b^2]$ over a calibration window or, per the paper, derived from TurboQuant-MSE statistics).
2. Greedily allocate integer bit widths to blocks by marginal gain on reconstruction MSE, subject to a target average bit budget.
3. Pack: high-energy blocks get 4 b/dim, low-energy ones get 2 or even 1.

The result is a *per-layer, per-head bit-allocation lookup table* — fixed once at calibration, zero overhead at inference.

### Other live techniques

- **KIVI / KVQuant / WKVQuant.** Per-channel (K) + per-token (V) static quantizers. Strong fp16 → INT4 baselines but don't exploit RoPE structure.
- **TurboQuant-MSE (TQ-MSE).** Statistically optimal uniform quantizer; Block-GTQ uses it as its primitive and only adds the allocation.
- **Mixed precision with fp16 recent-key buffer.** Keep the most recent K tokens in fp16; quantize the rest. Common in production; orthogonal to the bit-allocation question.

## Why it matters

- **Long-context cost is KV-bound.** A 70B model at 128k context spends most of its memory on KV. At K3V3 packed serving, Block-GTQ reports 3.24× compression and 1.34× faster decode than fp16 FlashAttention2 on a single H800; 56 GB → 20 GB peak memory at 128k.
- **Reasoning models suffer disproportionately at low bits.** Uniform TQ-MSE collapses to 0.0 / 0.0 on AIME 2024/2025 at K2V2 for DeepSeek-R1-Distill-Qwen-7B; Block-GTQ recovers 51.7 / 37.5 (vs fp16's 54.2 / 37.9).
- **Quality preservation enables longer contexts.** With low-bit KV, contexts that previously OOM'd become feasible (256k, 512k).
- **Complementary to MLA / sliding window / paged attention.** MLA reduces cache *width*; KV-quant reduces cache *depth (per-element bits)*; sliding window reduces cache *length*. They stack.

## Gotchas & tricks

- **Don't quantize V the same way as K.** V's attention contribution is linear in V (no QK-like rotation) — V can usually take 1 fewer bit than K at matched quality.
- **Outlier handling for V.** A small per-token scale (1 fp16 per token) eliminates the worst clipping at almost no memory cost.
- **Calibration set matters less than you'd think.** Energy scores transfer across domains; a small in-domain calibration set is enough.
- **Packed-cache serving.** To realize the compression in practice you need an attention kernel that reads packed bits. Block-GTQ ships one; FlashAttention3 has partial support.
- **Interaction with speculative decoding.** Different bit budgets per layer can confuse a draft model that shares the same KV cache. Match draft-model precision to target-model decoded layers.
- **RoPE-specific.** The block-energy logic assumes RoPE. For ALiBi, sinusoidal, or NoPE models the structure is different and a flat per-channel allocation is fine.

## Sources

- Paper: *RoPE-Aware Bit Allocation for KV-Cache Quantization* (Block-GTQ) — Liang, Zhang, Jia (HKUST / CUHK / Xiaomi MiMo), 2026 — [arXiv 2606.24033](https://arxiv.org/abs/2606.24033).
- Paper: *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* — Liu et al., 2024 — [arXiv 2402.02750](https://arxiv.org/abs/2402.02750).
- Paper: *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* — Hooper et al., 2024 — [arXiv 2401.18079](https://arxiv.org/abs/2401.18079).
