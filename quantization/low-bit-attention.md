# Low-Bit Attention (VC-Attention)
*Depth — training-free FP8 attention for Diffusion Transformers via value smoothing and a fused softmax cast.*

**TL;DR:** Low-bit attention has two orthogonal bottlenecks. **Accuracy** is limited by value outliers: unlike Q/K outliers, V outliers follow no fixed channel or spatiotemporal structure and prior smoothing techniques don't address them. **Speed** is limited by the FP32 exponential in softmax — Tensor Cores accelerate the two matmuls, but the softmax between them becomes the longest pipeline stage. **VC-Attention** (Li et al. 2026) fixes both, training-free: **V-Smooth** clusters value tokens online, quantizes only the residual after subtracting the block mean, and restores the mean from the row sum online softmax already maintains. **ExpCast-FP8** maps log-domain scores directly to E4M3 probability codes with one fused multiply-add.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [fp8](fp8.md), [_number-formats](_number-formats.md)
**Related:** [../architectures/mla.md](../architectures/mla.md) · [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)

---

## What it is

A training-free FP8 attention kernel targeting Diffusion Transformers, which now dominate video generation and whose long spatiotemporal sequences make attention the serving-cost bottleneck. Two coordinated techniques, one for accuracy (V) and one for speed (softmax); both preserve the mathematical operation up to quantization noise.

## How it works

### V-Smooth — value outlier handling

Prior work (SmoothQuant, QuaRot) smooths **Q** and **K** because their outliers correlate along channels. **V** outliers do not — they scatter unpredictably across channels and spatiotemporal position. V-Smooth applies three ideas:

1. **Online clustering.** Reorder value tokens with a lightweight streaming clustering pass, so tokens within a hardware block share statistics.
2. **Mean-subtracted residual quantization.** Quantize $(V - \bar V_{\text{block}})$ instead of $V$. The residual has smaller range, so the block's scale is tighter.
3. **Restore mean from the row sum.** The online softmax already maintains $\sum_j p_{ij}$; multiplying the (fixed) block mean by this sum recovers the mean-contribution to the output without a separate reduction.

### ExpCast-FP8 — softmax on the critical path

Log-domain attention scores are typically converted through an FP32 `exp()` and then re-quantized to FP8 for the probability matmul. ExpCast maps a log-domain score directly to its E4M3 probability code in a single fused multiply-add — no FP32 exponential, no format conversion.

## Why it matters

- **Attention is the DiT serving wall.** Every subsequent optimization pass on video-generation systems bumps into it. Removing both accuracy and speed bottlenecks in one training-free kernel is a rare compound win.
- **Speedups matter and stack.** Kernel: **1.46–1.59×** vs BF16 FlashAttention-4 on datacenter Blackwell/Hopper, **2.3–3.6×** on RTX PRO 6000 / RTX 5090. End-to-end clip generation: **1.13–1.19×** (datacenter) / **1.36–1.70×** (workstation).
- **Drops in.** Training-free — no fine-tune, no recipe change, no quality loss beyond documented FP8 numeric noise. Applies to Wan2.2, LongCat-Video, HunyuanVideo-1.5, MiniMax-H3.

## Gotchas & tricks

- **Online clustering is not free.** The clustering pass on V adds work; the paper reports it's net-positive at the studied sequence lengths but budget it into your throughput sums.
- **ExpCast is E4M3-specific.** The direct log-to-code mapping exploits E4M3's structure; porting to other 8-bit formats (E5M2, MXFP8) requires re-deriving the constants.
- **Block-mean restoration must use online-softmax row sums, not pre-softmax sums.** Using pre-softmax sums re-introduces the mean's normalization error and V-Smooth's accuracy gain evaporates.
- **Doesn't help decode-heavy LLMs.** DiTs have long, static sequences where value outliers are the dominant problem. Autoregressive LLM decode has different attention characteristics (KV cache reads dominate) — sparse KV decoding wins there, not V-Smooth.

## Sources

- Paper: *VC-Attention: Value Smoothing and Softmax Casting for Low-bit Attention* — Li, Zou, Zhang, Chen, Xi, Zhang, Zhu, Han, Zhang, Lin, Li, MIT / NVIDIA / CMU, 2026, arXiv 2609.15810.
- Baseline: *FlashAttention-4* — Dao (context: the BF16 baseline this improves on).
- Related: *SmoothQuant* — Xiao et al., 2022; *QuaRot* — Ashkboos et al., 2024 (Q/K smoothing predecessors).
