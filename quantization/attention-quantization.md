# Attention Quantization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Low-bit quantization of the attention module (Q, K, V, and the softmax-weighted output) is the last mile of LLM quantization: weights and MLP activations already tolerate 4-bit formats, but naive low-bit attention breaks fast. HyQuant proposes a **structural, hybrid-precision** scheme: keep a small set of **vertical-line tokens** (tokens many later tokens attend to heavily) and the **local sliding window** in high precision, quantize the rest aggressively, and fuse the mixed-precision layout into a single kernel that runs on both prefill and decode.

**Prereqs:** [attention](../fundamentals/attention.md), [_number-formats](_number-formats.md)
**Related:** [fp8](fp8.md), [../fundamentals/dca](../fundamentals/dca.md)

---

## What it is

Uniform-precision attention quantization degrades because a handful of tokens carry the mass of the attention distribution — quantize their K/V states into low bits and softmax outliers explode. Prior work smoothed the outliers per-tensor; attention-quantization instead selects, structurally, *which tokens* deserve high precision. The bulk of KV-cache and attention-state entries live in low bits (4-bit or lower), while structural outliers are preserved.

## How it works

**Two structural roles are protected:**

1. **Vertical-line tokens.** Tokens for which many later positions have high attention weight (they show up as vertical stripes in the attention matrix). These are typically the BOS token, punctuation, and a handful of pivot tokens per document. Kept in FP16/FP8.
2. **Local window.** The most recent $w$ tokens (sliding). These are attended to heavily by the current position and dominate the softmax; kept in FP16/FP8.

Everything else — the middle of the sequence — is quantized to low bit-width (INT4 or MXFP4).

**Selection.** Vertical-line tokens are chosen online with a lightweight scorer (running max attention received) and pinned in a small high-precision side buffer; the KV cache stores their index. Local-window pinning is trivial — the last $w$ positions.

**Fused kernel.** Prefill and decode both dispatch a mixed-precision attention op: (a) high-precision matmul over the pinned tokens, (b) low-precision matmul over the compressed bulk, (c) unified softmax across both partitions. The softmax normalization is exact — precision differs only in the inputs to the score computation.

## Why it matters

- **Unlocks long-context serving.** KV cache memory grows linearly in context; 4-bit KV cuts it 4× vs FP16, but only if attention is stable at 4-bit — which requires structural outlier protection.
- **Prefill *and* decode.** Prior attention-quantization schemes handle one path; HyQuant's fused kernel supports both, so quantization applies end-to-end during serving.
- **Composable.** Structural high-precision protection is orthogonal to weight and activation quantization — stacks with GPTQ/AWQ for weights and per-token-scale schemes for activations.

## Gotchas & tricks

- **Vertical-line count matters.** Too few and the softmax breaks; too many and the compressed savings vanish. Empirically a small number (10s) of persistent vertical-line tokens is enough.
- **Model-family differences.** Some architectures (MLA-style) have compressed K/V that already resist quantization better; the vertical-line phenomenon is strongest in standard MHA/GQA.
- **Window size $w$ interacts with RoPE.** Very short windows drop information that RoPE-encoded local structure relies on; keep $w \geq $ a few hundred.
- **Calibration data.** Pick vertical-line candidates on a diverse calibration set — using only in-domain data misses tokens that become vertical lines under out-of-distribution prompts.

## Sources

- Paper: *HyQuant: Hybrid-Precision Quantization for LLM Attention* — Ding, Xing, Zhang, et al., 2026 — [arXiv:2608.27875](https://arxiv.org/abs/2608.27875)
- Related: streaming-attention / attention-sink literature — where the vertical-line phenomenon was first characterized.
- Related: SmoothQuant, AWQ, GPTQ — orthogonal weight/activation quantization schemes that stack with HyQuant.
