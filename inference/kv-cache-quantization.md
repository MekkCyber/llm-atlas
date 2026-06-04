# KV-Cache Quantization

*Depth — compressing the K and V tensors of the attention cache to low bit-widths during autoregressive decoding.*

**TL;DR:** Long-context and long-CoT decoding is memory-bound on the KV cache. Quantizing the cache to 4-bit or 2-bit cuts memory and bandwidth roughly proportionally, but naive quantization breaks because (a) per-token magnitudes have heavy outliers and (b) under autoregressive decoding the per-step error *accumulates* over hundreds of generated tokens. Modern KV-quant methods address both via Hadamard rotation (to spread outliers) and variance-aware normalization (to bound per-step error).

**Prereqs:** [../quantization/_number-formats.md](../quantization/_number-formats.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [kv-cache-eviction.md](./kv-cache-eviction.md), [../quantization/fp8.md](../quantization/fp8.md), [../architectures/mla.md](../architectures/mla.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

For decoder transformers, attention reads back the K and V tensors of every previous token. At long context or long CoT, these dominate memory and bandwidth, especially on edge / single-GPU serving. KV-cache quantization stores K and V at low precision (typically 4-bit, increasingly 2-bit) while keeping weights and activations at their training precision.

**Two evaluation regimes — they differ.** Prefill-style benchmarks measure error on a fixed prompt, scoring perplexity once. Autoregressive-decoding benchmarks measure error after the model has generated hundreds of tokens off its *own* quantized cache. KVarN (Huawei, 2026) shows existing 2-bit baselines look fine on prefill but blow up on autoregressive decoding because errors compound.

## How it works

Modern KV-quant pipelines combine three ideas:

1. **Hadamard rotation.** Apply a fixed orthogonal Hadamard transform to K and V before quantization. This spreads per-token outliers across all dimensions, making the post-rotation tensor friendlier to uniform quantization. Standard since QuaRot / SpinQuant for weights and activations; KVarN ports it to the KV-cache regime.
2. **Variance-aware (dual-axis) normalization.** KVarN's main contribution: scale K and V along both the token axis *and* the head-dim axis using running variance, before quantizing. The dual-axis scaling pins per-token magnitude variance and stops the autoregressive-decoding error accumulation that single-axis methods miss.
3. **Per-channel or per-token scale.** Choice of granularity. Per-token (online, recomputed at each step) handles drift; per-channel (offline, fixed at calibration) is cheaper but more brittle on out-of-distribution inputs. KVarN is *calibration-free* — scales are computed online from the stream itself.

A typical 2-bit KV-cache decoder step:

```
K_new, V_new = compute_kv(layer, x_t)
K_new = quantize2bit(hadamard(K_new) * scale_K_t)
V_new = quantize2bit(hadamard(V_new) * scale_V_t)
append to cache
attn = softmax(Q_t · dequantize(K_cache)) @ dequantize(V_cache)
```

## Why it matters

- **Long-CoT reasoning is the killer app.** R1-class and Kimi-class reasoners decode for thousands of tokens; the KV cache is the binding constraint. 2-bit KV cuts cache memory 8× vs FP16 — turning a 32k-context single-GPU serving job from impossible to easy.
- **Composes with MLA / GQA.** [MLA](../architectures/mla.md) compresses K and V into a shared latent; KV quantization compresses what's left. Different axes; stack them.
- **Calibration-free wins for serving.** Production serving rotates prompts across domains; calibration on a static dataset overfits. Online variance normalization works on whatever distribution shows up.

## Gotchas & tricks

- **Prefill ≠ decode.** Methods tuned on prefill perplexity can be silently broken at decode; always evaluate on long-generation benchmarks (MATH500, AIME, HumanEval with full CoT).
- **K and V need different scales.** V tends to have larger outliers and more heavy-tail structure than K. Sharing scales is a common bug.
- **Asymmetric layers.** Early and late layers tolerate different bit-widths; mixed-precision KV (e.g. 4-bit on layer 0, 2-bit elsewhere) is a cheap win.
- **FlashAttention compatibility.** Storage format matters — dequant kernels must fuse cleanly with FlashAttention's tiled compute or you pay back the savings in HBM round-trips.

## Sources

- *KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks* — Muller et al., Huawei, 2026 — [arXiv:2606.03458](https://arxiv.org/abs/2606.03458) — primary source for autoregressive error-accumulation framing and calibration-free dual-axis normalization.
- *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* — Liu et al., 2024 — early per-channel-K, per-token-V baseline.
- *QuaRot / SpinQuant* — 2024 — Hadamard rotation lineage for outlier suppression.
