# FP4
*Depth — the 4-bit floating-point formats (E2M1, NVFP4, MXFP4) and their use in LLM serving and training.*

**TL;DR:** FP4 stores each value as a 4-bit float (1 sign + 2 exponent + 1 mantissa — only **16 distinct values total**) scaled by a shared block exponent. Alone it has far too little range or precision for LLM activations, so every practical variant pairs the element with a **block scale** at fine granularity. The two production variants are **MXFP4** (OCP-standard 32-element blocks with an 8-bit E8M0 scale) and NVIDIA's **NVFP4** (16-element blocks with a smaller FP8 scale). Native FP4 tensor-core support arrived with Blackwell; recent algorithm–system co-design (BitDecoding-FP4) brings FP4 to the KV cache without materially losing accuracy.

**Prereqs:** [_number-formats.md](_number-formats.md), [fp8.md](fp8.md)
**Related:** [../inference/kv-cache-quantization.md](../inference/kv-cache-quantization.md)

---

## What it is

FP4 is the smallest floating-point format in current LLM practice. The bare element is **E2M1**: sign + 2 exponent bits + 1 mantissa bit, giving values { ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 }. That is *not* usable on its own — the dynamic range is below one decade and the precision is ternary-ish. The usable formats wrap the element in a block scale:

- **NVFP4** — NVIDIA's format, 16-element micro-blocks, each block carrying one FP8 (E4M3) scale. Scale is floating point; block is small to control outliers.
- **MXFP4** — OCP microscaling standard, 32-element blocks, each carrying one **E8M0** (8-bit unsigned power-of-two) scale. Rescaling is a bit-shift in hardware.

Blackwell (B100/B200/B300) adds tensor-core support for both, making FP4 matmul run at FP8's throughput.

## How it works

For MXFP4 the decode is `x_i ≈ dequant(E2M1 element_i) × 2^(E - 127)` where `E` is the 8-bit shared exponent for the 32-element block. One memory load of 32 elements + 1 byte of scale supplies 32 floats. Matmul in the tensor core does element multiplies in 4-bit and accumulates in BF16 or FP32, folding the per-block scale into the accumulator at block boundaries.

For NVFP4 the scale is itself FP8 (E4M3), allowing non-power-of-two scales at the cost of a slightly more expensive dequant. 16-element blocks mean roughly 2× the scale overhead per value but better outlier containment per block.

Scaling choice is downstream of outlier distribution: 16-element NVFP4 blocks reduce the chance that a single outlier dominates a block, which is particularly important for activations in attention (queries, probabilities). 32-element MXFP4 blocks are cheaper and fit better for more uniform tensors (quantized MLP weights).

## Why it matters

A 4-bit KV cache is **4× smaller** than BF16 and roughly 2× smaller than FP8 KV. In long-context serving the KV cache dominates memory bandwidth and capacity — pushing it to FP4 reshapes the cost-per-token curve and enables longer contexts on fixed hardware. BitDecoding-FP4 (2026) reports **2.4–3.0× higher peak serving throughput than BF16 KV** on B300 for Qwen3 while staying **within 0.5 points (NVFP4)** or **1 point (MXFP4)** of BF16 accuracy. Weight-only FP4 inference is similarly a free memory win where quantization error is controllable.

## Gotchas & tricks

- 16 distinct values is unforgiving — small shifts in block scale distort entire segments of the tensor. The scale-granularity axis matters more than which FP4 variant you pick.
- Attention probabilities and queries are the hardest things to put in FP4; most practical schemes keep them in FP8 or BF16 and only drop keys/values to FP4 (see [kv-cache-quantization](../inference/kv-cache-quantization.md)).
- Pre-Blackwell hardware emulates FP4 and you pay the dequant cost. Don't promise FP4 speedups without native tensor-core support.
- Training in FP4 is still research; most results so far are inference quantization. MXFP4 training is the near-term frontier (similar story to the FP8 transition circa 2024).

## Sources

- Spec: *OCP Microscaling Formats (MX) v1.0* — Open Compute Project, 2023 — MXFP4, MXFP6, MXFP8, E8M0 scale.
- Paper: *Microscaling Data Formats for Deep Learning* — Rouhani et al., Microsoft, 2023 — the MX paper behind the OCP spec.
- Paper: *BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm–System Co-Design* — Du et al., Edinburgh / Tencent Hunyuan, 2026 — [arXiv:2611.09825](https://arxiv.org/abs/2611.09825).
- NVIDIA Blackwell architecture whitepaper — hardware FP4 tensor-core details.
