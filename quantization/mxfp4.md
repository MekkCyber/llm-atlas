# MXFP4
*Depth — 4-bit microscaling float format for weights (and sometimes activations).*

**TL;DR:** MXFP4 is the OCP microscaling FP4 format: 4-bit floats with a shared exponent block, typically over 32 elements. It is the default 4-bit weight format for NVIDIA Blackwell-class serving stacks and the target precision for open 4-bit releases (GPT-OSS at MXFP4, Hypernova-60B via QAH). Balances the dynamic range of a small exponent against the tiny per-element storage of 4 bits.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** [quantization-aware-healing](quantization-aware-healing.md), [../pre-training/fp8-training](../pre-training/fp8-training.md)

---

## What it is

A microscaling variant of FP4. Each 4-bit element (`E2M1`: 1 sign, 2 exponent, 1 mantissa) is dequantized as `element × 2^scale`, where the scale is a shared **E8M0 exponent** carried once per block of 32 elements. Total footprint per element: 4 bits + 1/32 of a byte for the block scale ≈ 4.25 bits.

## How it works

- **Block layout:** 32 consecutive values along the reduction axis share one E8M0 exponent (8-bit power of two).
- **Element format `E2M1`:** 8 representable positive values × sign bit = 16 states, of which one is zero.
- **Dequant on the fly:** GEMM kernels read the packed block + scale, dequantize per element inside the tensor core input path (Blackwell), and accumulate in higher precision (FP32 or FP16).
- **Weight-only vs A×W:** the common case is weight-only MXFP4 with FP16/BF16 activations. Full MXFP4×MXFP4 GEMMs exist on Blackwell but require calibrated activation scales.
- **Calibration:** post-training quantization typically uses per-block absmax to set the scale; QAT / QAH bakes low-precision awareness into the training loss.

## Why it matters

MXFP4 gives ≈4× weight-memory reduction over bf16 with much better dynamic range than pure INT4 — outliers survive because the exponent scale adapts per 32-element block. It is the target of the current 4-bit serving wave; a hardware-supported format with real production kernels changes what "4-bit-serving" means from lossy INT4 gimmick to first-class option.

## Gotchas & tricks

- Block size of 32 is a hardware choice, not a math preference. Larger blocks reduce scale storage but hurt outlier tolerance; smaller blocks cost too much overhead.
- Naive post-training MXFP4 (round to nearest, per-block absmax) works reasonably for large models but leaves 1–3% MMLU on the table vs QAT/QAH.
- Activations quantized to MXFP4 need per-block scales computed *online* (activations are runtime-dependent), which costs a small softmax-time reduction. Most serving stacks still keep activations at higher precision.

## Sources

- OCP microscaling formats spec (E2M1 + E8M0 shared exponent), 2024.
- Paper: *Quantization-Aware Healing: A Practical Recipe for Recovering Compressed, 4-Bit LLMs* — Ryskulov et al., 2026 — [arXiv:2608.20953](https://arxiv.org/abs/2608.20953)
