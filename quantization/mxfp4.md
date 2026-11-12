# MXFP4
*Depth — the OCP 4-bit microscaling format.*

**TL;DR:** MXFP4 is a 4-bit floating-point number format with a tiny per-block exponent. Each 32 elements are stored as E2M1 (1 sign, 2 exponent, 1 mantissa bits) — only **16 distinct values** per element — plus one 8-bit **E8M0** scale (a pure power of two) shared by the block. The shared scale gives it FP-like dynamic range at near-INT4 memory cost, and because the scale is a power of two, rescaling is a bit-shift in hardware.

**Prereqs:** [_number-formats.md](./_number-formats.md) (the taxonomy), [fp8.md](./fp8.md) (how block scaling works for FP8 first).
**Related:** [macaq.md](./macaq.md) (PTQ for MXFP4).

---

## What it is

One of the OCP-standardized **microscaling** (MX) formats. MXFP4's design choices:

| Field | Bits | Values |
| --- | --- | --- |
| Sign | 1 | ± |
| Exponent | 2 | 4 levels |
| Mantissa | 1 | 2 levels |
| Per-block scale (E8M0) | 8 | 2^(E−127), E ∈ [0, 254], E=255 is NaN |
| Block size | — | 32 elements |

Total: 4 bits per element + 8 scale bits / 32 ≈ **4.25 bits/weight** amortized. The 16 representable E2M1 values are {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}; multiplying by the block's E8M0 scale selects an actual numeric range.

## How it works

- **Store** each 32-element block as `[32 × 4-bit element][1 × 8-bit scale]`.
- **Decode** element $e$ with block scale $s$ as `value = decode_e2m1(e) × 2^(s − 127)`.
- **Encode** by choosing the scale that places the block's largest absolute value near E2M1's max (6.0), then rounding each element.
- **Rescale** in hardware by bit-shifting the exponent of the E8M0 scale — no floating-point multiply needed.
- **NaN** has a single encoding (scale = 0xFF); there is no ±∞.

The 32-element block aligns naturally with tensor-core tile widths, so matmul in MXFP4 can scale-multiply once per block instead of once per element.

## Why it matters

- **4× less memory than BF16, 2× less than FP8.** For inference-dominant workloads (KV cache aside) this is the current frontier format.
- **Dynamic range without per-tensor scale fragility.** A single per-tensor scale for INT4 is destroyed by one outlier channel; a per-32 E8M0 scale isolates outliers to their own block.
- **Hardware-native on Blackwell.** B100 / B200 tensor cores run MXFP4 matmul directly. On Hopper and earlier, MXFP4 is software-emulated — you pay the memory savings but not the throughput.
- **One standard across vendors.** The OCP spec means ONNX, vLLM, and TensorRT-LLM all agree on the layout, so a model quantized once runs anywhere.

## Gotchas & tricks

- **W4A4 is harder than W4A16.** Weights in MXFP4 is nearly free quality-wise with modern PTQ; activations in MXFP4 need outlier handling (rotations, per-channel scales). See [macaq.md](./macaq.md) for a layer-local approach.
- **The block size is 32, non-negotiable in the OCP spec.** Custom block sizes (128, 64) exist in research and in pre-OCP hardware; they are *not* MXFP4 even if people call them that.
- **E8M0 saturates.** At E = 0 the scale is 2^-127 ≈ 0, and at E = 254 it is 2^127. A block whose magnitude straddles both ends cannot be represented without re-scaling; this is rare in attention/MLP activations but shows up in router logits.
- **Only 16 element values** means uniform rounding error is much larger than for FP8. Most of MXFP4's quality comes from the *per-block scale search*, not from the element codes themselves.
- **Mixed precision is normal.** Keep sensitive layers (final LM head, routing logits, LayerNorm) in BF16 or FP8. Pure MXFP4 everywhere is rarely the right tradeoff.

## Sources

- Spec: *OCP Microscaling Formats (MX) v1.0* — Open Compute Project, 2023 — MXFP4 and E8M0 scale.
- Paper: *Microscaling Data Formats for Deep Learning* — Rouhani et al., Microsoft, 2023 — the MX paper behind the OCP spec.
- NVIDIA Blackwell architecture whitepaper — hardware support for MXFP4.
