# Number Formats

*Taxonomy — the numeric types modern LLMs actually use, from FP32 down to 4-bit and below.*

**TL;DR:** Every LLM tensor lives in one of a small zoo of numeric formats. They differ along three axes: **total bit width** (memory cost), **exponent vs mantissa split** (range vs precision tradeoff), and **scale granularity** (per-tensor, per-channel, per-block). For training today the defaults are BF16 with FP32 master weights; for FP8 training, 1×128 / 128×128 tile-scaled E4M3. For weight-only inference quantization, INT4 or NF4 is standard. Microscaling (MX) formats are replacing ad-hoc block scaling as the OCP standard from Blackwell onward.

**Related taxonomies:** none yet.
**Depth files covered here:** [fp8](fp8.md) · others as they land.

---

## The problem

A model weight is just a real number, but hardware stores it in finite bits. Halving the bit width doubles the memory budget and (usually) the matmul throughput — but the same bits can be spent very differently. **Floating-point** formats allocate some bits to an exponent (dynamic range) and the rest to a mantissa (precision); **integer** formats have no exponent but need an external scale; **block floating-point / microscaling** formats share one exponent across a block to get FP-like dynamic range at near-INT memory cost.

What goes wrong with the wrong format:
- Too little **range** → activation outliers overflow to inf or clip to the max, spiking loss.
- Too little **precision** → small updates round to zero, training stalls.
- Too coarse a **scale** → one outlier wrecks every value sharing its scale.

---

## The shared pattern

Every format here can be described by:

```
(bit width) = (sign) + (exponent bits) + (mantissa bits)        ← floating-point
(bit width) = signed/unsigned k-bit integer                     ← integer
(scale)     = per-tensor | per-channel | per-block | per-group  ← external scaling
```

A "value" is reconstructed as:

```
x ≈ decode(element) · scale
```

where `element` is the low-precision storage and `scale` is kept in higher precision (usually FP32, BF16, or the 8-bit E8M0 used by MX formats).

---

## Variants

| Format | Bits | Layout (s / e / m) | Approx. max | Approx. min normal | Typical use |
|---|---|---|---|---|---|
| FP32 | 32 | 1 / 8 / 23 | 3.4e38 | 1.2e-38 | master weights, loss accumulation |
| TF32 | 19 (stored as 32) | 1 / 8 / 10 | 3.4e38 | 1.2e-38 | Ampere tensor-core matmul, legacy |
| FP16 | 16 | 1 / 5 / 10 | 65 504 | 6.1e-5 | older mixed-precision training (pre-BF16) |
| BF16 | 16 | 1 / 8 / 7 | 3.4e38 | 1.2e-38 | **default training dtype today** |
| FP8 E4M3 | 8 | 1 / 4 / 3 | **448** | 2^-9 (subnormal) | FP8 forward activations/weights — see [fp8.md](fp8.md) |
| FP8 E5M2 | 8 | 1 / 5 / 2 | 57 344 | 2^-16 (subnormal) | FP8 gradients |
| MXFP8 | 8 per element + 8 scale / 32 | E4M3 or E5M2 elements, E8M0 block scale | inherits from element format, scaled | — | Blackwell-era FP8 training with hardware block scaling |
| FP6 (E3M2, E2M3) | 6 | 1 / 3 / 2 or 1 / 2 / 3 | ~28 or ~7.5 | varies | experimental; MXFP6 variant |
| FP4 (E2M1) | 4 | 1 / 2 / 1 | 6.0 | 1.0 (only 16 values total) | MXFP4 inference and emerging training |
| INT8 | 8 | signed integer | ±127 | — | GPTQ/SmoothQuant inference, per-channel or per-token scale |
| INT4 | 4 | signed integer | ±7 | — | GPTQ/AWQ weight-only inference |
| NF4 | 4 | 16 pre-defined levels (quantiles of N(0,1)) | symmetric | — | QLoRA weight quantization |
| MXFP4 | 4 per element + 8 scale / 32 | E2M1 elements, E8M0 block scale | 6.0 scaled | — | OCP microscaling 4-bit inference / training |

---

## How to choose

**Training master weights** → FP32. Nothing else has enough dynamic range for AdamW's `v` second moment and long-run accumulation.

**Training compute (forward/backward matmuls)** → BF16 is the safe default. FP16 has been effectively retired in new training runs because its 5-bit exponent is too narrow for gradients without loss scaling. FP8 (either E4M3/E5M2 hybrid or E4M3 everywhere with fine-grained tiles) is the frontier — see [fp8.md](fp8.md).

**MXFP8 vs ad-hoc FP8 tile scaling** → same idea, different packaging. Ad-hoc FP8 (what DeepSeek-V3 does) uses 1×128 / 128×128 tiles with FP32 scales. MXFP8 standardizes the scheme: block size 32, **E8M0** scale (an 8-bit unsigned "exponent-only" scale = power of two), elements in E4M3 or E5M2. Hardware acceleration for MXFP8 landed on Blackwell (B100 / B200) — on Hopper you pay software cost for the scale management.

**Optimizer states** → BF16 for `m` / `v` (as in DeepSeek-V3) saves 2× memory vs FP32 with no measurable quality loss. FP8 optimizer states is an open research area.

**Inference, weight-only quantization** → INT4 (GPTQ, AWQ) is the current default for 4-bit. NF4 (from QLoRA) is common in PEFT workflows. MXFP4 is taking over for hardware with native support.

**Inference, weight + activation quantization** → INT8 with per-channel weight scales and per-token activation scales (SmoothQuant pattern), or FP8 serving (vLLM / TensorRT-LLM).

**Sub-4-bit** → BitNet (ternary), QuIP (2-bit), mostly research. Don't use in production.

### The scale-granularity axis (independent of bit width)

| Granularity | Storage | Where it wins |
|---|---|---|
| Per-tensor | 1 scalar | toy workloads, pure INT8 inference where outliers are absent |
| Per-channel (row or column) | O(d) scalars | SmoothQuant-style weight quantization |
| Per-block (e.g. 128×128) | O(d²/block²) | FP8 training, legacy large-block schemes |
| Per-group along one dim | O(d · (d/group)) | GPTQ, weight-only inference |
| Microscaling (32-element, E8M0) | O(d/32) × 8 bits | MX formats — standardized hardware path |

Scale granularity matters **more than** the last bit of element precision in most LLM failure modes — outliers in a handful of channels are the common killer.

---

## E8M0 explained (the MX scale format)

```
8 bits, unsigned, no sign bit
Interpreted as a biased FP32 exponent: scale = 2^(E - 127)
E = 0        → 2^-127 (smallest scale, effectively zero)
E = 127      → 2^0   = 1
E = 254      → 2^127 (largest finite scale)
E = 255      → NaN (single NaN encoding, no ±∞)
```

Because the scale is a pure power of two, rescaling a block is a bit-shift in the exponent — free in hardware. That's the main reason MX formats are cheap to implement compared to FP32-scale schemes.

---

## Adjacent but distinct

- **Posit / Unum formats** — alternative number systems with tapered precision. Not used in any production LLM system as of 2026; kept here only for completeness.
- **Logarithmic number systems** — values stored as log-of-magnitude. Good for dynamic range, bad for addition. Mostly research.
- **Stochastic rounding** — not a format, a rounding strategy; can be applied to any of the above.

---

## Sources

- Paper: *FP8 Formats for Deep Learning* — Micikevicius et al., 2022 — E4M3 and E5M2 spec.
- Spec: *OCP Microscaling Formats (MX) v1.0* — Open Compute Project, 2023 — MXFP8, MXFP6, MXFP4, MXINT8, and E8M0 scale.
- Paper: *Microscaling Data Formats for Deep Learning* — Rouhani et al., Microsoft, 2023 — the MX paper behind the OCP spec.
- Paper: *QLoRA: Efficient Finetuning of Quantized LLMs* — Dettmers et al., 2023 — NF4 definition.
- Paper: *GPTQ: Accurate Post-Training Quantization* — Frantar et al., 2022 — INT4 with group-wise scales.
- Paper: *SmoothQuant* — Xiao et al., 2022 — per-channel weight / per-token activation INT8.
- NVIDIA Hopper and Blackwell architecture whitepapers — hardware support for FP8 and MXFP formats.
