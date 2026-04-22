# FP8
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An 8-bit floating-point format, standardized in two variants — **E4M3** (higher precision, narrower range) and **E5M2** (wider range, lower precision). Halves memory vs FP16/BF16, doubles tensor-core throughput on H100+. The tradeoff: range and precision are both small enough that naive per-tensor scaling is fragile, so practical use needs per-channel or per-tile scales.

**Prereqs:** none (just know what IEEE-754 floats are).
**Related:** mixed-precision training · matmul accumulation · per-tile quantization

---

## What it is

A floating-point format where each number takes 8 bits:

```
                      E4M3                     E5M2
 bit layout    [s | e e e e | m m m]     [s | e e e e e | m m]
 sign           1                         1
 exponent       4                         5
 mantissa       3                         2
 bias           7                        15
 max finite     448                      57344
 min positive   2^-9 ≈ 1.95e-3           2^-16 ≈ 1.53e-5
 (subnormal)
```

The two formats trade off range vs precision:
- **E4M3** — more mantissa bits → smaller quantization error, but small dynamic range. Used for **forward activations and weights**, where values are well-bounded.
- **E5M2** — more exponent bits → wider dynamic range (same as FP16), but coarser. Used for **gradients**, where outliers span many orders of magnitude.

Both formats were standardized in the 2022 "FP8 Formats for Deep Learning" paper by NVIDIA/Arm/Intel, and implemented in hardware starting with NVIDIA Hopper (H100) and AMD MI300.

---

## How it works

### The value formula

For a non-zero FP8 value with sign bit `s`, exponent field `E`, mantissa field `M`, exponent bias `bias`, mantissa width `m`:

**Normal** (`E ≠ 0`):

```
v = (-1)^s · 2^(E - bias) · (1 + M / 2^m)
```

**Subnormal** (`E = 0`):

```
v = (-1)^s · 2^(1 - bias) · (M / 2^m)
```

**Special case for E4M3**: the standard IEEE pattern would reserve `E = 0b1111` for ±∞ and NaN, giving a max of 240. The E4M3 FP8 spec sacrifices ±∞ (keeping only one NaN encoding) to reclaim those values for finite numbers — that's why max E4M3 = **448** and not 240. E5M2 keeps the standard ±∞ / NaN behavior.

### Concrete: reading an E4M3 bit pattern

Take `0 1000 101` = `s=0, E=8, M=5`.

```
v = 1 · 2^(8 - 7) · (1 + 5/8) = 2 · 1.625 = 3.25
```

### Quantization

To represent a higher-precision tensor `x` (FP32 / BF16) in FP8, pick a scale `s` and compute:

```
x_fp8   = round(x / s)    ← clipped to the FP8 representable range
x_recon = x_fp8 · s       ← when read back into FP32/BF16
```

The scale `s` is stored separately in higher precision (FP32 or BF16). Choosing it is the entire game:

```
s = max(|x|) / max_fp8      ← "absmax" scaling; fits the whole range of x
                              into [-max_fp8, +max_fp8]
```

`max_fp8 = 448` for E4M3, `57344` for E5M2.

### Scale granularity

A single scale per tensor is the cheapest but most fragile — one outlier makes every other value round to ~0. Common granularities, from coarse to fine:

| Granularity | Scale shape for a weight `W: [out, in]` | Cost | Robustness |
|---|---|---|---|
| Per-tensor | 1 scalar | ~free | low |
| Per-channel (row or col) | `[out]` or `[in]` | one per row/col | medium |
| Per-block (e.g. 128×128) | `[out/128, in/128]` | one per tile | high |
| Per-group along a dim | `[out, in/group]` | one per group | high |

Finer granularity = more scales stored alongside the FP8 tensor, but each scale covers a narrower slice so outliers are contained locally.

A concrete modern recipe (DeepSeek-V3): **1×128 tiles for activations** (one scale per token per 128-channel group) and **128×128 blocks for weights**.

### FP8 matmul

Hardware FP8 matmul on H100 etc. takes FP8 inputs and accumulates to higher precision:

```
C[i,j] = Σ_k  A[i,k] · B[k,j]        (A, B in FP8)
                                      (accumulator in FP32 or BF16)
```

The tensor core does a chunk of the sum in its native accumulator (FP22-ish on H100, FP16 partial on some hardware) and periodically promotes to FP32. If you want strict FP32 accumulation, you must force a flush every N inner-dim elements — most codes land on N = 128.

### Why FP8 works at all for LLMs

Weights and activations in trained transformers are approximately log-normal, so they fit well into a floating-point (not integer) representation. The main failure mode is **activation outliers** in specific channels of certain layers (attention output projections, some FFN-down projections). Per-tensor FP8 scaling gets wrecked by them; per-channel or per-tile scaling contains them.

---

## Why it matters

- **2× memory savings** over FP16/BF16 — directly translates to 2× larger batches, or a larger model on the same GPU.
- **2× tensor-core throughput** over FP16 on Hopper, and more on newer chips — wall-clock speedups of 1.5–1.9× are typical for matmul-dominated workloads.
- **Cheaper KV cache** and serving memory for inference-side quantization.
- Enables the end-to-end-FP8 training regime (with fine-grained scaling), which is where most frontier models are heading.

---

## Gotchas & tricks

- **E4M3's 448 max is not a typo.** It comes from sacrificing ±∞ to recover two extra finite exponents. If you compare against a naïve (E, m, bias) calculation you'll get 240 and be confused.
- **Use E4M3 for forward, E5M2 for backward — unless you have fine-grained scaling.** With 1×128 activation tiles the local dynamic range is bounded enough that E4M3 works for gradients too; that's what DeepSeek-V3 does.
- **Round-to-nearest-even** is the only rounding mode that doesn't bias training. Truncation introduces systematic drift.
- **Denormals/subnormals matter in FP8.** The subnormal range is a meaningful chunk of the representable numbers (unlike FP32 where you can often ignore them). Don't flush-to-zero.
- **Scale storage is not free.** A 128×128 block scale on a 7168×7168 weight matrix costs an additional 7168/128 × 7168/128 = 3136 FP32 scales ≈ 12 KB — negligible for weights, but per-tile activation scales recomputed every step add up in bandwidth.
- **Accumulation precision matters more than quantization precision.** Most FP8 failure stories trace back to tensor cores accumulating to FP16/BF16 over long inner dims. Force FP32 accumulation (periodic flush) if your hardware default isn't already FP32.
- **Not everything should be FP8.** Embeddings, the LM head, softmax, normalization, and the MoE router are almost always kept in BF16/FP32 — tight numerics, small FLOP share, nothing to gain.
- **E4M3 has one NaN, E5M2 has many.** E4M3's reclaiming of `0b1111_xxx` bit patterns means there's only one NaN encoding. Important if you're writing hand-rolled kernels that check for NaN.

---

## Sources

- Paper: *FP8 Formats for Deep Learning* — Micikevicius et al., NVIDIA / Arm / Intel, 2022 — the spec for E4M3 and E5M2.
- Paper: *Using FP8 for Deep Learning Training* — Sun et al., 2020 — one of the earlier explorations of 8-bit FP training before standardization.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — end-to-end FP8 training with 1×128 / 128×128 tile scaling and forced FP32 accumulation.
- Spec: OCP Microscaling Formats (MX) — 2023 — successor standards for per-block FP/INT formats.
- NVIDIA Hopper Tuning Guide — documents the hardware FP8 matmul path and accumulation behavior on H100.
