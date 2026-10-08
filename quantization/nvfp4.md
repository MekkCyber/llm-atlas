# NVFP4

*Depth — NVIDIA's Blackwell-generation 4-bit float: FP4 elements in blocks of 16 with an FP8 scale.*

**TL;DR:** NVFP4 is NVIDIA's variant of 4-bit floating-point. The element is **E2M1** (same 16 values as MXFP4), but the block size is **16** instead of 32 and the per-block scale is an **FP8 E4M3** value instead of the OCP E8M0 power-of-two. Finer blocks + a precision-rich scale means NVFP4 absorbs block-level outliers better than MXFP4, at the cost of 2× the per-block scale storage. Blackwell tensor cores do native NVFP4 matmul.

**Prereqs:** [fp4.md](./fp4.md), [_number-formats.md](./_number-formats.md)
**Related:** [fp8.md](./fp8.md)

---

## What it is

A block floating-point format standardized inside NVIDIA's Blackwell architecture. Within each contiguous block of 16 elements along the reduction dimension of a matmul, every element is an E2M1 FP4 value, and the block carries a single E4M3 FP8 scalar scale. Decode: `value = element · scale`.

The design knob relative to MXFP4 (OCP's microscaling 4-bit): shrink the block to make scale sharing less aggressive, upgrade the scale type to carry mantissa bits.

## How it works

Per 16-element block:

```
amax = max_i |x_i|
scale (E4M3) = quantize_fp8(amax / 6.0)        # 6.0 = FP4 max element
element_i (E2M1) = round(x_i / scale)
```

Decode on the matmul tensor-core:

```
accumulator += (element_i · scale_block) * (element'_j · scale'_block)
```

The accumulator is BF16 or FP32. The multiplication of two FP8 scales produces an FP16-range scale that the tensor core handles natively.

Compared to MXFP4:

| Property | MXFP4 | NVFP4 |
|---|---|---|
| Block size | 32 | 16 |
| Scale type | E8M0 (8 bits, power of 2) | E4M3 (8 bits, FP8) |
| Scale resolution | coarse (powers of 2) | fine (FP8 mantissa) |
| Scale bits per element | 0.25 | 0.5 |
| Outlier containment | one bad element wrecks block of 32 | half the blast radius; scale can represent exact-ish amax |
| Hardware | OCP standard, several vendors | NVIDIA Blackwell |

## Why it matters

- **Rollouts and inference at BF16-equivalent quality on Blackwell**. TRIAGE (2026) demonstrates native NVFP4 for RL rollouts on Qwen3-4B and Qwen3-30B-A3B, reaching up to 2.3× rollout throughput vs BF16 while preserving policy quality — after fixing training-inference mismatch (see [training-inference-mismatch.md](../post-training/training-inference-mismatch.md)).
- **The precision-rich scale is the key design choice.** When a block contains one outlier, an E8M0 scale has to round the amax to the nearest power of two (losing up to a factor of 2), while an FP8 scale can represent it to ~0.4% precision. For blocks of 16 this is enough to make unmodified training recipes work where MXFP4 would need outlier handling.

## Gotchas & tricks

- **Vendor lock-in on scale format.** MXFP4 is OCP-standard; NVFP4 is NVIDIA-specific. Code written for one needs a scale-format rewrite for the other.
- **Scale storage overhead is 2× MXFP4.** For block size 16 at 8 bits of scale, the effective per-element bit cost is 4 + 0.5 = 4.5. For MXFP4 it's 4 + 0.25 = 4.25. Not a dealbreaker but worth sizing memory bandwidth for.
- **Native tensor-core instructions require Blackwell.** No native NVFP4 matmul on Hopper or Ada; software fallback paths exist but lose most of the throughput benefit.
- **Mismatch bites hardest near convergence.** As the policy sharpens, the per-token logprob gap between FP4 rollout and BF16 train-step grows. Papers like TRIAGE formalize this and apply direction-aware corrections.

## Sources

- NVIDIA Blackwell Tensor Core architecture whitepaper, 2024 — NVFP4 definition and tensor-core instructions.
- Paper: *TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning* — Zhang et al., 2026 — [arXiv:2610.07043](https://arxiv.org/abs/2610.07043).
- Comparison baseline: *OCP Microscaling Formats (MX) v1.0* — MXFP4's block-32 / E8M0-scale design.
