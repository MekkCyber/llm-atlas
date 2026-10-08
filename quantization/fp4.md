# FP4 (E2M1)

*Depth — the 4-bit floating-point element used by MXFP4 and NVFP4, now the frontier for both inference and RL rollouts.*

**TL;DR:** FP4 is a 4-bit float with layout **1 / 2 / 1** (sign / exponent / mantissa), i.e. the **E2M1** encoding. It represents only 16 distinct values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. Alone it is useless for LLM tensors (way too little range) — in practice FP4 only ships inside a **microscaling block format** (MXFP4 at block size 32, NVFP4 at block size 16) where every block carries a higher-precision scale. The combination gives FP-like dynamic range at near-INT4 memory and bandwidth cost.

**Prereqs:** [_number-formats.md](./_number-formats.md), [fp8.md](./fp8.md)
**Related:** [nvfp4.md](./nvfp4.md), [fp8-training.md](../pre-training/fp8-training.md)

---

## What it is

FP4 is the 4-bit element in the microscaling family. The E2M1 encoding packs:

```
 1 sign bit
 2 exponent bits (bias 1)       → exponents in {−1, 0, 1, 2}
 1 mantissa bit                 → 1.0 or 1.5 fraction
```

The 16 representable magnitudes are the finite values of this format. Nothing more. There is no NaN or ±∞ encoding inside the element — those live in the block scale.

Scale granularity is where FP4 becomes usable. In **MXFP4** (OCP, block size 32) the scale is an 8-bit E8M0 power-of-two. In **NVFP4** (NVIDIA Blackwell, block size 16) the scale is an FP8 E4M3 value — twice the per-block storage cost, half the block, but much finer scale resolution.

## How it works

Decoding a single FP4 element inside its block:

```
value_i = element_i (E2M1) · block_scale
```

Encoding (per block of N elements):

```
amax = max_i |x_i|
block_scale = round_up_to_representable(amax / 6.0)   # 6.0 = FP4's max element
element_i = round(x_i / block_scale, to nearest E2M1 grid)
```

The round-up policy on the scale is important: round-down overflows outliers to ±6 · scale, which clips. In practice frameworks also apply **outlier handling** (clip, saturate, or spill to a side channel) because a single outlier in a block of 32 wrecks every other element's precision.

Hardware does matmuls directly in the format on Blackwell (B100/B200) and Hopper for MXFP4 — no software dequant-to-BF16-matmul path. The gain is both memory bandwidth and tensor-core throughput.

## Why it matters

- **4× bandwidth over FP16/BF16**, 2× over FP8, at close-to-dense quality for weight-only inference.
- **Enables FP4 rollouts in RL**. TRACE (2026) uses MXFP4 end-to-end — weights, activations, and KV cache — during RL rollouts and matches BF16 policy quality while running 5.4× faster. TRIAGE (2026) does the same with NVFP4 at 2.3× rollout throughput over BF16.
- **The default low-precision serving format post-Blackwell.** MXFP4 is baked into Blackwell tensor cores and the OCP spec, so cross-vendor interop should be better than for the FP8 generation (where everybody shipped a slightly different scaling scheme).

## Gotchas & tricks

- **FP4 element range is just ±6.** Anything outside the block's `6 · scale` bound is clipped. Choose the scale conservatively (round-up) and expect to redo the scale frequently.
- **The two FP4 flavors differ at the scale, not the element.** MXFP4 and NVFP4 both encode the element as E2M1; what changes is block size (32 vs 16) and scale format (E8M0 vs FP8 E4M3). Code written for one is a short rework away from the other.
- **Outliers still kill you.** Standard outlier mitigation (clipping, per-channel smoothing à la SmoothQuant, or an exception path) is as important as for INT4.
- **Gradient quantization is the hard problem.** Forward FP4 works; FP4 gradients require a lot more care because gradient dynamic range is much wider than activation range.
- **Rollout vs training-step mismatch.** Running the rollout in FP4 while training in BF16 introduces training–inference mismatch (see [training-inference-mismatch.md](../post-training/training-inference-mismatch.md)). This is a known problem in FP4-RL and the subject of papers like TRIAGE.

## Sources

- Spec: *OCP Microscaling Formats (MX) v1.0* — Open Compute Project, 2023 — MXFP4 definition.
- Paper: *Microscaling Data Formats for Deep Learning* — Rouhani et al., Microsoft, 2023.
- NVIDIA Blackwell architecture whitepaper, 2024 — NVFP4 (block size 16, FP8 scale).
- Paper: *TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models* — Wang et al., Alibaba, 2026 — [arXiv:2610.07767](https://arxiv.org/abs/2610.07767).
- Paper: *TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning* — Zhang et al., 2026 — [arXiv:2610.07043](https://arxiv.org/abs/2610.07043).
