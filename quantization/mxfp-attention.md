# MXFP Attention (causally-safe microscaled attention)
*Depth — attention kernels in MXFP8 / NVFP4 microscaled formats, with causal-leakage avoidance.*

**TL;DR:** NVIDIA Blackwell exposes native microscaled FP8/FP4 (MXFP8, NVFP4) at the attention matmul. Naively reusing a GEMM-style per-block scale in causal attention leaks future-token information into the scale of past tokens. A causally-safe attention kernel re-arranges the per-block scales so no scale depends on tokens past the current position, while still packing documents densely in one batch. FLAMIA (Yandex, 2026) is the reference implementation; its MXFP8 and mixed NVFP4/FP8 paths are drop-in stable for Llama-style pretraining and give end-to-end speedups on B200/B300.

**Prereqs:** [fp8.md](./fp8.md) · [_number-formats.md](./_number-formats.md) · [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [../pre-training/fp8-training.md](../pre-training/fp8-training.md) · [../pre-training/_training-stability.md](../pre-training/_training-stability.md)

---

## What it is

A microscaled attention kernel stores Q/K/V tiles in a low-precision element format (E4M3, E5M2, E2M1) plus one shared **block scale** per 32-element group (E8M0). The softmax-time matmul (`QK^T`, `P·V`) reads these as scaled elements; hardware multiplies by the shared scale in the exponent domain, essentially for free.

In GEMM this is fine: every element in a block is independent of every other element. In **causal attention** a block of `K` or `V` tokens may straddle the causal frontier of some queries. If that block's shared scale is set using *all* its tokens, then the scale that a past query sees has been influenced by future tokens — a correctness-breaking information leak.

## How it works

Three things change versus a plain GEMM microscaled kernel:

1. **Asymmetric scale layout along the causal axis.** Scales are computed over only those positions that are already visible to every query within the tile. For a tile of K tokens `[k_i, …, k_{i+31}]` attended by query `q_j`, the scale is derived from the subset `k_{i..min(j, i+31)}`, not the full block.
2. **Forward/backward scale consistency.** The backward pass regenerates (or re-reads) scales under the *same* causal visibility rule as the forward. Mismatched scales between forward and backward produce silently wrong gradients that look like a normal training-stability bug.
3. **Dense document packing is preserved.** Packed-sequence training (multiple documents concatenated into one sample) doesn't lose throughput — document boundaries are treated like additional causal cuts, and the scale layout accommodates them.

FLAMIA ships MXFP8 (E4M3 or E5M2 elements, E8M0 scale, block 32) and a mixed path that keeps `P·V` in NVFP4 while the softmax input stays FP8 for numeric headroom.

## Why it matters

Microscaled formats are Blackwell's compute-density story — up to 2× the throughput of FP16/BF16 flash-attention. Without a causally-safe kernel, teams either stayed in FP16/BF16 for attention (losing that throughput) or accepted silent correctness drift. A drop-in MXFP-attention kernel removes the last blocker to running an end-to-end pretraining run in microscaled precision on Blackwell, lowering the per-token cost of frontier training without a pretraining-loss regression on Llama-scale models.

## Gotchas & tricks

- **Mask the right axis.** Causal leakage hits the *key/value* side, not the query side. A scale derived from a query tile is always safe.
- **NVFP4 is tight.** E2M1 has only 16 representable values. Watch for softmax-output clipping before worrying about the matmul quality.
- **Double-check the backward.** Forward-only correctness is cheap to validate against FP32; a silent backward bug usually shows up as a slow loss divergence 10k+ steps in.
- **Sliding-window / document packing** need the same causal-safe treatment at every window / document boundary, not just the sample boundary.

## Sources

- Paper: *FLAMIA: FLAsh MIcroscaling Attention for Fast and Causally Safe Training on Blackwell* — Yandex Research, 2026 — [arXiv:2611.03344](https://arxiv.org/abs/2611.03344).
- Spec: *OCP Microscaling Formats (MX) v1.0* — Open Compute Project, 2023 — MXFP8, NVFP4, E8M0 scale.
- NVIDIA Blackwell architecture whitepaper — hardware support for MXFP8/NVFP4 tensor cores.
