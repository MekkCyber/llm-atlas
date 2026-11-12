# MACaQ
*Depth — fast, model-agnostic post-training quantization to MXFP4 by making each linear layer self-contained.*

**TL;DR:** A post-training quantization recipe that compresses weights *and* activations to MXFP4 (W4A4) **one layer at a time**. The trick is a block-Hadamard rotation plus per-channel scale that is applied online to activations and folded into weights, so neighboring layers never need to be touched together. Residual error is then shrunk by a Hessian-aware local code search called **Flip-Refinement**. Reaches lower perplexity than DuQuant++ and MR-GPTQ at ~11× lower quantization time, with ~2.3% decode overhead in vLLM, and quantizes a 122B-parameter model in 64 minutes on a single GPU.

**Prereqs:** [_number-formats.md](./_number-formats.md) (MXFP4, E8M0 scale), [fp8.md](./fp8.md) (block scaling intuition).
**Related:** [_number-formats.md](./_number-formats.md), [../inference/README.md](../inference/README.md).

---

## What it is

A 4-bit post-training quantization method targeting the OCP **MXFP4** format (E2M1 elements + E8M0 block scale, block size 32) for both weights and activations. Its defining property: every linear layer is quantized **independently** — no neighboring layers need to be loaded into memory, no cross-layer error propagation to track, no sequential dependency. That makes it both fast and trivially parallelizable across devices.

## How it works

Two parts: an inference-time transform, and an offline weight refinement.

**1. Online rotation + scale.** For each linear layer $y = Wx$, insert an invertible per-channel scale $S$ and a block-Hadamard rotation $H$ on the activation side:

$$ y = W x = \underbrace{W (S H)^{-1}}_{\tilde{W}}\,\underbrace{(S H) x}_{\tilde{x}} $$

Only $\tilde{W}$ is stored (the inverse is baked into it). At inference, apply $S$ then $H$ to $x$ before the matmul. The rotation decorrelates channels so MXFP4's 32-element blocks share a scale more efficiently; $S$ equalizes outlier channels. Nothing about the *rest of the network* changes.

**2. Flip-Refinement.** Starting from round-to-nearest $\tilde{W}$ in MXFP4, iterate over each 4-bit weight and consider the two neighboring MXFP4 codes (up / down one step). For each candidate flip, approximate the layer-output MSE change using a Hessian estimated from the layer's calibration inputs — closed-form O(1) per candidate. Accept if it reduces the estimated error. One or two passes suffice; each layer finishes in seconds.

## Why it matters

- **Per-layer calibration kills the memory ceiling.** GPTQ-family methods need the preceding layer's outputs as calibration; that creates a forward-pass dependency and, at 100B+ scale, forces offloading or sharding. MACaQ breaks the dependency — an H100 can quantize a 122B model in 64 minutes using 32.5 GB.
- **W4A4 at real inference cost.** The Hadamard + scale cost about 2.3% of decode time on Qwen3-32B in vLLM, so the memory savings actually translate into throughput.
- **Portable.** The same recipe applies to a diffusion backbone (FLUX.2 Klein 4B) with no architecture-specific tweaks — a sign the layer-local assumption holds beyond LLMs.

## Gotchas & tricks

- **Block-Hadamard, not full Hadamard.** A dense Hadamard would be O(d²) at inference; MACaQ uses the 32-element block size matching MXFP4's scaling block, so the rotation becomes a per-32-element butterfly. Forgetting this pins GPU utilization at the memory wall.
- **Flip-Refinement's Hessian is diagonal-approximated** and uses a small calibration set (a few hundred sequences). Scale this up only if perplexity plateaus — for most models it already matches global-search methods.
- **The method is sensitive to outlier-channel magnitudes before $S$ is applied.** Papers in this space traditionally fuse SmoothQuant-style scaling with the Hadamard; MACaQ's $S$ is exactly that fusion.
- **Activation quantization is where other W4A4 methods usually fail.** MACaQ treats $SH$ as an online transform on activations and reports no accuracy collapse — the rotation is what rescues activation quantization, not just a weight trick.

## Sources

- Paper: *MACaQ: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations* — Bich, Boretti, Chang, Zhuang, Cavigelli, 2026 — [arXiv:2611.10649](https://arxiv.org/abs/2611.10649).
- Spec: *OCP Microscaling Formats (MX) v1.0* — Open Compute Project, 2023 — MXFP4 and E8M0 scale.
- Prior art: *DuQuant* and *MR-GPTQ* — MACaQ's main W4A4 PTQ baselines.
