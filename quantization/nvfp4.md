# NVFP4 — NVIDIA's 4-bit float for training and RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** NVFP4 is NVIDIA's native 4-bit floating-point format on Blackwell and later (B100/B200/B300), used for both activations and weights (W4A4) with hardware-accelerated GEMMs. For inference it just works; for RL training it has repeatedly tripped on a **cross-engine execution gap** between sampler and learner — matching weights and formats isn't enough. The 2026 miniTE paper reframes the problem and ships kernels that close the gap: on Qwen3-30B-A3B, a single B300 node reaches **33,540 tok/s** for a full learner update, **1.39× BF16** and **1.47×** vanilla Transformer Engine (TE) NVFP4.

**Prereqs:** [_number-formats.md](./_number-formats.md), [fp8.md](./fp8.md)
**Related:** [../pre-training/fp8-training.md](../pre-training/fp8-training.md), [../post-training/_rl.md](../post-training/_rl.md), [../systems/README.md](../systems/README.md)

---

## What it is

NVFP4 is a 4-bit floating-point format with a shared scale per block, backed by Blackwell's native W4A4 Tensor Core matmul path. In the training stack it is used as the storage/compute format for both weights and activations in forward and backward passes, with higher-precision (BF16/FP32) buffers reserved for optimizer state and specific numerically sensitive ops.

Compared to FP8:

- **Memory.** ~2× smaller than FP8, ~4× smaller than BF16.
- **Compute.** W4A4 GEMM throughput roughly 2× FP8 on Blackwell.
- **Range.** Narrower exponent, more sensitive to scaling and outliers.

## How it works

- **Block scaling.** NVFP4 shares an FP exponent (scale) across a small block of elements (typical block size 16); each element stores a 4-bit value interpreted against that scale.
- **W4A4 pipeline.** The Tensor Core consumes NVFP4 weights and NVFP4 activations directly; results accumulate in higher precision and get requantized before the next GEMM.
- **Loss-sensitive ops.** Layer norms, residual-stream reductions, softmax, loss, optimizer updates stay in higher precision; the dial is which GEMMs promote and which stay native.
- **Calibration.** Blockwise scales are chosen either on-the-fly (dynamic) or from a short calibration pass; dynamic is standard for training.

## Why it matters

- **Training throughput.** The 1.39× over BF16 learner throughput on B300 is a meaningful fraction of the generational hardware uplift, free, from the format alone — once numerical stability is handled.
- **Memory.** Weights at 4-bit mean larger per-GPU models at the same memory, which eases parallelism complexity.
- **RL has been the hard case.** Sampler and learner must agree numerically. Matching the format doesn't match the *execution* — different kernels can produce different outputs for the same NVFP4 inputs. miniTE's "execution gap" framing is the first to make this an addressable engineering problem rather than an unfixable precision issue.

## Gotchas & tricks

- **Not just FP8 with fewer bits.** The range is tight enough that outlier handling (per-block scales, per-channel clipping) is load-bearing.
- **Cross-engine mismatch is the RL killer.** Sampler running vLLM/TE kernels, learner running a different custom stack, both consuming NVFP4 — divergence between their outputs breaks the PPO ratio and destabilizes training. See miniTE for a working fix.
- **Hardware-specific.** NVFP4 is a Blackwell+ format; earlier GPUs do not have the native path.
- **Watch the eval gap, not the loss curve.** Loss can look fine while eval scores drift; the paper reports that naive TE NVFP4 trains stably but underperforms on evals vs. miniTE's aligned-execution variant.

## Sources

- Paper: *miniTE: Diagnosing and Bridging the Execution Gap in Native NVFP4 Reinforcement Learning* — Li et al., 2026 — [arXiv:2610.23928](https://arxiv.org/abs/2610.23928) — ships kernels, diagnostics, and training configs; frames the cross-engine execution gap.
- NVIDIA Blackwell architecture docs — native NVFP4 Tensor Core path.
