# SoloQ
*Depth — calibration-free rotation-based PTQ for diffusion language models.*

**TL;DR:** Diffusion language models (dLLMs) break standard PTQ pipelines: activation statistics shift *across denoising steps*, so a quantizer calibrated at one step is wrong at another, and calibrating per step is expensive. SoloQ applies an analytic **structured rotation** to weights and activations that produces a step-invariant distribution, letting one quantizer serve all denoising steps. Combines the rotation with a lightweight per-step scale correction. Supports custom codebooks and NVFP4. 4-bit SoloQ beats calibration-based baselines; up to **2.61× peak-memory reduction** and **2.24× inference speedup** with NVFP4.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** [attention](../fundamentals/attention.md)

---

## What it is

A post-training quantization method specifically for diffusion LLMs (dLLMs like LLaDA, Mercury). "Calibration-free" means no representative calibration set is required — the rotation is computed analytically from the model weights themselves, so the recipe runs offline from a model checkpoint alone.

## How it works

Three stages:

1. **Structured rotation.** Apply an orthogonal rotation to both weight columns and activation rows so that the resulting distribution has predictable (approximately normal, step-invariant) statistics. The rotation is derived analytically — it doesn't need data to fit — and is fused into the surrounding linear layers at inference time, so there is no runtime overhead.
2. **Lightweight scale correction.** A single scalar per step rescales the quantized activations to absorb the residual step-dependent drift. Compute cost is negligible.
3. **KV-cache compression (block-diffusion variants).** When the dLLM is block-diffusion-style (commits a prefix and samples the rest), the persistent KV-cache is also quantized at commit time, using the same codec.

## Why it matters

Diffusion LLMs are gaining real traction (LLaDA, Mercury) but have had no solid quantization story — standard PTQ methods assume stationary activation distributions. Calibration-free matters practically: dLLM users typically don't have representative calibration sets, and the dynamic-across-steps behavior defeats small-sample calibration anyway. 4-bit NVFP4 inference with 2.6× memory savings pushes dLLMs toward being deployable on commodity hardware.

## Gotchas & tricks

- **Not for AR LLMs by default.** The method targets the step-wise distribution shift unique to diffusion sampling; applying it to a standard autoregressive LLM gives no advantage over existing rotation methods (QuaRot, SpinQuant).
- **Codebook choice.** The paper shows both custom codebooks and NVFP4; NVFP4 gives the headline speedups because it has hardware support, but custom codebooks match or beat its accuracy.
- **Block-diffusion-only KV compression.** The KV path is only relevant for block-diffusion dLLMs; pure diffusion LMs don't persist a KV between denoising steps.

## Sources

- Paper: *SoloQ: Calibration-Free Quantization for Diffusion Language Models* — Ghosh, Manjunath, Rhee, Kook, Xiao, Kim, Panda, 2026 — [arXiv:2610.07121](https://arxiv.org/abs/2610.07121)
- Related: *QuaRot*, *SpinQuant* (rotation-based PTQ for AR LLMs; different distribution-shaping problem).
