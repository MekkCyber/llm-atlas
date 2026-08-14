# Post-Training Quantization (PTQ)

*Taxonomy — algorithms that turn a trained high-precision LLM into a low-bit model without additional training.*

**TL;DR:** PTQ is how a trained model gets from BF16 down to INT4 (or below) fast, without a training run. The variants trade off along two axes: **do they need calibration data** (a small held-out set) and **do they optimize weights, activations, or both**. Vanilla RTN needs nothing but produces bad quality below 8 bits; calibration-based GPTQ / AWQ / SmoothQuant close most of that gap; ReRound and its cousins narrow the calibration-free gap further. Pair with the [_number-formats](_number-formats.md) taxonomy to pick a bit-width + algorithm.

**Related taxonomies:** [_number-formats](_number-formats.md)
**Depth files covered here:** [fp8](fp8.md) · [reround](reround.md)

---

## The problem

Training in FP32/BF16/FP8, serving that same model everywhere, is expensive: memory bandwidth caps throughput, quantization halves or quarters memory footprint, and the newest tensor cores double or quadruple throughput on low-bit paths. But naively casting to a low-bit format destroys quality — the failure mode is *outlier activations* and *ambiguous weight rounding* that compound across a deep stack. PTQ methods systematically decide **how to scale**, **how to round**, and **which subset stays high-precision** so that the deployed model matches the trained one on the tasks that matter.

## The shared pattern

Every PTQ method answers three questions:

1. **Granularity.** Per-tensor, per-channel, per-token, per-group — where do scale factors live?
2. **Rounding.** Round-to-nearest, GPTQ's second-order aware rounding, ReRound's midpoint-resolved rounding.
3. **Calibration.** None (RTN, ReRound), a small held-out set (GPTQ, AWQ, SmoothQuant), or synthetic data.

Fixing these three fixes the algorithm.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| RTN (round-to-nearest) | Nearest representable value, per-channel scale | Fast, zero setup — falls off cliff below 8 bits | 8-bit weight-only quant, small models |
| GPTQ | Iterative second-order-aware rounding using inverse-Hessian | Needs calibration data, one-shot but slower | 3–4 bit weight-only quant with quality budget |
| AWQ | Activation-aware channel-scaling before quant | Needs calibration data, extra scaling factors | Small calibration budget, weight-only 4-bit |
| SmoothQuant | Migrate activation outliers into weights via per-channel scaling | Enables W8A8 (activation quant), needs calibration | Serving throughput matters, INT8 W8A8 |
| [reround](reround.md) | Diffusion-guided tie-break for near-midpoint RTN weights | Calibration-free, diffusion model must be trained once | Privacy/no-calibration deployments at 3–4 bits |
| GGUF-style k-quants | Grouped scale + rounding heuristics used by llama.cpp | Extremely fast, CPU-friendly | Local / consumer-GPU inference stacks |

## How to choose

- **Server-side, calibration data available, quality is king** → GPTQ or AWQ for W4A16; SmoothQuant if you also want activation quantization.
- **Server-side, throughput-bound, activations expensive** → SmoothQuant → INT8 W8A8; consider [fp8](fp8.md) if H100+ available.
- **No calibration data / privacy-sensitive** → RTN at 8-bit; ReRound at 3–4 bit.
- **Consumer / local** → GGUF k-quants at 4–5 bits are the pragmatic default.
- **Frontier / MoE experts** → per-expert calibration + finer group sizes; hybrid schemes across expert weights and shared attention.

For most 4-bit weight-only quant runs, GPTQ or AWQ dominates on quality; RTN + ReRound is the *no-calibration* fallback that has become viable rather than a strict upgrade.

## Adjacent but distinct

- [_number-formats](_number-formats.md) — the underlying representations PTQ targets.
- Quantization-aware training (QAT) — trains with fake-quant nodes in the loop; higher quality at low bits but requires a training run.
- Extreme quantization (BitNet, ternary) — trains natively in a low-bit format rather than post-hoc quantizing.

## Sources

- Paper: *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — Frantar, Ashkboos, Hoefler, Alistarh, 2023.
- Paper: *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* — Lin et al., 2023.
- Paper: *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* — Xiao et al., 2023.
- Paper: *ReRound: Reconstructive Rounding to Resolve Midpoint Ambiguity in Calibration-Free LLM Quantization* — Hsieh & Kung, 2026 — [arXiv:2608.11045](https://arxiv.org/abs/2608.11045). See [reround](reround.md).

---

## Conventions

- **Filename:** `_ptq.md` (leading underscore — taxonomy).
- **Folder placement:** `quantization/`, sibling of [_number-formats](_number-formats.md).
- **Scope:** post-training quantization only; training-time (QAT, native low-bit training) belongs in [../pre-training/fp8-training.md](../pre-training/fp8-training.md).
