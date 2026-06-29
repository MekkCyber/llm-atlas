# ViQ — Text-Aligned Visual Quantized Representations
*Depth — a native-resolution discrete visual tokenizer that closes much of the gap to continuous CLIP-style encoders.*

**TL;DR:** Most visual tokenizers are either **continuous** (CLIP/SigLIP — semantically rich, awkward for unified text+vision LLMs) or **discrete** (VQ-VAE-family — fine-grained but text-misaligned). ViQ trains for text-alignment *first*, then quantizes — and conditions the codebook on spatial position so a single shared codebook covers any input resolution natively without retiling. Competitive accuracy with continuous encoders while accelerating training 20–70%; PSNR 22.73 / rFID 0.62 reconstruction.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md), [../quantization/_number-formats.md](../quantization/_number-formats.md)
**Related:** [../architectures/mla.md](../architectures/mla.md)

---

## What it is

A visual tokenizer designed for the "unified text+vision token" goal. The pitch: representing images as discrete tokens in the same vocabulary space as text simplifies multimodal modeling and makes training cheaper. But naively discretizing image features loses too much information — and existing discrete tokenizers (VQ-VAE descendants) aren't aligned with text semantics, so the multimodal LLM has to bridge two distributions.

ViQ aims for the best of both: **discrete tokens that are semantically text-aligned**, with **native-resolution input handling** (no fixed-size tiling).

## How it works

Two training stages:

1. **Text-aligned pre-training.** Train a continuous vision encoder with a CLIP-style contrastive objective against text. Same shape as SigLIP / CLIP — produces semantically rich continuous features.
2. **Feature discretization.** Quantize the continuous features into a codebook with two key innovations:
   - **Proximal representation learning.** A regularizer that keeps the continuous features close to their nearest codebook entry during stage 2 — stabilizes VQ training, avoids the dead-code problem.
   - **Position-aware quantization.** The codebook lookup is conditioned on spatial position, so the same codebook serves any input resolution. Token *embeddings* depend on `(content, position)`; effective codebook capacity scales with resolution without growing the actual codebook.

The final output is a sequence of discrete codebook indices that can be embedded into the same vocabulary as text tokens.

## Why it matters

- **A real candidate for the unified text+vision LLM stack.** If the multimodal LLM can consume discrete vision tokens with the same next-token-prediction objective as text, no special-cased continuous encoder is needed — no projection layer, no separate optimizer, no two-tower complexity.
- **Native-resolution.** Production multimodal systems take wildly varying input sizes; ViQ handles them without resampling or tiling.
- **Training speed: 20-70% faster** than continuous-encoder baselines at competitive quality.
- **Reconstruction strong enough for dual use:** PSNR 22.73, rFID 0.62 — the same tokens serve both understanding and generation, which is the right shape for any-to-any multimodal models.

## Gotchas & tricks

- Stage 1 needs the standard CLIP-scale paired text-image corpus; the contribution is the stage-2 quantization recipe and the position-aware codebook lookup.
- Position-aware quantization is the load-bearing piece; ablating it back to position-agnostic VQ kills the native-resolution property and reverts to fixed-grid tiling.
- Proximal regularizer hyperparameter is sensitive — too strong and the discretization is too tight; too weak and codebook collapse returns.
- Codebook size choices interact with vocabulary expansion in the downstream LLM — needs to be designed jointly with the LLM's embedding budget.

## Sources

- Paper: *ViQ: Text-Aligned Visual Quantized Representations at Any Resolution* — Yu, Liu, Yang, Dong, Qian, Lu, Hu, Rao, 2026 — [arXiv:2606.27313](https://arxiv.org/abs/2606.27313). Tencent HY Vision / Tsinghua / NTU / CAS.
