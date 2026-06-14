# Holistic Visual Tokenizer
*Depth — a single ViT that tokenizes both images and video into one latent space for unified multimodal models.*

**TL;DR:** Unified Multimodal Models (UMMs) typically bolt separate encoders onto an LLM — one for images, one for video. A **holistic visual tokenizer** uses a single ViT for both modalities, with shared latents that the LLM amortizes across. The tradeoff is reconstruction: doing video well usually hurts image quality and vice versa. Hydra-X (2026) shows the design choices that resolve this: **frame-level causal temporal attention** for reconstruction, **hierarchical temporal compression** for semantics, and a lightweight decompressor trained against joint image+video teacher supervision.

**Prereqs:** *(none)*
**Related:** [README.md](README.md)

---

## What it is

A unified tokenizer for visual inputs in a UMM, with three requirements:

1. **One encoder** for both images and video, not two.
2. **Shared latent space** so the LLM treats image tokens and video tokens uniformly.
3. **Both reconstruction and semantics** — the same latents must support generation (decoded back to pixels) and understanding (read by the LLM).

---

## How it works

### Temporal attention design

Two ablations from Hydra-X drive the design:

- **Frame-level causal temporal attention** is sufficient for visual reconstruction. Each frame's tokens attend to prior-frame tokens causally; full spatiotemporal attention actually *degrades* reconstruction.
- **Full spatiotemporal attention** is worse than causal frame-level — counterintuitive but consistent with information-bottleneck arguments.

### Temporal compression

- **Hierarchical temporal compression** (multi-stage downsampling) beats single-step compression. Stages let the model preserve different frequency bands of motion.

### Decompressor

A lightweight head upsamples the temporally-compressed features and is supervised by joint image+video teacher outputs. This forces the compact latent space to carry both modalities' semantics.

### Editing in the tokenizer

A practical finding: image editing pipelines work better when source-target interaction happens **at the latent level inside the tokenizer**, not at the semantic level inside the LLM. Editing consistency and convergence both improve.

---

## Why it matters

- **Stack simplification.** One encoder instead of two reduces parameter count, training complexity, and inference latency.
- **Cross-modal transfer.** Latents amortized across image and video carry shared priors, lifting both modalities.
- **Editing pipelines change.** Moving the source-target join below the LLM is a useful design heuristic for any image-editing UMM.

---

## Gotchas & tricks

- **Capacity allocation matters.** A too-small tokenizer can't carry both modalities; the Hydra-X 7B scale is what they validate.
- **Teacher choice for the decompressor.** Joint image+video teachers are the contribution — using only an image teacher loses video semantics, and vice versa.
- **Decode latency.** A multi-stage decompressor adds inference cost vs. simpler image-only tokenizers; budget accordingly.
- **Doesn't include audio.** Truly any-to-any UMMs still need a separate audio encoder; the tokenizer is visual-only.

---

## Sources

- Paper: *Native Unified Multimodal Models with Holistic Visual Tokenizers* (Hydra-X) — Zhang et al., Tencent Hunyuan + Nanjing U. + Shanghai AI Lab, 2026 — [arXiv:2606.13289](https://arxiv.org/abs/2606.13289).
