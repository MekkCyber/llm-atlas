# Mage-VAE
*Depth — the one-step diffusion-style VAE released with Mage-Flow.*

**TL;DR:** A lightweight image tokenizer for latent diffusion transformers. Encoder and decoder are each a single diffusion-style step (no iterative denoising), regularized with an *anchor-latent* term so the latent space stays compatible across variants. Achieves reconstruction quality on par with strong public VAEs while cutting tokenization compute by more than an order of magnitude.

**Prereqs:** *(none in the current graph — assumes familiarity with VAEs and latent diffusion)*
**Related:** [../architectures/mla.md](../architectures/mla.md), [rectified-flow won't yet be in graph]

---

## What it is

The bottleneck tokenizer for a latent DiT is usually a heavy multi-stage VAE — costly to train, hard to swap, and expensive at inference. Mage-VAE targets that specific cost. It is a lightweight VAE whose encoder and decoder each perform *one* diffusion-style pass, not a full multi-step denoise, giving a fast, high-fidelity tokenizer purpose-built for a 4B-scale image generator.

## How it works

Two ingredients:

1. **One-step diffusion-style encode/decode.** Both encoder and decoder use a diffusion-shaped block but are trained to produce the full output in a single step. This preserves the qualitative benefits of diffusion decoders (sharpness, texture faithfulness) without paying for iterative sampling.
2. **Anchor-latent regularization.** During training, the latent representation is pulled toward an anchor distribution — typically a frozen reference VAE's latent statistics — so the tokenizer stays interoperable with existing DiTs and downstream Turbo variants can share latents.

Trained jointly with the Mage-Flow DiT under rectified flow matching, then reused across Base / RL-aligned / Turbo generation-and-editing variants without retraining the tokenizer.

## Why it matters

Tokenizer cost is a hidden multiplier on every diffusion training and inference pass. Cutting it by ~10× at equal reconstruction quality is a system-level win that compounds across the full training pipeline. Anchor-latent regularization also matters strategically: it means the tokenizer isn't a one-shot design decision — you can swap DiTs above the same latent space.

## Gotchas & tricks

- One-step decoders are notoriously fragile — the anchor-latent term is doing quiet work stabilizing the latent geometry.
- Reconstruction fidelity matches "strong public VAEs" on standard benchmarks; whether Mage-VAE preserves the fine-grained editing latents needed for instruction editing at extreme resolutions is a separate question.
- The one-step step is a diffusion-shaped block, not a plain conv encoder — architectural detail that gets easily lost when reproducing.

## Sources

- Paper: *Mage-Flow: An Efficient Native-Resolution Foundation Model for Image Generation and Editing* — Microsoft Mage Team, 2026 — [arXiv:2607.19064](https://arxiv.org/abs/2607.19064)
