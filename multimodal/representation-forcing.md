# Representation Forcing

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training technique for unified multimodal models (UMMs) that removes the frozen, separately-pretrained VAE from the image branch. Instead of decoding pixels through an external latent space, the model is *forced* to autoregressively predict high-level visual-representation tokens as intermediates, then run pixel diffusion in the same backbone conditioned on those tokens. Perception outputs and generation targets become the same object — closing the VAE-bottleneck quality gap end-to-end.

**Prereqs:** *(no in-graph prereq files yet — assumes familiarity with unified multimodal models and diffusion transformers)*
**Related:** [README](README.md)

---

## What it is

Unified multimodal models try to handle both perception (image → text) and generation (text → image) inside one network. In practice, almost all current UMMs still bolt a frozen VAE onto the image branch: image pixels are encoded by the VAE into a compact latent, the backbone operates on latents, and the VAE decodes back to pixels at the end. The VAE is a separate pretraining; the gradient never reaches it.

The bottleneck is structural. The model never learns *its own* compressed representation. End-to-end optimization is blocked. Removing the VAE naively (predict pixels directly) collapses quality, because the backbone has to simultaneously model fine-grained pixel noise and the high-level structure of the image.

Representation Forcing (RF) decouples those two skills by adding an intermediate step: the model first emits visual-representation tokens (something like the features a perception encoder would produce), then runs pixel diffusion conditioned on those tokens within the same backbone.

---

## How it works

### Two passes in one backbone

For an image to generate, the backbone runs two passes:

1. **Representation pass.** Autoregressively predict a sequence of visual-rep tokens given the text prompt (and any prior context). These tokens are trained to match a target representation drawn from a perception objective — what a vision encoder would extract from the ground-truth image.
2. **Pixel pass.** With the predicted rep tokens kept in context, run diffusion on pixel-space tokens. The rep tokens condition every pixel-pass step.

Both passes share weights. The model learns to compress images into rep tokens (perception) and to expand rep tokens into pixels (generation) end-to-end — without an external autoencoder.

### Training objective

Loss is a sum of (a) rep-prediction loss (autoregressive cross-entropy or feature-matching against the perception target) and (b) pixel-diffusion loss. The two are computed in the same forward pass; the model is forced (hence "Representation Forcing") to commit to a rep token sequence that supports good pixel-pass diffusion.

### Why the VAE bottleneck closes

The rep tokens are *part of the model*, not a separate frozen module. Gradients propagate from pixels through the diffusion pass back into the rep-prediction pass and into the perception-side weights. The backbone's perception and generation halves are co-optimized.

---

## Why it matters

- **End-to-end UMMs.** No external VAE means the architecture is a single trainable system. Architecture choices, losses, and training data are no longer split across two artifacts.
- **Generation matches the VAE-based SOTA** while perception *improves over* the VAE-based variant — the rep-prediction pass acts as a useful perception auxiliary task.
- **Pixel-space training becomes tractable at scale.** RF is the missing component that makes pixel-space UMMs competitive without resorting to a frozen latent space.

---

## Gotchas & tricks

- **The rep-target choice is load-bearing.** Pick a perception target that is genuinely informative about the image (DINO-style self-supervised features, CLIP, etc.). A weak rep target means weak conditioning during the pixel pass.
- **Token length grows.** Rep tokens + pixel tokens are both in-context. Budget the rep-token count carefully — too many wastes context, too few under-conditions the pixel pass.
- **Distinct from VAE-style tokenization.** RF predicts rep tokens *autoregressively*, in the LM head, with categorical or feature-matching loss. A VAE encodes to a fixed latent in one shot. Don't confuse them.
- **Co-training requires balanced sample budget.** If only generation data is supervised, the rep-prediction skill drifts. Mix in perception data so the rep-prediction head stays well-calibrated.

---

## Sources

- Paper: *Representation Forcing for Bottleneck-Free Unified Multimodal Models* — Lin, Yang, Zhao, Xiao, He, Zhao, Ding, Wang, Wang, Zhang, Fan, Liu, 2026 — HKU / ByteDance Seed / CUHK / Nanjing U. / Tsinghua. Introduces representation forcing; reports pixel-space RF matches VAE-based SOTA on generation and outperforms on understanding.
