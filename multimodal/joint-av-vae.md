# Joint Audio-Video VAE with Cross-Modal Alignment
*Depth — a VAE that produces aligned audio and video latent spaces for downstream joint generation (OmniVAE).*

**TL;DR:** A jointly-trained audio-video VAE that gives the two modalities an *aligned* latent geometry, so downstream text-to-audio-video models don't have to learn cross-modal synchronization from scratch. On top of standard reconstruction, OmniVAE adds a **segment-level audio-video contrastive loss** and **distills features from strong per-modality semantic encoders** into each latent stream.

**Prereqs:** [../multimodal/README.md](../multimodal/README.md)
**Related:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)

---

## What it is

Most audio-video generative systems train separate VAEs per modality and stack a large diffusion model on top to learn the sync. That model bears the entire cross-modal-alignment burden. OmniVAE moves alignment *into* the tokenizer stage: the encoded latents already respect temporal-semantic correspondence between audio and video, so the downstream generator inherits a much easier problem.

## How it works

Three losses trained jointly:

1. **Reconstruction.** Standard VAE reconstruction on both modalities — video pixels and audio spectrograms — with modality-specific encoders and decoders sharing a joint training loop.
2. **Segment-level audio-video contrastive.** Split clips into short segments; pull the audio and video latents of temporally-aligned segments together, push apart mismatched pairs. Captures the temporal-semantic correspondence that "the sound of the door slam happens at the frame of the door slam."
3. **Semantic distillation.** Distill features from pretrained per-modality semantic encoders (a strong audio encoder into the audio latent, a strong video encoder into the video latent). Each latent stream inherits downstream learnability from these teachers.

The output is a pair of latent spaces with aligned geometry, ready to feed a downstream text-to-audio-video diffusion or autoregressive model.

## Why it matters

Cross-modal sync has been the last-mile problem for generative video with audio. Building the alignment into the VAE — rather than asking a big downstream generator to figure it out end-to-end — is a cleaner factoring, mirrors the "align first, generate second" pattern that made CLIP-conditioned image diffusion work, and reportedly improves both generation quality and sync accuracy on downstream text-to-A/V.

## Gotchas & tricks

- Segment length is the critical knob — too short and the contrastive loss over-fits to noise; too long and the temporal alignment becomes coarse.
- Semantic distillation weights need to be small enough that reconstruction still dominates; too aggressive and the latents drift away from being invertible.
- Contrastive negatives should be *within-clip* time-shifted pairs, not cross-clip random pairs, to force the model to learn temporal correspondence rather than category-level separation.
- A frozen text encoder feeding the downstream generator still benefits — the aligned latents make text→A/V easier regardless of the text side.

## Sources

- Paper: *OmniVAE: An Audio-Video VAE with Cross-Modal Alignment for Joint Generation* — Zhan et al., 2026 — [arXiv:2607.23855](https://arxiv.org/abs/2607.23855)
