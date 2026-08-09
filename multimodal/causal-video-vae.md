# Causal Video VAE

*Depth — 3D VAE tokenizer for video whose temporal dimension is causally masked, so latent frames form a valid autoregressive prefix for downstream generation.*

**TL;DR:** Latent-diffusion video stacks need a tokenizer that maps raw pixels to a compact latent. A **causal** 3D VAE enforces that the latent at time `t` depends only on frames `≤ t` — matching the causal structure that downstream text-to-video transformers (and streaming inference) require. KVAE-3D (Kandinsky Lab, 2026) ships two causal variants at 4×16×16 and 4×8×8 spatiotemporal compression, alongside sibling KVAE-2D (image) and KVAE-Audio (continuous full-band 48 kHz, 50 Hz × 64 ch) tokenizers with matched latent geometries.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [README.md](README.md)

---

## What it is

A 3D convolutional VAE that encodes video into a spatiotemporal latent block, with a strict **temporal causality constraint**: the encoder for latent slice `t` may only look at input frames `≤ t`. Concretely, all temporal convolutions and attention layers in the encoder are masked / padded to the past.

Two purposes this serves:

- **Prefix-friendly decoding.** A downstream latent-video AR / diffusion model can be trained on prefixes of the latent block and generate future latents step-by-step without leaking information from the future during training.
- **Streaming inference.** At generation time, the decoder can produce the next latent slice from the current prefix without re-encoding the whole clip.

The paper also matches latent geometries across the KVAE family (audio 50 Hz × 64 ch; image and video with shared channel counts) so a single text-conditioned diffusion transformer can share architecture across modalities.

---

## How it works

**Encoder.** Stack of 3D convolutions + attention with causal temporal padding. Spatial downsampling by a factor of 8 or 16; temporal downsampling by 4.

**Latent block.** For video shape `[T, H, W, 3]`, latent is `[T/4, H/16, W/16, C]` (for the 4×16×16 variant) or `[T/4, H/8, W/8, C]` (for 4×8×8).

**Decoder.** Non-causal 3D convolutions expand the latent back to pixels. Causality is only required in the encoder — the decoder can look anywhere within the emitted latent.

**Loss.** Standard VAE reconstruction loss + KL to a prior + (typically) an adversarial term for perceptual sharpness. Multi-modality training shares the latent-shape conventions across audio / image / video tokenizers.

**Compression ratios.** 4×16×16 = 1024× spatiotemporal compression; 4×8×8 = 256×. The trade is reconstruction quality vs. downstream generation cost.

## Why it matters

- **Enables streaming latent video generation.** Non-causal video VAEs force the downstream generator to see the whole clip at once — no streaming, no online extension.
- **Compatible with AR video transformers.** Causal latents match the structure AR training assumes; no leakage during teacher-forced training.
- **Shared latent geometry across modalities.** Any-to-any generation stacks benefit from tokenizers with matched channel counts and matched compression ratios; the KVAE family is designed this way.

## Gotchas & tricks

- **Causal decoders vs. causal encoders.** Only the encoder needs to be causal for training-time correctness; a non-causal decoder gives sharper reconstructions. The KVAE-3D variants use causal encoders + non-causal decoders.
- **Compression ratio bounds downstream quality.** 4×16×16 is aggressive — enough to keep long generations tractable, but detail loss is real. Choose based on target duration and downstream model capacity.
- **Temporal padding conventions matter.** For finite clips, the first few latents lack full temporal context; either pad the input or accept boundary artifacts.
- **VAE + adversarial loss balancing.** Common instability across VAE-tokenizer training; too much adversarial loss produces artifacts, too little produces blur. The KVAE report acknowledges typical VAE-training pitfalls but does not publish an exhaustive ablation.

## Sources

- Paper / report: *KVAE: Family of Tokenizers for Multimodal Generative Models* — Kandinsky Lab, 2026 — [arXiv:2608.05798](https://arxiv.org/abs/2608.05798). Ships KVAE-Audio (continuous 48 kHz, 50 Hz × 64 ch), KVAE-2D (image, 8× spatial × 32 ch), and KVAE-3D (video, causal, 4×16×16 and 4×8×8).
