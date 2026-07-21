# Audio Tokenizers for Music / Speech LLMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Audio-LLM stacks convert raw waveform into a discrete token stream for autoregressive modeling. Two broad approaches: **acoustic tokens** (low-level, high-bitrate, preserve fidelity — RVQ codec style) and **semantic tokens** (high-level, low-bitrate, preserve musical/linguistic structure). Modern audio-LLMs (Qwen-Music, 2026) increasingly factor the pipeline as *semantic-token LLM* + *generative acoustic renderer* — the LLM plans structure at low bitrate, the renderer recovers detail.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md), [../fundamentals/bpe.md](../fundamentals/bpe.md)
**Related:** [representation-autoencoders.md](representation-autoencoders.md)

---

## What it is

Two families of audio tokens:

| Family | Bitrate | What it preserves | Reconstruction | Example |
| --- | --- | --- | --- | --- |
| **Acoustic (codec)** | High (1.5–24 kbps) | Waveform fidelity, timbre, spatial | Direct decoder | EnCodec, SoundStream (RVQ) |
| **Semantic** | Low (0.4–1 kbps) | Melody, phonemes, structure | Needs a rendering model | HuBERT-clusters, Qwen-Music-Tokenizer |

A single codebook of semantic tokens is dramatically smaller than an RVQ codec stream (many codebooks per frame) — cheaper for an LLM to model — but it *cannot be decoded back to fidelity audio directly*. It needs an acoustic renderer conditioned on the semantic stream.

Qwen-Music's tokenizer produces a 25 Hz single-codebook stream of "Music Semantic Tokens" that keeps melody and structure but drops acoustic detail. A separate **Qwen-Music-Render** stage does generative stereo rendering to restore fidelity.

## How it works

The three-stage audio-LLM pattern:

1. **Semantic tokenizer.** Encode waveform into a low-rate discrete stream. Typical construction: a self-supervised speech / music encoder (HuBERT, EnCodec-semantic, MERT) whose continuous embeddings are quantized (k-means or VQ) to a single or small-codebook stream at ~25 Hz.
2. **Semantic-token LLM.** Autoregressive model over the semantic stream — trained on huge multilingual audio corpora. Domain-specific tricks (e.g. **Melody-CoT** in Qwen-Music) plan structure before full-song generation.
3. **Generative renderer.** Conditioned on the LLM's semantic-token output, generate the acoustic waveform (typically as an acoustic-codec stream then vocoded, or directly via a diffusion / flow-matching decoder to raw audio). Renderer is optimized purely for fidelity, disjoint from the LLM's compute path.

The factoring pays because most of the model's parameters and inference cost sit in the LLM — which now runs on a very compact stream — while the fidelity restoration is off-loaded to a smaller, purpose-built model.

## Why it matters

- **Compute cost is dominated by the LLM.** Reducing the token rate 20–100× (from RVQ multi-codebook to single-codebook semantic) makes autoregressive audio modeling tractable at LLM scale.
- **Structural planning becomes possible.** Melody-CoT and similar tricks require the LLM to reason over musical structure — feasible at 25 Hz, hopeless at 1500 Hz acoustic frame rate.
- **The pattern generalizes across modalities.** Semantic-tokens-LLM + acoustic-renderer is now the default for music (Qwen-Music), speech (VALL-E variants, Voicebox), and increasingly video (semantic latents + diffusion decoder — see [representation-autoencoders.md](representation-autoencoders.md)).

## Gotchas & tricks

- **Semantic-tokenizer choice is a strong prior.** A tokenizer optimized for speech phonemes will discard music-relevant harmony; music tokenizers may waste bits on timbre irrelevant to structure. Pick per-domain.
- **Single-codebook vs. multi-codebook.** Multi-codebook (residual VQ) squeezes more info per frame but forces the LLM to model correlated streams — usually not worth the complexity for a semantic tokenizer.
- **Renderer trained separately from LLM.** Joint training is expensive and unstable; the two-stage pattern (freeze tokenizer, train LLM; then train renderer on paired data) is the working recipe.
- **Frame rate is a tradeoff knob.** 25 Hz keeps LLM sequences short; 50 Hz preserves more melodic detail at 2× LLM cost.
- **Multilingual coverage requires huge data.** Qwen-Music uses 5M+ hours of multilingual music — a scale that only industrial datasets reach.

## Sources

- Paper: *Qwen-Music Technical Report* — Qwen team, Alibaba, 2026 — [arXiv:2607.11699](https://arxiv.org/abs/2607.11699).
- Reference: *SoundStream* — Zeghidour et al., Google, 2021 — canonical RVQ acoustic codec.
- Reference: *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units* — Hsu et al., 2021 — canonical semantic tokenizer for speech.
- Reference: *MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training* — Li et al., 2023 — the music-domain analog of HuBERT.
