# Omni-Modal Dialogue with Visual Thought Plan
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A framework for jointly generating *coordinated* text, personalized speech, and reference-conditioned video in a dialogue setting. The system emits a **Visual Thought Plan** first — a compact prediction of scene, emotion, and motion — which then conditions both the speech and video branches. **Multi-codebook speech units** serve as the shared acoustic-temporal interface across modalities. A **distilled streaming student** replaces a slower teacher video generator to reach ~1.293× real-time factor. Introduced in *Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence* (2026).

**Prereqs:** [README.md](README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

"Omni-modal dialogue" here means: input can be any of text/audio/image, and the **output includes video** — a talking, animated visual presence — in addition to text and speech. Two hard problems:

1. **Cross-modality coordination.** Text, speech, and video must agree on emotion, timing, and content. A joint autoregressive transformer emitting all modalities is one option but scales badly and is hard to align.
2. **Real-time video generation.** Video models are slow; a real-time video dialogue partner needs ≥1× real-time factor.

Ex-Omni-2D addresses both with a **plan modality** and a **shared codebook interface**, plus **distillation** for the streaming video generator.

## How it works

**Visual Thought Plan (VTP).** Before the modality-specific decoders emit anything, the model produces a compact plan predicting **scene**, **emotion**, and **motion** for the upcoming turn. Both the speech and video branches read the VTP, so they share the same underlying decision about what the response should look and feel like.

**Multi-codebook speech units as shared interface.** Rather than treating speech as raw waveform or one codebook, the paper uses **multiple discrete codebooks** whose combined tokens act as a shared *acoustic-temporal* representation. The video branch subscribes to the same codebook stream — timing and prosody in speech are directly available to the video generator, so lip sync and expression follow naturally.

**Distilled streaming student video generator.** A slower teacher video model is distilled into a **streaming student** that emits video incrementally as speech is generated, achieving **~1.293× real-time factor** at the reported resolution. Distillation targets both output quality and streaming stability.

## Why it matters

- **Plan-first pattern for omni-modal.** The VTP is a small, cheap token stream that gates the expensive modalities. Cleaner than forcing one autoregressive transformer to jointly emit text + audio + video, and easier to debug (you can inspect the plan).
- **Shared codebook > two independent codes.** A single shared acoustic-temporal representation removes an entire class of synchronization bugs between speech and video.
- **Streaming distillation as a general recipe.** Applies beyond dialogue: any expensive video generator can be distilled into a streaming student if you can supply timing signals from another modality.

## Gotchas & tricks

- **Real-time factor is resolution-dependent.** 1.293× is at the paper's reported resolution; higher resolutions likely push the student below real-time.
- **VTP quality is the ceiling.** A bad plan corrupts both modalities. Training the plan predictor deserves at least as much attention as the modality decoders.
- **Multi-codebook = more tokens.** The shared representation is richer but the token budget grows; the streaming pipeline has to be provisioned for the larger context.

## Sources

- Paper: *Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence* — Zhang, Li, Tang, Yu, Guo, arXiv 2608.10720, 2026.
