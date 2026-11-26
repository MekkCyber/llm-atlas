# Codec-Guided Token Pruning for Video Diffusion
*Depth — use classical video-codec signals (motion vectors, residuals) to drop redundant condition tokens in DiTs.*

**TL;DR:** Video diffusion transformers consume long condition-token streams (depth maps, pose, edges, audio, …). Most of that information is redundant across time because real video is locally smooth. CodecOPSD reuses the motion vectors and residuals that video codecs already compute: tokens in regions the codec says haven't moved get pruned from the condition stream. An on-policy self-distillation step trains a DiT adapter to work with the pruned conditions. Across 16 condition types, ~66% of condition tokens are dropped with little quality loss, ~2× denoising speed, and up to 73% KV-cache memory reduction.

**Prereqs:** [inference/README.md](README.md)
**Related:** [fp8.md](../quantization/fp8.md) · [multimodal/README.md](../multimodal/README.md)

---

## What it is

A pruning method for the *condition* stream of a video diffusion transformer (not the latent stream, not the model weights). The key observation: classical video codecs — HEVC, AV1 — have already computed, at zero extra cost to the diffusion stack, exactly which pixels moved and which didn't. Motion vectors tell you where each block came from in the previous frame; residuals tell you how different it is. Regions with zero motion and small residuals are effectively repeat tokens — the diffusion model doesn't need them re-emitted. CodecOPSD drops them, then uses on-policy self-distillation to teach a DiT adapter to work from the pruned condition stream without quality regression.

## How it works

1. **Codec pass.** Run the video through a codec (or decode a codec stream you already have). Extract per-block motion vectors `MV(x, y, t)` and residuals `R(x, y, t)`.
2. **Token-level pruning mask.** For each condition token, check its spatial-temporal location. If the local codec signal says "no motion and small residual," mark the token for drop. Thresholds are per-condition-type (depth, pose, edge, audio, …).
3. **Adapter.** A small LoRA-style DiT adapter is trained to take the pruned condition stream and produce the same output as the non-pruned version.
4. **On-policy self-distillation.** The adapter is trained *against the DiT's own output under the full condition stream* — the student is the pruned-condition DiT, the teacher is the full-condition DiT, both are the same base model. On-policy because samples are drawn from the student's current denoising trajectory, not a static set.
5. **At inference,** run the codec-informed pruning and the adapter together. KV cache for condition tokens shrinks by the fraction pruned (~66%); denoising throughput roughly doubles.

Reported across 16 condition types (depth, pose, edge, sketch, audio, text, …).

## Why it matters

- **Free signal.** Codec motion/residual is computed by every video pipeline; reusing it costs nothing.
- **Memory is the bottleneck.** Video DiTs at serve time are condition-token-memory-bound, not compute-bound. 73% KV-cache reduction translates directly to concurrent-stream capacity.
- **Compatible with existing bases.** Adapter is small; you don't retrain the DiT. Drop-in to production video stacks.
- **Generalizes beyond video.** Any modality with a cheap "did this change?" oracle (audio onsets, sensor deltas, document diff) can use the same recipe.

## Gotchas & tricks

- **Codec quality matters.** Lossy codec residuals undercount change; the pruner then drops tokens that shouldn't be dropped. Use a high-bitrate codec pass for the mask, even if the delivery codec is low-bitrate.
- **On-policy self-distillation is load-bearing.** Off-policy or static distillation data lets the student miss failure modes the pruned conditioning induces. Must sample from the student.
- **Per-condition thresholds.** A single global motion/residual threshold works poorly; depth and pose tolerate much more pruning than edges or text. Tune per condition type.
- **Boundary artifacts.** Heavy pruning at motion/no-motion boundaries creates subtle rendering seams. Dilate the "keep" mask slightly to leave a buffer.
- **Not compatible with condition-free generation.** Zero-shot (no conditioning) video generation has no condition stream to prune; this method only helps conditional pipelines.
- **Composable with quantization.** The condition stream can be fp8/int8-quantized on top of being pruned; the two compress orthogonal axes.

## Sources

- Paper: *On-Policy Codec Distillation for Efficient All-in-One Video Generation* — part of the work done at Adobe Research (internship), 2026 — [arXiv:2611.31184](https://arxiv.org/abs/2611.31184).
