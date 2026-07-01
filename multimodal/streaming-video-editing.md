# Streaming Diffusion Video Editing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Real-time streaming video editing (12.66 FPS) via **causal, frame-by-frame** diffusion editing that preserves backgrounds and non-edited regions over long horizons. Two design pillars: (1) a **three-stage distillation pipeline** that transfers editing capability from a powerful bidirectional foundation model to an efficient unidirectional streaming editor, and (2) an **AR-oriented mask cache** that reuses region-related computation across frames.

**Prereqs:** [README.md](./README.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md), [../inference/README.md](../inference/README.md)

---

## What it is

Video editing with diffusion models has been non-streaming: the editor sees the whole clip, edits everything, then plays back. Two things break in real-time / streaming use: latency (can't wait for the whole clip) and stability (frame-by-frame editors drift). LiveEdit produces a *streaming* editor that runs frame-by-frame with strong content preservation.

## How it works

**Three-stage distillation (bidirectional → causal → streaming).**
1. **Bidirectional foundation editor.** Start from a strong bidirectional video-editing model that sees past and future frames — the teacher.
2. **Causal editor (Stage 2).** Distil into an editor that only sees past frames. Quality drops from bidirectional; the loss is limited by aligning the causal editor's outputs with the teacher's on the same clips.
3. **Streaming editor (Stage 3).** Distil the causal editor into a streaming variant that emits one edited frame per input frame at low latency. Long-horizon stability is preserved because the two earlier stages seeded the causal representation.

Progressive distillation is the trick — going bidirectional → streaming in one step loses too much quality.

**AR-oriented mask cache.**
- Video edits are usually **region-scoped** — inpaint a person, swap a background, replace an object.
- The mask cache stores masked-region features from previous frames and reuses them when the current frame's edit region overlaps.
- Per-frame work drops to *the edit region only* rather than the full frame; the unedited background is served from cache.
- Analogous to a KV cache for text generation — same "reuse the parts that didn't change" idea, adapted to region-scoped video edits.

**Reported performance.** 12.66 FPS with SOTA visual quality among streaming baselines. Long-clip background stability holds across the paper's benchmark.

## Why it matters

- **Unlocks interactive / AR use cases.** Real-time video editing is a prerequisite for AR pipelines, live-broadcast effects, and creator tools — none of which tolerate offline latency.
- **Distillation recipe generalises.** The bidirectional → causal → streaming ladder is a template for any offline diffusion model that needs a streaming counterpart.
- **Mask cache as a serving primitive.** The region-scoped cache is a specific instance of "cache what doesn't change per token/frame" — one of the recurring ideas across text KV caching, prompt caching, and now video.

## Gotchas & tricks

- **Region overlap governs cache hit rate.** If edits are large / whole-frame, the cache degenerates and per-frame work approaches the uncached baseline.
- **Long-horizon drift is limited but not eliminated.** Distilled streaming still accumulates tiny errors; 30+ second clips may need occasional re-anchoring against a bidirectional pass.
- **12.66 FPS is on the paper's hardware.** Absolute FPS transfers roughly with FLOPs; expect scaling with GPU generation.
- **Edit prompts change the cache validity.** A prompt change invalidates the mask cache; interactive workflows should expect a 1–2 frame warmup after edit changes.

## Sources

- Paper: *LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing* — 2026 — [arXiv:2606.26740](https://arxiv.org/abs/2606.26740).
