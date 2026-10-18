# OmniCompass
*Depth — dual-track omni-modal video benchmark.*

**TL;DR:** OmniCompass is a holistic video benchmark with two tracks: OmniCompass-VL (vision + text input) and OmniCompass-AV (joint audio + visual input). Each track scores temporal grounding, spatial grounding, and open-ended QA on the same video pool, so audio's contribution is isolable. 50 MLLMs are evaluated side by side. Open-source MLLMs lag proprietary ones most severely on audio-visual understanding, which previous VLM benchmarks mostly ignore.

**Prereqs:** [README.md](README.md), [multimodal/README.md](../multimodal/README.md)
**Related:** [ifeval.md](ifeval.md), [mmlu.md](mmlu.md)

---

## What it is

A benchmark that treats audio as a first-class input modality for video understanding and reports per-track gaps so a release can be scored on audio-visual understanding without a separate suite. "Grounding" here means both *when* (temporal) and *where* (spatial) a claim is anchored.

## How it works

- Same video source material across both tracks; OmniCompass-AV adds the native audio channel to each clip.
- Three task types per track: temporal grounding (start/end timestamps), spatial grounding (bounding boxes / region references), and open QA. Scored with per-task metrics standard in the video-QA literature.
- 50-model side-by-side includes major open-source (LLaVA-NeXT-Video, Qwen2-VL, etc.) and proprietary (Gemini, GPT-4o) baselines.

## Why it matters

Audio has been an afterthought in VLM evaluation; most "video" benchmarks mute the track. OmniCompass gives a shared yardstick to grade genuinely omni-modal (A+V+T) models, which is where the frontier is moving (Gemini 2, GPT-4o).

## Gotchas & tricks

- The gap on OmniCompass-AV reflects partly *data access* (audio-paired video is harder to curate than text-captioned video), not only architecture — don't over-read model-family rankings.
- Spatial grounding metrics are sensitive to annotation granularity; cross-model comparisons within one metric are more reliable than cross-metric aggregates.
- Dual-track design means a VL-only model can still be scored on OmniCompass-VL for a direct comparison with its omni-modal successor.

## Sources

- Paper: *OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding* — Liu et al., HK PolyU / OPPO / Colorado Mines / NTU / SJTU, 2026 — [arXiv:2610.18829](https://arxiv.org/abs/2610.18829)
