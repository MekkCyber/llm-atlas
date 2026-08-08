# GST-Bench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Video-VQA benchmark for **global spatial awareness** in VLMs. Models watch long egocentric synthetic video streams (6,790 minutes total) and answer spatial-reasoning questions from novel viewpoints or produce top-down maps from first-person frames. Fills the gap in the VLM eval landscape between local single-view spatial perception (BLINK, VSR) and full embodied planning.

**Prereqs:** none.
**Related:** [../multimodal/README.md](../multimodal/README.md) · [README.md](./README.md)

---

## What it is

Existing VLM spatial benchmarks are dominated by **local** perception: given one or a handful of frames, answer about depth, orientation, or object relations. GST-Bench targets the **global** capability — does the model maintain an internal spatial representation that survives a long horizon and generalizes to viewpoints it never saw?

## How it works

**Data generation.** 6,790 minutes of synthetic egocentric video with known ground-truth 3D geometry. Because scenes are synthesized, every question can be automatically graded against the true spatial layout — no human annotation bottleneck.

**Question templates.**
- **Novel-viewpoint reasoning:** "You're standing at position X facing direction Y. What do you see in front of you?" — the model has seen the room only from other viewpoints.
- **Egocentric → top-down mapping:** given the video, produce (or verify) a top-down room sketch.
- **Multi-object spatial relations across the horizon:** "Is object A north of object B given the sequence of rooms you traversed?"

**Grading.** Deterministic against the ground-truth geometry.

**Scale.** 22 VLMs evaluated zero-shot. Best model: **42.68** vs **79.08** human baseline — a large gap.

## Why it matters

- **Prerequisite for embodied agents.** VLAs, VLNs, and computer-use agents all need a global map; GST-Bench is the isolated capability test.
- **Long-horizon stress test.** Gap grows with horizon length and viewpoint novelty — GST-Bench won't saturate soon.
- **Synthetic-ground-truth grading.** No annotator ceiling; benchmark can scale with model progress without new labeling.

## Gotchas & tricks

- **Synthetic ≠ real distribution.** Model rankings on GST-Bench may not perfectly transfer to real egocentric video (Ego4D). Report both when possible.
- **Frame subsampling matters.** Some VLMs process only k frames; performance drops sharply with aggressive subsampling. Report throughput at fixed frame budget for fair comparison.
- **Coordinate-frame prompting** is fiddly. Small ambiguities in "north" or "your left" can be interpreted differently by models — the paper specifies exact prompt conventions.
- **VLM tokenizer/encoder capacity** likely dominates on long videos — separating "the model can't hold the context" from "the model can't reason spatially" needs careful ablation.

## Sources

- Paper: *GST-Bench: Can VLMs Develop Global Spatial Awareness from Video?* — Huang et al., ByteDance Seed / ZJU / NUS, 2026 — [arXiv:2608.05747](https://arxiv.org/abs/2608.05747).
