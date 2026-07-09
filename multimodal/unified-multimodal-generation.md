# Unified multimodal generation (SenseNova-Vision)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reformulate every "computer vision" task — detection, OCR, keypoints, segmentation, depth, surface normals, point maps, camera pose — as **generation from a unified multimodal model**. No task-specific prediction heads: users specify tasks through natural-language instructions (plus optional visual prompts), the model outputs text, images, or mixed outputs. Backed by a purpose-built instruction-response corpus. Competitive with task-specific baselines across structured perception *and* dense geometric prediction from a single architecture.

**Prereqs:** *(none — sits atop UMM basics)*
**Related:** [README.md](./README.md)

---

## What it is

Traditionally, "computer vision" splits into families with specialized heads: classification (linear head), detection (box regression + class), segmentation (mask head), depth (dense regression head), etc. Each family has its own pretraining, architectures, and eval conventions.

Unified multimodal generation collapses all of these into one paradigm: **any vision task is a generation task from a UMM**. Detection outputs are text tokens ("`<box>x1,y1,x2,y2</box>`") or masks (image outputs). Depth is an image output. Keypoints are text coordinates. Segmentation is an image output. All expressed in the UMM's native text + image output spaces.

## How it works

**Task specification.** A natural-language instruction + optional visual prompt (e.g., a marker, a click, a bounding box). Example: "Segment the red car in this image" + image prompt → the model outputs a mask image.

**Output space.** Text tokens, image tokens, or mixed. Structured outputs (boxes, keypoints) live in text; dense outputs (masks, depth, normals) live in image tokens. The UMM's tokenizer/decoder handles both.

**No task heads.** Zero task-specific parameters. The entire "which task is this" logic lives in the natural-language instruction interpretation.

**SenseNova-Vision Corpus.** Large-scale instruction-response corpus covering all target tasks (detection, OCR, keypoints, segmentation, depth, surface normals, point maps, camera pose estimation). This is the load-bearing artifact: without it, the UMM has no way to learn task-specific output conventions.

## Why it matters

- **One backbone, no CV heads.** Kills the artifact-explosion problem where each new task requires a new head, a new training pipeline, and its own maintenance overhead.
- **Scope: perception *and* geometry.** Prior task-unifying work (Pix2Seq, Painter, UniPose) covered structured perception (detection, segmentation, poses). SenseNova-Vision adds dense geometric prediction (depth, normals, point maps, camera pose) — the frontier of "what a UMM can absorb."
- **Zero-shot task composition** becomes natural. A prompt combining "detect the cat, estimate its depth, and describe its pose" runs in one model with one output. Composition across CV tasks was previously an ensemble problem.
- **The vision-model-as-separate-artifact era is ending.** If UMMs can absorb geometry, the design pattern of "vision encoder → language model" starts looking like transitional infrastructure.

## Gotchas & tricks

- **The corpus is doing most of the work.** Success depends on having instruction-response data covering every target task with high quality. Scaling to new tasks means scaling the corpus, not adding new architectures — but corpus curation is not free.
- **Task-specific baselines still win in absolute quality** on well-defined narrow benchmarks. UMM matches or approaches; task heads with dedicated training + arch choices remain the ceiling for pure metric maximization.
- **Structured output parsing is fragile.** Text outputs like "`<box>x1,y1,x2,y2</box>`" require post-processing; if the model emits malformed text, the downstream evaluation breaks. Training must penalize format violations.
- **Dense outputs eat tokens.** A high-resolution depth map as image tokens costs many tokens per prediction; not efficient for real-time dense prediction pipelines. Task heads still win on throughput here.
- **Not a substitute for a proper vision encoder** in every setting. For pure feature-extraction downstream tasks (representation learning for other pipelines), a UMM is overkill.

## Sources

- Paper: *Vision as Unified Multimodal Generation* — Li, Deng, Chen, et al., SenseTime / NTU / CUHK / PKU / SJTU / ZJU, 2026 — [arXiv:2607.06560](https://arxiv.org/abs/2607.06560).
- Related: *Pix2Seq* — Chen et al., 2022 — object detection as sequence generation.
- Related: *Painter / SegGPT* — Wang et al., 2023 — visual prompting for CV tasks.
