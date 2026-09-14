# Synthetic Block-Stacking for LVLM Spatial Reasoning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Fine-tune LVLMs for 3D spatial reasoning on a **compact procedurally generated block-stacking dataset** instead of expensive real-scene spatial-QA data with dense geometric annotations. A 15k-sample corpus with 3D-to-2D projection, viewpoint transformation, and structural combination transfers to real-world spatial benchmarks.

**Prereqs:** [README](README.md), [_data-curation](../data/_data-curation.md)
**Related:** [_data-curation](../data/_data-curation.md)

---

## What it is

Spatial intelligence in LVLMs — reconstructing and reasoning about 3D structure from 2D images — has been trained via real-scene QA datasets that need dense geometric labels (depth, normals, camera pose). Those labels are expensive, noisy, and dependent on an external perception pipeline. This technique replaces them with **fully synthetic Blockworld data**: colored blocks stacked programmatically, rendered from controlled viewpoints, with ground-truth spatial questions generated from the same simulator.

The dataset (SpatialBlock-15k) covers three primitive skills — 3D-to-2D projection, viewpoint transformation, structural combination — and includes controlled color modulation so the model learns anchor-based reasoning under visually complex conditions.

## How it works

- **Procedural generation.** Sample block configurations (position, orientation, color); render each from N viewpoints with controlled color modulation.
- **Automatic QA generation.** From the ground-truth 3D scene, mint spatial questions: "which block is behind X?", "if seen from angle Y, what's on top?", "which face is visible?", etc.
- **Two supervision modes.** Direct-answer (predict the answer token) and reasoning-based (predict a CoT that ends in the answer). Reasoning-based supervision wins.
- **Fine-tune, don't pretrain.** The LVLM is already vision-capable; SpatialBlock-15k is a targeted spatial-skill fine-tune, small enough to run cheaply on top of any open LVLM.

## Why it matters

If a compact procedural corpus transfers to real scenes, spatial reasoning becomes a data-*engineering* problem rather than a data-*collection* problem. It removes the dependence on external perception modules for labels, cuts cost by orders of magnitude, and gives a clean curriculum for testing which primitive spatial skills each LVLM lacks.

## Gotchas & tricks

- **Reasoning-based supervision beats direct-answer.** Direct-answer fine-tuning is faster but under-learns the underlying 3D structure and generalizes worse to real scenes.
- **Color modulation is not decoration.** It acts as a controlled distractor that forces anchor-based reasoning; ablations show removing it hurts real-scene transfer.
- **Small ≠ weak.** 15k samples is enough — larger synthetic corpora saw diminishing returns in the paper's ablations, suggesting curriculum coverage matters more than volume.
- **Watch for texture / lighting gaps.** Pure Blockworld has no photorealistic textures; combining with a light real-image auxiliary set closes the domain gap further.
- **Skill isolation is the design goal.** Each of the three skills (projection / viewpoint / composition) should stay identifiable in the data so you can diagnose which one your model is missing.

## Sources

- Paper: *SpatialBlock: Enhancing Spatial Intelligence in LVLMs via Synthetic Block-Stacking Problem* — Ryu, Kim, Yang (KAIST), 2026 — https://arxiv.org/abs/2609.07064
- Code + data: https://github.com/rsoohyun/SpatialBlock
