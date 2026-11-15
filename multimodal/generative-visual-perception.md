# Generative Visual Perception (Open Vision Banana)
*Depth — using an image-generation model as a universal perception head, plus legend-ensembling test-time scaling.*

**TL;DR:** Treat perception tasks (segmentation, referring expression, reasoning segmentation) as **conditional image generation**: have the model paint the labels as colored overlays over the input image. **Open Vision Banana (OVB)** is a fully open reproduction of this recipe, with controlled studies of the design axes (color encoding, prompt design, parameter adaptation, task balancing). It introduces **legend ensembling**, a training-free test-time scaling trick where the color legend is varied at inference and predictions are ensembled. OVB surpasses Vision Banana's reported Cityscapes zero-shot (73.9 vs 69.9 mIoU), and legend ensembling adds up to +8.7 mIoU on Cityscapes and +15.0 on COCO-Object.

**Prereqs:** *(none)*
**Related:** [../post-training/_post-training](../post-training/_post-training.md)

---

## What it is

A perception paradigm: instead of a classification head over pixels, the model *generates an image* where the task's labels are visually encoded (e.g., a legend "red = road, blue = sky") and painted over the input. Parsing the generated image recovers the labels.

The appeal: one generative backbone can do segmentation, referring expressions, reasoning segmentation, instance masking, and keep its image-editing abilities — no task-specific heads.

## How it works

### Core recipe

- **Condition** a base image-generation model on {image, task instruction, color legend}.
- **Generate** the output image with labels painted as colored overlays.
- **Decode** labels back to masks/boxes by color matching against the legend.
- **Adapt parameters** — the paper ablates where to train (full, LoRA, specific blocks), finding that light adaptation preserves the base model's editing ability.

### Design axes studied

- **Color encoding**: palette choice affects disambiguation on fine classes.
- **Prompt design**: how the legend is described in text influences compliance.
- **Parameter adaptation**: full fine-tune vs adapter choices.
- **Task balancing**: segmentation, referring, reasoning must be weighted so one doesn't crowd out the others.
The axes *interact* — one-at-a-time tuning finds a suboptimal recipe.

### Legend ensembling (test-time)

At inference, run the model multiple times with different color legends (e.g., red/blue/green vs purple/orange/yellow), decode each output, and ensemble the label votes per pixel. Pure inference-time trick — no retraining. Reported gains: up to +8.7 mIoU on Cityscapes, +15.0 on COCO-Object.

## Why it matters

- **Fully open reproduction.** Dataset pipeline, training code, and checkpoints released — closes the reproducibility gap on the generation-as-perception line.
- **Preserves base abilities.** Light adaptation means the resulting model still edits images well, so one checkpoint does both generation and perception.
- **Legend ensembling is portable.** The test-time scaling trick applies to any generation-based perception setup, not just OVB.

## Gotchas & tricks

- Instance segmentation is the weakest axis — color-based decoding struggles when many instances share a class; OVB trails Vision Banana here.
- Decoding depends on the generated image faithfully placing legend colors; with aggressive editing prompts the colors can drift and the parser fails.
- Legend ensembling costs K× inference. The per-mIoU-point speedup is worse than a single run; trade-off governed by deployment budget.

## Sources

- Paper: *Open Vision Banana: A Fully Open Recipe for Generative Visual Perception* — Pariza et al., UTN / DLR / AIST / U. Tübingen / TUM / valeo.ai, 2026 — arXiv 2611.15844 — https://arxiv.org/abs/2611.15844
- Baseline comparison: Vision Banana (Google proprietary).
