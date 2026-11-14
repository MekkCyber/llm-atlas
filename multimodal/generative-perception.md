# Generative Visual Perception
*Depth — repurposing a pretrained image generator as a dense perception model by generating a color-coded legend over the scene.*

**TL;DR:** A pretrained image generator (e.g., a large text-to-image model) can be turned into a multi-task perception model — segmentation, depth, etc. — without training a dedicated perception head. The trick is to **prompt the generator to overlay a color-coded legend** on the scene: each segmentation class or depth bucket gets a color, and the generator paints the correct color at each pixel. Open Vision Banana shows an open-recipe version of this that beats closed Vision-Banana zero-shot on Cityscapes, and introduces **legend ensembling** — majority-voting across many sampled color legends at test time — as a free quality boost.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

A perception paradigm in which a *generative* model acts as the perceiver. For a task like semantic segmentation:

1. Define a **color legend** mapping each target class (road, car, pedestrian) to an RGB color.
2. Build a prompt that describes the task and the legend: "paint each pixel the color corresponding to its class, using legend {…}."
3. Run the image generator conditioned on the input image and the prompt. The output is an RGB image whose pixels encode the segmentation.
4. Decode: assign each pixel to the legend color it's nearest to.

The same scaffold handles depth (grayscale or coded palette), surface normals, referring segmentation, and other dense tasks — only the prompt and legend change.

## How it works

### Prompt + legend as the task specification

The model sees the input image through its image conditioning pathway and a text prompt that describes:

- Which task to perform.
- The color legend (either enumerated in text or rendered as a swatch strip).
- Any task constraints (e.g., "ignore sky in the inner mask").

Because the base model is a generator, it has strong priors on how colors blend at object boundaries — which is also useful for perception because boundary labels are what's usually hard.

### Legend ensembling (test-time augmentation)

For a fixed task, sample $N$ different color legends uniformly at random from a palette with good perceptual separation. For each legend, run the generator, then decode the per-pixel class. Majority-vote the class per pixel across the $N$ legends.

This is a decoder-side ensemble — the model weights don't change. The gain comes from the fact that the generator's errors are *color-specific* (e.g., "misreads green as olive"), and shuffling the color assignment decorrelates those errors across legends.

### Parameter-adaptation choices

Open Vision Banana ablates three choices together: (i) color encoding scheme, (ii) prompt template, (iii) which parameters to fine-tune (frozen vs. LoRA vs. full). They interact — a good color encoding depends on the prompt template, and the best tuning strategy depends on both. Isolated tuning is misleading.

## Why it matters

- Decouples "perception model" from "specialized perception architecture." The generator stays general; only the prompt changes per task.
- Demonstrates that multi-task perception can share one model without multi-task heads — the legend is the task specification.
- Legend ensembling is a clean example of the kind of free test-time compute perception has lacked, analogous to self-consistency in text LLMs.

## Gotchas & tricks

- Color palettes with poor perceptual separation (e.g., two near-red classes) will consistently fail; the legend has to be perceptually uniform.
- Legend ensembling saturates: beyond ~8–16 legends, marginal gain is tiny relative to compute.
- Sky, ignore-region, and off-canvas pixels need an explicit background color — otherwise the generator will "fill" and corrupt the decoding.
- Boundary quality depends on the generator's upscaling head. Models with crisp text/edge priors (SDXL class) outperform softer generators on fine-structure tasks.

## Sources

- Paper: *Open Vision Banana: A Fully Open Recipe for Generative Visual Perception* — Pariza et al., 2026 — https://arxiv.org/abs/2611.15844
