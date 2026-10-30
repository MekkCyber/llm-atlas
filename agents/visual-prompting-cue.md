# Visual Prompting Cues
*Depth — overlays on an input image (red circles, numbered boxes, arrows, blurs) that steer a frozen multimodal model's attention without retraining it.*

**TL;DR:** A **visual prompting cue** is a visible mark drawn onto a model's input image before inference — a red circle, a numbered box, a translucent arrow, a selective blur. It is the image-side analogue of a chain-of-thought prompt or a set-of-marks label. The frozen VLM already attends to visually-salient regions; the cue engineers that saliency to point at a specific element. Used throughout modern GUI agents: PreView (Hou & Xu, 2026) uses a red circle + textual hint to lift Qwen3-VL / OS-Atlas grounding without any fine-tune. The cue family generalizes to referring-expression tasks, grounding benchmarks, and plain visual QA.

**Prereqs:** [../multimodal/gui-grounding.md](../multimodal/gui-grounding.md)
**Related:** [computer-use-agents.md](computer-use-agents.md)

---

## What it is

A deterministic image-processing step applied before an input is passed to a frozen VLM:

```
original_image, hint_coord ──► cue_overlay ──► annotated_image ──► VLM
```

Common cue types:

| Cue | Shape | Signal to the VLM |
| --- | --- | --- |
| **Red circle** | thin red ring around candidate | "look here" |
| **Numbered box** | labeled bounding box | "this is element 3" (set-of-marks) |
| **Arrow** | arrow pointing to candidate | directional attention |
| **Dim-elsewhere** | darken all pixels outside ROI | hard-crop without losing context |
| **Zoom-crop** | zoomed insert in a corner | both global and local view |

The overlay is *outside* the model. No training, no inference cost beyond a few milliseconds of CPU drawing.

---

## How it works

### Mechanism

Modern VLMs (Qwen-VL, GPT-4V, Claude 3.x Vision) attend disproportionately to visually-salient regions during image encoding. A high-contrast overlay creates saliency *where you choose*. The VLM then naturally spends more capacity on the marked region, improving referring-expression accuracy, grounding precision, and tool-use coordinates.

The effect is empirical, not architectural: it works because of training-time co-occurrence between visual saliency and relevance.

### Pairing with text cues

Visual cues alone can be ambiguous (which of two red-circled items?). Modern systems pair them with a text hint:

```
input image: [annotated with red circle at (x, y)]
text:        "The marked element may be the Send button. Confirm and click."
```

This narrows the ambiguity without requiring a precise click prediction from the overlay generator.

### Modality-level fit

The cue sits in a middle layer of the agent stack:

```
screenshot ──► small grounding model ──► candidate coord
                                               │
                                               ▼
                                        draw red circle
                                               │
                                               ▼
                                     frozen VLM reasons & clicks
```

The grounding model proposes; the VLM disposes. Visual cue is the hand-off format.

---

## Why it matters

- **Training-free lift.** No fine-tune, no new model. Deploy with a few lines of OpenCV or Pillow.
- **Transparent and controllable.** You can see exactly what the model sees. Debugging is pixel-visible.
- **Composable.** Red circle + numbered boxes + crop-and-zoom all stack. Each cue family addresses a different failure mode.
- **Generalizes beyond GUIs.** Image QA, visual referring expressions, chart reading, document VQA — anywhere a frozen VLM would benefit from an attention prior.

---

## Gotchas & tricks

- **Over-marking is counterproductive.** Too many overlays cause the VLM to attend to the overlays themselves, not the underlying content. One or two cues per image.
- **Color choice matters per backbone.** Red circles work on most VLMs; some models respond better to blue or green. If you can afford a one-time sweep, do it.
- **Overlays can occlude.** Thick circles can hide the element you are pointing at. Keep lines thin and offset slightly from the target.
- **Modality-shift risk.** If the training distribution never saw overlays, large overlays can push the image off-distribution. Keep marks small relative to the image.
- **Doesn't fix a wrong proposal.** The cue amplifies the proposer; it does not replace it. If the grounding model is wrong about where to point, the cue misleads.

---

## Sources

- Paper: *PreView: Enhancing GUI Grounding with Preprocessed Views and Cues* — Hou & Xu, 2026, [arXiv 2610.33645](https://arxiv.org/abs/2610.33645) — red-circle + hint recipe.
- Paper: *Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V* — Yang et al., 2023 — the numbered-box ancestor.
- Paper: *Visual Prompting via Image Inpainting* — Bar et al., 2022 — early visual-prompting literature.
