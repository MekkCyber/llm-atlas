# Compositional Edit-Prior Distillation (CEPD)
*Depth — distilling edit quality from a non-layered teacher editor into a layered student that outputs RGBA layers.*

**TL;DR:** When you add a structured-output head (RGBA layers) to a pretrained image editor, edit quality typically regresses because the new objective fights the original one. CEPD distills a non-layered teacher editor into the layered student so the student keeps the teacher's edit prior while also learning to produce reusable layers. The edited canvas and its component layers share one token sequence, which lets attention correlate the composite with each layer during training and inference.

**Prereqs:** none
**Related:** [attention.md](../fundamentals/attention.md)

---

## What it is

A distillation recipe for augmenting a pretrained diffusion-based image editor with a *layered* output format (edited canvas + per-component RGBA layers) without losing edit fidelity. CEPD treats the pretrained non-layered editor as a frozen teacher and matches its edit-prior behavior on the student even as the student learns to emit layers.

## How it works

1. **Shared token sequence.** The student represents the edited canvas and each contributing RGBA layer as positions in a single unified token sequence. Attention is unrestricted across layers and canvas.
2. **Teacher distillation.** For each training edit, run the frozen non-layered teacher and extract its predicted canvas distribution. The student is trained to match this canvas distribution via a distillation loss (KL on logits or a feature-matching loss on hidden states).
3. **Layer objective.** Separately, a reconstruction / dataset loss supervises the per-layer RGBA outputs against ground-truth layer annotations (LayerEditor-100K).
4. **Compositional constraint.** A soft loss requires the layers to composite back to the canvas, aligning the two objectives.

Loss = λ_distill · L_canvas_distill + λ_layer · L_layer_recon + λ_composite · L_composite.

## Why it matters

- **Keeps the base editor's edit skill.** Naive multi-objective fine-tuning of a pretrained editor degrades edit quality; CEPD avoids this by treating the pretrained model as a prior.
- **Produces reusable layers.** The output is directly consumable by downstream editing pipelines (Figma / Photoshop-style workflows) — not just a flat edited pixel grid.
- **General template.** The pattern — distill from a non-structured teacher while adding a structured head — is reusable for any "add structured output to a strong generative base" problem.

## Gotchas & tricks

- Loss weighting is sensitive: too high `λ_distill` and the layer head doesn't learn; too low and canvas quality regresses. The paper reports a schedule that warms up the layer objective after the distillation path is established.
- The composition loss is important as a regularizer; without it, layers can drift from the canvas (visible compositing seams).
- Teacher data distribution matters — if the teacher is weaker in a region of edit space, distillation can inherit that weakness.
- Not restricted to layer outputs; a similar recipe can graft any structured head onto a pretrained diffusion editor.

## Sources

- Paper: *Editing Images into Reusable Layers via Compositional Edit-Prior Distillation* — Xue et al., 2026 — [arXiv:2610.24738](https://arxiv.org/abs/2610.24738)
