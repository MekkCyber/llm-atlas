# Controllable Generation

*Taxonomy — steer a pretrained diffusion / flow-matching model with extra visual conditions (depth, edges, segmentation, pose, identity, …).*

**TL;DR:** Pretrained image generators take text but not detailed spatial conditions. Controllable generation adds those conditions without retraining the backbone. The dominant pattern is a **dual-branch** architecture: freeze the main generator, train a small side network that encodes the condition and injects features. Variants differ in *where they inject*, *how the side branch is trained*, and *what theoretical reading* they admit (score-decomposition vs feature-fusion).

**Related taxonomies:** [_diffusion-distillation.md](_diffusion-distillation.md)
**Depth files covered here:** [likelihood-score-alignment.md](likelihood-score-alignment.md)

---

## The problem

A frozen pretrained diffusion / flow-matching model produces images from text. Real applications need to *additionally* condition on spatial structure: edges, depth, segmentation, pose, identity. Retraining the whole model per condition is wasteful. We want a small adapter per condition that snaps onto the same frozen backbone.

## The shared pattern

The dominant pattern across techniques is **dual-branch**:

1. **Main network** — frozen pretrained generator; carries the prior over natural images.
2. **Side network** — trained encoder for the condition; outputs features.
3. **Injection** — side features are added to or fused with intermediate features of the main network at one or several layers.

A score-based reading: the main network supplies the unconditional score `∇ log p(x)`, the side network implicitly supplies the likelihood score `∇ log p(c | x)`, so the combined output approximates the conditional score `∇ log p(x | c)`. Most prior work treats this as empirical; LISA makes it explicit.

## Variants

| Technique | Side-branch architecture | Injection | Training pressure | When it wins |
| --- | --- | --- | --- | --- |
| ControlNet | Trainable copy of encoder half | Add to skip connections | Standard diffusion loss | Heavy-control tasks (depth, edge), maximum capacity |
| T2I-Adapter | Lightweight CNN per condition | Add at fixed levels | Standard diffusion loss | Cheap deployment, many conditions |
| IP-Adapter | Image-prompt encoder + cross-attention | Cross-attend at attention layers | Standard diffusion loss | Identity / image-prompt conditioning |
| Uni-ControlNet | Shared side branch across many conditions | Modulated injection | Multi-condition diffusion loss | Unifying many conditions into one adapter |
| [likelihood-score-alignment](likelihood-score-alignment.md) | Any dual-branch | Same as base method | Standard loss **+ explicit alignment to approximated likelihood score** | Faster convergence + better disentanglement, drop-in for any dual-branch design |

## How to choose

- **Single condition, high fidelity needed:** ControlNet. Heaviest but strongest.
- **Many conditions, cheap deployment:** T2I-Adapter family or Uni-ControlNet variants.
- **Identity / image-prompt:** IP-Adapter. Cross-attention is the right pattern for "this should look like *that*."
- **Already using a dual-branch:** stack [LISA](likelihood-score-alignment.md) — it's a regularizer that accelerates training and improves disentanglement at zero inference cost.
- The "best" injection layer is usually all of them, modulated by a small scalar per layer; ablations in the original papers tend to converge on this.

## Adjacent but distinct

- **Fine-tuning a LoRA on (condition, image) pairs**: bakes the condition into the backbone for *one* fixed condition type. Less flexible than dual-branch.
- **Classifier guidance / classifier-free guidance**: steering via a separately-trained classifier or an unconditional model. Different mechanism; condition is a label, not a spatial structure.
- **Distillation-based control**: train a student that incorporates the control end-to-end; cheaper at inference but doesn't reuse the frozen backbone.

## Sources

- Paper: *Adding Conditional Control to Text-to-Image Diffusion Models* — Zhang & Agrawala, 2023 — ControlNet.
- Paper: *T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for T2I Diffusion Models* — Mou et al., 2023.
- Paper: *IP-Adapter* — Ye et al., 2023 — image-prompt conditioning.
- Paper: *Uni-ControlNet* — Zhao et al., 2023.
- Paper: *LISA: Likelihood Score Alignment for Visual-condition Controllable Generation* — Wang et al., 2026 — score-aligned regularizer for dual-branch designs.
