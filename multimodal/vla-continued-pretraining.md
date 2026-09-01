# VLA Continued Pretraining (VLAct)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A representation-centric recipe for turning a strong Vision-Language Model into a Vision-Language-Action (VLA) model via **continued pretraining**, rather than joint-pretraining from scratch on ever more robot data. Preserves the base VLM's vision-language priors while introducing **shared action semantics across embodiments**, achieving strong sim and unseen-embodiment transfer under a limited compute budget.

**Prereqs:** [README.md](README.md), [../pre-training/mid-training.md](../pre-training/mid-training.md)
**Related:** [../pre-training/model-souping.md](../pre-training/model-souping.md) · [../architectures/README.md](../architectures/README.md)

---

## What it is

The dominant VLA recipe is joint-pretraining a large VLM+action model on a heterogeneous mix of robot trajectories, egocentric video, and vision-language corpora — data-hungry and hard to scale outside the largest labs. VLAct argues the bottleneck is **representation**, not data volume: given a strong VLM, a small continued-pretraining phase with the right structure is enough.

## How it works

Two guiding principles:

1. **Preserve the VLM prior.** Keep the vision-language alignment and world knowledge already in the base VLM — don't overwrite it in the process of adding actions. Weight-space regularization, careful learning-rate scheduling, and layer-freezing choices all serve this end.
2. **Introduce shared action semantics.** Extend the model with an action tokenization / semantics layer that is **shared across embodiments** — the same action tokens mean the same underlying primitives regardless of robot morphology. This lets the model transfer between embodiments without a bespoke head per platform.

The training phase is short compared to joint pretraining — it's *continued* pretraining rather than from-scratch, and diversity of embodiments in the phase matters more than raw robot-trajectory volume.

## Why it matters

- Reaches strong performance across simulation and **unseen** embodiments under a limited compute budget, outperforming data-scaling baselines that keep adding robot trajectories to joint pretraining.
- Turns VLA construction into a recipe applicable to any strong VLM, not only ones large labs pretrain from scratch — opens per-lab embodiment fine-tuning as a viable path.
- Frames the field's next question as "what's the right representation to bridge V/L and action?" rather than "how much more robot data can we collect?"

## Gotchas & tricks

- Losing the VLM prior is the failure mode to avoid — a naïve continued-pretraining phase with too high a learning rate erases the vision-language alignment.
- Shared action semantics only pay off if the embodiments actually share primitives at the level chosen; too-fine tokenizations degrade to per-embodiment vocabularies.
- Composes with adapters and per-embodiment fine-tuning as a final step; VLAct is the shared middle layer.

## Sources

- Paper: *Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models* — Yang et al., CUHK / HKU, 2026 — [arxiv](https://arxiv.org/abs/2608.27550)
