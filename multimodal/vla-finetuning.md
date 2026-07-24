# VLA Finetuning without Representation Drift (Anchor-Align)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Fine-tuning a pretrained VLM on robot demonstrations via behavior cloning (BC) is the standard VLA recipe, but BC progressively **overwrites** the pretrained visual and semantic representations. Co-training on web image-text data doesn't fix it — language and action losses live on separate observations, leaving VLAs with language–action misalignment. Anchor-Align adds two objectives: **Vision-Language Anchoring** (layer-wise distillation from a frozen VLM copy) and **Language-Action Alignment** (each action becomes a discrete motion-direction label jointly trained with language on the same robot observation). On a real xArm7, success rates roughly double.

**Prereqs:** [README.md](./README.md)
**Related:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md), [../post-training/dpo.md](../post-training/dpo.md)

---

## What it is

A **vision-language-action (VLA) policy** is a VLM whose head has been retargeted from token prediction to action prediction, then fine-tuned on robot demonstrations. The two standard failure modes:

- **Representation drift.** BC fine-tuning on narrow robot data overwrites the broad pretrained representations. The model gets good on training scenes but loses OOD generalization.
- **Language–action misalignment.** Co-training on web image-text data helps *some* but the language loss and action loss are computed on *different observations* — the model never has to reconcile the two on the same input.

Anchor-Align addresses both with two auxiliary objectives added to standard BC.

## How it works

**Vision-Language Anchoring.** Keep a **frozen copy** of the pretrained VLM. During BC fine-tuning, distill **layer-wise representations** from the frozen copy into the student. This is closer to knowledge distillation than a regularization term — the frozen copy acts as an "anchor" that prevents the student's representations from drifting off the pretrained manifold.

**Language-Action Alignment.** Convert each action target into a **discrete motion-direction label** (e.g. "move left", "rotate wrist clockwise"). Jointly train language prediction and action prediction on the **same robot observation** — the model has to produce both a natural-language description of the intended motion and the underlying action from the same image. This forces cross-supervision between the two heads on shared inputs, which co-training on web data alone cannot do.

Both objectives are added to the BC loss with tuned weights. Architecture-agnostic — the paper validates on two different VLA backbones.

## Why it matters

- **BC destroys representations** is a widely observed VLA failure mode with no clean fix before this. Anchor-Align gives one, works on two different VLA architectures — evidence it's a general recipe rather than an architecture-specific hack.
- **Real-robot doubling.** 28% → 54% and 37% → 60% real-world success across two backbones is a large effect for an auxiliary-loss change.
- **Sim results are consistent.** LIBERO-PRO, LIBERO-Plus, CALVIN all show OOD-perturbation and long-horizon gains, arguing the mechanism transfers beyond the specific hardware.

## Gotchas & tricks

- The frozen VLM copy doubles memory during training. Layer-wise distillation can be done with a subset of layers to reduce cost.
- Discretizing actions into motion-direction labels loses precision. Anchor-Align uses the discretization only for the language-alignment auxiliary; the primary action head still predicts continuous actions.
- Weighting is important: too much anchoring and the model can't learn robot behavior at all; too little and the drift returns. Paper's default weights are a starting point.
- Does *not* address the data-scarcity problem — you still need enough robot demonstrations to learn the primary BC objective.

## Sources

- Paper: *Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment* — Dalal, Patel, Jain, et al., 2026 — [arXiv:2607.13429](https://arxiv.org/abs/2607.13429)
- Benchmarks: LIBERO-PRO, LIBERO-Plus, CALVIN (used for the simulation-scale ablations).
