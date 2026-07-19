# Model Merging

*Taxonomy — combining several trained checkpoints into one final model without retraining.*

**TL;DR:** After training multiple checkpoints (different seeds, data mixes, or post-training regimes), combine them into one model at deployment time. The oldest and simplest recipe is uniform weight averaging (souping); modern variants add per-parameter selection, sign-consistency filters, or spectral cleanups so the combined model preserves each source's specialty without their cross-domain interference. Universally cheap — a one-time linear-algebra pass on the checkpoints, no gradient steps.

**Related taxonomies:** [_lr-schedules](_lr-schedules.md), [_training-stability](_training-stability.md)
**Depth files covered here:** [model-souping](model-souping.md) · [spectral-rewiring](../post-training/spectral-rewiring.md)

---

## The problem

Big-lab training runs produce many candidate checkpoints: seeds of the same recipe, multiple SFT/DPO/RL variants, domain-specialist fine-tunes. Shipping one and discarding the rest wastes information; ensembling them at inference multiplies serving cost. Model merging asks: is there a single set of weights that captures most of what the whole family knows?

The underlying constraint is loss-landscape geometry. Weight-space averaging only works when the checkpoints live in the same basin — different random inits typically don't, but checkpoints sharing a pretrained parent typically do. Modern RL post-training breaks even this: RL updates can have large orthogonal components that clash across domains.

## The shared pattern

Every merging method does the same three moves:

1. **Register.** Bring candidate checkpoints into a shared parameter frame (usually via a common ancestor — the shared pretrained base).
2. **Combine.** Aggregate weights (average, weighted sum, per-parameter select).
3. **Clean.** Optionally remove components that would cause interference (sign disagreements, orthogonal RL residuals, out-of-basin drift).

The variants differ in *what* they combine and *what* they clean.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [model-souping](model-souping.md) | Uniform or greedy average of shared-parent checkpoints | Fails across different inits; can degrade if a bad candidate is included | Shipping the best of multiple SFT candidates |
| Task Arithmetic (no depth file yet) | Add or subtract *task vectors* Δθ = θ_task − θ_base | Requires clean task vectors; cross-task collisions | Composing capabilities from multiple domain fine-tunes |
| TIES-Merge (no depth file yet) | Trim, elect sign, then merge — resolves per-parameter conflicts | Extra hyperparameters (trim threshold, sign rule) | Merging many task vectors with different scales |
| DARE (no depth file yet) | Drop and rescale a fraction of Δθ before averaging | Only sensible for post-hoc post-training deltas | Merging LoRA / SFT deltas at large numbers |
| [spectral-rewiring](../post-training/spectral-rewiring.md) | Project each Δθ onto the base's dominant spectrum, drop residual | Requires the base checkpoint; k must be tuned | Merging RL-post-trained checkpoints |

Link techniques with a depth file; leave others as plain text until a depth file lands.

## How to choose

The modern default for **SFT-only checkpoints** sharing a parent is uniform souping (or greedy souping if you have validation compute) — nothing has beaten it consistently at large scale for that regime.

For **multiple domain fine-tunes** (task arithmetic setting), TIES-merge is the current best-scoring default; DARE is a plug-in preprocessing step that often stacks with it.

For **RL post-training checkpoints**, plain averaging tends to interfere. Spectral rewiring is the newest cleanup pass designed for that case: project each RL Δθ onto the base's spectral core before averaging.

Variants combine: SAR → TIES-merge → uniform average is a legitimate stack, and each stage removes a different source of interference.

## Adjacent but distinct

- **Ensembling at inference.** Runs all models and averages logits. Different problem — pays per-request cost forever. Merging pays once.
- **Distillation.** Trains a new student to imitate an ensemble. Requires gradient steps and data. Merging is a straight arithmetic operation on the state dicts.
- **Mixture-of-experts.** Multiple *specialists* selected per token at inference. Different beast — merging collapses them into one dense model.

## Sources

- Paper: *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time* — Wortsman et al., 2022 — foundational souping.
- Paper: *Editing Models with Task Arithmetic* — Ilharco et al., 2023 — task-vector arithmetic.
- Paper: *TIES-Merging: Resolving Interference When Merging Models* — Yadav et al., 2023.
- Paper: *DARE: Language Models are Super Mario* — Yu et al., 2024 — drop-and-rescale preprocessing.
- Paper: *Spectral Rewiring for Exploration, Purification, and Model Merging* — Yu et al., 2026 — RL-checkpoint cleanup via spectral projection.
