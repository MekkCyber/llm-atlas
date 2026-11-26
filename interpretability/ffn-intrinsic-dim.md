# Layer-wise Intrinsic Dimensionality of FFNs
*Depth — PCA-based per-layer measurement of how much room each FFN actually uses, and a distillation use case.*

**TL;DR:** Apply PCA to the input of each FFN down-projection in an LLM; define the layer's intrinsic dimensionality as the fewest principal components needed to reconstruct the activation at a fixed cosine similarity (e.g., 0.99). Across 58 open-weight LLMs, middle layers consistently need more dimensions than edge layers. The profile is stable across English/Chinese calibration data. Allocating distillation-student width according to this profile beats uniform width at matched total parameters (+2.3 to +2.5 points on the two teachers with the most uneven profiles).

**Prereqs:** [interpretability/README.md](README.md)
**Related:** [transformer-block.md](../architectures/transformer-block.md)

---

## What it is

A model-agnostic way to measure how much of each FFN's input space is actually populated. The intuition: if a layer's inputs live on a low-dimensional subspace, that layer is using less of its allocated width than a layer whose inputs fill the full space. PCA on a calibration sample of activations at the down-projection input gives a per-layer spectrum; the intrinsic dimension is the index at which cumulative reconstruction cosine similarity crosses a threshold.

## How it works

For layer `ℓ`, run a calibration set through the model and collect the activations at the input of the FFN down-projection (i.e., *after* the up-projection, before the final linear):

```
X_ℓ ∈ R^{N × 4d}         # N calibration tokens × FFN inner width
```

Compute PCA: `X_ℓ ≈ U_k · Σ_k · V_k^T` for the top `k` components. The intrinsic dimension `d_ℓ` is the smallest `k` such that the rank-`k` reconstruction achieves mean cosine similarity ≥ 0.99 with the original activations. The paper sweeps thresholds; 0.99 is the headline. The resulting `(d_ℓ)` is a per-layer profile, typically U-shaped — low at edges, high in the middle — but with family-dependent kinks.

## Why it matters

- **Robust measurement.** Insensitive to calibration data: English and Chinese calibration give the same profile. This is important — it means the signature is a property of the trained weights, not the probe.
- **Direct use in distillation.** When building a smaller student, allocate FFN width per layer according to `(d_ℓ)` of the teacher, not uniformly. +2.3 to +2.5 points on a five-task average on the two teachers with the most uneven profiles, at matched total parameters. The gain persists through supervised fine-tuning.
- **A principled width-allocation signal.** Depth-vs-width debates in model-sizing usually appeal to intuition. This is a measurable alternative: measure the teacher, allocate student width to match.
- **Cheap to compute.** PCA on a few thousand calibration tokens per layer. Runs in minutes on a laptop given cached activations.

## Gotchas & tricks

- **Threshold choice shifts the profile.** 0.99 gives a sharper shape; 0.95 flattens it. Fix the threshold up-front and report ablation.
- **Cosine similarity, not MSE.** Reconstruction in cosine space ignores norm; the FFN input norms vary a lot across layers, so cosine is the fair comparator.
- **58-model study; hasn't been reproduced yet outside the paper.** Treat the specific +2.3/+2.5 numbers as paper-only until replication lands.
- **Intrinsic dim ≠ effective capacity.** A low intrinsic-dim layer is under-using its width — but reducing width there costs representation; the signal is *relative* allocation, not absolute removal.
- **Middle-layers-are-heavy isn't universal.** Some teachers have high middle *and* high top layers; family (MoE vs. dense), training data, and tokenizer all matter. Measure the specific teacher.
- **Not a pruning criterion directly.** Low intrinsic dim does not mean "prune these neurons" — the layer is using a low-rank subspace of its width, but which neurons span that subspace is not identified by this measurement.

## Sources

- Paper: *Measuring Layer-wise Intrinsic Dimensionality of FFNs in LLMs via PCA* — (single-author, affiliation not shown on HF page), 2026 — [arXiv:2611.32782](https://arxiv.org/abs/2611.32782).
