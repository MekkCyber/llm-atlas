# Model Merging

*Taxonomy — combine multiple trained model checkpoints into one, without retraining, by manipulating their weights.*

**TL;DR:** Given several checkpoints sharing an ancestor (SFT variants, RL-tuned experts, seed replicates), produce one final model by combining their weights. The design axis is *what part of each checkpoint's update to keep*: everything (souping), a projected subspace (SAR / spectral rewiring), a sign-consistent slice (TIES), or a sparsified core (DARE). All are training-free; they differ in what they discard and why.

**Related taxonomies:** —
**Depth files covered here:** [../pre-training/model-souping](../pre-training/model-souping.md) · [spectral-rewiring](spectral-rewiring.md)

---

## The problem

Training runs are expensive; multiple runs are common (seed sweeps, safety ablations, domain experts). Serving them all is a non-starter. Naively averaging their weights ("soup") often helps — but breaks when candidates come from meaningfully different recipes because their updates interfere.

Model merging is the class of *training-free* recipes that consolidate multiple checkpoints into one. The variants exist because "just average" is not always the right combining function.

## The shared pattern

Every method operates on the difference $\Delta W_i = W_i - W_{\text{base}}$ of each candidate from a shared base $W_{\text{base}}$, then combines the deltas back into a merged model:

$$W_{\text{merged}} = W_{\text{base}} + f(\Delta W_1, \ldots, \Delta W_N)$$

The choice of $f$ is the taxonomy: identity + average (souping), projection onto a base-aligned subspace (SAR), sign-consistent sparsification (TIES), random dropout + rescaling (DARE), or task-arithmetic linear combinations.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [../pre-training/model-souping](../pre-training/model-souping.md) | Uniform (or greedy) average of weights | Assumes candidates share a basin; cross-recipe merges get noisy | Seed sweeps, hyperparameter sweeps, same-recipe replicates |
| [spectral-rewiring](spectral-rewiring.md) | Project each Δ onto the base model's top singular subspace, keep only that | Requires SVD of the base; per-matrix rank choice | Multi-expert RL merging; releasing suppressed capabilities |
| TIES (no depth file yet) | Sign-align deltas across experts, drop conflicting entries | Discards information at conflicts; sensitive to threshold | Merging experts with known task conflicts |
| DARE (no depth file yet) | Randomly drop delta entries and rescale the survivors | Randomness → variance; needs multiple runs to stabilize | Compressing many similar deltas |
| Task arithmetic (no depth file yet) | Add / subtract task-vector deltas with signed coefficients | Only works when tasks are near-independent | Compositional skill combination |

## How to choose

For **same-recipe candidates** (seed variance, SFT replicates), start with **uniform souping** — it's the strongest simple baseline and beats fancier methods when candidates already share a basin.

For **RL-post-trained experts** (math expert + code expert + general instruction expert, all from the same base), start with **spectral rewiring**. Souping tends to smear out the RL improvements; SAR projects each expert's update onto the base's spectral subspace so aligned contributions add and orthogonal noise cancels.

For **explicit task combination** (e.g. "add math skill to a base"), task arithmetic gives you a signed knob; TIES helps when tasks are in tension.

Combining strategies is fine: soup a group of same-recipe replicates first, then merge the resulting per-expert souped checkpoints via SAR.

## Adjacent but distinct

- **Distillation** — trains a new model to imitate the ensemble; not training-free.
- **LoRA composition** — combines *low-rank adapters* rather than full checkpoints; different substrate.
- **Weight-space ensembling at inference** — routes to different checkpoints per input rather than merging; not a single served model.

## Sources

- Paper: *Model soups* — Wortsman et al., 2022.
- Paper: *Spectral Rewiring for Exploration, Purification, and Model Merging* — SIA-Lab of Tsinghua AIR & ByteDance Seed, 2026.
- Paper: *TIES-Merging* — Yadav et al., 2023.
- Paper: *DARE* — Yu et al., 2023.
- Paper: *Editing Models with Task Arithmetic* — Ilharco et al., 2023.
