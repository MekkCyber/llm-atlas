# Model Merging

*Taxonomy — combining multiple trained checkpoints into a single deployable model, without additional training.*

**TL;DR:** Model merging turns "we trained several candidates" into one final model — no distillation, no ensembling at inference time, just a weight-level combination. The simplest instance is uniform weight averaging (Model Souping); modern variants prune to the aligned subspace of the base's weights before merging (Spectral Rewiring) or align neurons across differently-initialized checkpoints (Git Re-Basin). The unifying observation: successful post-training updates concentrate in a low-dimensional subspace, and the merging operation lives or dies by how well it respects that subspace.

**Related taxonomies:** *(none directly; see [_training-stability](_training-stability.md) for stability of the training runs whose outputs get merged)*
**Depth files covered here:** [model-souping](model-souping.md) · [../post-training/spectral-rewiring](../post-training/spectral-rewiring.md)

---

## The problem

Modern training pipelines produce many candidate checkpoints per release: seed sweeps, hyperparameter variations, SFT-vs-DPO branches, RL experts across domains. Each candidate has cost real compute. Shipping only the single best throws the others' work away. Ensembling at inference is expensive. Distilling one checkpoint into another is a full training run.

Model merging asks whether the *weights* themselves can be combined cheaply — one linear operation on state dicts — and still produce a checkpoint that matches or beats the best single candidate.

## The shared pattern

Every merging technique follows the same recipe:

1. Start from $N$ checkpoints $\{\theta_1, \ldots, \theta_N\}$ that share architecture.
2. Choose (or learn) weights or projections that describe *how* to combine them.
3. Emit a single $\theta_{\text{merge}} = f(\theta_1, \ldots, \theta_N)$.
4. Serve the merged model. No further training required.

Techniques differ on step 2 — how they decide *what* to combine and *how much*. Every variant is fighting the same underlying constraint: naive averaging assumes candidates lie in the same loss basin, which is only true when they share initialization or an early-training checkpoint.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [model-souping](model-souping.md) (uniform / greedy) | Element-wise average of candidates sharing a parent | Fragile across recipes; assumes shared basin | Ship-time consolidation of many candidates from the same run |
| [Spectral Rewiring (SAR)](../post-training/spectral-rewiring.md) | Project each candidate's delta into the base's top singular subspace before merging | One extra SVD per matrix; picks $k$ | Merging RL-post-trained experts across domains; purifying interference |
| Git Re-Basin (Ainsworth et al. 2022) | Permutation-align neurons across differently-initialized checkpoints, then average | Solving the assignment problem is expensive at scale | Merging across independent runs, not just shared-init |
| Task Arithmetic (Ilharco et al. 2022) | Represent each fine-tune as a task vector $\Delta\theta$; add/subtract task vectors | Requires a shared base; interference between tasks | Combining multiple domain-specific fine-tunes |
| SLERP | Spherical interpolation instead of linear; preserves norm | Only two-way; rarely tuned well at LLM scale | Interpolating between two related checkpoints |
| TIES-Merging (Yadav et al. 2023) | Trim, elect sign, disjoint merge — reduces sign conflicts between task vectors | Extra hyperparameters; can under-merge | Task-vector merging with strong interference |

Link techniques with a depth file; leave others as plain text until a depth file lands.

## How to choose

**For consolidating multiple candidates from the same training run** (seeds, LR sweeps, SFT ablations): uniform [model-souping](model-souping.md) is the default. Greedy souping is strictly better if you have a held-out set and one candidate might be broken.

**For merging RL-post-trained experts across domains** (math expert + code expert + agent expert): use [Spectral Rewiring (SAR)](../post-training/spectral-rewiring.md). Naive souping of RL-expert deltas is brittle because the orthogonal residue of each update carries cross-domain interference; SAR removes it before averaging.

**For merging across different initializations** (rare in production but useful for research): Git Re-Basin or OT Fusion align neurons before averaging. Rarely used at LLM scale because the assignment problem is expensive.

**For adding/subtracting task capabilities** (positive: SFT-A + SFT-B; negative: remove a bad-behavior tune): Task Arithmetic. Add TIES-Merging when the task vectors have strong sign conflicts.

The techniques compose. A common LLM release pattern: seed-souping for stability, then task-arithmetic (or SAR) to combine domain experts, then a final SFT pass over the merged model to polish.

## Adjacent but distinct

- **Ensembling** — running multiple models at inference and combining outputs. Different tradeoff: no merged weight file, multiply inference cost by $N$.
- **Distillation** — training a student to imitate a teacher (or an ensemble). A full training run, not a merge.
- **LoRA merging** — a special case of task arithmetic where the "candidates" are LoRA adapters. The merge is simpler because LoRAs are already low-rank.
- **Mixture of Experts** — routes different inputs to different sub-networks; no weights are ever merged, they're all kept and selected between.
- **[Model souping in Kimi k1.5's long2short](../post-training/reasoning/long2short.md)** — souping is used *as one of four* long-to-short compression methods; it's the training-free baseline the other methods are compared to.

## Sources

- Paper: *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time* — Wortsman et al., 2022 — the canonical souping paper.
- Paper: *Spectral Rewiring for Exploration, Purification, and Model Merging* — Yu et al., 2026 — the spectral-projection variant that unlocks robust cross-domain merging.
- Paper: *Git Re-Basin: Merging Models modulo Permutation Symmetries* — Ainsworth et al., 2022 — permutation alignment across different inits.
- Paper: *Editing Models with Task Arithmetic* — Ilharco et al., 2022 — task vectors as a merging primitive.
- Paper: *TIES-Merging: Resolving Interference When Merging Models* — Yadav et al., 2023.
- Paper: *Averaging Weights Leads to Wider Optima and Better Generalization (SWA)* — Izmailov et al., 2018 — single-run predecessor to model souping.

---

## Conventions

- **Filename:** `_model-merging.md` (leading underscore — taxonomy).
- **Folder placement:** `pre-training/`, same folder as [model-souping](model-souping.md). Spectral Rewiring lives in `post-training/` because the merge operates on RL deltas, but the taxonomy links both.
- **Scope:** any technique that combines multiple trained checkpoints into one, without further training. Distillation and ensembling live in adjacent categories.
