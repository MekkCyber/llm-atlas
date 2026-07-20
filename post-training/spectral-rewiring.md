# Spectral Rewiring (SAR)
*Depth — post-hoc, training-free editing of RL post-training deltas by projection onto the base model's dominant singular subspace.*

**TL;DR:** RL post-training changes weights by a delta $\Delta W = W_{\text{rlhf}} - W_{\text{base}}$. **Subspace-Aligned Rewiring (SAR)** claims the *reasoning-effective* part of $\Delta W$ lives in a low-rank subspace aligned with $W_{\text{base}}$'s top singular directions — and everything orthogonal to that subspace is either noise (suppresses test-time-scaling) or interference (blocks multi-domain merging). Keep the aligned projection, discard the rest. No retraining.

**Prereqs:** [../pre-training/model-souping.md](../pre-training/model-souping.md), [grpo.md](grpo.md)
**Related:** [rlvr.md](rlvr.md), [rejection-sampling.md](rejection-sampling.md)

---

## What it is

SAR is a **post-hoc, training-free edit** applied to an RL post-trained checkpoint. Given a base model $W_{\text{base}}$ and an RL-tuned model $W_{\text{rlhf}}$, form $\Delta W = W_{\text{rlhf}} - W_{\text{base}}$, project $\Delta W$ onto the subspace spanned by $W_{\text{base}}$'s top singular directions, and add the projection back to $W_{\text{base}}$. The output is a cleaner, more merge-friendly RL model.

The paper positions SAR as the fix for two deployment-relevant pathologies of dense full-parameter RL updates: **suppressed reasoning** (premature test-time-scaling saturation) and **cross-domain interference** (multi-domain training or model merging collides).

## How it works

For each weight matrix (attention Q/K/V/O, MLP up/down), compute the base model's SVD:

$$W_{\text{base}} = U \Sigma V^\top$$

Pick the top-$k$ singular vectors (both left and right). Project the delta:

$$\Delta W_{\parallel} = U_k U_k^\top \Delta W V_k V_k^\top$$

and set:

$$W_{\text{SAR}} = W_{\text{base}} + \Delta W_{\parallel}$$

Everything orthogonal to the top-$k$ base subspace is thrown away. Rank $k$ is chosen small — the paper reports ~0.58% of total parameter count is enough. There is no gradient step, no retraining, no calibration data required (except to pick $k$).

For multi-expert merging, project each expert's own $\Delta W_i$ onto its shared base subspace and add:

$$W_{\text{merged}} = W_{\text{base}} + \sum_i \Delta W_{i,\parallel}$$

Because each expert's kept component lives in the same base-aligned subspace, contributions add without the destructive interference that plagues naive weight averaging.

## Why it matters

- **>99% of RL-post-training gains preserved** while retaining only ~0.58% of parameters worth of update — a strong claim about where the useful information actually sits.
- **Reactivates high-k test-time scaling.** Suppressed reasoning shows up as pass@k saturating early; SAR restores the exploration tail on math benchmarks.
- **Multi-domain purification.** SAR of a mixed-domain RL model releases suppressed coding capability without harming math or instruction following — the orthogonal components were actively fighting each other.
- **Beats standard merging baselines** across expert consolidation, sometimes even beating the best single-domain expert.
- **Training-free.** Cheap enough to be a post-processing step on every RL release.

## Gotchas & tricks

- **The base's SVD must match your target weight layout.** For MoE layers, project each expert independently; naïve full-parameter SVD across experts is meaningless.
- **Pick $k$ per-matrix, not global.** Different layers have different effective rank; a single $k$ across all matrices leaves quality on the table.
- **This is training-free but not calibration-free.** The choice of $k$ is a hyperparameter tuned on a validation set. Underspecified: the paper's exact $k$-selection protocol needs to be reproduced from code.
- **Cross-init merging is still off-limits.** SAR helps expert merging when experts share $W_{\text{base}}$. Different pretraining runs → different SVD frames → SAR alone won't rescue.
- **Related to but distinct from LoRA.** LoRA constrains the *update* to be low-rank up front; SAR post-hoc extracts the base-aligned low-rank part from a full-rank update.

## Sources

- Paper: *Spectral Rewiring for Exploration, Purification, and Model Merging* — Yu, Gao, Wu, Song, Ma, Zhang, Zhou — SIA-Lab of Tsinghua AIR & ByteDance Seed, 2026 — introduces SAR.
- Paper: *Model soups* — Wortsman et al., 2022 — baseline merging method SAR beats.
