# Spectral Rewiring (SAR)
*Depth — post-hoc editing that keeps only the base-spectrum-aligned part of an RL update.*

**TL;DR:** After RL post-training, decompose the update Δθ into the component that lies in the base model's dominant spectral subspace and the orthogonal residual, then keep only the aligned component. The kept part carries essentially all the reasoning gain; the discarded orthogonal residual is what saturates test-time scaling and clashes with other domain fine-tunes at merge time. A one-shot edit, no retraining.

**Prereqs:** [_rl](_rl.md), [grpo](grpo.md)
**Related:** [../pre-training/model-souping](../pre-training/model-souping.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md), [rlvr](rlvr.md)

---

## What it is

Let `θ_0` be a base (pre-RL) checkpoint and `θ = θ_0 + Δθ` be an RL-post-trained one. Subspace-Aligned Rewiring (SAR) chooses the `k` dominant spectral directions of the base model's parameter geometry, projects Δθ onto their span, and reconstructs:

$$
\theta_{\text{SAR}} = \theta_0 + \Pi_{U_k} \Delta\theta
$$

where `U_k` is the top-k spectral basis of the base. Everything orthogonal to `U_k` is dropped. Applied per-layer or per-block in practice.

## How it works

The paper's core empirical claim is that the *reasoning-effective* component of an RL update concentrates in the base's spectral core. Two symptoms of the leftover orthogonal residual:

1. **Test-time scaling saturation.** Long-CoT gains plateau earlier than they should — the orthogonal residual suppresses further reasoning depth.
2. **Merge-time interference.** When averaging several domain-specific RL checkpoints, the orthogonal residuals from different domains do not cancel; they collide.

SAR removes both by dropping the residual. Because the projection is defined by the *base* model's spectrum, multiple RL fine-tunes projected onto the same `U_k` share a compatible representation for downstream averaging.

## Why it matters

- **Preserves reasoning gains** while restoring headroom for test-time scaling.
- **Cleans RL updates for model merging.** Souping and TIES-merge assume checkpoints live in a shared basin; SAR reprojects RL updates so that assumption holds better.
- **Post-hoc.** No retraining, no gradient step — a one-time linear-algebra pass over the checkpoint.

## Gotchas & tricks

- **Choice of k matters.** Too small and reasoning gains erode; too large and the residual returns. Tune per model on a validation reasoning benchmark.
- **Per-layer spectra differ.** Applying a single global `k` is worse than choosing per-layer or per-block `k` from the base model's layer-wise spectra.
- **Base checkpoint required.** SAR is defined against `θ_0`; if you only have `θ`, you can only approximate it (e.g. via SVD of `θ` alone), and the guarantees degrade.
- **Composable with souping.** SAR-cleaned checkpoints soup more cleanly than raw RL checkpoints (that is the paper's main merging result).

## Sources

- Paper: *Spectral Rewiring for Exploration, Purification, and Model Merging* — Yu et al., 2026 (SIA-Lab of Tsinghua AIR & ByteDance Seed).
- Related: [../pre-training/model-souping.md](../pre-training/model-souping.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md).
