# Distribution-wise rewards
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RL fine-tuning of image / video generators with **sample-wise** rewards (e.g. an aesthetic reward per image) reliably produces reward hacking and mode collapse — every sample chases the same reward-maximizing point. A **distribution-wise reward** scores a *batch* against the target data distribution (FID-flavored, or divergence to a reference set) rather than each sample individually, so mode diversity becomes part of the objective. Made cheap by a **subset-replace** estimator that keeps a rolling reference set and updates the distributional statistic incrementally.

**Prereqs:** [_rewards](_rewards.md), [_rl](_rl.md)
**Related:** [grpo](grpo.md), [../pre-training/model-souping](../pre-training/model-souping.md)

---

## What it is

Standard RL for generative models scores each generated sample independently:

$$
r_i = R(x_i)
$$

with $R$ a per-sample scorer (aesthetic model, CLIP alignment, learned RM). The policy gradient then pushes *every* sample toward high $R$, which — combined with a stochastic policy — collapses samples into a narrow mode. Reward hacking follows: the policy finds the tightest distribution that scores well.

A **distribution-wise reward** replaces this with a signal computed over the *joint* distribution of a batch:

$$
r_{\text{batch}} = -D\!\big(\mathrm{batch},\;\mathrm{reference}\big)
$$

where $D$ is a divergence (FID-style, MMD, or a discriminator-based estimator) between the generated batch and a reference set that approximates the true data distribution. High reward means "the generated batch looks like real data," not "each sample scores highly."

---

## How it works

**Subset-replace estimator.** Naïve implementation recomputes $D$ from scratch every step — prohibitive for RL. Instead:

1. Maintain a rolling **reference set** of generated samples.
2. Each step, replace a small subset of the reference with newly generated samples.
3. Compute the distributional statistic *incrementally* — only the changed samples affect it.

This is cheap enough to run inside a policy gradient loop, and it keeps the reference set moving toward the current policy's distribution while still being a valid signal.

**RL over merging coefficients.** As a companion trick, the authors also apply RL to the coefficients used when merging post-hoc model checkpoints (souping). Because RL for diffusion introduces stochastic-differential-equation dynamics that mismatch the deterministic inference-time solver, RL-tuned merging coefficients help absorb the mismatch.

---

## Why it matters

- **Fights mode collapse at the reward level, not with regularizers.** Sample-wise reward + entropy bonus is a patch; distributional reward is a fix.
- **Concrete gains.** FID-50K: **SiT 8.30 → 5.77**, **EDM2 3.74 → 3.52** — big margins on standard diffusion baselines while preserving qualitative diversity.
- **Reusable estimator.** The subset-replace trick is applicable to any batch-level metric (KID, precision/recall of manifolds) as long as the metric supports incremental update.
- **Complements distillation / caching accelerators.** Because it's an RL fine-tuning recipe, it composes with fast-sampling techniques for the base model.

---

## Gotchas & tricks

- **Reference set drift.** If you replace too many samples per step, the reference set collapses to the current policy and the reward saturates. Small replace fractions (~1%) work well.
- **FID-flavored metrics have bias for small batches.** Use robust variants (KID, unbiased FID) or grow the reference set to a size where bias is tolerable.
- **RL step still needs a policy-gradient estimator through the sampler.** For diffusion, use SDE-style back-propagation-through-time or score-based tricks. The paper's RL-tuned merging coefficients partially work around train/inference mismatch.
- **Signal is per-batch, not per-sample.** All samples in a batch share the same reward — like GRPO's response-level advantage. Fine for image / video generation; not the shape you want for token-level LLM tuning.
- **Not for text.** The distributional-reward idea in principle applies to text, but reference-set divergences over text distributions are much harder to estimate.

---

## Sources

- Paper: *Optimizing Visual Generative Models via Distribution-wise Rewards* — 2026 — [arXiv:2607.02291](https://arxiv.org/abs/2607.02291).
- Related: [grpo](grpo.md) — the per-response advantage pattern this reward integrates into; [model-souping](../pre-training/model-souping.md) — post-hoc merging targeted by the RL-tuned coefficient trick.
