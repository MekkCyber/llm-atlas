# Distribution-wise Rewards
*Depth — RL fine-tuning generative models against a *batch-level* reward that scores a whole distribution instead of per-sample scores.*

**TL;DR:** Conventional RL fine-tuning of image (and text) generators uses per-sample rewards $r(x)$ and averages over the batch. That signal makes every sample optimize *the same direction independently*, encouraging mode collapse and reward-hacking artifacts. **Distribution-wise rewards** replace $r(x)$ with a score $R(\{x_1, \ldots, x_B\})$ over a whole generated batch, judged against a reference distribution. To keep it affordable, a **subset-replace** estimator scores only the swapped-in generated fraction of a large reference set. Improves FID-50K from 8.30 → 5.77 on SiT and 3.74 → 3.52 on EDM2 while preserving diversity.

**Prereqs:** [_rewards.md](_rewards.md), [ppo.md](ppo.md), [grpo.md](grpo.md)
**Related:** [_rl.md](_rl.md), [dpo.md](dpo.md)

---

## What it is

Per-sample rewards evaluate each generation in isolation: $r(x_i)$, then aggregate. Two failure modes:

1. **Mode collapse.** Every sample optimizes toward the reward-max mode; batch diversity vanishes.
2. **Reward hacking.** The reward model has blind spots; the policy finds them one sample at a time.

Distribution-wise rewards observe that "match the reference distribution" is a *set-level* property. Score a whole batch of generated samples against a reference set with a distributional metric (MMD, FID-like, or a learned discriminator over sets). The gradient then pushes samples toward *filling out* the reference distribution, not toward one point.

## How it works

**Setup.** Keep a large reference set $\mathcal{R}$ of real samples (fixed). At each RL step, generate a batch $\mathcal{G}$ of policy samples.

**Subset-replace estimator.** Rather than recomputing a full distributional distance $D(\mathcal{G}, \mathcal{R})$ from scratch — expensive — replace a small subset of $\mathcal{R}$ with $\mathcal{G}$ to form $\mathcal{R}' = (\mathcal{R} \setminus \mathcal{S}) \cup \mathcal{G}$. Score:

$$
R_{\text{dist}} = -\, D(\mathcal{R}', \mathcal{R})
$$

Because only the swapped fraction changes, $D$ can be updated incrementally at the cost of $|\mathcal{G}|$ evaluations per step, not $|\mathcal{R}|$. Backprop this reward through the standard policy-gradient update (PPO / GRPO family).

**Merging-coefficient RL.** As a bonus, the paper also RL-optimizes the coefficients of a *post-hoc model merge* between a base and fine-tuned checkpoint. The distribution-wise reward is used directly on the merged model's outputs, sidestepping the train/inference mismatch that stochastic-differential-equation (SDE) sampling introduces during standard RL fine-tuning.

## Why it matters

- **Directly targets diversity.** The reward function *sees* the batch's spread against reality; if the batch collapses, the reward collapses. No indirect entropy bonus needed.
- **Real gains on FID.** SiT 8.30 → 5.77 and EDM2 3.74 → 3.52 — meaningful improvements on well-tuned models, without diversity regressions.
- **Composable with existing RL stacks.** The estimator is a swap-in reward function. PPO / GRPO / DPO on top work unchanged.
- **Applies beyond images.** The batch-vs-reference framing works for any generative modality (text, audio, video) where per-sample reward-hacking is a concern.

## Gotchas & tricks

- **Batch size $B$ is a real hyperparameter.** Too small and the distributional score is high-variance; too large and it's expensive and slow to update. Above ~2048 generated samples per step tends to be needed for stable MMD-style estimators.
- **The distributional metric must be estimated correctly.** Naive MMD estimators from small batches are dominated by bias; use unbiased or careful kernel choices.
- **Reference set drift.** If the reference distribution changes (e.g. training data mixture shifts), stale $\mathcal{R}$ silently mis-scores. Refresh periodically.
- **Not a substitute for a good reward *model*.** If the underlying scoring function is bad, distribution matching just optimizes toward that bad distribution more diversely.

## Sources

- Paper: *Optimizing Visual Generative Models via Distribution-wise Rewards*, 2026 — [arXiv:2607.02291](https://arxiv.org/abs/2607.02291).
- Related: *ReFL / DPO for diffusion* — per-sample RL baselines this improves upon.
- Related: *MMD / Sliced-Wasserstein* — the distributional metrics on which the subset-replace estimator is built.
