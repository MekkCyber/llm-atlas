# Flow-DPPO — RL for flow-matching models

*Depth — divergence-proximal PPO for flow / diffusion image and video generation.*

**TL;DR:** When you RL-tune a flow-matching model, the per-denoising-step policy is **Gaussian** — so you can compute the KL between old and new policies in closed form instead of estimating it from a single-sample ratio like PPO does. Flow-DPPO swaps PPO's ratio clip for an exact-KL trust region and adds an asymmetric mask that only blocks gradients that *simultaneously* exceed the divergence threshold *and* push away from the trusted region. Cleaner trust region, fewer wasted updates, stable multi-epoch training.

**Prereqs:** [ppo](ppo.md), [grpo](grpo.md), [_rl](_rl.md)
**Related:** [_rl-trust-regions](_rl-trust-regions.md) · [token-level-trust-region](token-level-trust-region.md)

---

## What it is

Recent flow-matching image and video generators are RL-tuned by treating the denoising chain as a Markov decision process and applying PPO-style updates (Flow-GRPO, CPS). The problem: PPO's ratio $r_t = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ is a **single-sample noisy estimate** of the true policy divergence. In text PPO this is tolerable because the action space is discrete and we have a clean probability; in flow models, where each step samples from a Gaussian, the ratio is a *high-variance estimate* of an analytically-computable quantity.

Flow-DPPO replaces the noisy estimate with the exact closed-form KL between Gaussian per-step policies, and replaces the hard ratio clip with an *asymmetric divergence mask* that masks out only the bad updates, not all of them.

## How it works

### Closed-form KL per step

For two Gaussian per-step policies $\mathcal{N}(\mu_\text{old}, \Sigma_\text{old})$ and $\mathcal{N}(\mu_\theta, \Sigma_\theta)$:

$$ \mathrm{KL}_t = \frac{1}{2}\left[ \log\frac{|\Sigma_\theta|}{|\Sigma_\text{old}|} - d + \mathrm{tr}(\Sigma_\theta^{-1}\Sigma_\text{old}) + (\mu_\theta - \mu_\text{old})^\top \Sigma_\theta^{-1}(\mu_\theta - \mu_\text{old}) \right] $$

No Monte-Carlo estimate, no variance, computed per denoising step from the network's predicted moments.

### Asymmetric divergence mask

The standard ratio clip kills *all* updates outside the trust region — including useful ones that happen to be moving *into* the trust region. Flow-DPPO's mask blocks the gradient only when both of these hold for a token:
- the KL exceeds the divergence threshold $\tau$, *and*
- the candidate update moves the policy *further* from the trusted region.

Updates that exceed the threshold but move *back toward* the trusted region are allowed through. The result: fewer wasted updates, more efficient use of each rollout.

### The objective

$$ L^\text{Flow-DPPO} = -\mathbb{E}_t\left[ A_t \cdot \mathbb{1}[\text{not masked}_t] \cdot \log \pi_\theta(a_t \mid s_t) \right] - \beta \cdot \mathbb{E}_t[\mathrm{KL}_t] $$

where the mask is the asymmetric one above. Compared to ratio-clip PPO, all three pieces are different: exact KL, asymmetric mask, no ratio at all.

## Why it matters

- **Reflows are now the dominant text-to-image / text-to-video objective.** RL-from-reward is the standard final stage. Getting the trust region right is essential.
- **Fixes a real bug in Flow-GRPO/CPS.** The ratio-clip pathology is structural to flow models, not a tuning issue. Flow-DPPO is the principled fix.
- **Avoids catastrophic forgetting.** The asymmetric mask preserves more in-distribution capability than ratio clip while still optimizing reward.
- **Enables multi-epoch RL.** Ratio-clip schemes degrade after one epoch on the same rollouts; Flow-DPPO trains stably across epochs.

## Gotchas & tricks

- **Threshold $\tau$ is the new $\epsilon$.** Tune carefully — too small starves the update, too large loses the trust-region guarantee.
- **Per-step KL is per *denoising* step**, not per pixel. Don't accumulate naively across the whole chain — that overcounts.
- **Multi-objective rewards balance better.** The paper reports improved balance across simultaneous reward signals (aesthetic + alignment + safety), attributing it to the gentler trust region preserving more capability dimensions.
- **Inherits the PPO problem of off-policy drift past epoch 1**, but with much shallower degradation than ratio-clip.

## Sources

- Paper: *Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models* — Ping et al., XJTU / Tencent Hunyuan / NUS, 2026 — [arXiv 2606.11025](https://arxiv.org/abs/2606.11025).
- Background: *Flow-GRPO* — applies GRPO to flow matching.
- Background: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017.
