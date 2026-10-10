# ReSPO (Reshaped Sequence Policy Optimization)
*Depth — an α-divergence sequence-level objective that replaces PPO's clipped ratio to fix sign-dependent gradient starvation in off-policy RLVR.*

**TL;DR:** When RLVR reuses rollouts across many policy updates, PPO's clipped ratio starves gradients asymmetrically: it suppresses under-generated **positive** responses at the *low*-importance-weight tail while letting over-generated **negative** responses dominate the *high*-weight tail. **ReSPO** replaces the clip with a smooth, two-branch, sequence-level kernel derived from an α-divergence variational objective plus an exponential variance-control tilt — the positive branch keeps nonzero gradient weight on under-generated positives, the negative branch suppresses over-generated negatives. Accelerates early optimisation and improves held-out performance under rollout reuse on Qwen3 dense and MoE models.

**Prereqs:** [ppo.md](ppo.md), [grpo.md](grpo.md), [rlvr.md](rlvr.md)
**Related:** [reasoning/online-policy-mirror-descent.md](reasoning/online-policy-mirror-descent.md) · [_rl.md](_rl.md)

---

## What it is

A drop-in replacement for the clipped surrogate in GRPO-style RLVR. Instead of clipping the importance ratio $r = \pi_\theta / \pi_{\theta_{\text{old}}}$ at $[1-\epsilon, 1+\epsilon]$, ReSPO reshapes the per-sequence gradient weight with a sign-aware, continuous kernel.

The underlying problem is specific to **off-policy rollout reuse**: as training proceeds, $\pi_{\theta_{\text{old}}}$ drifts from $\pi_\theta$, and PPO's clip asymmetrically kills gradient signal on the responses that matter most for learning (new positives that the current policy can't yet produce fluently).

## How it works

Derive the loss from an α-divergence variational objective to an exponential-reward target, then take a *sequence-level* reshaped weight $w(r, A)$:

- **Positive branch** ($A > 0$): $w$ stays bounded away from 0 even when $r$ is small — low-probability positive rollouts keep driving the policy toward them.
- **Negative branch** ($A < 0$): $w$ decays smoothly to 0 for large $r$ — high-probability negative rollouts don't blow up the gradient.

An **exponential variance-control tilt** on the weighted loss keeps gradient variance bounded, replacing PPO's clip-as-variance-reducer role with a principled bound derived from α-divergence stability.

The outer structure (sampling $G$ rollouts per prompt, baseline normalisation, KL to reference) is unchanged from GRPO; only the per-sequence surrogate changes.

## Why it matters

Rollout reuse is the single biggest cost knob in RLVR — reusing rollouts across more policy updates lets you spend compute on longer, more diverse trajectories instead of regeneration. But reuse amplifies off-policy drift, which is exactly where PPO's clip was never designed to work. ReSPO is a principled replacement for the clipped surrogate in that regime, with the two-branch asymmetry addressing the specific failure mode (positive-tail starvation) rather than papering over it.

Likely to become a reference point when designing RLVR pipelines that stretch rollouts across many optimiser steps or across epochs.

## Gotchas & tricks

- The α hyperparameter interpolates between KL-regularised objectives and more aggressive variational forms; defaults in the paper apply to Qwen3, but tune on a smaller run first.
- The variance-control tilt depends on the empirical gradient-norm distribution — recalibrate when switching rollout budget, prompt mix, or reward scale.
- Doesn't substitute for the KL-to-reference term — keep it.
- On near-on-policy data (small rollout reuse), ReSPO and PPO converge; the gain is specifically in the off-policy / long-reuse regime.

## Sources

- Paper: *ReSPO: Reshaped Sequence Policy Optimization for Gradient Starvation in Off-Policy Learning* — Ban, Hsieh et al., UCLA, 2026 — [arXiv:2609.35433](https://arxiv.org/abs/2609.35433).
