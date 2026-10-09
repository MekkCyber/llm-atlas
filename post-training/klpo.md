# KLPO: KL-Regularised Policy Optimisation

*Depth — a critic-free, importance-weight-free RL algorithm for asynchronous LLM post-training that anchors the KL term at the sampler.*

**TL;DR:** PPO/GRPO regularise the current policy toward a *frozen reference* model; the KL penalty is between $\pi_\theta$ and $\pi_{\text{ref}}$. KLPO anchors the KL at the **sampler** — whatever actually produced the on-trajectory rollouts, which under async RL is a slightly stale version of $\pi_\theta$. Fit a log-ratio optimality condition by least squares on the sampler's own trajectories; no importance weights, no critic, one rollout per prompt. SPPO, GPO, REBEL, and BPO fall out as special cases.

**Prereqs:** [_rl](_rl.md), [ppo](ppo.md), [grpo](grpo.md)
**Related:** [rlvr](rlvr.md) · [dpo](dpo.md) · [online-policy-mirror-descent](reasoning/online-policy-mirror-descent.md)

---

## What it is

In PPO:

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}\!\left[ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t \right] - \beta \cdot D_{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

The KL is between the *current* policy and the *frozen reference*. KLPO replaces this with a KL between the current policy and the **sampler** $\pi_{\text{samp}}$ — the specific checkpoint whose rollouts are in the batch. Under synchronous RL $\pi_{\text{samp}} = \pi_{\theta_{\text{old}}}$ and the two formulations agree; under asynchronous RL they diverge.

KLPO's objective can be written as the solution to an optimality condition on the log-ratio $\log(\pi_\theta / \pi_{\text{samp}})$, which the paper fits with ordinary least squares on sampler trajectories.

## How it works

1. Collect one rollout per prompt from a slightly stale sampler $\pi_{\text{samp}}$.
2. For each token, construct a target log-ratio from the reward signal and the KL regulariser.
3. Fit $\log \pi_\theta - \log \pi_{\text{samp}}$ to this target by least squares.
4. Update $\theta$; the sampler lags by a few steps.

Three structural consequences:

- **No critic.** The target log-ratio replaces the value function.
- **No importance weights.** The KL anchor absorbs the staleness that importance ratios would otherwise correct.
- **One rollout per prompt.** No group-relative baseline needed; the regulariser is per-token, not per-group.

### Why it unifies prior methods

- Setting the regulariser's shape recovers **SPPO** (self-play PPO).
- Changing the fit to KL-divergence instead of log-ratio recovers **GPO**.
- Dropping the regulariser recovers **REBEL**.
- Setting the reward to be preference-based recovers **BPO**.

## Why it matters

Async rollout is now the default at frontier scale (DeepSeek, Qwen, open-RLHF stacks). The sampler is never actually $\pi_\theta$; the gap grows with pipeline depth. PPO's frozen-reference KL ignores this and PPO's importance ratios try to compensate — brittle in practice. KLPO's sampler-anchored KL is the cleaner formulation and matches multi-rollout baselines with a single rollout per prompt.

## Gotchas & tricks

- **Staleness ceiling.** The sampler can only lag so far; the paper suggests capping the gap at a few policy updates.
- **Regulariser weight $\beta$.** Treats like PPO's: too low and the policy drifts; too high and learning stalls. The sampler anchor makes the sensitivity sharper.
- **Reward shape.** KLPO treats the reward as a per-token or per-trajectory scalar — doesn't automatically give you dense signal. Pair with [self-retrospection-distillation](self-retrospection-distillation.md) for dense gradients when needed.
- **Not a free lunch for synchronous RL.** When $\pi_{\text{samp}} = \pi_{\theta_{\text{old}}}$ exactly, KLPO ≈ PPO without the clip. The win is specifically in async.

## Sources

- *On KL-Regularized Policy Optimization* — Yifan Zhang — 2026 — [arXiv:2610.08963](https://arxiv.org/abs/2610.08963)
