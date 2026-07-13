# UP (Unbounded Positive Asymmetric Optimization)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A drop-in replacement for the PPO/GRPO **clipped surrogate** that **removes the clip on positive-advantage updates** while keeping it on negative-advantage updates. The claim: symmetric clipping bounds off-policy drift (good) but also suppresses large positive updates on newly-discovered high-reward trajectories (bad — kills exploration). UP restores exploration by letting positive-advantage gradients flow unbounded, and preserves stability by keeping negative-advantage updates pessimistic. Introduced by Fan et al. (ByteDance Seed), 2026 (arXiv 2607.06987), presented as a universal objective across PPO, GRPO, and downstream RL variants.

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md)
**Related:** [rlvr.md](./rlvr.md) · [_rl.md](./_rl.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md)

---

## What it is

A one-line modification to the clipped policy-gradient surrogate that treats positive and negative advantages **asymmetrically**. Standard PPO/GRPO applies the same clip on both sides of the importance ratio, which is *pessimistic* by design — but pessimism on the positive side is exactly what suppresses the model from committing to a newly-found high-reward path. UP argues that unbounded upside + clipped downside is the right shape for LLM RL, where the exploration bottleneck is more pressing than the stability bottleneck.

## How it works

Recall the PPO clip. With importance ratio $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ and advantage $A_t$:

$$
L_{\text{PPO}} = \mathbb{E}\!\left[\, \min(\,r_t A_t,\; \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t\,)\, \right]
$$

The $\min$ formulation caps upside (for $A_t > 0$) *and* caps downside (for $A_t < 0$).

**UP splits by sign of the advantage.** For $A_t > 0$, drop the clip entirely — use the raw $r_t A_t$. For $A_t \le 0$, keep PPO's clipped-and-min form as-is (still pessimistic on the negative side).

Schematically:

$$
L_{\text{UP}} = \mathbb{E}\!\left[\, \mathbb{1}[A_t > 0] \cdot r_t A_t \;+\; \mathbb{1}[A_t \le 0] \cdot \min(r_t A_t,\; \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t)\, \right]
$$

Everything else is unchanged: the KL penalty to $\pi_{\text{ref}}$, the group-relative advantage estimator (in GRPO), the rollout schedule. UP is a one-line loss modification.

The paper positions this as a **universal objective** — applied uniformly across PPO, GRPO, and their reasoning-RL descendants without algorithm-specific tuning.

## Why it matters

- **Explores harder without destabilizing.** LLM RL's central pathology is that the model converges to a narrow behavior mode too fast — the symmetric clip contributes to that by capping how far the policy can move toward a newly-found positive reward. UP frees the positive side while keeping the safety on the negative side.
- **Structurally orthogonal to other clip variants.** Papers like DAPO, VAPO, LOOP, and Dr.GRPO debate *where* to clip. UP debates *which side*. It composes with those choices — you can UP-ify DAPO or VAPO trivially.
- **One-line loss change.** No new hyperparameters beyond what PPO/GRPO already ship. Deployable in any existing RL post-training stack in minutes.

## Gotchas & tricks

- **Exploration ≠ stability.** Removing the positive-side clip can blow up if the reward function is noisy or hackable. UP assumes negative-advantage updates plus the KL penalty are enough to catch it; if your reward RM is fragile, this may not hold.
- **Interacts with rollout freshness.** The whole point of the ratio clip is bounding off-policy drift. With very stale rollouts, unbounded positive updates on a rare good trajectory can push the policy far — favor short rollout-to-update cycles when UP is on.
- **Behavior at very small advantages.** Near $A_t = 0$ the sign-based split creates a small discontinuity in gradient. In practice it's dominated by noise and the paper reports it as harmless; worth watching if you see unusual gradient spikes.
- **Not the same as "removing the clip entirely."** Full unclipped surrogate has been known to be unstable for years — UP's story is specifically that the *asymmetry* is what preserves stability.

## Sources

- Paper: *UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma* — Fan, Liu, Huang, Liu, Lin (ByteDance Seed), 2026 — [arXiv 2607.06987](https://arxiv.org/abs/2607.06987).
- Foundational: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017.
- Foundational: *DeepSeekMath* (introduces GRPO) — Shao et al., 2024.
