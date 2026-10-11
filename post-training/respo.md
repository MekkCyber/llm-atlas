# ReSPO
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A policy-gradient algorithm for Reinforcement Learning with Verifiable Rewards (RLVR) that replaces the PPO-style **clip** on the sequence importance-sampling (IS) weight with a smooth, sign-dependent *kernel*. The reshape is designed so that gradients survive on *under-generated positive* responses (where clipping would zero them out) and are attenuated on *over-generated negatives* (where clipping lets them dominate). On Qwen3 dense and MoE backbones, ReSPO reports faster early learning and higher late-training scores than GRPO / GSPO under rollout reuse. Introduced in 2026.

**Prereqs:** [grpo.md](grpo.md), [ppo.md](ppo.md), [_rl.md](_rl.md), [rlvr.md](rlvr.md)
**Related:** [../systems/partial-rollouts.md](../systems/partial-rollouts.md) · [_rewards.md](_rewards.md)

---

## What it is

Rollout reuse is the dominant lever for cheap RL post-training: generate $K$ rollouts per prompt once, then take several policy-gradient steps on them before refreshing. Each additional step makes the data more off-policy, which — under clipped surrogates like GRPO or GSPO — causes two specific failure modes:

- **Gradient starvation on positives.** A positive response whose sequence IS weight $r = \pi_\theta/\pi_{\theta_\text{old}}$ fell below $1-\epsilon$ gets clipped to zero. The policy stops being rewarded for a response it is undergenerating relative to its own training.
- **Over-amplification on negatives.** A negative response whose $r$ grew well above $1+\epsilon$ keeps dominating the update when it is clipped only on the positive side.

ReSPO replaces the clip with a kernel $f(r)$ that is non-zero on both sides but shaped so the sign-conditional behavior above is reversed.

## How it works

- **Sequence IS weight.** As in GSPO, use the whole-sequence IS weight $r_i = \pi_\theta(o_i \mid q) / \pi_{\theta_\text{old}}(o_i \mid q)$ rather than per-token ratios.
- **Reshape, don't clip.** The surrogate objective uses a smooth kernel $f(r)$ in place of $\mathrm{clip}(r, 1-\epsilon, 1+\epsilon)$. The kernel is sign-dependent on the advantage $A_i$:
  - For $A_i > 0$: $f(r)$ keeps weight on $r < 1$ (so under-generated positives still receive gradient).
  - For $A_i < 0$: $f(r)$ bounds weight on $r > 1$ (so over-generated negatives do not dominate).
- **Everything else as GRPO.** KL to the reference model, group-relative advantage, same reward sources (rule verifiers, mixed rewards).

## Why it matters

- Fixes the specific pathology that limits rollout-reuse economics with GRPO/GSPO; the authors show the limit is lower than it needs to be.
- Keeps the attractive properties of sequence-level RL (one scalar per rollout, no critic) while recovering gradient signal on the examples that matter most for learning.
- Reported stability and headroom gains on Qwen3 dense and MoE suggest the reshape is not architecture-specific.

## Gotchas & tricks

- The kernel is a hyperparameter: the paper proposes one shape; others are compatible with the same reasoning and may fit different reward distributions.
- Rollout reuse past some horizon still causes drift; ReSPO lifts the ceiling, it does not remove it. Expect to still refresh rollouts periodically.
- Mixing ReSPO with learned reward models rather than rule verifiers is untested in the paper; preference-RM gradients have different statistics than binary verifier gradients.

## Sources

- Paper: *ReSPO: Reshaped Sequence Policy Optimization for Gradient Starvation in Off-Policy Learning* — Ban, Hsieh et al., UCLA, 2026 — [arXiv:2609.35433](https://arxiv.org/abs/2609.35433).
