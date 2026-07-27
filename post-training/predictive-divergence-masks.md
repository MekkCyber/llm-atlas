# Predictive Divergence Masks (PDM)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For long-CoT RL, most tokens in a rollout are boilerplate — connective phrasing, restatements, chain-of-thought filler where the policy barely differs from its reference. Computing gradients on every token is wasteful. **Predictive Divergence Masks** identify the tokens where current policy and reference policy actually diverge, and apply the RL update *only there*. Cuts per-step compute without hurting reasoning-benchmark accuracy.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md), [long-cot-rl.md](./reasoning/long-cot-rl.md).
**Related:** [ppo.md](./ppo.md) · [rlvr.md](./rlvr.md) · [_rewards.md](./_rewards.md)

---

## What it is

A per-token gradient mask applied to a GRPO/PPO-style RL loss. For a rollout token $x_t$ with current policy $\pi_\theta$ and reference $\pi_{\text{ref}}$, the mask is

$$
m_t = \mathbb{1}\big[ \text{KL}\big(\pi_\theta(\cdot\mid x_{<t})\,\|\,\pi_{\text{ref}}(\cdot\mid x_{<t})\big) > \delta \big]
$$

and only tokens with $m_t = 1$ contribute to the policy gradient. Low-divergence tokens (where the policy hasn't yet moved off the reference) are excluded — they carry near-zero information for the update anyway.

## How it works

Per RL step:

1. Roll out under current policy, log per-token log-probs.
2. Score per-token KL vs reference (already computed for the KL penalty term).
3. Threshold at $\delta$ to build the mask $m$.
4. Apply the standard GRPO objective **only over masked tokens**; skip backward for the rest.

Skipped tokens still cost the forward pass (the KL score is computed there) but avoid the backward pass — the dominant cost in long-CoT training where sequence lengths run into tens of thousands.

## Why it matters

Long-CoT RL is bottlenecked by rollout length: a single 30k-token trace with a full backward pass swamps GPU memory and step time. Selective-token masking is the RL analogue of sparse attention — it targets compute at the tokens that actually matter for the policy update. And unlike heuristic length penalties, PDM's mask is *policy-derived*: it adapts as the policy drifts, focusing on wherever the current update is doing real work.

## Gotchas & tricks

- **Threshold $\delta$ is critical.** Too high → most tokens masked out, updates get sparse and unstable. Too low → no compute savings. Paper reports the tradeoff curve; a threshold that keeps ~10-30% of tokens is a reasonable starting point.
- **Reference drift.** If the reference model is frozen (typical), the mask concentrates on tokens where the policy has drifted farthest. If the reference is periodically refreshed (à la some PPO variants), mask semantics change.
- **Compose with clipping and KL penalty.** PDM is orthogonal to PPO's clip ratio and to the KL penalty on advantages — it acts on which tokens contribute to *both* the clip and the KL, but at the mask level, not by rescaling.
- **Not the same as length penalty.** Length penalty biases toward shorter outputs; PDM biases *compute* toward informative tokens without changing what outputs are preferred.

## Sources

- Paper: *Predictive Divergence Masks for LLM RL* — Zhou, Yao, Qi, Ping, Tang, Wang, Pang, 2026 — [arXiv:2607.10848](https://arxiv.org/abs/2607.10848)
