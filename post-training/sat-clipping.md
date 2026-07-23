# SAT (Staleness-Adaptive Trust Region) clipping
*Depth — a PPO clip variant that tightens only on stale, high-mismatch tokens.*

**TL;DR:** In asynchronous LLM RL, rollouts come from a policy several optimizer steps behind the trainer. PPO's clip is a per-sample surrogate, not a full-policy constraint, so high-staleness updates are weakly controlled. SAT uses the sampled log-ratio as a staleness proxy, identifies high-mismatch tails inside each batch, and **contracts only the sign-selected endpoint of the PPO interval** on those tokens — leaving ordinary tokens alone. Enables near-flat AIME24 accuracy from lag 1 to lag 8 on Qwen3-30B-A3B.

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md)
**Related:** [../systems/partial-rollouts.md](../systems/partial-rollouts.md), [rlvr.md](./rlvr.md)

---

## What it is

Asynchronous RL for LLMs decouples rollout generation (on serving stacks like SGLang) from optimizer updates (on trainers like Megatron). Rollouts arrive **stale**: sampled from a policy version behind the current trainer weights. PPO clipping is defined per-sample and gates outward updates on the sample's own ratio — it doesn't measure or bound the true policy divergence.

SAT is a targeted fix: keep PPO's clip on ordinary tokens, but tighten it *only* where staleness is measurably hurting.

## How it works

For each token in a batch:

1. **Staleness proxy.** The (detached) sampled log-ratio between the current policy and the behavior policy is used as a per-token proxy for how stale that token's sample is.
2. **Tail identification.** A kernel over log-ratios flags a high-mismatch tail — the tokens whose staleness stands out within the batch.
3. **Asymmetric contraction.** On tail tokens, SAT contracts *only the outward endpoint* of the nominal PPO interval — the endpoint corresponding to the sign of the advantage. Inward moves are untouched. Non-tail tokens keep the vanilla PPO clip.

The paper proves two properties: **local interval containment** (SAT's clip is always inside PPO's) and **pointwise pessimism** (per-token, SAT's update magnitude is bounded above by PPO's).

## Why it matters

- **Async RL is how you scale LLM post-training economically.** Fixing its dominant failure mode without slowing the trainer is a big deal.
- **Global tricks (smaller LR, larger clip, replay) trade off convergence.** SAT is local, so it doesn't degrade well-behaved tokens.
- **Composes with routing replay.** In MoE, staleness and expert-routing inconsistency are two independent instabilities; SAT targets the first, routing replay the second.

Result: on Qwen3-30B-A3B-Base, SAT-GSPO w/ R3 hits AIME24 avg@8 = **35.83 at lag 1** and **34.79 at lag 8** — near-flat, where vanilla clipped RL collapses at moderate lag.

## Gotchas & tricks

- Staleness proxy is on the *sampled* log-ratio, not the full-policy KL — cheap to compute but noisier per token; the kernel scaling absorbs that.
- Asymmetric clipping means SAT contracts the endpoint pointing "in the direction of the advantage." Getting the sign wrong inverts the whole method.
- The gains show up specifically at high lag; at lag 1 SAT looks like a small win over baselines.

## Sources

- Paper: *Stale but Stable: Staleness-Adaptive Trust Regions for Stabilizing Asynchronous Reinforcement Learning* — Yang et al., 2026 — [arXiv:2607.18722](https://arxiv.org/abs/2607.18722)
