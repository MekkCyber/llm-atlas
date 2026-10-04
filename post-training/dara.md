# DARA (Density-Aware Reward Aggregation)
*Depth — a density-adaptive weighting for multi-reward RL that fights sparse-reward crowding.*

**TL;DR:** When RL post-training uses several rewards (verifier correctness + tool-use format + language-consistency + …), frequently-firing rewards dominate the policy gradient and crowd out sparse-but-important ones. **DARA** (Zheng et al., 2026) weights each reward by `1 / √(density + ε)` within every training batch, under-weighting dense signals and amplifying rare ones. On tool-calling and math-reasoning RLVR runs this cuts steps-to-target by 26–65% versus fixed-weight baselines, without hurting final performance.

**Prereqs:** [_rewards.md](./_rewards.md), [_rl.md](./_rl.md), [rlvr.md](./rlvr.md)
**Related:** [grpo.md](./grpo.md) · [rl-prompt-curation.md](./rl-prompt-curation.md)

---

## What it is

Multi-reward RL writes the per-rollout scalar as a weighted sum:

$$
R(q, o) = \sum_{k=1}^{K} w_k \cdot r_k(q, o)
$$

Fixed $w_k$ is standard, hand-tuned per project. The problem: $r_k$ varies in *how often it fires* with a non-zero signal. A format reward is usually 1; a verifier reward is 0 or 1 depending on the problem; a safety reward fires only on unsafe-looking outputs. Over a batch of $G$ rollouts, dense rewards contribute many non-zero gradient terms; sparse ones contribute few.

DARA replaces fixed $w_k$ with an estimate of each reward's *density* in the current batch, then uses an inverse-square-root correction.

## How it works

For each training batch and reward $k$:

1. **Estimate density.** $\rho_k = \frac{1}{|\text{batch}|} \sum_{(q, o)} \mathbb{1}[r_k(q, o) \neq r_k^{\text{neutral}}]$ — the fraction of rollouts where $r_k$ gave a non-neutral signal.
2. **Compute DARA weight.** $w_k^{\text{DARA}} = \frac{w_k^{\text{prior}}}{\sqrt{\rho_k + \varepsilon}}$ with $\varepsilon$ a small positive constant (prevents blow-up when $\rho_k = 0$).
3. **Aggregate.** $R^{\text{DARA}}(q, o) = \sum_k w_k^{\text{DARA}} \cdot r_k(q, o)$.
4. **Normalize per batch** so the aggregated reward's expected scale is stable (otherwise GRPO's $\sigma$-normalization partially does this).
5. **Feed into GRPO / PPO** as the per-rollout reward.

The inverse-square-root (rather than inverse) matches variance-weighting from classical multi-task learning: it protects against overcorrecting when $\rho_k$ is tiny but noisy.

## Why it matters

- **Pairs with GRPO without extra machinery.** DARA is a reward-aggregation preprocessor; the RL optimizer is unchanged.
- **Removes a hand-tuning axis.** Fixed $w_k$ selection has been a dark art; DARA replaces it with an in-training estimator.
- **Especially impactful for agentic RLVR.** Agent tasks have one rare terminal success reward competing with many dense format/compliance rewards — exactly where sparse crowding bites.
- **Reported results:** steps-to-target drop by 26% (compliance) and 65% (reasoning), with competitive final performance.

## Gotchas & tricks

- **$\varepsilon$ selection.** Too small ⇒ sparse rewards explode on first firing; too large ⇒ the correction disappears. Paper tunes empirically; a reasonable default is $\varepsilon = 0.01$.
- **Density needs enough rollouts.** $\rho_k$ is noisy for batches < ~16 rollouts per prompt; combine with GRPO's $G$ large enough to stabilize.
- **Not a reward-hacking fix.** Amplifying sparse rewards also amplifies any adversarial signal buried in them; audit each $r_k$ independently.
- **Interaction with warm-up.** Dense format rewards drive early training; aggressively correcting them from step 0 can slow format convergence. A density correction applied only after warm-up is a safe default.
- **Not for anti-correlated rewards.** If two rewards oppose each other, DARA still weights them by density independently; use a learned router (e.g., Adaptive Reward Routing in audio-video diffusion) instead.

## Sources

- Paper: *Make Sparse Rewards Count: Density-Aware Reward Aggregation for Multi-Reward RL* — Zheng et al., 2026 — [arXiv:2610.00574](https://arxiv.org/abs/2610.00574).
- Related: multi-task learning variance-balancing literature (GradNorm, Kendall et al. "Multi-Task Learning Using Uncertainty"), which the inverse-sqrt scaling echoes.
- Related: *Adaptive Reward Routing for Joint Audio-Video Diffusion* — Yang et al., 2026 ([arXiv:2609.37200](https://arxiv.org/abs/2609.37200)) — complementary learned-router approach to multi-reward RL.
