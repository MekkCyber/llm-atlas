# RIPO — Riemannian Isometric Policy Optimization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A drop-in replacement for PPO-Clip (and GRPO's clip) that fixes a geometric mismatch: standard clipping treats the policy as a Euclidean vector when it actually lives on a Riemannian manifold. The Euclidean clip overconstrains updates in low-probability regions and overshoots in high-probability regions, collapsing exploration. RIPO enforces an **intrinsic** bound on each update instead — up to 60% AIME24 improvement over GRPO in the source paper.

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [rlvr.md](./rlvr.md), [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md), [reasoning/online-policy-mirror-descent.md](./reasoning/online-policy-mirror-descent.md)

---

## What it is

PPO-Clip caps the policy-ratio `π_θ / π_θ_old` inside `[1-ε, 1+ε]` per token. Since the ratio is a coordinate-wise measure of change, the clip is implicitly **Euclidean on probability vectors**. But a policy is a distribution — the natural distance is KL, and the geometry is Riemannian with the Fisher metric.

Concretely: a rare token at `p = 0.01` and a common one at `p = 0.5` get the same additive clip window, so the rare token can barely grow before being clipped, while the common one can absorb large multiplicative moves. Exploration collapses because low-probability actions never get room to become high-probability.

RIPO replaces the Euclidean clip with an **isometric-on-the-manifold** update rule: each step moves the policy by a bounded *intrinsic* distance (Fisher / KL), independent of where on the simplex it starts. Low-probability tokens now get room proportional to their intrinsic distance, not their absolute size.

## How it works

The surrogate rewrites the PPO ratio in a coordinate system where equal Euclidean radii correspond to equal Fisher radii. In practice, this looks like:

- **Log-space clipping** on the token log-probability change, bounded symmetrically around 0.
- A **manifold-consistent trust region** so the effective step size is invariant to the base probability.
- Same GRPO frontend as before — sample G rollouts per prompt, use the group's mean/std as the advantage baseline — only the clip surrogate changes.

The bias-variance analysis in the paper argues RIPO gives a strictly better trade-off than the Euclidean clip: the update magnitude no longer depends on the arbitrary scaling of the probability coordinates.

## Why it matters

GRPO carries PPO-Clip verbatim; every reasoning-RL pipeline built on GRPO inherits the exploration-collapse failure mode. If the RIPO fix holds at scale, it becomes a mandatory swap for RLVR runs on hard-exploration benchmarks (math contests, code with sparse verifiers).

## Gotchas & tricks

- The paper reports improvements on competition math (AIME24) where exploration matters most; gains may be smaller on easier / denser-reward benchmarks.
- Implementation is a one-file change to the GRPO loss — no changes to sampling, verifier, or reference model needed.
- Watch for KL blow-up early in training: RIPO's larger effective step in low-probability regions can push the reference-KL penalty above its usual regime.

## Sources

- Paper: *Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization* — Guo et al., 2026 — [arXiv:2607.10169](https://arxiv.org/abs/2607.10169)
- Prior art: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017 — the PPO baseline RIPO critiques.
