# LoGRA — Low-Rank Gradient Sketches for LLM RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RL post-training (PPO/GRPO) is memory-bound: optimizer state and gradients for a 27B policy don't fit on one 8-GPU node under dense Adam. LoGRA keeps the useful learning signal in **low-rank gradient sketches** and pairs them with a **predicted-KL step controller** that estimates the policy change *before* applying an update and shrinks the step if it would be too large. Result: 45.7% average training-memory reduction and stable 27B RL on 8 GPUs for 1,100+ steps.

**Prereqs:** [ppo.md](ppo.md), [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [rlvr.md](rlvr.md) · [rejection-sampling.md](rejection-sampling.md) · [../pre-training/_training-stability.md](../pre-training/_training-stability.md)

---

## What it is

Two independent memory- and stability-savers for RL post-training, meant to be used together:

1. **Low-rank gradient sketches** — store accumulated gradient information in a factored, low-rank representation during the policy update, and use the same compact form for policy synchronization across data-parallel workers.
2. **Predicted-KL step control** — before committing an update, cheaply estimate the KL divergence between the pre- and post-update policies; if it exceeds a target, scale the step down.

## How it works

- **Sketching.** Each gradient tensor is projected into a rank-$r$ sketch via a fixed random projection (or an online SVD-maintained basis). The sketch replaces the full gradient everywhere it's accumulated, broadcast, or applied, cutting optimizer-state memory roughly $d/r$-fold.
- **Policy sync via sketches.** When RL workers synchronize updated policies, they exchange sketches instead of full parameter deltas, saving all-reduce traffic.
- **KL-step controller.** For each candidate update $\Delta\theta$, approximate $\mathrm{KL}(\pi_{\theta+\Delta\theta} \,\|\, \pi_\theta)$ using a cheap second-order estimate on a mini-batch, then rescale $\Delta\theta$ so the predicted KL hits a target. Analogous in spirit to natural-gradient / trust-region methods, but applied *per update* without the full Fisher.

The two pieces compose: sketches reduce the memory needed to hold the update; the controller keeps the (possibly noisier) sketched updates from blowing up policy.

## Why it matters

- Enables **27B RL on 1 node** where dense Adam OOMs. For labs without datacenter-scale budgets, this is the difference between running the experiment and not.
- Memory savings (**-45.7% average**) come **without quality loss** on reasoning benchmarks.
- Composes with any PPO/GRPO backbone — it's a drop-in optimizer+step-size pair, not a new RL algorithm.
- Open-source via the Molt library, which lowers reproduction cost.

## Gotchas & tricks

- **Sketch rank is a quality knob.** Too low → signal loss in the gradient; too high → memory savings vanish. Paper uses rank schedules; defaults are per-layer.
- **KL target replaces step size.** Treat the KL target as the real hyperparameter and let step magnitudes vary. This is close to how trust-region methods tune.
- **Policy-sync savings only materialize at multi-worker scale.** Single-worker RL only sees the optimizer-state gain.
- **Early training is noisier.** Sketched updates have more variance early when gradients haven't aligned to a useful subspace. Expect the first few hundred steps to look slower; convergence catches up.

## Sources

- Paper: *LoGRA: Scaling LLM Reinforcement Learning with Low-Rank Gradient Sketches* — 2026 — [arXiv:2610.06647](https://arxiv.org/abs/2610.06647).
- Code: Molt library (per paper).
