# Muon Optimizer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** **Muon** is a matrix-aware optimizer for training deep networks. Unlike AdamW (element-wise adaptive step sizes on flat parameter vectors), Muon operates on *matrix-shaped* parameters (attention W_Q/W_K/W_V/W_O, FFN weights) and uses a Newton–Schulz-based orthogonalization step to shape updates in matrix-geometry-aware ways. It's been competitive with AdamW at pretraining scale, and recent work (Muon-agentic-RL, 2026) shows it can transfer to RL post-training under the right advantage-estimator and learning-rate combination.

**Prereqs:** [_lr-schedules.md](_lr-schedules.md), [_training-stability.md](_training-stability.md)
**Related:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

Two conceptual moves distinguish Muon from AdamW:

1. **Treat matrix-shaped parameters as matrices.** AdamW's per-parameter EMAs of gradient and squared gradient ignore parameter shape — a $d \times d$ weight matrix is treated as $d^2$ independent scalars. Muon keeps matrix structure and applies matrix-level updates.
2. **Orthogonalize the update via Newton–Schulz.** Given the momentum-smoothed gradient matrix $G$, Muon replaces the raw update with $\text{orthogonalize}(G)$ — the closest orthogonal matrix, computed cheaply via a few Newton–Schulz iterations. The orthogonal update has bounded spectral norm, which controls update magnitude in a matrix-natural way.

Applied to matrix parameters only. Scalars, biases, LayerNorm parameters, and embeddings typically stay on AdamW.

## How it works

The Muon step (simplified):

```
for each matrix-shaped param W in model:
    G  = grad(W)                      # gradient
    M  = beta * M_prev + (1-beta)*G   # momentum EMA
    O  = orthogonalize(M)             # Newton–Schulz, ~5 iterations
    W -= lr * O
```

Orthogonalization via Newton–Schulz uses only matrix products (no SVD), making it practical inside a training step. The update $O$ has all singular values equal to 1 — a maximally isotropic step in matrix space.

Muon's insight over AdamW: the *shape* of the update matters, not just its element-wise magnitudes. An orthogonal update explores parameter space in a way that doesn't collapse into a few dominant directions — a common failure mode of AdamW at very large scale.

Practical wins reported at pretraining scale include matching AdamW's final loss with less wall-clock (Muon's per-step cost is higher but step count needed is lower) and better training stability on large models.

## Why it matters

- **Structurally different from AdamW.** Not a hyperparameter tweak — a different update geometry. Gains and losses relative to AdamW are shape-driven.
- **Transfers to post-training RL, conditionally.** Muon-agentic-RL (2026) reports GiGPO + Muon (hidden weights only) on ALFWorld / Qwen2.5-0.5B lifting final-window validation success from 0.290 → 0.546 (+88%); AdamW controls at matched high LR retain no post-update success. Effect strength depends on advantage estimator (GRPO / GiGPO / GraphGPO) and LR — not a universal win.
- **Composable with existing schedules.** Works with cosine, WSD, or custom LR schedules — the optimizer is orthogonal to schedule choice.

## Gotchas & tricks

- **Matrix params only.** Applying Muon to biases / scalars / embeddings breaks it. Standard practice: Muon on 2D+ params, AdamW on the rest.
- **Newton–Schulz iteration count is a tradeoff.** 5 iterations is common; fewer is faster but noisier, more is diminishing returns.
- **LR is not directly comparable to AdamW's.** Because the update is orthogonal-normalized, the effective step size vs. AdamW is different — expect to re-tune LR by 3–10×.
- **RL sensitivity.** Under RL post-training, Muon's benefit depends on the advantage estimator and LR in a non-obvious way. Muon-agentic-RL reports pockets where Muon dominates and others where AdamW is comparable — do the sweep, don't assume.
- **Multi-seed validation is thin so far.** Most Muon-RL numbers are single-seed exploratory; treat them as directional until multi-seed replication lands.
- **Not yet the default at scale.** As of 2026 Muon is the leading challenger to AdamW at pretraining scale but not the industry default — worth trying on new runs, not worth ripping out AdamW pipelines that work.

## Sources

- Paper: *Muon: An Efficient Optimizer for Neural Networks* — the original Muon proposal (Jordan et al., 2024).
- Paper: *When Does Muon Help Agentic Reinforcement Learning?* — 2026 — [arXiv:2607.16169](https://arxiv.org/abs/2607.16169) — RL post-training extension.
- Background: *Adam: A Method for Stochastic Optimization* — Kingma & Ba, 2015 — the baseline Muon is compared against.
