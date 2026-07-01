# Muon
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pretraining optimiser that replaces AdamW's per-parameter second-moment adaptation with an **orthogonalised** update over each weight matrix, computed via a Newton–Schulz iteration on the momentum. Empirically matches or beats AdamW on LLM pretraining at competitive wall-clock, and — importantly for systems — is robust to **stale gradients**, unlike AdamW. Increasingly the default choice when the training schedule is asynchronous or when scaling laws demand a tighter optimiser floor.

**Prereqs:** [_training-stability.md](./_training-stability.md), [README.md](./README.md)
**Related:** [../systems/async-pipeline-parallel.md](../systems/async-pipeline-parallel.md), [_lr-schedules.md](./_lr-schedules.md)

---

## What it is

AdamW is the modern default: per-parameter moving averages of first and second gradient moments, applied element-wise. It's simple and it scales, but it accumulates second-moment state that assumes gradients are a smooth function of *current* parameters. Under noise, staleness, or ill-conditioned weight matrices, that assumption breaks.

Muon reframes the update as a **matrix-level** operation: for each 2-D weight matrix, take the momentum-smoothed gradient and *orthogonalise* it before applying. The result is invariant to per-column scaling and less brittle to noise in individual coordinates.

## How it works

**Momentum on the raw gradient.**
```
M_t = β · M_{t-1} + (1 − β) · G_t
```

**Newton–Schulz orthogonalisation.** For each weight matrix, compute an approximate polar factor of `M_t`:
```
U = NewtonSchulz(M_t, iterations=k)   # k ≈ 5
```
`U` is close to the orthogonal projection of `M_t` (essentially `M_t (M_t^T M_t)^{-1/2}`) but computed with only matmuls, no explicit SVD. Cost: `O(k · d^3)` per weight matrix, small relative to the forward/backward.

**Apply the update.**
```
θ_t = θ_{t-1} − η · U
```
No per-parameter second-moment estimate. Learning rate `η` is set once per layer/tensor group.

**Bias and 1-D parameters.** Muon is only applied to 2-D weights; biases, embeddings, and layer-norm gains use AdamW as a fallback.

**Why it's robust to staleness.** AdamW's failure under gradient delay is dominated by the second-moment estimate drifting off the current parameter neighbourhood. Muon has no such state — orthogonalising the momentum is a *local* operation that doesn't compound past mistakes about the loss surface's coordinate scaling.

## Why it matters

- **Reopens asynchronous training schedules.** The [async-pipeline-parallel](../systems/async-pipeline-parallel.md) result shows PipeDream-2BW works if you swap AdamW → Muon, previously written off because of AdamW.
- **Matches or slightly beats AdamW on pretraining wall-clock** at competitive step count in the 1B–10B regime.
- **Cheaper state.** No second-moment tensor; memory savings comparable to Lion or Sophia families.

## Gotchas & tricks

- **Newton–Schulz iteration count is load-bearing.** Too few and orthogonalisation is inexact (unstable updates); too many and the compute tax eats the wall-clock win. `k = 5` is the community default at pretraining scale.
- **Learning-rate schedule is *not* a drop-in from AdamW.** Peak LR and warmup shape differ; expect a re-tune.
- **1-D parameters still need an adaptive optimiser.** Applying Muon uniformly to biases/embeddings degrades quality — hybrid setup with AdamW on those is standard.
- **Not a preference-optimisation replacement.** Muon is a pretraining optimiser; RL post-training pipelines (GRPO etc.) usually stick with AdamW because the gradient shape differs.
- **Empirical, not fully theoretical.** Convergence guarantees are strongest in narrow settings (e.g. under one-step delay in the async-pipeline-parallel paper); broader theory is active work.

## Sources

- Related: *Muon* pretraining optimiser — Jordan et al. (community effort, 2024–2025). Original repo and writeup circulated as blog posts before formal writeups.
- Paper: *One-Step Gradient Delay Is Not a Barrier for Large-Scale Asynchronous Pipeline Parallel LLM Pretraining* — Zmushko et al., 2026 — [arXiv:2606.30634](https://arxiv.org/abs/2606.30634) — the async-robustness analysis that motivated the depth file being extracted here.
