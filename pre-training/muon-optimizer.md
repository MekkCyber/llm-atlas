# Muon Optimizer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Optimizer for matrix-valued parameters (attention / MLP weights) that applies **Newton-Schulz orthogonalization** to the momentum update before taking the step. The step direction has equalized singular values rather than a wildly skewed spectrum — empirically faster convergence than AdamW on large LMs and, as recent work shows, more robust in **long-horizon RL** where credit assignment produces highly asymmetric gradients.

**Prereqs:** [_training-stability.md](./_training-stability.md), [_lr-schedules.md](./_lr-schedules.md)
**Related:** [fp8-training.md](./fp8-training.md) · [wsd-schedule.md](./wsd-schedule.md) · [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

AdamW scales the gradient element-wise by a running second-moment estimate — it does not account for the matrix structure of a weight tensor. For matrix-valued parameters, element-wise scaling can leave the update direction's singular spectrum very skewed: a single singular direction dominates the step, and the effective update rank is small.

Muon treats each 2D weight tensor as a matrix and **orthogonalizes** its momentum update, giving a step whose singular values are all equal. The intuition is a Newton-step-like preconditioner on the Fisher of a matrix parameter.

## How it works

Per matrix parameter $W \in \mathbb{R}^{m \times n}$:

1. Maintain momentum $M \leftarrow \mu M + G$, where $G$ is the gradient.
2. **Orthogonalize** $M$ to produce $O = \mathrm{orth}(M)$ — a matrix with the same column span but all singular values equal to 1. Computed cheaply via a **Newton-Schulz iteration** (typically 5 iterations of a cubic polynomial on $M$, no SVD).
3. Update: $W \leftarrow W - \eta \cdot O$.

Non-matrix parameters (biases, norms, embeddings) stay on AdamW. The hybrid optimizer is sometimes called Muon-AdamW.

The orthogonalization makes the step **spectral-norm-bounded**: $\|O\|_2 = 1$. Learning rate directly controls the maximum singular value of the update rather than its Frobenius norm.

## Why it matters

- **Faster pretraining.** Reported wall-clock and token-efficiency wins on 1B–7B+ language models vs tuned AdamW baselines.
- **Robust in long-horizon RL.** Agent RL at 100+-step horizons produces gradients where a few steps carry most of the signal — a highly non-uniform singular spectrum. Muon's spectral equalization keeps the update from being dominated by a single direction. *Long-Horizon RL Needs Muon* (2026) shows the combination of Muon + token-averaged loss + truncated IS + length/tool penalties trains successfully where AdamW stacks collapse.
- **No tuned per-matrix learning rates.** Spectral-norm bound makes LR behave consistently across differently-sized matrices.
- **Cheap.** Newton-Schulz is 5 small matmuls per step per matrix — fraction of a percent overhead.

## Gotchas & tricks

- **Matrix-only.** 1D parameters (biases, LN weights) must stay on another optimizer; mixing is standard practice.
- **Newton-Schulz coefficients matter.** The 5-iteration cubic variant is a specific choice; off-the-shelf polynomial coefficients from the Muon paper work for well-conditioned $M$.
- **Momentum coefficient.** Typical $\mu = 0.95$, similar to Adam but the orthogonalization makes the effective behavior different; Muon tolerates higher $\mu$.
- **Learning rate scale.** Because the step is spectral-norm-bounded, effective LR is roughly the Frobenius norm of the update divided by $\sqrt{\min(m, n)}$ — a sanity-check conversion is useful when porting from AdamW.
- **Weight decay.** Decoupled WD still applies; typical values match AdamW.
- **Interaction with FP8.** Orthogonalization in low precision needs care — numerical conditioning of $M$ in FP8 can introduce drift. Recent Muon-FP8 recipes use BF16 for the Newton-Schulz step and FP8 for everything else.
- **Not a silver bullet for post-training.** Short-horizon RL (reasoning, single-turn tasks) sees smaller gains from Muon than long-horizon agent RL — the gradient skew the method targets is less present.

## Sources

- Paper: *Long-Horizon RL Needs Muon* — Ruan, Lin, Chen, Zhou, Wei, Huang, 2026 — [arXiv:2610.20256](https://arxiv.org/abs/2610.20256). Combines Muon with token-averaged loss, truncated IS, and length/tool-error penalties on SWE-style benchmarks.
- Paper: *Muon: Momentum Orthogonalized by Newton-Schulz* — Keller Jordan et al., 2024 — the original optimizer.
