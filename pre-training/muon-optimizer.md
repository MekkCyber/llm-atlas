# Muon Optimizer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An optimizer that replaces AdamW on 2D weight matrices with an update step defined by the **matrix sign** of the momentum buffer (approximated via a few Newton–Schulz iterations of `M @ (M M^T)^{-1/2}`). This orthogonalizes the update direction, keeping singular values of the effective update near 1 and matching Maximal Update Parameterization (µP) width-transfer behavior for free. Adopted by Kimi (Moonshot), used in the DeepSeek family of recipes, and now the standard optimizer paired with µP for large MoE runs.

**Prereqs:** [_lr-schedules.md](./_lr-schedules.md), [_training-stability.md](./_training-stability.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md) · [mup.md](./mup.md) · [../case-studies/kimi-k1-5.md](../case-studies/kimi-k1-5.md)

---

## What it is

Muon is a matrix-aware optimizer for the 2D parameters of a transformer (linear-layer weights). For scalars, biases, embeddings, and gains, it falls back to AdamW. On 2D weights it applies an update whose direction is the **matrix sign** of the gradient momentum — i.e. the SVD of the momentum with its singular values replaced by 1.

## How it works

Per step, for each 2D weight `W`:

```
G_t = ∇L(W_t)                    # gradient
M_t = μ · M_{t-1} + G_t          # momentum (typical μ = 0.95)
O_t = msign(M_t)                 # orthogonalize
W_{t+1} = W_t − η · O_t          # apply LR
```

The orthogonalization `msign(M) = U V^T` where `M = U Σ V^T` is not computed via SVD. Muon uses **five Newton–Schulz iterations** on `M_t` with a fixed cubic polynomial `p(x) = a x + b x³ + c x⁵`, chosen so `p(x) → sign(x)` on `[-1, 1]`. Cost is a handful of matmuls, no eigendecomposition. Momentum is stored in full precision (typically BF16); no second moment is needed.

Auxiliary parameters (biases, RMSNorm gains, embedding tables) stay on AdamW.

## Why it matters

- **Compute-per-token efficiency.** Achieves lower loss than AdamW at matched tokens across LLM pretraining benchmarks; equivalently, hits AdamW loss with 30–40% fewer tokens in the Kimi report.
- **Width-transfer for free.** Because updates are unit-singular-value, per-block update magnitude scales predictably with width — a Muon-parameterized model transfers optimal LR across widths in the same way a µP-parameterized AdamW model does, without needing a separate width-dependent LR scale.
- **No second moment.** Halves optimizer-state memory vs AdamW (no `v_t`), which matters for MoE where expert weights dominate parameter count.

## Gotchas & tricks

- **Weight decay.** Standard Muon adds decoupled weight decay on `W` alongside the orthogonalized update; without it, weight norms grow because the update is scale-free.
- **LR magnitudes differ from AdamW.** Muon optimal LR is roughly `1e-2` to `3e-3` — orders of magnitude higher than AdamW's `1e-4`, because the update direction is normalized. Don't reuse AdamW LRs.
- **Newton–Schulz stability.** The iteration diverges if the input's spectral norm exceeds a bound. Muon rescales `M_t` by its Frobenius norm (or an estimate) before the iteration.
- **Only for 2D weights.** Applying orthogonalization to 1D or embedding tensors makes no sense. All modern implementations gate the Muon path on tensor rank.
- **MoE + Muon.** For MoE runs, Muon runs on each expert's weights independently. µP-style width transfer with MoE-and-MLA-adapted parameterization has been validated at 155B/17B (Kim et al., 2026).

## Sources

- Paper: *Muon: An Optimizer for Hidden Layers in Neural Networks* — Jordan et al. / Keller Jordan, 2024 — the original optimizer.
- Report: *Kimi K1.5* — Moonshot AI, 2025 — production LLM run, Muon at scale.
- Paper: *Let's Scale Step by Step* — Kim et al., 2026 — µP + Muon + MLA adaptation for large-scale MoE, R²=0.95 LR extrapolation.
- Code: https://github.com/KellerJordan/Muon
