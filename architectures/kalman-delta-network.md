# Kalman Delta Networks
*Depth — linear-attention with uncertainty-aware writes, derived from Kalman filtering.*

**TL;DR:** Delta-rule linear-attention updates a fixed-size associative memory without tracking how confident the current estimate is. Kalman Delta Networks (KDN) recast the memory as a linear-Gaussian state-space model; the Kalman gain then weights each write by accumulated evidence. Two GPU-friendly approximations — Diagonal KDN and Isotropic KDN — remain scan-compatible and give consistent perplexity and downstream gains over SOTA linear-attention at 750M and 1.3B pretraining.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [transformer-block.md](transformer-block.md)
**Related:** [_linear-attention.md](_linear-attention.md)

---

## What it is

An improved write rule for linear-attention associative memories. Where DeltaNet-style layers learn an overwrite strength from the current token embedding alone, KDN also tracks a covariance-like uncertainty and modulates each write by it — writes into low-evidence regions of memory dominate, writes into well-estimated regions are damped.

## How it works

Model the memory `S_t` (a matrix associative memory) as a linear-Gaussian state-space model:

```
S_t = A_t · S_{t−1} + ε_t          (state transition + process noise)
k_t v_tᵀ ≈ H_t S_t + η_t           (observation via the current key/value)
```

The Kalman filter gives the optimal recursive posterior:

- Predict: propagate mean **and** covariance one step.
- Update: Kalman gain `K = P Hᵀ (H P Hᵀ + R)⁻¹` weights the residual write by predictive covariance `P` and observation noise `R`.

The **Delta rule** falls out as the special case where covariance is replaced by an isotropic per-token surrogate and covariance tracking is dropped.

Exact tracking needs a dense state-dependent Riccati recursion — poorly suited to GPU-parallel linear-attention scans. KDN introduces two scan-friendly approximations:

- **Diagonal KDN** — project each one-step posterior onto a diagonal Gaussian via online mean-field variational inference; per-dimension uncertainty.
- **Isotropic KDN** — one uncertainty scalar per head. Even cheaper.

Both give **Möbius-map** recursions for the uncertainty term, so the whole update composes under an associative scan with logarithmic parallel depth — matching the throughput class of DeltaNet / Mamba.

## Why it matters

- **Principled write rule.** Instead of hand-designing an overwrite gate, derive it from filtering theory; DeltaNet and its variants become recoverable special cases.
- **Consistent quality gains.** At 750M and 1.3B controlled pretraining, KDN variants improve perplexity and mean downstream accuracy over the strongest linear-attention baselines.
- **Constant-memory decoding preserved.** No blow-up in the recurrent state; scan complexity unchanged.
- **Frames the next design axis** for linear-attention: uncertainty tracking, not just gating.

## Gotchas & tricks

- **Diagonal is the default;** isotropic is a fallback when memory pressure dominates.
- **Numerical stability of the Möbius recursion** matters — use the log-domain parametrisation the paper describes.
- **Comparison caveat:** gains are demonstrated at ≤1.3B; scaling to 10B+ has not been reported yet.
- **Complementary to gating.** KDN's uncertainty gate is orthogonal to Gated DeltaNet's forget gate; the paper shows compositions.

## Sources

- Bui, Huang, Ying — *Kalman Delta Networks: Uncertainty-aware Associative Memory* — 2026 — https://arxiv.org/abs/2609.07816
- Schlag et al., *Linear Transformers Are Secretly Fast Weight Programmers* (DeltaNet) — 2021 — https://arxiv.org/abs/2102.11174
- Yang et al., *Gated Linear Attention Transformers* — 2023 — https://arxiv.org/abs/2312.06635
