# Skaling law
*Depth — coupling model size and data with a single interaction exponent.*

**TL;DR:** Chinchilla-style scaling laws assume model size $N$ and training data $D$ contribute to loss independently. This underestimates loss in the *data-scarce* extreme and overestimates it in the *overtraining* extreme. The Skaling law introduces one extra parameter — an interaction exponent coupling $N$ and $D$ — that cuts mean absolute percentage error (MAPE) **1.5–3×** across both interpolation and extrapolation, and lets you extrapolate full-grid loss surfaces with **~10× less compute** using a sparse low-compute grid.

**Prereqs:** [README.md](README.md), [mid-training.md](mid-training.md)
**Related:** [_scaling-laws.md](_scaling-laws.md), [wsd-schedule.md](wsd-schedule.md)

---

## What it is

A scaling law is a fitted functional form $L(N, D)$ giving expected training loss as a function of parameter count and token count. Kaplan (2020) coupled them; Chinchilla (2022) decoupled them into separable additive terms. Both extremes are systematically wrong at the far corners of $(N, D)$ space.

Skaling reintroduces a single coupling exponent that lets $N$ and $D$ interact, without moving all the way back to Kaplan's tight coupling.

## How it works

The classical Chinchilla form:

$$
L(N, D) \approx E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

Terms in $N$ and $D$ enter additively — no interaction. Skaling adds one interaction exponent $\gamma$ that couples them:

$$
L_{\text{Skaling}}(N, D) \approx E + \frac{A}{N^\alpha \cdot D^{\gamma}} + \frac{B}{D^\beta}
$$

(One published parameterization; the paper explores variants.) The extra $\gamma$ term captures the fact that at extreme regimes — heavy overtraining ($D \gg N$) or data-starved ($D \ll N$) — the two contributions don't decompose cleanly.

Fitting is done with the same least-squares Huber loss protocol as prior work; the extra parameter has negligible fit cost.

**Sparse-grid extrapolation.** Because the Skaling law extrapolates more accurately, you can fit it from a small number of low-compute $(N, D)$ points and extrapolate to the frontier without running the frontier training grid. Reported: ~10× less compute for the same full-grid extrapolation quality.

## Why it matters

- **Bigger frontier bets get more accurate.** Every large training run is an extrapolation from smaller sweeps; a 1.5–3× MAPE reduction directly reduces the "we trained at the wrong point" risk.
- **Sparse-grid sweeps become viable.** ~10× compute reduction for the sweep itself lets small teams do compute-scaling planning previously reserved for frontier labs.
- **Restores the Kaplan-vs-Chinchilla dialectic.** Kaplan had coupling but wrong form; Chinchilla had form but no coupling. Skaling is one clean parameter that captures what both were reaching for.

## Gotchas & tricks

- **The interaction exponent is task-family-specific.** Different loss functions (LM vs code) may fit different $\gamma$'s.
- **Sparse-grid sweeps still need the *right* corner.** The paper restricts to low-compute regimes; picking a bad corner can still mislead extrapolation.
- **Doesn't address optimizer / schedule effects.** LR schedule (see [wsd-schedule.md](wsd-schedule.md)) still matters and isn't captured in the functional form.
- **Not a training recipe.** It's a *predictor*. Use to allocate compute across $(N, D)$; combine with WSD or similar for actual training.

## Sources

- Paper: *Skaling: Chinchilla's Exponents Meet Kaplan's Coupling* — Videau, Youbi-Idrissi, Lopez-Paz, Ahuja, FAIR at Meta, 2026 — arXiv:2608.07222.
- Prior: *Scaling Laws for Neural Language Models* — Kaplan et al., 2020.
- Prior: *Training Compute-Optimal Large Language Models* — Hoffmann et al., 2022 (Chinchilla).
