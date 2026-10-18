# SAE Additivity Residuals (SAR)
*Depth — measuring how far a sparse-autoencoder decomposition is from behaviorally additive.*

**TL;DR:** Sparse autoencoders (SAEs) decompose activations into sparse features. The hope is that effects on behavior add up: steer feature A plus feature B = sum of individual effects. SAR measures how much of behavior variation is left over after individual contributions, and after pairwise interactions, and so on up to a chosen order. Pairwise terms close most of the gap; third-order parameter-matched random terms don't help further. SAE coordinates beat random rotations of the same subspace.

**Prereqs:** [README.md](README.md)
**Related:** [sleeper-agents.md](../safety/sleeper-agents.md)

---

## What it is

A quantitative test of a long-standing SAE claim: "each feature controls one behavior." SAR formalizes it as a decomposition of behavior variance into orders of feature interaction and reports residuals after each order.

## How it works

For a scalar behavior metric `b` (toxicity, sentiment, factuality) and a steering-vector family parameterized by feature coordinates `c_1, …, c_K`:

1. Collect `b(c)` across many random coordinate combinations.
2. Fit an order-`k` ANOVA-style decomposition: `b(c) ≈ b_0 + Σ_i f_i(c_i) + Σ_{i<j} g_{ij}(c_i,c_j) + …`
3. Report residual variance after each order; this is SAR of that order.
4. Compare against random rotations of the SAE basis (same dimension, same sparsity, scrambled directions). Learned SAE directions should have *smaller* first-order residual.

## Why it matters

The dream of SAEs — "edit one feature, change one thing" — is quantifiable, and this paper says it's partially true but not sufficient. Pairwise interactions are often necessary for a faithful steering vector; practitioners should include them (adding pairwise terms raised rank fidelity from 0.5874 → 0.7790 in reported steering).

## Gotchas & tricks

- Behavior metric must be scalar and continuous — binary metrics make the decomposition degenerate.
- Residual magnitude depends on behavior noise; always normalize by the total variance of `b` for cross-behavior comparison.
- Third-order terms *parameter-matched to random* don't help, but structured third-order terms (e.g. known-circuit triples) might. The paper argues against a general cubic correction, not against targeted ones.

## Sources

- Paper: *Behavioral Compositionality of Sparse Autoencoder Representations* — Mei & Sala, University of Wisconsin–Madison, 2026 — [arXiv:2610.21261](https://arxiv.org/abs/2610.21261)
