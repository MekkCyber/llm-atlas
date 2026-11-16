# FOAM
*Depth — a memory-efficient SOAP variant.*

**TL;DR:** SOAP (Shampoo-with-Adam) is one of the strongest second-order optimizers for LLM pre-training but drags a doubled optimizer state. FOAM — Factored One-sided Adam-Moment — does two things: precondition only along the input axis of each weight matrix, and replace the second-moment accumulator with a rank-1 (Adafactor-style) factorization in the preconditioned coordinate system. Matches SOAP and Muon on 124M–760M LLM pre-training with ~50% of SOAP's optimizer state and near-Muon wall-clock.

**Prereqs:** [z-loss.md](z-loss.md)
**Related:** [../pre-training/_training-stability.md](../pre-training/_training-stability.md), [../pre-training/_lr-schedules.md](../pre-training/_lr-schedules.md)

---

## What it is

A second-order LLM optimizer in the Shampoo / SOAP family. SOAP keeps per-axis preconditioners for each weight matrix and runs Adam in that preconditioned coordinate system. FOAM targets SOAP's two main costs — doubled state and preconditioner maintenance — while keeping its convergence behavior.

## How it works

Two changes to SOAP:

1. **One-sided (input-only) preconditioning.** SOAP learns preconditioners along both the input (left) and output (right) axes of a weight matrix. FOAM keeps only the left-side preconditioner. Empirically the right-side one contributes less once the loss is well-behaved; dropping it halves the preconditioner state.
2. **Factored second moment.** The Adam-style second-moment accumulator is replaced with a rank-1 outer-product factorization (as in Adafactor) *after* transforming into the preconditioned basis, where the second moment is approximately low-rank. This halves the remaining state again.

Net: ~50% of SOAP's extra state, and near-Muon runtime because fewer matmuls per step.

## Why it matters

Two bars have held second-order optimizers back from routine LLM pre-training use: memory footprint (full-shard state doesn't fit) and hyperparameter fragility. FOAM clears the first (comparable to AdamW state) and reports the second: validation loss is stable over a wide learning-rate range, so you can pick an LR without expensive sweeps at large scale.

## Gotchas & tricks

- "One-sided" means input-side. Flipping which side is preconditioned does *not* give the same behavior for pre-training transformers.
- The factored second moment is applied in the preconditioned basis, not raw parameter space. Doing it in raw space kills the gains.
- Reported results are up to 760M; behavior at frontier scale is not demonstrated.
- Preconditioner update cadence (how often the left-axis stats are refreshed) is the main cost knob — too infrequent and FOAM drifts toward AdamW behavior; too frequent and wall-clock regresses back toward SOAP.
- Like other Shampoo-family methods, FOAM is sensitive to beta schedules in the first few thousand steps; the LR-range tolerance only kicks in once the preconditioner estimates stabilize.

## Sources

- Paper: *Factored One-sided Adam-Moment for Practical and Scalable SOAP* — NAVER team, 2026 — [arXiv:2611.16905](https://arxiv.org/abs/2611.16905)
- Code: github.com/naver-ai
