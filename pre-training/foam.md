# FOAM — Factored One-Sided Adam-Moment SOAP
*Depth — lightweight second-order optimizer that trims SOAP to Muon-class speed with half the state.*

**TL;DR:** FOAM is a lightweight variant of the SOAP second-order optimizer. Two simplifications: (1) **one-sided preconditioning** — keep the input-side factor, drop the output-side factor; (2) replace SOAP's dense rotated second-moment buffer with a **row-and-column factored** estimator. Under a K-FAC-style assumption with diagonal output curvature both changes are nearly exact; in practice the authors observe ~15–20% relative deviation but still match SOAP and Muon on validation loss across 124M–760M LLM pretraining with ~50% of SOAP's optimizer state at near-Muon wall clock. Robust across a 3× LR range.

**Prereqs:** *(none — assumes familiarity with Adam-family optimizers)*
**Related:** [_training-stability](_training-stability.md)

---

## What it is

Second-order and spectral optimizers — Shampoo, Muon, SOAP — accelerate LLM pretraining by preconditioning the gradient with a learned curvature approximation. **SOAP** uses two-sided (input × output) rotation of the parameter's second-moment statistics, which gives strong convergence but expensive state and compute per step.

FOAM asks: can we keep SOAP's convergence while cutting its cost to Muon's level? Answer: yes, by dropping the output-side rotation and factoring the second-moment buffer.

## How it works

Let g be the gradient of a weight matrix W ∈ ℝ^{d_in × d_out}.

- **One-sided preconditioning** — SOAP rotates g by two preconditioners P_in and P_out. FOAM keeps P_in only. Under the K-FAC assumption that the output-side curvature is near-diagonal, P_out is close to identity, so dropping it is nearly exact.
- **Factored second moment** — SOAP keeps a dense rotated running second moment of size d_in × d_out. FOAM replaces it with two smaller factors r ∈ ℝ^{d_in} and c ∈ ℝ^{d_out} (row and column estimates), approximating the dense buffer by the outer product rc^⊤.
- **Update** — standard Adam-style update using P_in-rotated gradient and the factored second moment; one rotation per step instead of two.

Net effect: ~½ the optimizer state of SOAP and roughly Muon-class wall-clock throughput.

## Why it matters

- **Makes second-order affordable at LLM scale.** Optimizer state is a hard constraint at frontier scale; halving SOAP's state while matching its loss is a practical win.
- **Robustness to LR.** Reported to perform well across a 3× LR range, which reduces hyperparameter-sweep cost compared to Adam/Muon.
- **Clear theoretical story.** Both simplifications are tied to a named assumption (diagonal output curvature) with a measurable deviation, not just engineering intuition.

## Gotchas & tricks

- The K-FAC assumption's 15–20% deviation is reported as not harmful at the tested scales (124M–760M); it has not been independently verified at frontier scale (≥7B) in the paper.
- "Near-Muon speed" assumes an efficient implementation of the factored moment; a naive implementation can easily lose the speed advantage.
- As with Muon, FOAM preconditions matrix parameters; vector / 1-D parameters still use Adam or a scalar fallback.

## Sources

- Paper: *Factored One-sided Adam-Moment for Practical and Scalable SOAP* — NAVER, 2026 — arXiv 2611.16905 — https://arxiv.org/abs/2611.16905
- SOAP (comparison baseline): *SOAP: Improving and Stabilizing Shampoo using Adam* — Vyas et al.
- Code: github.com/naver-ai (referenced in paper).
