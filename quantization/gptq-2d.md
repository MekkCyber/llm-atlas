# GPTQ-2D
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A cubic-time algorithm for the *two-sided* generalisation of GPTQ, where a nonsingular basis matrix acts on both the left and the right of the weight residual. The naive vectorised solve is $O(n^4)$; GPTQ-2D produces the identical output in $O(n^3)$ by rounding anti-diagonals of the weight matrix in parallel.

**Prereqs:** [gptq](gptq.md)
**Related:** [_number-formats](_number-formats.md), [fp8](fp8.md)

---

## What it is

Standard GPTQ minimises a *one-sided* quadratic reconstruction error $\| W X - \hat{W} X \|_F^2$: only the input covariance $XX^\top$ appears. A **two-sided** version instead uses two fixed nonsingular bases $L$ (rows) and $R$ (cols):

$$
\| L (W - \hat{W}) R \|_F^2
$$

so both input and output sensitivity shape the objective. When $R = I$ this collapses back to one-sided GPTQ. Two-sided calibration is provably tighter but had a compute problem.

## How it works

**The naive vectorisation.** Flatten $W$ into a vector; the two-sided objective becomes a one-dimensional quadratic form whose Gram matrix is the Kronecker product $R^\top R \otimes L^\top L$. Babai / GPTQ applies verbatim — but the vector is length $n^2$, so the sweep is $O(n^4)$.

**The GPTQ-2D observation.** The Kronecker structure means that entries of $W$ on the same **anti-diagonal** are decoupled in the appropriate ordering — rounding one does not perturb the others *at that step*. So each anti-diagonal can be rounded fully in parallel.

**The algorithm.**

```
for each anti-diagonal d in W (there are ~2n of them):
    for each entry (i, j) with i + j = d, in parallel:
        round W[i, j] to the nearest grid point using the current residual
    propagate the rounding errors of anti-diagonal d into anti-diagonals d+1, d+2, ...
    via a Kronecker-structured update
```

Each anti-diagonal costs $O(n^2)$ work; there are $O(n)$ anti-diagonals; total $O(n^3)$. The output is **bit-identical** to the naive $O(n^4)$ sweep — this is not an approximation but a re-scheduling.

## Why it matters

- **Unlocks two-sided calibration at frontier LLM scale.** Two-sided rounding is provably tighter than one-sided for a given bit-width — the compute cost was the only reason it wasn't already the default.
- **Retains GPTQ's post-training-only, weight-only story.** No fine-tuning, no gradients, drop-in replacement for GPTQ.
- **Composes with modern number formats.** Two-sided rounding shines at 3-bit and MXFP4 regimes where residual error is what limits quality.

## Gotchas & tricks

- **Choosing the right bases $L$, $R$.** The paper's choice is derived from input covariance (as in GPTQ) *and* output-side Hessian info — the two-sided story is only useful if $R$ is picked to actually capture output sensitivity.
- **Anti-diagonal parallelism is embarrassingly parallel per diagonal but strictly sequential across diagonals.** Batching multiple weight matrices helps hide the sequential barrier.
- **Numerical damping** on the Kronecker Gram is still needed, same as one-sided GPTQ. Damp each factor separately, not the Kronecker product directly.
- **Same calibration-set caveats as GPTQ.** 128–512 samples of representative traffic; domain leakage flatters results.

## Sources

- Paper: *GPTQ-2D: Cubic-Time Two-Sided Adaptive Rounding* — Chen, Hoefler, Alistarh et al. (IST DASLab), arXiv:2607.27042, 2026.
- Paper: *GPTQ* — Frantar et al., 2022 — the one-sided ancestor.
- Paper: *A New Nearest-Plane Algorithm* — Babai, 1986 — the lattice-algorithmic root that both share.
