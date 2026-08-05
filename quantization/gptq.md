# GPTQ
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Post-training weight-only quantization for LLMs. Round the weight matrix one column at a time, and after each column redistribute the rounding error into the still-unquantized columns via the inverse Hessian of the input activations. Cheap enough to run on a single GPU in a few hours per LLM. Equivalent to Babai's nearest-plane algorithm applied to a lattice whose Gram matrix is the activation covariance.

**Prereqs:** [_number-formats](_number-formats.md)
**Related:** [fp8](fp8.md)

---

## What it is

Given trained FP16 weights $W \in \mathbb{R}^{r \times c}$ and a small calibration set of inputs $X$, GPTQ produces a quantized $\hat{W}$ (INT4, INT3, MXFP4, etc.) that minimises the layer-wise reconstruction error

$$
\| W X - \hat{W} X \|_F^2
$$

in a single sweep over the columns of $W$, at a cost of $O(c^3)$ per output row. No fine-tuning, no gradient descent, no re-training. Runs on the pretrained checkpoint plus ~128 calibration samples.

## How it works

Let $H = X X^\top$ be the Hessian of the reconstruction loss and $H^{-1}$ its Cholesky-inverse. Process columns $j = 1, \dots, c$ in order. For each column:

1. Round each entry $W_{ij}$ to its nearest quantization grid point $\hat{W}_{ij}$.
2. Compute the rounding error $e_i = W_{ij} - \hat{W}_{ij}$.
3. Add a correction $e_i \cdot [H^{-1}]_{j, j+1:c} / [H^{-1}]_{j,j}$ to the still-unquantized columns $j+1, \dots, c$ of row $i$ — this is the linear-combination of remaining weights that best compensates for the rounding just done.

The update pattern is exactly **Babai's nearest-plane algorithm** on the lattice defined by $H$, and — equivalently — a Cholesky-factorised OBQ (Optimal Brain Quantization) sweep. The trick that makes it scale to LLMs is doing the Cholesky-in-place on $H^{-1}$ and processing entire *blocks* of columns at once in fp16, so the whole procedure is bandwidth-bound rather than compute-bound.

## Why it matters

- **Preserves quality at 3–4 bits per weight** — where naïve round-to-nearest and per-tensor scaling collapse.
- **Runs in hours, not days** — on a single A100 for a 70B model. No optimizer state, no gradients.
- **Weight-only, activation-in-FP16** — plays cleanly with existing inference stacks; no activation-quantization headaches.
- Is the de-facto baseline every subsequent LLM PTQ paper (AWQ, SqueezeLLM, OmniQuant, QuaRot, SpinQuant, GPTQ-2D) compares against.

## Gotchas & tricks

- **Column order matters** — activations-aware ordering (largest diagonal of $H$ first) is more numerically stable than left-to-right for very low bit-widths.
- **Group scales** — a single scale per column is fragile at 4 bits. Groups of 128 along the input dim (per-group scales) are standard and add negligible overhead.
- **Calibration set** — 128–512 samples of ~2k tokens is typical. Domain leakage into calibration data flatters GPTQ results; keep the set close to intended deployment traffic.
- **Numerical stability of $H^{-1}$** — add a small damping $\lambda \cdot \mathrm{tr}(H)/c$ to the diagonal before inverting. Standard trick, hidden in most implementations.
- **The one-sided objective is the limitation** — GPTQ only accounts for the input covariance, not the output-side sensitivity. See [gptq-2d](gptq-2d.md) for the two-sided extension that removes that limit at the same asymptotic cost.

## Sources

- Paper: *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — Frantar, Ashkboos, Hoefler, Alistarh (IST Austria), 2022.
- Paper: *Optimal Brain Compression* — Frantar & Alistarh, 2022 — OBQ, the direct ancestor.
- Paper: *A New Nearest-Plane Algorithm* — Babai, 1986 — the lattice-algorithmic root.
- Code: `IST-DASLab/gptq` and the widely-used `AutoGPTQ` runtime library.
