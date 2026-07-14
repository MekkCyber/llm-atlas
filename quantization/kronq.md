# KronQ
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-training quantization framework that fixes a hidden assumption of GPTQ-style methods — that all output channels contribute equally to the layer-wise reconstruction loss. KronQ folds the **gradient covariance** into the objective via a **Kronecker-factored Hessian** approximation, then uses that signal for (a) bidirectional incoherence processing (rotating *both* input and output dimensions) and (b) a new sensitivity metric for inter-layer mixed-precision allocation. On 2-bit LLaMA-3-70B, KronQ stays at 7.93 PPL where GPTQ diverges past 2000.

**Prereqs:** [_number-formats.md](./_number-formats.md), [fp8.md](./fp8.md)
**Related:** [../pre-training/fp8-training.md](../pre-training/fp8-training.md)

---

## What it is

Second-order post-training quantization methods (GPTQ, GPTAQ) build their per-layer quantization objective from **activation covariance** only:

$$
L_{\text{layer}} = \| W X - \hat W X \|_F^2
$$

where $X$ is the calibration activation. This implicitly weights all output channels equally. KronQ argues that under a Kronecker-factored approximation of the true block Hessian, the loss also depends on the **gradient covariance** $G$ on the output side:

$$
H \approx A \otimes G, \qquad A = X X^\top, \qquad G = \mathbb{E}[\nabla_y \nabla_y^\top]
$$

So a quantization scheme that respects $H$ needs to see both factors.

---

## How it works

### Bidirectional incoherence processing

The existing input-side trick (QuIP/QuaRot/GPTAQ) is to left-multiply weights and activations by a random orthogonal $R_A$ so that outliers get "smeared" across the input dimension:

$$
\hat W' = \hat W R_A^\top, \quad X' = R_A X
$$

KronQ symmetrizes this on the output side using $G$: another orthogonal $R_G$ derived from the gradient covariance is applied to output channels, reducing weight magnitude variance across the *output* dimension as well. The two rotations compose:

$$
\hat W'' = R_G^\top \hat W R_A^\top
$$

Result: fewer per-tile outliers in both dimensions, which is what enables 2-bit quantization to remain finite.

### Kronecker-Hessian sensitivity for mixed precision

For layer $\ell$, KronQ estimates a scalar sensitivity from the traces of both factors:

$$
s_\ell = \mathrm{tr}(A_\ell) \cdot \mathrm{tr}(G_\ell)
$$

Layers with high $s_\ell$ are allocated more bits; low-$s_\ell$ layers get pushed to 2-bit. The joint trace is a much better ranking signal than activation-magnitude heuristics used by prior work, because it captures how much the *loss* actually depends on that layer's weights.

---

## Why it matters

- **Extreme low-bit weight-only quantization stays viable.** On 70B-class models, 2-bit weight-only was the boundary where GPTQ/GPTAQ diverged. Keeping perplexity finite at 2-bit unlocks serving at ~2× the memory efficiency of 4-bit.
- **Layer-wise mixed precision gets a principled sensitivity metric.** Previous per-layer bit allocation was mostly heuristic; the Kronecker-Hessian trace makes it an approximation to the actual second-order loss.
- **The trick is orthogonal to the underlying number format.** Bidirectional rotation and Hessian-aware allocation can be layered on top of INT2/INT3/INT4/MXFP4 quantizers.

---

## Gotchas & tricks

- **You need gradients on the calibration set.** Unlike GPTQ, KronQ requires a backward pass over the calibration data to estimate $G$. Cheap compared to the model itself, but not free.
- **Kronecker approximation is exact only for MLPs with i.i.d. inputs.** For attention layers with structured outputs, $H \approx A \otimes G$ is an approximation whose quality varies. The paper reports it works in practice, but expect layer-specific behavior.
- **Bidirectional rotation increases scale-storage.** You now store two rotation matrices per layer alongside quantized weights. Small overhead for a $d \times d$ orthogonal (or Hadamard-approximated) rotation, but not zero.
- **2-bit success is model-family-dependent.** LLaMA-3-70B works; smaller or more instruction-tuned models sometimes have narrower recovery margins at 2-bit.

---

## Sources

- Paper: *KronQ: LLM Quantization via Kronecker-Factored Hessian* — Lee, Li, Yin, Panda, USC — [arXiv:2607.07964](https://arxiv.org/abs/2607.07964).
- Predecessor: *GPTQ: Accurate Post-Training Quantization for Generative Pretrained Transformers* — Frantar et al., 2022 — the layer-wise second-order quantization baseline this paper refines.
- Related: *QuIP / QuaRot / GPTAQ* — the input-side random-rotation lineage that KronQ extends to both dimensions.
