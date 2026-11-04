# SVD-based LLM Compression (LEMS + KFAC-SVD)
*Depth — weight-only LLM compression via low-rank SVD, with a layer-wise error model for rank allocation and KFAC statistics for decomposition.*

**TL;DR:** SVD-based weight compression replaces each weight matrix with a rank-`r` factorization `UΣV^T`. Hard parts are (a) choosing each layer's rank under a global budget and (b) picking the decomposition itself. **LEMS** (TUM / BMW, 2026) models per-layer compression error as local-approximation error + a propagation bias to downstream layers, then solves an integer linear program for the budget split. **KFAC-SVD** replaces Fisher-based decomposition (which rank-collapses under heavy compression) with Kronecker-factored curvature over token-wise activations. Together: up to +4.8 pp zero-shot accuracy over prior rank-allocation strategies across Mistral / Qwen3 / Llama 3 to 70B, and ~15% average perplexity improvement over Fisher-SVD.

**Prereqs:** [_number-formats.md](./_number-formats.md) · [fp8.md](./fp8.md)
**Related:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)

---

## What it is

SVD compression replaces `W ∈ ℝ^{m×n}` with `W ≈ UΣV^T` where `U ∈ ℝ^{m×r}`, `Σ ∈ ℝ^{r×r}`, `V ∈ ℝ^{n×r}` for some rank `r < min(m, n)`. Memory and matmul cost drop from `mn` to `r(m + n)`. Unlike GPTQ/AWQ/NF4 the compression is **format-free** — no low-precision integer or float — so it doesn't need quantization-aware kernels.

The method splits into:

- **Rank allocation** — given a total parameter budget, how many ranks does each weight matrix get?
- **Decomposition** — given `r`, which rank-`r` matrix is closest to `W` in the sense that matters for downstream task quality?

Both are open problems in prior work; LEMS attacks the first, KFAC-SVD the second.

## How it works

### LEMS — Layer-wise Error Modeling Search

Model per-layer compression error as a sum:

```
err(layer_i, r_i) = local_error(W_i, r_i) + propagation_bias(layer_i, r_i)
```

- `local_error` — the spectral tail cut off at rank `r_i`, weighted by an activation-based importance.
- `propagation_bias` — how the compression at layer `i` perturbs downstream layer `i+1, i+2, …` outputs. Coarse-grained, estimated from calibration forward passes.

Given these per-layer errors, LEMS casts the "allocate total rank budget across layers" problem as an integer linear program (ILP) minimizing total end-of-network error. ILP solves in seconds for a typical 30-80 layer model and allocates many more ranks to early/sensitive layers than uniform-split baselines.

### KFAC-SVD

Fisher-based SVD weights the decomposition by the Fisher information, which at high compression ratios shows **rank collapse** — all weight is piled onto one or two singular components, and the rest of the layer is wasted. KFAC replaces the Fisher by the token-wise Kronecker-factored curvature:

```
F ≈ A ⊗ G    where A is input-activation covariance, G is pre-activation gradient covariance
```

KFAC is tighter per layer (it uses per-token statistics) and does not collapse — the decomposition keeps a richer rank spread, which is what matters at aggressive compression.

## Why it matters

- **Calibration-light.** Needs activation statistics on a small calibration set, no gradient-based fine-tuning. Quick to apply to an unfamiliar model.
- **Hardware-agnostic.** No new kernel is needed; the compressed model's matmuls are plain low-rank factorizations.
- **Closes the gap to quantization.** +4.8 pp zero-shot accuracy brings SVD compression close to INT4 weight-only quantization at matched memory budget, on models where no quantization kernel is available (new architectures, hardware without native INT4 support).
- **Reusable framework.** The LEMS layer-wise-error view (local error + propagation bias) is reusable for any budget-allocation problem: pruning, mixed-precision quantization, LoRA rank allocation.

## Gotchas & tricks

- **Rank-allocation ILP needs a decent calibration set.** Too small and the propagation-bias estimate is noisy.
- **KFAC memory.** The input-activation covariance for large hidden dims is `d × d`, which at `d = 16k` is nontrivial. Shard or accumulate online.
- **Not useful for MoE experts.** Experts are already small; ranking budget across experts is dominated by the few hot ones.
- **Combine cautiously with quantization.** SVD + INT4 on top can stack losses; test the full pipeline before committing.

## Sources

- Paper: *Advancing SVD-based LLM Compression via Layer-Wise Error Model Search* — Thoma et al., TUM / BMW / Glasgow / TU Wien, 2026 — [arXiv:2610.32878](https://arxiv.org/abs/2610.32878). Project page: https://lems-svd.github.io/
- Paper: *Optimizing Full-Weight / Low-Rank Approximation (SVD-LLM, ASVD)* — prior-art SVD baselines referenced by LEMS.
