# Test-time regression compressor (TTR-Compressor)
*Depth — framing sequence compression as a closed-form ridge regression computed at inference time.*

**TL;DR:** TTR-Compressor replaces a learned MLP-style sequence compressor with a *closed-form regression solve per input*. The input tokens are the regressors; the targets and regularizer are learned end-to-end; the compressed representation is the exact ridge-regression solution for the current input. The learned parts are the *loss*, not the mapping, so compression is deterministic and data-adaptive without an extra forward pass. Shown useful for both block-wise sparse attention and VLM visual-token merging — the latter matches an MLP merger with 55.6% fewer visual tokens.

**Prereqs:** [attention.md](../fundamentals/attention.md)
**Related:** [mla.md](../architectures/mla.md)

---

## What it is

A general operator for "compress this sequence now, under a learned criterion". Instead of learning the compression function directly, TTR-Compressor learns the *targets* of a ridge regression problem whose solution is the compressed representation. For each input, the solution is computed in closed form at inference time — no learned MLP mapping, no sequential dependency inside the compressor.

## How it works

1. **Inputs.** A sequence of `n` token embeddings forms the design matrix `X`.
2. **Learned targets.** A small learned module produces a target matrix `Y` and a per-feature regularizer `λ` conditioned on the input (or on a lightweight query vector).
3. **Closed-form solve.** The compressed representation is `W* = (XᵀX + λI)⁻¹ XᵀY`. The solve is per-input and parallelizable; the regression's targets `Y` and penalty `λ` carry all the learned signal.
4. **End-to-end training.** Gradients flow through the closed-form solution to `Y` and `λ` via standard implicit-function differentiation.

The resulting operator is parameterized entirely by `Y` and `λ`; capacity is set by the dimensionality of `Y`.

## Why it matters

- **Data-adaptive without a per-token forward pass.** The solve is matrix-level; no autoregressive compression step.
- **Two independent wins:** improves RULER retrieval under block-wise sparse attention over attention-based compressors; matches a standard MLP visual-token merger in VLMs with 55.6% fewer visual tokens.
- **Composable.** Can drop into any pipeline that already has a learned compressor — KV cache compression, long-context compressors, video/token-merging stages.
- **No extra sequential dependence.** The compressor doesn't add a recurrence, which keeps throughput high.

## Gotchas & tricks

- The `λ` ridge term is critical for numerical stability — too small and the solve blows up, too large and the compression smooths out. Learning it per-input is what makes the operator robust.
- Implicit differentiation needs care at the matrix inverse; standard autodiff frameworks handle it, but memory for `(XᵀX + λI)⁻¹` can be a bottleneck at long sequence length.
- For block-wise sparse attention, chunk size × target dimensionality sets the compression rate; the paper's wins are at moderate rates, not aggressive 10× compression.
- Not a replacement for KV-cache quantization — complements it.

## Sources

- Paper: *Rethinking Sequence Compression with Test-time Regression* — 2026 — [arXiv:2610.31574](https://arxiv.org/abs/2610.31574)
