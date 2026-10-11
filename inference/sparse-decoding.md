# Decoding-Aware Pruning (SparseDecoding)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** When pruning an LLM for faster decoding, calibrate on the *model's own generated tokens* rather than on a static web-text corpus. SparseDecoding (2026) shows that activations during generation drift away from those on C4-style calibration data, and that fixing calibration source — plus a bitmask-indexed N:M sparse matrix-vector kernel — yields up to 1.48× end-to-end decoding speedup on A100 with better long-form generation quality than standard pruning.

**Prereqs:** [../quantization/_number-formats.md](../quantization/_number-formats.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [token-level-routing.md](token-level-routing.md) · [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

Weight-pruning for LLM inference traditionally calibrates on a fixed text source (C4, WikiText). Calibration gives per-layer activation statistics that the pruner uses to decide which weights are safe to zero. SparseDecoding argues that this calibration is systematically wrong for *decoding*: the model is going to serve tokens it generates itself, whose activation distributions differ from the ones in the calibration corpus.

The paper's two contributions are a *data-side* fix (use self-generated tokens for calibration) and a *kernel-side* fix (a fast N:M sparse matrix-vector kernel using bitmask indexing).

## How it works

- **Decoding-aware calibration.** Run the base model to generate completions on representative prompts. Collect activation statistics on those self-generated tokens. Prune the weights using those statistics with a standard N:M mask (e.g., 2:4).
- **Bitmask-indexed kernel.** Each pruned matrix is stored as a dense value array plus a bitmask indicating which positions survive. The matrix-vector kernel traverses the bitmask with fixed-step indexing, so the sparse mat-vec stays regular and cache-friendly on A100-class SMs.
- **Combined deployment.** Pruned weights + decoding-time kernel; no retraining required.

## Why it matters

- Pruning papers routinely report "quality preserved" on perplexity benchmarks and then regress on long-form generation. The calibration gap explains why — perplexity benchmarks test on static text; the deployed model serves self-generated text.
- The kernel is production-shaped: fixed-step traversal is friendly to Tensor Core execution, so the paper closes the common "theoretical sparsity ≠ real speedup" gap for N:M patterns.

## Gotchas & tricks

- Calibration-prompt diversity matters: narrow prompts produce narrow activations and over-prune the long-tail features.
- Combining with quantization (e.g., FP8 weights + N:M mask) requires re-calibrating in the chosen number format, not just at FP16.
- The kernel assumes contiguous bitmask traversal; irregular N:M patterns (varying $N$ per block) break the fixed-step traversal.

## Sources

- Paper: *SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference* — Westlake University, 2026 — [arXiv:2610.12327](https://arxiv.org/abs/2610.12327).
