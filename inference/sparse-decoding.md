# Decoding-Aware Pruning (SparseDecoding)
*Depth — N:M weight pruning calibrated on self-generated decode activations, plus a bitmask kernel that converts sparsity into throughput.*

**TL;DR:** LLM weight pruning is traditionally calibrated by passing a fixed text corpus through the dense model and recording activation statistics. But at inference, the model sees its *own* generated tokens, not the calibration corpus — the pruned pattern mis-fires. **SparseDecoding** (2026) collects calibration activations *during* autoregressive generation (skipping prefill), and ships an **N:M sparse matrix-vector kernel** with bitmask indexing and fixed-step traversal. Up to **1.48× end-to-end decoding speedup on A100** at matched quality.

**Prereqs:** [../quantization/_number-formats.md](../quantization/_number-formats.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [README.md](README.md)

---

## What it is

Weight pruning for the decoding-stage pipeline specifically. Two coupled ideas:

1. **Decoding-aware calibration.** Build the per-layer Hessian / activation covariance from tokens the dense model generates autoregressively, not from a static text snippet. The prefill phase is excluded because its activation statistics differ from decode-time.
2. **Decoding-aware kernel.** A hand-written N:M sparse matvec kernel optimised for the memory-bound single-token decode regime. Bitmask indexing avoids per-row metadata loads; fixed-step traversal keeps DRAM accesses regular and prefetcher-friendly.

Both pieces are necessary — a decoding-calibrated pattern without a fast kernel is just FLOP accounting; a fast kernel on a mis-calibrated pattern underperforms.

## How it works

**Calibration.** Pick a seed prompt distribution; run the dense model to generate $N$ tokens per prompt; record per-layer activation outer products *only on decode steps*. These feed a layer-wise Hessian-weighted pruning objective (standard OBQ/OBS style) that picks which weights to zero while minimising expected output error on the decode distribution.

**N:M structure.** Weights are pruned in groups of $M$ consecutive values, keeping exactly $N$ non-zeros per group (typically 2:4 or 1:4). N:M is hardware-friendly on A100 Sparse Tensor Cores.

**Kernel.** For each weight tile:
- Store a compact bitmask (one bit per weight) and the compressed non-zero values.
- Iterate columns at a fixed stride; use the bitmask to gate the matvec accumulation.
- No per-row metadata loads → no divergent memory access patterns.

End-to-end, decode throughput rises with the sparsity ratio until memory bandwidth is no longer the bottleneck.

## Why it matters

Decode is **memory-bandwidth-bound**: every weight parameter streams in once per token. Weight pruning only pays off if (a) the pattern aligns with real decode activations, and (b) the sparsity is cashed in by a kernel that actually skips the zeros. SparseDecoding is the first paper that nails both halves of the loop for autoregressive LLMs — and the calibration-distribution-shift point is a general lesson for any post-training compression technique applied to models that are then run in self-generation mode.

## Gotchas & tricks

- Decode-time activations drift as generation proceeds (early vs. late tokens). Collect calibration data across the whole generation length, not just the first $k$ tokens.
- 2:4 is the sweet spot on A100; 1:4 and higher ratios often cost quality unless retraining is added.
- Prefill still runs dense in SparseDecoding — the sparse weights are only used in decode. Mixed storage requires careful memory layout.
- Bitmask kernels still underperform on short sequences where launch overhead dominates; prefer them in long-decode workloads.

## Sources

- Paper: *SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference* — Westlake University, 2026 — [arXiv:2610.12327](https://arxiv.org/abs/2610.12327).
