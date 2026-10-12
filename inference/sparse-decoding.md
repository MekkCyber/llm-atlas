# SparseDecoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Training-free N:M weight sparsity for LLM decoding, calibrated on **decode-time activations** instead of fixed pre-training text. Paired with a bitmask-indexed sparse GEMV kernel. Beats fixed-text-calibrated pruning on long-form generation and delivers up to 1.48× end-to-end decoding speedup across Llama-3.1-8B / 3.3-70B and Qwen3-14B/32B on A100.

**Prereqs:** [README.md](./README.md), [../quantization/_number-formats.md](../quantization/_number-formats.md)
**Related:** [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

A pruning + kernel pair aimed at the memory-bound regime of LLM decoding. The pruning algorithm is standard N:M (keep N nonzeros per group of M columns), the novelty is the **calibration data**: activations collected while the model is *actually decoding* its own tokens, not activations collected on fixed text snippets.

## How it works

1. **Warm up.** Run the base model autoregressively on representative prompts, logging per-layer activations from the self-generated continuation.
2. **Mask selection.** Choose per-matrix N:M masks that minimise the error on those decode-time activations rather than on fixed-text activations (which have a different distribution — crucially, no long self-generated tails).
3. **Store the mask as a bitmask.** The sparse GEMV kernel reads the bitmask and dispatches only the N live weights per group, with lightweight metadata that fits in on-chip memory.
4. **Serve.** At decode time the pruned weight matrices sit at a smaller memory footprint; the kernel's bandwidth win is what delivers the speedup.

## Why it matters

Pruning for LLM inference is a well-trod area, but the "activation distribution" has usually meant "activations on The Pile." For autoregressive decoding, those are off-distribution: the model is reading its own output, which has sharper tails and different per-layer statistics. Calibrating to the real decode distribution recovers quality lost by fixed-text calibration at the *same* sparsity, which is close to a free win. The result lands as **1.48× end-to-end speedup with better WritingBench / ClassEval scores** than fixed-text baselines.

## Gotchas & tricks

- Decode-time calibration needs representative prompts; a tiny unrepresentative warmup can bake the wrong mask in.
- The gain assumes a memory-bound regime (small batch, long generation). At large batch, decoding is compute-bound and the kernel win shrinks.
- N:M sparsity on A100 benefits from 2:4 structured hardware support; different hardware (consumer GPUs, MI300) has different sparse-GEMV ceilings — benchmark locally.
- Combining with weight-only quantisation is possible but the authors don't fully characterise the interaction; expect some calibration-order sensitivity.

## Sources

- Paper: *SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference* — Westlake, 2026 — [arXiv:2610.12327](https://arxiv.org/abs/2610.12327)
