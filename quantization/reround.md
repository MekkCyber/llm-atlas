# ReRound — Reconstructive Rounding
*Depth — calibration-free post-training quantization that resolves round-to-nearest's midpoint ambiguity with a diffusion-guided reconstruction.*

**TL;DR:** Standard round-to-nearest (RTN) makes an arbitrary tie-break for weights sitting near a quantization interval's midpoint, and those arbitrary choices compound across a layer. ReRound (2026) trains a **conditional diffusion model** to reconstruct plausible continuous weight distributions, then uses that reconstruction to decide the rounding side for the ambiguous weights only. A tolerance metric picks which weights go through diffusion vs. plain RTN; a singular-value spectrum comparison selects the best candidate. Beats RTN at 3/4-bit for smaller LLMs, competitive with GPTQ/AWQ *without* a calibration set, and adds zero inference-time cost.

**Prereqs:** [_number-formats](_number-formats.md)
**Related:** [_ptq](_ptq.md), [fp8](fp8.md)

---

## What it is

A **post-training weight quantization** method that inherits from the RTN family — no calibration data required — but replaces the naive tie-break at bin midpoints with a learned decision. Positioned as a drop-in for scenarios where GPTQ/AWQ-style calibration is expensive or infeasible (rapidly-changing weights, privacy-sensitive corpora, low-latency deployments).

## How it works

Per weight matrix, per row:

1. **Tolerance sweep.** Score each weight by distance from the nearest quantization midpoint. Weights well inside a bin round via plain RTN — the tie is not ambiguous.
2. **Diffusion reconstruction.** For the near-midpoint weights, run a conditional diffusion model (trained once, offline) that produces multiple plausible continuous reconstructions of the row.
3. **Candidate ranking.** For each reconstruction, form a candidate quantized matrix by picking the rounding side that agrees with the reconstruction; compare the candidate's singular-value spectrum against the original full-precision matrix and pick the closest.
4. **Emit.** Store the resulting integer weights. Inference proceeds as if it were RTN — no diffusion at runtime, no calibration state carried into deployment.

The whole procedure is one-shot per model.

## Why it matters

- **Calibration-free.** No held-out data, no calibration-set contamination worries, no per-deployment recalibration. Big deal for privacy-sensitive and rapidly-updated models.
- **Zero inference overhead.** The diffusion model runs only at quantization time; the deployed model is a plain low-bit integer.
- **Meaningful at 3–4 bits.** That's the regime where RTN's midpoint failures matter — high-bit RTN is already fine, calibration-based methods dominate at the 2-bit extreme.
- **Fills a hole in the [_ptq](_ptq.md) taxonomy** — the "smarter rounding, no calibration" branch between vanilla RTN and calibration-heavy methods.

## Gotchas & tricks

- **The diffusion model must be trained on representative weight distributions.** A model trained on weights from a different architecture family transfers poorly.
- **Only near-midpoint weights benefit.** For most weights the diffusion step is wasted — the tolerance metric matters a lot.
- **Singular-value spectrum is a heuristic proxy.** It works, but per-row spectra can be misleading for MoE or attention-projection layers with structured sparsity; validate on end-task metrics.
- **Doesn't compose trivially with activation quantization.** ReRound targets weights; for activation-quantization regimes (INT8 W8A8, FP8 W8A8) pair with an activation calibration method.
- **Small-model finding may not extrapolate.** Reported wins are on smaller LLMs; large-model behavior at low bit-widths is worth re-checking before shipping.

## Sources

- Paper: *ReRound: Reconstructive Rounding to Resolve Midpoint Ambiguity in Calibration-Free LLM Quantization* — Hsieh & Kung, Harvard SEAS, 2026 — [arXiv:2608.11045](https://arxiv.org/abs/2608.11045) — the tolerance-gated diffusion-reconstruction method and 3/4-bit RTN comparison.
