# T²Quant (temporal task-aware quantization for diffusion LMs)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Diffusion Language Models (DLMs) have **task- and timestep-dependent** quantization sensitivity: different downstream tasks need different precision at different denoising steps. T²Quant detects this sensitivity *without* ground-truth labels or task-specific metrics, then solves a budget-constrained mixed-precision allocation across denoising steps. Across 3 DLMs × 6 benchmarks, up to **+6.7 pp** accuracy over existing quantization methods at the same bit-width budget (Technion · Ben-Gurion INSIGHT Lab, 2026).

**Prereqs:** [_number-formats.md](_number-formats.md), [fp8.md](fp8.md)
**Related:** [README.md](README.md)

---

## What it is

A mixed-precision quantization framework specifically for DLMs' iterative denoising. The core observation is that quantization error is *not* uniform along the denoising chain: early steps may need higher precision than late steps for one task, and the pattern reverses for another. A fixed per-model precision schedule wastes budget; T²Quant solves per-step precision allocation per task, with a label-free sensitivity signal driving the decision.

## How it works

- **Sensitivity signal (label-free).** Measure per-step activation sensitivity to quantization *without* needing labels or task-specific metrics. The paper's signal correlates well with downstream degradation across tasks and timesteps.
- **Budget-constrained allocation.** Given a fixed average-bit-width budget across steps, solve for per-step activation precision that minimises predicted sensitivity. A small combinatorial / LP-style problem; cheap to solve.
- **Composable with weight quantization.** T²Quant targets activation precision; it pairs with existing weight-quantization schemes (GPTQ-style, AWQ-style). Activation precision is the new degree of freedom.
- **Evaluation.** 3 DLMs × 6 benchmarks (reasoning, generation, QA). T²Quant improves existing quantization *methods* — meaning it's applied on top of each — by up to 6.7 pp average accuracy at the same budget.

## Why it matters

- **DLMs need their own quant story.** Autoregressive-LM quantization doesn't account for the denoising-step structure, which is a huge free variable in DLMs.
- **Label-free is the deployment win.** A quant scheme that needs no labelled data per task can be applied to models post-release without a reference set.
- **Template for timestep-aware mixed precision.** The same template probably applies to image/video diffusion models, which have the same temporal structure.
- **Pairs cleanly with existing stacks.** No need to replace the weight-quant method; T²Quant is a layer above.

## Gotchas & tricks

- **Sensitivity signal has to generalise.** The paper's signal works on evaluated DLMs; new architectures (e.g. non-masked DLMs) may need re-validation.
- **Allocation granularity matters.** Allocating per step is best; coarser (per group of steps) saves solve time at some accuracy cost. The paper quantifies this tradeoff.
- **Activation-precision changes per step mean the inference stack must support dynamic bit-widths.** Not every serving runtime handles this smoothly.
- **The 6.7 pp headline is a maximum, not an average.** Per-task gains vary; some tasks see much smaller improvements.

## Sources

- Paper: *Different Tasks, Different Timesteps: Temporal Task-Aware Quantization for Diffusion Language Models* — Avivi, Rahimi, Baskin, Technion · Ben-Gurion INSIGHT Lab, 2026 — [arXiv:2610.31198](https://arxiv.org/abs/2610.31198).
- Project page: https://ofekavivi.github.io/T2Quant/
