# T2Quant
*Depth — temporal, task-aware activation-precision allocation for diffusion language models.*

**TL;DR:** Quantization sensitivity in diffusion language models (dLMs) varies along two axes at once: the task and the denoising step. T2Quant exploits this with a *label-free* sensitivity signal to allocate activation precision across denoising steps under a fixed bit-width budget, giving up to 6.7 pp average-accuracy gains over prior quantization at the same budget.

**Prereqs:** [_number-formats.md](../quantization/_number-formats.md), [fp8.md](../quantization/fp8.md)
**Related:** [attention.md](../fundamentals/attention.md)

---

## What it is

A per-task, per-denoising-step mixed-precision scheme for dLMs. Rather than one activation-precision schedule per model, T2Quant chooses a schedule that spends more bits on the denoising steps the current task is sensitive to and saves bits on the steps it isn't. The sensitivity signal is label-free so the scheme can be applied without ground-truth evaluation data at calibration time.

## How it works

1. **Measure sensitivity per step** on a small calibration set. For each denoising step `t`, perturb activations with simulated quantization noise and record the shift in the model's own output distribution. No labels needed — the signal is self-referential.
2. **Group by task.** The sensitivity-over-steps curve differs by task; T2Quant clusters tasks (or conditions per prompt) and keeps one schedule per cluster.
3. **Allocate bits.** Under a global average bit-width budget, assign higher precision to high-sensitivity steps and lower precision to low-sensitivity ones. Standard integer programming or greedy heuristics suffice.
4. **Quantize.** Run inference with the allocated per-step activation precision. Weight quantization is orthogonal.

## Why it matters

- dLMs are starting to match autoregressive quality; serving cost dominates adoption, and quantization is the main lever.
- Uniform-precision schedules waste bits on insensitive steps and clip accuracy on sensitive ones. Allocating by *when it matters* is strictly dominant.
- Label-free sensitivity is the practical bit: calibration without labelled downstream data is a prerequisite for deployment on arbitrary user tasks.
- Reported gains: up to +6.7 pp average accuracy at the same bit budget across three dLMs and six benchmarks.

## Gotchas & tricks

- The sensitivity signal is a proxy — correlation with downstream accuracy is strong but not perfect. Validate on a held-out benchmark before shipping.
- Cluster granularity trades schedule count against fidelity. Too few clusters and the allocation reverts to uniform; too many and the calibration data runs thin.
- Early denoising steps often dominate sensitivity; a quick heuristic is "high precision on the first 20% of steps, low on the rest" — but the paper's data-driven schedule beats this.
- Works alongside weight quantization (GPTQ/AWQ), not instead of it; the two combine.

## Sources

- Paper: *Different Tasks, Different Timesteps: Temporal Task-Aware Quantization for Diffusion Language Models* — Avivi, Rahimi, Baskin, 2026 — [arXiv:2610.31198](https://arxiv.org/abs/2610.31198)
