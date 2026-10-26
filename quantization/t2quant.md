# T2Quant

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Activation-precision allocation for **diffusion language models** that redistributes bits *across denoising steps* rather than across layers. Per-(task, step) quantization sensitivity is estimated **label-free** (no ground-truth needed), then bits are allocated under a fixed per-token budget. On 3 diffusion LMs across 6 benchmarks, T2Quant lifts prior quantization methods' accuracy by up to **+6.7 pp at matched budget**.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** [README.md](README.md)

---

## What it is

Autoregressive LLM quantization usually allocates precision across *layers* or *channels*: Hessian-weighted bit allocation (GPTQ), activation-sensitivity-weighted channels (SmoothQuant), etc. The time dimension is a non-issue — there's one forward pass per token.

Diffusion LMs are different. A sequence is produced by many denoising passes over the same positions; quantization error accumulates over the schedule. The authors show two things:

1. **Sensitivity to activation quantization varies strongly across denoising steps.** Early steps (where the sequence is still mostly masked) and final steps (where predictions commit) are more sensitive than middle steps.
2. **This sensitivity also varies across tasks.** The same step may be high-sensitivity on math and low-sensitivity on open-ended writing.

T2Quant builds a label-free estimator of per-(task, step) sensitivity and uses it to allocate activation bits across the denoising schedule under a fixed average budget.

---

## How it works

### Label-free sensitivity estimation

For each (task, denoising step), T2Quant estimates sensitivity without ground-truth labels or task metrics. The signal comes from the model's own prediction stability: run the step at full precision and at a candidate low precision, measure divergence in output logits for that step. Aggregate across a small calibration set drawn from the task's prompt distribution.

Key property: no labels are needed. This is important because diffusion LMs are often deployed on tasks where ground-truth is expensive or unavailable (open-ended generation, long-form writing).

### Budget-constrained bit allocation

Given per-step sensitivity $s_{t}$ and an average activation bit-width budget $\bar{B}$, T2Quant chooses per-step bit widths $B_t$ that minimize total quantization error subject to:

$$
\frac{1}{T}\sum_t B_t = \bar{B}
$$

Steps with high $s_t$ get more bits (e.g. INT8 or FP8); steps with low $s_t$ get fewer (INT4). The allocation is per-task — the same model running two different tasks uses two different bit schedules.

### Composition with existing quantization methods

T2Quant does not replace per-tensor or per-channel quantization — it sits *on top* and shifts the bit budget across the time axis. Any method that already supports per-tensor precision choice can be made T2Quant-aware by sampling its budget from the step-schedule.

---

## Why it matters

- **First diffusion-LM-native quantization optimizer.** Prior methods were ports of autoregressive-LM quantization that ignored the schedule.
- **Up to +6.7 pp average accuracy at matched budget** across 3 diffusion LMs × 6 benchmarks — a free lunch when the allocator is used.
- **Label-free.** Composable with calibration data drawn purely from the deployment-time prompt distribution; no annotation step.
- **Orthogonal to layer-level methods.** Stacks cleanly on top of GPTQ, SmoothQuant, AWQ, etc.

---

## Gotchas & tricks

- **Per-task calibration overhead.** The sensitivity estimator must be re-run per task class. For a general-purpose chat deployment, this means either calibrating on a representative task mixture or re-calibrating per domain.
- **Budget granularity matters.** If the hardware only supports two activation precisions (e.g. FP8 and FP16), T2Quant degenerates to a schedule-switching decision rather than a continuous allocation. Finer per-step precisions (INT4/INT6/INT8) expose more benefit.
- **Weight quantization is unaffected.** T2Quant allocates *activation* bits. Weight quantization is independent and should be chosen with standard methods.
- **Not for autoregressive LMs.** The method's premise is multi-pass denoising. Autoregressive decoding has one pass per token; the time axis the allocator exploits doesn't exist there.
- **Sensitivity is noisy at low NFE.** When the denoising schedule has very few steps (NFE = 4–8), the per-step sensitivity signal becomes unreliable and T2Quant's gains shrink. The method works best at mid-to-high NFE.

---

## Sources

- Paper: *Different Tasks, Different Timesteps: Temporal Task-Aware Quantization for Diffusion Language Models* — Avivi, Rahimi, Baskin, Technion / BGU (INSIGHT Lab), 2026 — [arXiv:2610.31198](https://arxiv.org/abs/2610.31198).
- Precedents: *GPTQ* (Frantar et al. 2023), *SmoothQuant* (Xiao et al. 2023), *AWQ* (Lin et al. 2023) — layer- and channel-level quantization methods T2Quant composes with.
