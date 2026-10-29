# MoE Channel Selection
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A no-calibration recipe for cutting the memory bandwidth cost of **MoE decoding at low batch sizes**, where expert-weight loading from HBM is the bottleneck. Replaces threshold-based channel selectors (which need offline calibration) with a per-token **top-$k$ over FFN channels**, and pairs it with a **data-free channel reader** built from the model's own weights: a ternary sketch screens channels and a low-rank map reconstructs the ones skipped. Reads **17.8%** of routed-expert weights while keeping **98.1%** of full accuracy and runs MoE blocks **1.6–1.9× faster** than optimized vLLM at batch 1.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md) · [../architectures/deepseek-moe.md](../architectures/deepseek-moe.md)

---

## What it is

In MoE decoding, each token is routed to a small subset of experts, but each selected expert is a full FFN whose weights must be loaded from HBM. At **batch 1**, this load is the critical path — compute is cheap, bandwidth isn't. Prior work ("channel selection") loads only the *channels* each token actually needs inside an expert, using a threshold on predicted channel importance scores. The threshold is calibrated on an offline dataset.

Kim and Hwang (2026) argue the calibration step is unnecessary and brittle:

- Calibrated thresholds don't actually control the amount of data read (distribution shift at inference changes what crosses the threshold).
- On Gemma 4, threshold selection causes a **repetition-loop failure mode** that top-$k$ selection does not exhibit.

Their replacement needs no calibration data at all.

## How it works

Two parts:

### 1. Top-$k$ channel selection (data-free)

Per token, score channels inside an expert by $|x_t \cdot W_{\text{in},:,c}|$ (or an equivalent cheap scalar), then **load the top $k$**. $k$ is a fixed fraction (e.g., 18% of channels). No threshold, no calibration. Deterministic read volume regardless of input distribution.

### 2. Channel reader built from weights (data-free)

A two-stage read:

- **Ternary sketch.** A ternary ($-1, 0, +1$) matrix derived from the expert's weights screens channels: a sparse multiply gives a conservative upper bound on each channel's contribution, letting the kernel skip clearly inactive ones *before* paying for the FP16/BF16 load.
- **Low-rank reconstruction.** For channels the top-$k$ step skipped, a small learned low-rank map (fit directly from weights, no data) reconstructs an approximation. This recovers quality lost to the hard $k$ cutoff.

Both stages are derived from the expert's own weights at load time — no activation data, no calibration pass.

## Why it matters

- **Removes the calibration step.** Deployment-time MoE optimizations usually depend on a representative calibration dataset. This one does not — drop-in across workloads.
- **Fixes a Gemma 4 failure mode.** Threshold-based selectors can trigger repetition loops on long contexts in Gemma 4; top-$k$ does not. The failure mode is specific but illustrates why distributional calibration is fragile.
- **Real speedup.** Triton kernels run MoE blocks **1.60–1.90× faster than optimized vLLM at batch 1** across three post-trained MoE models, at **98.1%** of full-model accuracy and **17.8%** of routed-expert weight reads.
- **Low-batch is the dominant case.** Single-user interactive serving of frontier MoE models runs at batch 1 or 2. Memory bandwidth there is the dominant cost.

## Gotchas & tricks

- **Top-$k$ is per-token, not per-prompt.** The read volume is deterministic *per token*, but prompts with many tokens still read a lot in aggregate. The win is per-token latency.
- **$k$ interacts with expert granularity.** Fine-grained MoE ([deepseek-moe.md](../architectures/deepseek-moe.md), many small experts) has more routed experts per token, so the absolute savings compound.
- **The ternary sketch is derived once at load.** No per-step cost beyond the sparse multiply itself. It's weights-side preprocessing, not activation-side.
- **Low-rank reconstruction cost.** Must be small enough that the saved bandwidth beats the extra compute. The paper picks ranks that stay well under this crossover.
- **Doesn't help prefill.** Prefill is compute-bound; channel selection targets the memory-bound decode phase.
- **Composes with quantization.** Weight-only quantization reduces the volume per channel; channel selection reduces the *number* of channels. Both compose.

## Sources

- Paper: *Channel Selection for Mixture-of-Experts Decoding Needs No Calibration Data* — Kim, Hwang, 2026 — [arXiv:2610.37386](https://arxiv.org/abs/2610.37386)
