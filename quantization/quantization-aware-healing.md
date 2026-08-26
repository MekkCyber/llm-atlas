# Quantization-Aware Healing (QAH)
*Depth — a single-stage distill-from-source recipe for recovering compressed 4-bit LLMs.*

**TL;DR:** Deploying cheap LLMs today usually means both *structural compression* (fewer params) and *aggressive quantization* (4-bit). The standard pipeline stacks them — prune → distill → quantize → QAT — leaking quality at every hop. QAH replaces the whole chain with one step: distill a compressed, quantized student directly from the *uncompressed original* teacher. Demonstrated on GPT-OSS 120B → 60B at MXFP4 (Hypernova-60B), matching or beating bf16 on 7/9 benchmarks.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md), [mxfp4](mxfp4.md)
**Related:** [../pre-training/fp8-training](../pre-training/fp8-training.md), [../post-training/reasoning/r2-opd](../post-training/reasoning/r2-opd.md)

---

## What it is

A one-shot recovery recipe for models that are simultaneously compressed and quantized. Instead of restoring quality in multiple staged passes (each with its own hyperparameter surface and data mixture), QAH treats the compressed-quantized model as a "wounded" version of the original and heals it via distillation at the target precision.

## How it works

- Start from an uncompressed, full-precision teacher (e.g. GPT-OSS 120B at bf16).
- Produce a **compressed student** at the target parameter count via width/depth/expert reduction.
- **Cast the student to the target precision** (e.g. MXFP4) *before* healing — so the healing loss is computed at deployment precision.
- Distill from the bf16 teacher onto the 4-bit compressed student in a single training stage. Loss is a standard KL/CE distillation objective over teacher logits.
- No intermediate "recover-then-quantize" step; the model is quantized during healing.

## Why it matters

Multi-stage pipelines each lose a little quality and compound. QAH's one-shot recipe makes the deployment-time student the *only* thing being optimized, at *deployment precision*, against the strongest possible teacher. Result: half the params, 4-bit weights, and headline numbers match or beat bf16 on most benchmarks.

## Gotchas & tricks

- Distilling at MXFP4 requires stable low-precision training — practical only with a mature 4-bit training kernel. If the fused kernel is not ready, staged healing (bf16 then quantize) may still be cleaner.
- Coverage matters: reported benches include 7-of-9 wins vs bf16, meaning **2 benches regress**. QAH is not free; check per-benchmark deltas for your task.
- Data mixture for the healing stage should match the teacher's original mixture, not the student's downstream domain — the goal is to reproduce the teacher, not fine-tune away from it.

## Sources

- Paper: *Quantization-Aware Healing: A Practical Recipe for Recovering Compressed, 4-Bit LLMs* — Ryskulov et al., 2026 — [arXiv:2608.20953](https://arxiv.org/abs/2608.20953)
