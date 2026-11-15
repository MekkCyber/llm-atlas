# Early Exit — Nested Sub-Models with a Learned Exit Gate (NestOne)
*Depth — one training run that produces a spectrum of depth-efficiency tradeoffs via nested prefix sub-models.*

**TL;DR:** Converting a general LLM into an efficient *System One* decision model (single-pass, structured output) usually means either keeping the full backbone (slow) or distilling a smaller model (which tends to mimic the full model's shallow behavior rather than develop new decision skill). **NestOne** jointly trains the full LLM with **nested prefix sub-models** that share the final layers and the decision head, using self-distillation across the nesting. A lightweight **learned exit gate** reads each intermediate hidden state and decides whether to stop early. One training run yields the whole depth-vs-accuracy curve; across seven decision benchmarks NestOne matches or beats separately-distilled smaller models at each depth.

**Prereqs:** *(none)*
**Related:** [../post-training/fine-tuning/README](../post-training/fine-tuning/README.md)

---

## What it is

A training recipe that produces a *family* of depth-truncated decision models from one checkpoint, plus a gate that picks a depth per input at inference time. The motivating observation: for decision tasks, *information concentrates in middle layers*, so a prefix of the network is a viable decision model if the head is trained to read from it.

## How it works

- **Nested sub-models.** During training, cuts at k = {k₁, k₂, …} intermediate layers each define a "sub-model" that uses the backbone's first k layers and the (shared) final layers + decision head. All sub-models share the same head.
- **Self-distillation.** The full model's decision distribution is the teacher; each sub-model is trained to match it. This differs from training a smaller model from scratch (which tends to imitate shallow behavior) because every sub-model inherits the full-model head.
- **Exit gate.** A small MLP reads each intermediate hidden state h_k and emits a confidence score. If the score passes threshold τ, the forward pass stops and the decision head reads h_k. Otherwise, continue.
- **Inference.** Per-input depth selection — easy inputs exit early, hard inputs run deeper. One checkpoint serves all depths.

## Why it matters

- **One training run, many deployment points.** Operators can pick a depth-accuracy tradeoff per environment without retraining.
- **Beats separate-distillation baselines.** Because sub-models share the head and see the full backbone's gradient during joint training, they learn decision representations that don't exist in isolated-distillation smaller models.
- **Composes with quantization.** Early exit is orthogonal to weight-side compression; both can stack.

## Gotchas & tricks

- Threshold τ sets the speed-accuracy tradeoff and is calibrated per domain; one τ across tasks under-utilizes the gate.
- Only beneficial when hidden-state confidence is well-calibrated; on tasks where confidence is miscalibrated, the gate exits too early and accuracy suffers.
- The "middle layers carry the decision" observation is specific to structured-decision tasks and does not necessarily hold for open-ended generation.

## Sources

- Paper: *From Large Language Models to Efficient System One Models via Adaptive Computation* — NestOne, 2026 — arXiv 2611.16321 — https://arxiv.org/abs/2611.16321
