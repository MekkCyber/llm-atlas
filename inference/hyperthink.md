# HyperThink
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Replaces long chain-of-thought traces at inference with a *parameter edit*. A small hypernetwork reads the question and predicts a per-query delta over a chosen set of bias parameters; a vector-quantized decoder restricts those deltas to a finite reusable codebook. The adjusted model writes concise answers directly. Trained by distilling the base model's own thinking-mode outputs. The latency gap widens as the budget tightens — most effective in low-latency regimes.

**Prereqs:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md)
**Related:** [long2short](../post-training/reasoning/long2short.md), [length-penalty](../post-training/reasoning/length-penalty.md)

---

## What it is

A text-to-parameter hypernetwork that trades generated reasoning tokens for a one-shot parameter adjustment. Lives at inference time: given a query, HyperThink predicts bias-parameter deltas before the model decodes an answer. Not a per-task LoRA (chosen offline) and not a prompt tuning (soft prompt in the input); it is a *query-conditioned* edit over the base model's biases.

## How it works

Three components:

1. **Hypernet encoder.** A small encoder reads the query (and, optionally, a short task tag) and produces a latent "edit intent."
2. **VQ codebook decoder.** The latent is quantized against a finite codebook, and the chosen codes map to bias-parameter deltas on a selected subset of layers.
3. **Edited base model.** The base LLM's biases are modified in-place by the predicted delta; the model then decodes a direct answer (no thinking traces).

Training objective: distill the base model's own thinking-mode outputs. Thinking-mode generates CoT + final answer; HyperThink is trained so that the *edited* model produces the same final answer directly, matching the thinking-mode distribution over answers but with no intermediate tokens.

The VQ step is critical: it keeps the parameter-space edits in a small reusable set, so the hypernet effectively picks from reasoning "patterns" rather than inventing a bespoke edit per query.

## Why it matters

Reasoning compute is overwhelmingly in the generated-token count. Moving it into parameter space is an aggressive form of long-to-short compression:

- At inference, pay the hypernet's single forward pass instead of the thousands of CoT tokens.
- The gap vs thinking-mode shrinks as the latency / token budget tightens — HyperThink wins exactly where thinking-mode is unaffordable.
- Matches thinking-mode accuracy on math and general reasoning in the paper's experiments.

Positions parameter-space reasoning as a legitimate deployment path alongside train-time length penalties and decoder-side early-exit.

## Gotchas & tricks

- VQ codebook size is a sensitive hyper-parameter: too small, the hypernet can't express query-specific edits; too large, it overfits and loses the "reusable pattern" guarantee.
- Which bias parameters to edit matters. The paper restricts to a small subset (per-layer biases) to keep the hypernet lightweight.
- Not a free lunch for *all* reasoning: tasks that require genuinely multi-step tool calls or verifier feedback cannot be collapsed into a single edit.
- Needs a thinking-mode base model to distill from; a direct-answer-only base model has no supervision source.

## Sources

- Paper: *HyperThink: Text-to-Parameter Hypernetworks for Efficient Reasoning* — Kim et al., KAIST/NYU, 2026 — [arXiv:2610.03039](https://arxiv.org/abs/2610.03039)
