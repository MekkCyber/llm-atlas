# Zero-or-T thinking
*Depth — a learned per-request decision to pay for CoT, or skip it entirely.*

**TL;DR:** Every question gets full CoT at the same price today, even when reasoning doesn't help. Zero-or-T trains one model to run in two modes — zero generated reasoning tokens, or a budget T of them — and attaches a small block that predicts, per-request, the *value of reasoning*. The model pays for CoT only when that predicted value exceeds a configurable price. Trained end-to-end with Calibrated Decision Policy Optimisation (CDPO), without per-decision labels.

**Prereqs:** [long-cot-rl.md](long-cot-rl.md)
**Related:** [length-penalty.md](length-penalty.md), [long2short.md](long2short.md)

---

## What it is

A gating mechanism bolted onto a reasoning LLM. The gate is a lightweight predictor that estimates, from the question alone, whether running CoT for a fixed budget T will improve expected answer quality enough to justify the cost. The underlying model is trained to answer competently in *both* modes — zero tokens of reasoning or exactly T tokens — so the gate's output is a meaningful knob.

## How it works

Three components:

1. **Dual-mode policy.** The reasoning LLM is trained so that both "answer immediately" and "think for T tokens then answer" are valid, competitive behaviors. Not a cascade of two models; one model, two modes selected by a control token.
2. **Value-of-thinking predictor.** A small block reads the question embedding and outputs an estimated marginal reward from running CoT. At serve time, this is compared to a configured price; above threshold → think, below → answer immediately.
3. **CDPO training.** Calibrated Decision Policy Optimisation trains the predictor from *episodic* rewards only (final answer correct or not), without needing per-decision ground-truth labels of whether thinking helped. Calibration keeps the predictor's output comparable to the price in the same units — otherwise the threshold doesn't mean anything.

## Why it matters

A clean, trainable version of "think only when you need to," turning per-query compute into a learned decision rather than a hand-tuned policy. For serving stacks that already have budgeted-generation knobs, the price threshold is a direct lever on average cost. On JevBench: +0.7% accuracy over a prior token-free system at −72.5% median latency.

## Gotchas & tricks

- The dual-mode training is the hard part. If "answer immediately" isn't competitive on easy questions, the gate learns to always think.
- CDPO calibration depends on reward-scale consistency across tasks. Mixing benchmarks with very different scoring (accuracy vs F1 vs reward model) breaks the price comparison.
- The predictor is small by design. Making it large drifts into "predict correctness of the final answer" — a different and harder problem.
- "T tokens" is a fixed budget — truncate mid-sentence rather than let the model drift past. The dual-mode training should include the truncation behavior so the answer still parses.
- The price threshold is deployment-side state. A router that changes it at runtime (to meet an SLA) will see the gate's hit-rate shift, which is the point — but downstream KV-cache planning has to account for it.

## Sources

- Paper: *The Value of Thinking: Learning When to Reason in Hybrid Decision Models* — Xu, Fang, Ammar, Luo, Wang (Liverpool · Huawei Noah's Ark · UCL), 2026 — [arXiv:2611.16961](https://arxiv.org/abs/2611.16961)
