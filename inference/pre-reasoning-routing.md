# Pre-Reasoning Routing
*Depth — skip chain-of-thought when a pre-reasoning forward pass already commits to a confident answer.*

**TL;DR:** For reasoning-tuned LLMs, the answer is often decided **before** the CoT begins — a single pre-reasoning forward pass recovers the committed answer and its confidence. Route: if the pre-reasoning distribution over answers is sharp, skip CoT and return the direct answer; if it's spread out, decode the full CoT. Reported by Sarfati et al. (2026) to save **30–47% of generated tokens** with no accuracy loss on forecasting.

**Prereqs:** *(none)*
**Related:** [../interpretability/representation-pooling-probes](../interpretability/representation-pooling-probes.md), [../safety/cot-monitoring](../safety/cot-monitoring.md)

---

## What it is

An inference-time router for reasoning models that trades CoT tokens for a cheap signal read out of the model's pre-reasoning state. Motivated by the empirical observation that reasoning-tuned LLMs frequently *pre-commit* to an answer at the start of decoding — the CoT then rationalizes it rather than deriving it.

Applies wherever the model was trained to always emit CoT before an answer (o1/R1-style reasoning models, forecasting fine-tunes, math reasoners).

## How it works

### The pre-reasoning read

Force the model to output the answer immediately, without CoT. Some ways to do this:

- **Forced answering.** Prepend a template that skips reasoning ("Answer only, no explanation: …").
- **Answer-token peek.** Run the forward pass to the position where the model would normally emit `<answer>`, and read the distribution over the next answer tokens before any CoT tokens.

Either way you get a distribution over candidate answers plus an implicit confidence (entropy or top-1 margin).

### The route

- **Sharp distribution** (low entropy, high top-1 margin) → return the pre-reasoning answer. No CoT.
- **Diffuse distribution** → run the full CoT and return that answer.

A threshold on entropy or margin defines the split. On the paper's forecasting workload, this saves **30–47%** of generated tokens with no accuracy loss.

## Why it matters

- **Serving cost.** CoT is the dominant cost per query for reasoning models. Cutting a third to a half of it at no accuracy cost is a large cost lever.
- **Latency.** Same argument for user-facing latency — skipping the CoT preamble makes confident answers return in ~1 forward pass instead of thousands of tokens.
- **Compatible with existing models.** No re-training; just a pre-processing step + a threshold. Ship it on top of any reasoning model.
- **Complements CoT monitoring.** The routing signal is orthogonal to whether the CoT is faithful — for the pre-committed cases the CoT was never doing the work anyway.

## Gotchas & tricks

- **Threshold tuning is task-specific.** Sharp on one workload isn't sharp on another. Tune the threshold on a held-out set per deployment.
- **Some tasks legitimately need CoT.** Multi-step arithmetic and multi-hop reasoning benefit from decoding — routing must not skip CoT on hard items. The spread-based route handles this in aggregate but can fail per-item.
- **Not everything is pre-committed.** The finding is strongest for forecasting and other "single categorical answer" tasks. Free-form generation and creative reasoning don't have a well-defined pre-answer distribution.
- **Confidence miscalibration remains.** The pre-reasoning distribution is the model's *stated* pre-answer confidence — which the same paper shows is often miscalibrated. Pairing this router with an activation probe ([representation-pooling-probes](../interpretability/representation-pooling-probes.md)) gives a better-calibrated threshold source.

## Sources

- Paper: *What LLM Forecasters Know but Don't Say: Probing Internal Representations for Calibration and Faithfulness* — Sarfati et al., Eternis / Cornell, 2026 — arXiv 2607.08046.
- Related: [representation-pooling-probes](../interpretability/representation-pooling-probes.md) — calibrated signal for setting the routing threshold.
