# Superposition Linearity Hypothesis
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** If you linearly mix the embedded tokens of two independent text streams at a Transformer's input, its next-token distribution comes out as roughly a linear mix of the two individual conditional distributions. The property is intrinsic to the Transformer architecture (visible at random init), degrades through pretraining, and can be restored by a lightweight fine-tune with a mixing regularizer. A guided decoder can then peel the two conditional distributions apart, generating two coherent continuations from a single forward pass.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [README.md](README.md)

---

## What it is

A structural property of Transformer blocks: they preserve linear structure of their inputs up to some measurable deviation. Concretely, for embedded inputs `e_A` and `e_B`, `f(α·e_A + (1−α)·e_B) ≈ α·f(e_A) + (1−α)·f(e_B)` in output-distribution space. The paper names this the **Superposition Linearity Hypothesis** and turns it from an interp curiosity into a decoding capability.

## How it works

The measurement protocol:
1. Take two token sequences, embed independently.
2. Form the input superposition `x_mix = α·e_A + (1−α)·e_B` at the model's input.
3. Compare the model's next-token distribution `p_mix` to `α·p_A + (1−α)·p_B` via KL.

Empirically, random-init Transformers exhibit low KL (linearity is a **structural** feature of attention + MLP composition), pretraining raises it (nonlinearity accumulates as capabilities grow), and a **lightweight fine-tune** with a mixing regularizer that penalizes the KL brings it back down.

Given a linearity-restored model, the **guided decoder** samples tokens conditioned on `p_mix` while using per-stream priors to route each generated token back to its parent stream. The end result is two coherent continuations, one for each source prompt, from a single forward pass through the mixed input.

## Why it matters

- Recasts an interpretability finding as a **decoding-time capability**: near-free parallel generation via input superposition.
- Suggests pretraining *loses* a useful structural property that fine-tuning can recover cheaply, which is a general shape worth watching.
- Connects to feature-superposition literature: representation superposition inside layers is well studied; input-level activation superposition through the whole stack is the same idea at a different scale.

## Gotchas & tricks

- The KL between `p_mix` and the linear mix is not zero even at random init — linearity holds *approximately*, and papers should report the deviation, not claim exact equality.
- Fine-tuning with the mixing regularizer risks capability regression on standard tasks; the paper reports "lightweight" tuning, so watch for a real quality-vs-linearity tradeoff at larger fine-tune budgets.
- The guided decoder assumes clean per-stream priors; if the two streams are near-identical, disentanglement degrades.
- Not the same as **speculative decoding** — there is no draft model here; the parallelism is intrinsic to a single forward pass on a mixed input.

## Sources

- Paper: *Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs* — Tikhonov, Korznikov, Mikhalchuk, Dragunov, Rahmatullaev, Druzhinina, Razzhigaev, Oseledets, Tutubalina — 2026 — [arXiv:2609.29845](https://arxiv.org/abs/2609.29845).
