# Self-Speculative Decoding

*Depth — speculative decoding where the drafter is induced from the target model itself.*

**TL;DR:** Standard speculative decoding needs a separate small drafter pretrained or distilled from the target — extra weights, extra training, and constant drift when the target changes (e.g. during RL post-training). **Self-speculative decoding** induces the drafter *from the target itself* — typically by quantizing the target to a cheaper precision or by attaching extra heads to the target's middle layers. No separate model, no drift, no synchronization. Critical for accelerating RL rollouts where the target policy is updating every step.

**Prereqs:** [speculative-decoding](speculative-decoding.md), [fp8](../quantization/fp8.md)
**Related:** [partial-rollouts](../systems/partial-rollouts.md), [grpo](../post-training/grpo.md)

---

## What it is

Speculative decoding's drafter-acquisition problem: a fresh model is expensive to make, and once made it ages — the longer you train it, the more its distribution drifts from the target. Self-speculative decoding sidesteps that problem by deriving the drafter from the *current* target each step, automatically tracking the target's distribution.

## How it works

Two common derivation strategies:

**Quantized self-draft.** Run the target weights at lower precision (FP8 → FP4, or weight-only INT4). The quantized copy is much faster to evaluate (smaller weight reads, more flops per byte), shares the target's tokenizer and architecture, and produces a draft distribution close to the target's. Acceptance rate stays high because the quantized target is, structurally, the target's own approximation of itself.

**Layer-skip self-draft.** Run the target's early layers + LM head as the drafter. The shallow forward is cheap; verification is the full-depth target forward over $k$ candidates. Used in LayerSkip and related variants.

Verification and acceptance are standard speculative decoding (single batched target forward, modified rejection sampling, guaranteed-identical output distribution).

For **RL rollouts** specifically, an extra control loop helps: monitor active batch size and toggle speculation off when the regime becomes compute-bound (large batches saturate flops, so the $k$-candidate verification stops being free). Adapt draft length $k$ to the running acceptance rate so a degrading drafter doesn't tank latency.

## Why it matters

- **No drafter pretraining.** Just quantize. Saves the entire separate-model training run that vanilla SD needs.
- **Auto-tracks the policy.** Critical in RL post-training: the target model updates every step, but the quantized drafter is *always derived from the current target*, so acceptance stays high without re-distilling.
- **Same memory footprint.** No extra weights resident on GPU.
- **Production-shippable.** EfficientRollout reports up to 19.6% rollout latency reduction and 12.7% end-to-end RL step reduction in real training, with no quality loss.

## Gotchas & tricks

- **Acceptance rate depends on the quantization scheme.** FP8 self-draft usually gives $\alpha > 0.7$; aggressive 2-bit quantization can drop $\alpha$ below the break-even point.
- **Compute-bound regimes still hurt.** A system-aware toggle is essential — the only way to recover the wins consistently across an RL run is to turn SD off when active batch sizes are large.
- **Long high-temperature generations** (RL rollouts) test the drafter harder than typical serving. Acceptance-aware draft-length adaptation keeps the right $k$ for the current regime.

## Sources

- Paper: *EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts* — Kim, Lee, Oh, Galim, Kim, Hooper, Singh, Gholami, Koo, Kang, 2026 — [arXiv:2606.18967](https://arxiv.org/abs/2606.18967).
- Paper: *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding* — Elhoushi et al., 2024.
- Paper: *Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding* — Zhang et al., 2023.
