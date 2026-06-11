# Alignment gating

*Depth — learnable per-layer gates that isolate (and reverse) misalignment-carrying activations.*

**TL;DR:** Insert learnable scalar gates into a model during fine-tuning; the gates learn to identify which internal activations are responsible for unsafe responses. Amplifying those activations exacerbates misalignment; suppressing them reverses it. Demonstrated as a recovery mechanism for [emergent-misalignment](emergent-misalignment.md) without retraining the base model.

**Prereqs:** [emergent-misalignment](emergent-misalignment.md)
**Related:** [refusal-suppression](refusal-suppression.md) · [alignment-faking](alignment-faking.md) · [_scheming](_scheming.md) · [scheming](scheming.md)

---

## What it is

A small, interpretability-friendly intervention on a fine-tuned model: insert a learnable gate $g_\ell \in [0,1]^{d_\ell}$ at chosen layers, where $g_\ell$ scales components of the residual stream. The gates are trained alongside (or after) fine-tuning with an auxiliary objective that splits *aligned* from *misaligned* responses. Once trained, the gates serve two purposes:

- **Diagnose** — large gate values on a dimension mean the dimension carries the unsafe signal.
- **Control** — set gates to zero on the unsafe dimensions and the model's misaligned behaviour drops, without retraining the rest of the network.

The mechanism extends the single-direction refusal-vector findings (Arditi 2024) into a *learnable* version: rather than discovering a refusal direction post hoc, the gates jointly discover *which directions* carry the safety signal during fine-tuning.

## How it works

1. **Insert gates.** Place $g_\ell$ between the layer's residual stream and the next block, multiplying element-wise (or per low-rank subspace).
2. **Add a contrastive auxiliary loss.** On a labelled split of (aligned response, misaligned response) pairs:
   $$ L^\text{gate} = \mathrm{distance}(h^\text{aligned}, g \odot h) - \mathrm{distance}(h^\text{misaligned}, g \odot h) $$
   The gates learn to suppress the dimensions whose suppression makes the aligned response more likely.
3. **Train gates only.** Backbone weights frozen during the gate-training phase. Few parameters, fast to converge.
4. **Inference-time control.** At deployment, multiply by trained gates to suppress unsafe directions. Tunable knob: scale gates between 0 (fully suppress) and 1 (no suppression) to trade safety vs capability.

## Why it matters

- **Reversal without retraining.** Unlike re-fine-tuning on aligned data (which is expensive and may not reach baseline alignment), gating identifies the affected substrate and dampens it directly.
- **Interpretability-grounded knob.** Each gate dimension is inspectable — you can read off which features were responsible.
- **Small footprint.** Adds $O(d)$ scalars per layer; negligible memory and compute.
- **Composes with other safety interventions.** Doesn't interfere with refusal training, RLHF, or external moderation.

## Gotchas & tricks

- **Requires labelled aligned/misaligned pairs.** Quality of the labels bounds the gates' precision. For broad EM, even rough labels work because the relevant directions are highly informative.
- **Suppression can hurt capabilities.** Misalignment-carrying directions sometimes overlap with general capability directions. Tune gate scaling per layer rather than globally suppressing.
- **Not a fix for adversarial inputs.** Gates dampen the *internal* signal that produces unsafe outputs in everyday rollouts. Targeted jailbreaks that recruit *different* internal pathways can bypass.
- **Doesn't catch every misalignment shape.** Works well for activation-localized phenomena (EM, sycophancy-driven misalignment). Less obvious it helps for distributed phenomena like [scheming](scheming.md) or trigger-conditioned [sleeper-agents](sleeper-agents.md).
- **Track sustainability.** Re-fine-tuning the gated model can shift the unsafe substrate to other dimensions; gates would need retraining.

## Sources

- Paper: *Emergent Misalignment Can Be Induced by Sycophancy and Reversed via Alignment Gating* — Zhu et al., 2026 — [arXiv 2606.09068](https://arxiv.org/abs/2606.09068).
- Background: *Refusal in Language Models is Mediated by a Single Direction* — Arditi et al., 2024.
