# Latent Probes for Deployment-Time Control
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A lightweight side-attached "control layer" that reads hidden-state trajectories from a **frozen** LLM/VLM and predicts *deployment-time* control decisions. Two heads: a **Capability Head** ("can this model solve the instance, or should we defer?") and a **Resolution Head** ("respond directly, ask a clarifying question, invoke a tool, or abstain?"). Trained only on latent traces from the same frozen backbone — no backbone changes, no retraining.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** none yet

---

## What it is

A **probe** in the mechanistic-interpretability sense (a small classifier trained on frozen internal activations), repurposed as a *runtime controller* rather than an explanatory tool. The insight is that control-relevant information — will this model succeed? does it need a tool? should it abstain? — is already present in the hidden-state trajectory *before* generation finishes.

## How it works

- **Backbone stays frozen.** The base LLM/VLM is not modified in any way.
- **Trace collection.** Hidden states from selected layers are logged during generation (a subset of layers is usually enough).
- **Two heads.**
  - **Capability Head:** binary/probability output — "solve locally" vs. "defer to a stronger model."
  - **Resolution Head:** categorical output over {Direct Answering, Clarification, Tool Use, Abstention}.
- **Training data.** Latent traces labeled with the correct control decision. Because the backbone doesn't change, one training pass produces a probe that stays valid until the backbone is swapped.
- **Inference time.** The probes are cheap; they can be evaluated on *partial* generations, enabling **early handoff** — routing to a stronger model mid-generation once confidence drops.

## Why it matters

Multi-model production stacks — cheap-first + strong-fallback — currently commit to routing *before* the cheap model runs. Reading the cheap model's own latents mid-generation to decide "still confident?" is a strictly stronger signal. Because probes are decoupled from the backbone, they're also swap-friendly when the backbone version changes.

## Gotchas & tricks

- **Layer choice matters.** Different layers surface different features; ablate empirically per backbone.
- **Distribution-shift is the real risk.** Probes trained on lab traces can miss decision-relevant regimes in production; monitor and retrain the probe when the deployment distribution drifts.
- **"Defer" and "tool use" are cheap wins; "abstain" is dangerous.** Abstention should be conservative; miscalibrated abstention becomes a user-facing failure mode.
- **Post-hoc adaptation ≠ zero risk.** Cheap to attach but still adds a control-plane failure mode; monitor the probe head as a first-class deployment component.

## Sources

- Paper: *Multi-Head Latent Control: A Unified Interface for LLM Agent Decision Making* — Ghasemabadi, Chen, Rashidi, Niu, 2026 — [arXiv:2607.14277](https://arxiv.org/abs/2607.14277).
