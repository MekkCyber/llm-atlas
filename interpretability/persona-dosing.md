# Persona Dosing — Calibrated Activation Steering for Graded Trait Control
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Activation steering normally sets intervention strength via an opaque coefficient. Persona Dosing (PersonaDose) trains a description-conditioned flow-based controller (FLAS) and *calibrates* its flow time against measured trait expression — so a user can request a specific mean intensity ("show trait X at level 60") and get it, not just "more" or "less."

**Prereqs:** [../safety/refusal-suppression.md](../safety/refusal-suppression.md)
**Related:** [../safety/needle-backdoor-removal.md](../safety/needle-backdoor-removal.md)

---

## What it is

Jin et al. (2026) separate two things conflated in activation-steering literature: the *range* of behaviours a controller can reach, and the *accuracy* of hitting a requested intervention level within that range. PersonaDose is a controller plus a calibration procedure that makes graded trait control (e.g. "60% assertive") work on a measured behavioral scale rather than an arbitrary scalar.

## How it works

**Two stages.**

1. **Train a shared description-conditioned FLAS controller.** The FLAS (flow-based activation steering) controller takes a *persona description* and a *flow time* $\tau$ as inputs and produces an activation edit. Training uses persona-labeled responses, but **training responses are not paired with target intensities** — the controller learns a continuous family of edits per description, not a lookup table.

2. **Calibrate flow time against measured trait expression.** After training, probe the controller at a grid of $\tau$ values, measure trait expression under the Persona Vectors metric, and fit a monotone mapping $\tau \leftrightarrow \text{measured intensity}$. At inference, a user request "intensity 60" is mapped through this calibration to the right $\tau$.

The calibration separates the controller's intrinsic capacity from the request-mapping accuracy. Reachable targets within the controller's range can be hit; targets outside the range are flagged as unreachable rather than silently approximated.

## Why it matters

- **Headline numbers at the Persona Vectors coherence floor (75):** PersonaDose lifts core-trait expression over contrastive activation addition by **33.2 / 18.3 / 17.8 points** on Llama-3.1-8B / Qwen3-8B / Gemma-3-4B.
- **Targeting accuracy:** across seven trained traits, calibrated requests achieve 4.7–6.2 point mean error over 14–22 reachable targets (out of 28 per model).
- First activation-steering method to let practitioners actually request a *level*, not just a direction. Enables fine-grained safety work: "halve the sycophancy" becomes a measurable, repeatable intervention.
- Shares interp machinery with activation-subspace defences (see [../safety/needle-backdoor-removal.md](../safety/needle-backdoor-removal.md)) — both treat "edit this axis, don't edit that one" as the core primitive.

## Gotchas & tricks

- The coherence floor doesn't hold for *every* trait on held-out questions — targeting within the reachable range is reliable, but out-of-distribution contexts may degrade coherence.
- Calibration is per-model and per-trait; adding a new trait requires re-probing the controller.
- "Reachable" is a measured property; a trait whose extremes exceed the controller's range is flagged, not silently clipped.
- Only 14–22 of 28 calibration targets per model are reachable — the controller's intrinsic range is narrower than the requested range; practitioners should check reachability before relying on a particular intensity.

## Sources

- Paper: *Persona Dosing: Calibrated Activation Steering for Graded Trait Control* — Jin, Wang, Deng, Chen, Zhang, Zhang, Shen, 2026 — [arXiv:2609.36388](https://arxiv.org/abs/2609.36388).
- Context: *Persona Vectors* — the measurement framework PersonaDose calibrates against.
- Context: *Contrastive Activation Addition (CAA)* — Panickssery et al., 2023 — the activation-steering baseline PersonaDose beats on expression magnitude.
