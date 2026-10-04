# Activation Steering
*Depth — inference-time intervention that adds a direction to a layer's residual stream to shift behavior.*

**TL;DR:** **Activation steering** adds a precomputed vector $v_c$ (a *steering vector*, representing some concept — honesty, refusal, a persona, a sentiment) to the residual stream at one or more layers at inference time. The policy response then shifts along that concept's direction without any weight update. The vector is usually recovered as the mean activation difference between contrast prompts (e.g., *honest answer* vs *dishonest answer*).

**Prereqs:** [README.md](./README.md)
**Related:** [../safety/refusal-suppression.md](../safety/refusal-suppression.md) · [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

Residual-stream intervention of the form:

$$
h'_\ell = h_\ell + \alpha \cdot v_c
$$

where $h_\ell$ is the layer-$\ell$ residual-stream activation at a chosen token position (often the last prompt token), $v_c$ is a unit vector representing concept $c$, and $\alpha$ scales the intervention. Negative $\alpha$ steers *away* from the concept.

Variants:

- **Mean-difference steering.** $v_c = \overline{h}_\text{positive} - \overline{h}_\text{negative}$ across a contrast dataset. Simplest, most common.
- **CAA (Contrastive Activation Addition).** Averaged across multiple contrast pairs with normalization. Panickssery et al., 2024.
- **Persona vectors.** $v_c$ derived from role-play prompts for a trait like *agreeable* or *verbose*. Zou et al., 2023 (Representation Engineering).
- **Trained steerers.** A small controller network produces $v_c$ as a function of the input or desired level (e.g., [persona-dosing](persona-dosing.md)).

## How it works

1. **Contrast dataset.** Collect pairs where one side exhibits the concept, the other doesn't.
2. **Extract activations.** Run both prompts through the frozen LM; capture the chosen layer's hidden state at the chosen position.
3. **Form the vector.** Take the mean difference; L2-normalize.
4. **Apply at inference.** Add $\alpha \cdot v_c$ to $h_\ell$ for the chosen token range (whole prompt, whole generation, or a span).

Choice of layer and position matters: middle layers (around 15–25 for a 32-layer model) typically carry the richest concept representations; applying at every generation token tends to over-steer.

## Why it matters

- **Train-free control.** Changes behavior without SFT or RL, so it composes with any model.
- **Fast iteration.** New behaviors cost a few hundred contrast prompts, not a training run.
- **Dual-use for safety.** Steering can enforce refusals or *remove* them ([refusal-suppression](../safety/refusal-suppression.md)); the primitive is the same.
- **Debugging tool.** When a concept vector produces the expected behavior shift, it is evidence the model represents that concept as a direction — a mechanistic interpretability win.

## Gotchas & tricks

- **$\alpha$ is model- and layer-specific.** No single $\alpha$ generalizes; always sweep.
- **Coherence collapse.** Large $\alpha$ (even unit-norm $v_c$) breaks fluency. "Persona Vectors" literature defines a *coherence floor* to measure the safe range.
- **Trait leakage.** A persona vector for *agreeable* often also shifts *verbose* and *polite* — concepts aren't orthogonal in activation space.
- **Non-stationary intervention.** Steering at every token can cause the activation distribution to drift out of what the next layer expects; applying only at a specific token position is often more stable.
- **Not graded by default.** Vanilla steering is a binary on/off; to request "60% honest," you need calibration (see [persona-dosing](persona-dosing.md)).
- **Contrast quality is everything.** If the contrast dataset confuses the concept with surface features (length, formatting), $v_c$ steers for those instead.

## Sources

- Paper: *Representation Engineering: A Top-Down Approach to AI Transparency* — Zou et al., 2023 — introduces reading/writing concept directions.
- Paper: *Activation Addition* — Turner et al., 2023 — the simplest steering recipe.
- Paper: *Steering Llama 2 via Contrastive Activation Addition* — Panickssery et al., 2024 — the CAA variant and persona-vector baselines.
- Paper: *Persona Dosing: Calibrated Activation Steering for Graded Trait Control* — Jin et al., 2026 — [arXiv:2609.36388](https://arxiv.org/abs/2609.36388). Graded, calibrated variant.
