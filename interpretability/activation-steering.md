# Activation steering
*Depth — inference-time behavioral control by adding a vector to hidden states.*

**TL;DR:** Fit a vector in a model's residual stream that shifts its behavior toward or away from some concept (helpfulness, a specific human value, refusal). At inference time, add ±α·v to the residual at one or more layers. It's the lightweight alternative to RLHF/DPO: no gradient step, one vector per behavior, and steering strength is a knob.

**Prereqs:** [attention.md](../fundamentals/attention.md), [transformer-block.md](../architectures/transformer-block.md)
**Related:** [refusal-suppression.md](../safety/refusal-suppression.md), [cot-monitoring.md](../safety/cot-monitoring.md), [recognition-refusal-misalignment.md](../safety/recognition-refusal-misalignment.md)

---

## What it is

Post-hoc control over a fine-tuned LM without retraining. A steering vector is a direction in the residual stream that, when added, biases outputs along a behavioral axis: refuse vs. answer, one persona vs. another, one value vs. its opposite. Compared to prompting it's more reliable across paraphrases; compared to RLHF/DPO it's cheaper (no rollouts, no reward model) and reversible.

## How it works

Two families of methods to obtain the vector `v`:

- **Distribution-driven** (e.g. CAA — Contrastive Activation Addition, SphericalSteer, ODESteer): collect paired activations `h⁺, h⁻` from prompts exhibiting vs. not exhibiting the target behavior; set `v = mean(h⁺) − mean(h⁻)` at a chosen layer (usually a middle transformer block). Variants project to a unit sphere or solve an ODE in activation space.
- **Behavior-centric** (e.g. COLD-Steer, BiPO): optimise `v` directly so that generation under `h ← h + α v` maximises a behavior-reward, using preferences or a classifier. Behaves more like preference optimisation compressed into a single vector.

Apply at inference by adding `α v` to the residual at a fixed set of layers on every token. `α` is the steering strength; sign chooses direction. Multi-behavior steering stacks additive vectors.

## Why it matters

- **Cheap alignment lever:** one vector per behavior, no fine-tuning; useful for deployment-time customisation and for probing what a model "knows how to do" versus what it does by default.
- **Testable representations:** steering isolates a direction responsible for a behavior, giving a mechanistic hypothesis you can validate with causal interventions.
- **Value structure:** distribution-driven steering vectors have been shown to encode theory-consistent geometry of human values, while behavior-centric ones behave more like shortcuts even when they steer as well (Abootorabi et al., 2026).

## Gotchas & tricks

- **Layer choice matters:** middle layers (roughly 40–60% depth) usually win; too early is noisy, too late doesn't propagate through the rest of the stack.
- **Instruction tuning hurts geometry:** post-SFT/RLHF models steer as well behaviourally but their vectors align less with theoretical value structure than base models.
- **α is nonlinear:** too small = no effect; too large = OOD activations, degraded coherence. Sweep `α ∈ [0.5, 3.0]` in units of ‖v‖.
- **Composition is fragile:** stacking many vectors interferes; distribution-driven vectors compose more predictably than behavior-centric ones.
- **Not a jailbreak fix:** steering along a refusal-related direction changes refusal behavior, but if refusal is trained on a nearly-orthogonal direction to a competing "recognition" direction, steering one won't touch the other (see [recognition-refusal-misalignment.md](../safety/recognition-refusal-misalignment.md)).

## Sources

- Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition* — 2023 — https://arxiv.org/abs/2312.06681
- Turner et al., *Activation Addition: Steering Language Models Without Optimization* — 2023 — https://arxiv.org/abs/2308.10248
- Abootorabi et al., *Steering Geometry: Validating Human Value Geometry in LLM Steering Space* — 2026 — https://arxiv.org/abs/2609.06289
