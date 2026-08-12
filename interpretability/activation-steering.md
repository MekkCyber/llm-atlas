# Activation Steering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** At inference time, add a fixed **steering vector** to the model's activations at a chosen layer to push generations toward or away from a target concept — no fine-tuning, no prompt engineering. Steering vectors are computed as the difference between activations on paired contrasting prompts, or read off directly from SAE decoder columns for a specific feature.

**Prereqs:** [README.md](README.md), [sae.md](sae.md)
**Related:** [inherently-interpretable-training.md](inherently-interpretable-training.md)

---

## What it is

Fine-tuning is the standard way to change a model's behavior; prompt engineering is the standard way to steer it at runtime without changing weights. Activation steering is a third option: intervene directly on the residual stream at inference time by adding a precomputed vector.

The vector encodes a concept ("helpful", "concise", "refuses harmful requests", "prefers formal tone"). Adding it amplifies the concept in outputs; subtracting it suppresses. No fine-tuning is needed, and the intervention is per-request and reversible.

## How it works

**Constructing a steering vector.** Two dominant recipes:

- **Contrastive activation addition (CAA).** Collect paired prompts that differ only in the target concept — e.g. "The response was helpful" vs "The response was unhelpful." Run both through the model, take the mean activation at a chosen layer for each set, and compute the difference:
  ```
  v = mean(activation | concept=+) - mean(activation | concept=-)
  ```
- **SAE-derived.** If an SAE has been trained on the target layer, the decoder column for feature `i` is *directly* a steering vector for that feature — no contrastive prompt collection needed.

**Applying at inference.** At the chosen layer during a forward pass:

```
h_layer ← h_layer + α · v
```

`α` is the steering scale — small values (`α ∈ [0.5, 3]` for CAA vectors) are typical. Layer choice matters: middle layers usually steer semantic concepts; late layers steer output style.

## Why it matters

- **Cheap, reversible behavior control.** No training run, no prompt-space redesign. Toggle on/off per request.
- **Composable.** Multiple steering vectors sum linearly (subject to interference); you can stack "concise" + "formal" + "refuses harmful" at inference.
- **Interpretability sanity check.** If a claimed feature vector doesn't actually steer the model's behavior when injected, the feature interpretation was probably wrong. Steering is the falsifiable half of SAE feature analysis.
- **Safety lever.** Refusal-related steering vectors have been used both for safety enhancement (amplify refusals for dangerous requests) and for red-teaming (subtract the refusal vector to demonstrate jailbreak vulnerabilities).

## Gotchas & tricks

- **α is a scalpel.** Too small → no visible effect. Too large → collapses generation into concept-babble ("helpful helpful helpful..."). Sweep `α` per vector per layer.
- **Layer choice is empirical.** Common wisdom: middle layers (say layers 12–20 of a 32-layer model) steer semantics; late layers steer output surface form. Try several.
- **Interference between stacked vectors.** Additive composition works up to ~2–3 concurrent vectors; more starts producing incoherent outputs. Orthogonalizing vectors helps but not fully.
- **Vector transfer across models fails.** A steering vector computed for Model A does not work for Model B — activations are model-specific.
- **Doesn't survive fine-tuning.** Post-hoc steering vectors from the base model don't apply cleanly to a fine-tuned descendant — layers may shift what they encode.
- **Inherently interpretable models expose native steering.** Steerling-style models make steering a first-class API, sidestepping the layer / scale / composition tuning.

## Sources

- Paper: *Activation Addition: Steering Language Models Without Optimization* — Turner et al., 2023 — the original CAA formulation.
- Paper: *Steering Llama 2 via Contrastive Activation Addition* — Rimsky et al., 2024 — CAA at scale on Llama-2 with safety-relevant vectors.
- Paper: *Refusal in LLMs is mediated by a single direction* — Arditi et al., 2024 — safety-critical single-vector steering.
- Related: [sae.md](sae.md) — SAE features as pre-derived steering vectors.
