# Probing vs Steering Directions
*Depth — the direction that best reads a concept is not the direction that best writes it.*

**TL;DR:** A probe finds a hyperplane in activation space that *classifies* a concept (truthful vs not). A steering vector finds a direction that, when added to activations, *induces* that concept. The field often assumed these are the same direction. They are not: steering prefers directions near the mean-difference between the two classes; probes prefer task-discriminative directions, which can be nearly orthogonal. Probe directions transfer across models; steering directions do not.

**Prereqs:** [activation-steering](activation-steering.md), [README](README.md)
**Related:** [refusal-suppression](../safety/refusal-suppression.md)

---

## What it is

Two uses of a single "concept direction" $v$ in a model's residual stream:

| Use | Criterion | Typical method |
| --- | --- | --- |
| **Probe (read)** | $v^\top h$ linearly separates concept-positive from concept-negative activations | Logistic regression, linear SVM |
| **Steer (write)** | $h + \alpha v$ makes the model behave *as if* concept-positive | Mean difference, CAA, ITI |

Papers routinely fit a probe and then use its weight vector to steer — an implicit assumption that the two tasks share an optimum. The 2026 paper above tests this directly.

## How it works

For each concept (truthfulness, sentiment) and each of three instruction-tuned LLMs:

1. Fit a probe on contrast activations; record the probe's weight vector $v_{\text{probe}}$.
2. Fit a steering vector $v_{\text{steer}}$ independently (mean of positive activations minus mean of negative).
3. Evaluate both as probes (classification AUC) and both as steerers (behavior-change rate at matched intervention strength).
4. Measure cross-model transfer: apply $v$ fit on model $A$ to model $B$.

The decomposition: a probe is free to pick directions aligned with narrow decision margins even when those directions carry little behavioral mass. A steering vector that nudges the model toward the positive-class region of activation space is approximately a translation along $\bar{h}_+ - \bar{h}_-$. These targets differ by a rotation that grows with class anisotropy.

## Why it matters

- **Explains replication failure.** Many published activation-steering interventions came from *probes*; they underperform purpose-built steering vectors and some fail entirely.
- **Changes safety engineering.** Refusal / jailbreak-direction work that uses probe-derived vectors should refit with steering-aware objectives.
- **Transfer asymmetry.** Probes transferring across models but steering vectors not transferring is a strong constraint on cross-model safety interventions — you can detect a concept in a new model from an old probe, but to *suppress* it you have to refit.
- **Application: controlled abstention.** Steering toward an "unknown" direction on PopQA mostly converts uncertain errors into abstentions rather than correct answers — a usable safety lever once the right direction is in hand.

## Gotchas & tricks

- **Match the task to the optimization.** For reading: optimize classification loss. For writing: optimize an actual behavior-change target (e.g., KL between steered and reference outputs).
- **Don't use probe weights as steering vectors by default.** At minimum, compare against the mean-difference baseline.
- **Transfer only probes.** Use probes for cross-model detection; refit steering vectors in each model.
- **Strength matters more for steering.** The optimal $\alpha$ for behavioral change depends on the direction norm; probes don't share this sensitivity.

## Sources

- Paper: *Do Probing and Steering Favor the Same Directions Within and Across Language Models?* — Sadeghi & Becker (Fraunhofer IAIS), 2026.
- Paper: *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model* — Li et al., 2023.
- Paper: *Representation Engineering: A Top-Down Approach to AI Transparency* — Zou et al., 2023.
