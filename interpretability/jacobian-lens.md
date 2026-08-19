# Jacobian Lens
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A probing method that measures how much a particular activation is causally responsible for the network's output, by looking at the **Jacobian** of the output with respect to that activation. Introduced with the "gathering, not admission" story (UCSC, 2026), the lens separates *readability* (can this component's contents be read from the logits?) from *usage* (does perturbing this component actually change behavior?). Shows that three components can shift readout logits within 12% of each other while differing 7.4× in behavioral impact — a warning shot for readout-only interpretability.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../interpretability/README.md](README.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

Most mech-interp today reads latent variables via a **linear readout** — project an activation into vocabulary or feature space via the unembedding matrix and see what the model "would say" if it stopped here (logit lens, tuned lens). The Jacobian lens does something different: it directly measures the *sensitivity* of the model's output to a chosen internal activation.

For a chosen intermediate activation $h$ and an output metric $y$ (a logit, a feature, a downstream loss), compute

$$
J_{y \leftarrow h} = \frac{\partial y}{\partial h}
$$

The norm of $J$ tells you how much perturbing $h$ moves $y$ — a direct behavioral proxy.

## How it works

1. Pick the activation of interest — a residual-stream position, an attention head's output, an MLP neuron, an SAE feature.
2. Forward-pass a batch of prompts through the model; capture the activation.
3. Compute the Jacobian $\partial y / \partial h$ where $y$ is the metric of interest (single logit, KL-shift when the activation is zeroed, downstream loss on a task).
4. Aggregate — norm, spectrum, projection onto vocabulary directions — depending on the question.

Compared to activation patching, the Jacobian lens is *linear* in the model — it measures local sensitivity, not the full non-linear effect of a full swap. But it's cheap and it's what the gradients already know.

## Why it matters

- **Readout ≠ usage.** The gathering-not-admitted paper shows three components can shift the vocabulary readout within 12% of each other but differ **7.4×** in Jacobian-lens behavioral impact. A large chunk of the mech-interp literature that concludes "this component encodes X" from readout shifts alone needs revisiting.
- **Local, cheap, comparable across activations.** A gradient-based lens is one forward + one backward pass; you can use it to compare many components on the same footing.
- **Combines with SAEs.** For a chosen SAE feature, the Jacobian into an output metric gives an evidence-based measure of feature importance rather than just "the feature fires on these tokens."

## Gotchas & tricks

- **Local, not causal.** The Jacobian is a first-order Taylor expansion. If the component's real effect is highly non-linear (a switch that flips), the Jacobian understates it. Combine with activation patching for confirmation on important claims.
- **Metric choice matters.** $\partial \text{logit}_\text{answer} / \partial h$ is not the same as $\partial \text{KL}(\text{full model} \,\|\, \text{ablated model}) / \partial h$; different metrics find different components.
- **Aggregation choice matters.** Norm, largest singular value, projection onto a specific direction — each tells you a different story. Publish the aggregation.
- **Not a lie detector for readout lenses.** Readout lenses ask "is this readable here?"; that's still a valid question. The Jacobian lens is a *different* question, and the mech-interp literature needs to be explicit about which one it's asking.

## Sources

- Paper: *Gathered, Not Admitted: How Attention Brings a Latent Variable into Verbalizable Form* — Parsa Mazaheri — arXiv:2608.15022 — 2026 (UC Santa Cruz).
- Contrast: *Logit Lens* — nostalgebraist, 2020 — the readout-based ancestor.
- Related method: *Tuned Lens* — Belrose et al., 2023 — learned readouts, still readout-based.
