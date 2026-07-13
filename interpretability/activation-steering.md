# Activation Steering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Control an LLM's behavior at inference by **adding a fixed vector to hidden activations** at some layer, without changing any weights. The vector — the **steering direction** — is extracted from the model's own activations by contrasting examples of the target behavior vs. its complement (mean-of-differences, difference-of-means, or a linear probe). Injected during forward pass, it nudges generation toward the behavior. Complements neuron-level ablation (which localizes to individual units) — steering vectors capture behavior that lives **distributed** across many neurons.

**Prereqs:** [../fundamentals/attention.md](./../fundamentals/attention.md)
**Related:** [_steering.md](./_steering.md) · [../safety/cot-monitoring.md](./../safety/cot-monitoring.md)

---

## What it is

An intervention technique. Given a trained LLM you don't want to fine-tune, activation steering asks: is there a **direction in activation space** that, when added to the hidden state, reliably makes the model exhibit a specific behavior — refuse, be honest, generate in a particular dialect, use a particular style? For many behaviors the answer is yes, and the direction can be extracted with a handful of contrastive examples.

Contrast with:

- **Fine-tuning** — cheaper (no gradient updates, no dataset), but less capable.
- **Prompting** — often overrides steering entirely; steering is what you use when prompts don't hold.
- **Neuron ablation** — surgical to a specific unit; misses distributed features.
- **SAE feature editing** — the more principled cousin; steers along *learned* dictionary features rather than raw activation directions.

## How it works

**Extract the direction.**

1. Collect a small paired set: prompts + generations for the target behavior ($B_+$) vs. a matched contrast ($B_-$). Typical sizes: 32–1024 pairs.
2. Cache the residual-stream activations at a chosen layer $\ell$ for each example.
3. Compute the mean activation for each set: $\bar h_{+}, \bar h_{-}$.
4. The steering vector is $v = \bar h_{+} - \bar h_{-}$ (difference-of-means), sometimes normalized.

**Inject at inference.**

For each forward pass, add $\alpha \cdot v$ to the residual stream at layer $\ell$:

$$
h_\ell \leftarrow h_\ell + \alpha \cdot v
$$

with $\alpha$ a scalar (positive to amplify the target behavior, negative to suppress). $\alpha$ and $\ell$ are hyperparameters; typical practice is a grid search on a small validation set.

**Variants:**

- **Multi-layer injection** — inject the same $v$ at every layer, or use per-layer vectors.
- **Per-token vs. per-sequence** — inject on the prompt only, on every generation token, or on the first generation token.
- **CAA (Contrastive Activation Addition)** — the systematic-name for the difference-of-means version.

## Why it matters

- **Deployment lever without touching weights.** No fine-tuning, no dataset preparation beyond a handful of contrastive examples. A model already in production can be steered.
- **Interpretability doubles as control.** The same direction that steers the model *is* evidence about how the behavior is represented — that's why the technique is filed under interpretability, not deployment.
- **Handles distributed features.** Neuron-level ablation only touches localized behaviors. Many real-world behaviors (dialect, tone, refusal, stance) are entangled across many neurons and only show up as a *direction*.
- **Practical example.** The Arabic-dialect-steering paper (Elozeiri et al., 2026, [arXiv 2607.03936](https://arxiv.org/abs/2607.03936)) shows the technique working for a real deployment need — Arabic LLMs overproduce MSA; steering along an extracted dialect direction shifts generation to target dialects without any dialect-specific fine-tuning.

## Gotchas & tricks

- **$\alpha$ is fragile.** Too small: no effect. Too large: model degenerates (repetition, off-distribution tokens). The workable range is narrow and behavior-specific.
- **Layer choice matters.** Behaviors have preferred layers — often mid-to-late for high-level behaviors, early for surface-level. Grid search a handful of layers first.
- **Interacts with prompting.** A strong system prompt can wash out steering. Test with realistic prompts, not just neutral ones.
- **Difference-of-means is a baseline, not the ceiling.** Learned steering vectors (small linear probes trained to separate $B_+$ from $B_-$) often outperform mean-of-differences. If you have the labeled data, learn the direction.
- **Not the same as "concept editing" / ROME.** Those methods *modify weights* to change what a specific fact returns. Steering keeps weights frozen and adjusts activations at run time.
- **Sparse-neuron amplification is complementary, not alternative.** Some fraction of a behavior lives in localized neurons; the rest is distributed. Combining neuron amplification with directional steering (as the Arabic paper does) exposes both parts.

## Sources

- Paper: *Steering Language Models With Activation Engineering (CAA)* — Rimsky et al., 2023 — foundational contrastive activation addition.
- Paper: *Representation Engineering: A Top-Down Approach to AI Transparency* — Zou et al., 2023 — the broader interpretation-as-intervention framing.
- Paper: *Can Dialects Be Steered Like Languages? Sparse Neurons and Distributed Directions in Arabic LLMs* — Elozeiri, Abassy, Kallas, Dalvi, Nakov, Inui, Durrani (MBZUAI / QCRI), 2026 — [arXiv 2607.03936](https://arxiv.org/abs/2607.03936) — real deployment case for dialect control.
- Related: *Inference-Time Intervention (ITI)* — Li et al., 2023 — steering aimed at truthfulness.
