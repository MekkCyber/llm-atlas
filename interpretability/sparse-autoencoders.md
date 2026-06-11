# Sparse autoencoders (SAEs)

*Depth — learn a sparse, overcomplete decomposition of a model's activations.*

**TL;DR:** Train a small autoencoder on a frozen LLM's residual-stream activations with a sparsity constraint on the hidden code, so that each input activation decomposes into a *few* active *features* out of many possible ones. The features turn out to be (mostly) monosemantic — interpretable, human-labelable concepts the model uses internally. SAEs have become the default interpretability microscope for frontier models and increasingly double as *control knobs* via [activation steering](activation-steering.md).

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [activation-steering](activation-steering.md) · [alignment-gating](../safety/alignment-gating.md)

---

## What it is

An LLM's residual stream is dense and polysemantic — most neurons activate for many unrelated concepts. SAEs decompose each activation $x \in \mathbb{R}^d$ into:

$$ z = \mathrm{Enc}(x) \in \mathbb{R}^D, \quad \hat{x} = \mathrm{Dec}(z) \approx x $$

with $D \gg d$ (overcomplete) and only a few entries of $z$ nonzero per input (sparse). The columns of the decoder are the *learned features*. The features are interpretable enough that humans (or LLMs) can label them: "Golden Gate Bridge," "Python f-string syntax," "first-person plural pronouns in legal text."

## How it works

### Architecture

Single-hidden-layer autoencoder over a chosen layer's activations:

$$ z = \sigma(W_\text{enc} x + b_\text{enc}), \quad \hat{x} = W_\text{dec} z + b_\text{dec} $$

The decoder columns are the **dictionary atoms** — direction vectors in the LLM's residual stream that the SAE believes are interpretable feature axes.

### Sparsity mechanisms

The key design choice. Variants:

- **L1 (vanilla SAEs).** Soft sparsity via $\lambda \cdot \|z\|_1$ penalty. Simple; suffers from "dead features" (never activate).
- **TopK SAEs.** Hard sparsity: keep only the largest $k$ entries of $z$ per input. Eliminates the L1 shrinkage bias; lower reconstruction error at equal $k$.
- **BatchTopK / Gated SAEs.** TopK applied across a batch, or learnable gates. Generally the SOTA recipes as of 2026, used in the cited TTS paper for instance.
- **Jump-ReLU SAEs.** Threshold + ReLU instead of TopK; smoother for downstream gradient use.

### Training data

Collect activations from the target LLM by running it on a large diverse corpus, recording the chosen layer's residual stream for each token. Typically 100M–1B tokens of activations. Train the SAE to reconstruct these with the sparsity penalty/constraint.

### Feature labelling

Once trained, label each feature by examining the inputs that maximally activate it. *Auto-interp* pipelines feed top-activating examples to an LLM and ask "what concept does this feature represent?" → a single-sentence description per feature.

## Why it matters

- **Default interpretability microscope.** Anthropic, DeepMind, EleutherAI, T-Tech, and others have published SAEs on frontier-scale models. The recipe scales.
- **Causally testable.** Unlike post hoc attention visualizations, SAE features can be *intervened on* via [activation steering](activation-steering.md): clamp the feature's activation high → the model emits the corresponding behaviour.
- **Generalizes beyond text.** Recent work applies SAEs to multimodal residual streams (vision+language, TTS) — the same recipe, the same kind of interpretable features.
- **Substrate for safety tools.** Refusal directions, deception features, EM-affected dimensions all map onto SAE features in principle. Tools like [alignment gating](../safety/alignment-gating.md) are downstream of this.

## Gotchas & tricks

- **Reconstruction error vs sparsity is a Pareto frontier.** Lower $k$ → more interpretable features, worse reconstruction → worse downstream causal claims. Pick the operating point honestly.
- **Polysemantic features survive.** A few SAE features still fire for unrelated concepts. Increasing $D$ helps; doesn't eliminate.
- **Feature splitting / merging.** Train SAEs at multiple widths $D$ — features split (one big feature becomes several specific ones) or merge (specific features collapse) across scales. Take results from multiple widths together.
- **Layer choice matters.** Mid-residual-stream layers (around layer ~⅔ of the network) yield the most interpretable features for most LLMs. Early layers are too low-level; final layers are too output-shaped.
- **Auto-interp labels can lie.** The labelling LLM tends to over-confidently name features. Sanity-check with *causal* tests — does steering the feature reproduce the labelled behaviour?
- **Don't overstate.** "Interpretable" means "human-labelable concept" — not "the model literally uses this representation as a discrete symbol." SAE features are useful coordinates, not ground truth.

## Sources

- Paper: *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* — Bricken et al., Anthropic, 2023.
- Paper: *Scaling Monosemanticity* — Templeton et al., Anthropic, 2024 — SAEs on Claude 3 Sonnet.
- Paper: *Scaling and evaluating sparse autoencoders* — Gao et al., OpenAI, 2024 — TopK SAEs.
- Paper: *Gated Sparse Autoencoders* — Rajamanoharan et al., DeepMind, 2024.
- Paper: *Interpreting and Steering a Text-to-Speech Language Model with Sparse Autoencoders* — Koriagin et al., T-Tech, 2026 — BatchTopK SAEs on a TTS LM — [arXiv 2606.10029](https://arxiv.org/abs/2606.10029).
