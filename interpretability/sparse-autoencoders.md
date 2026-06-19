# Sparse Autoencoders (SAEs)

*Depth — learn an over-complete, sparse dictionary that decomposes residual-stream activations into interpretable features.*

**TL;DR:** Train a wide single-layer autoencoder $h \to \text{ReLU}(W_e h + b) \to W_d \cdot \text{features}$ on a transformer's residual-stream activations, with a sparsity penalty (L1 or top-k) that forces most features inactive per token. The hope: each non-zero feature corresponds to a single monosemantic concept (a word sense, a syntactic role, a circuit input). SAEs became the dominant approach to mechanistic interpretability after 2023, although their reliability as **intervention** handles is now contested.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [sae-interventions](sae-interventions.md), [bag-of-dims](bag-of-dims.md)

---

## What it is

A targeted dictionary-learning method: assume each residual-stream activation is a sparse sum of a small number of underlying concept directions out of a much larger learned dictionary, and recover both the dictionary and the per-token sparse codes. Distinct from PCA (no orthogonality), distinct from k-means (continuous codes), distinct from probing (unsupervised).

## How it works

Given residual stream activations $h \in \mathbb{R}^d$:

$$z = \text{ReLU}(W_e h + b_e) \in \mathbb{R}^D, \quad D \gg d$$

$$\hat{h} = W_d z + b_d$$

Loss combines reconstruction MSE with a sparsity term:

$$\mathcal{L} = \| h - \hat{h} \|_2^2 + \lambda \| z \|_1$$

L1 sparsity is the original; **top-k SAEs** (Anthropic's variant) replace L1 with a hard top-k mask on $z$ — cleaner sparsity, no shrinkage bias.

Variants: **JumpReLU SAEs** add a learned threshold per feature, **gated SAEs** decouple "which features are active" from "how much they're active." All share the same recipe: train on a snapshot of model activations, inspect features by examining what tokens / contexts maximize each $z_i$.

## Why it matters

- **Decomposition.** SAEs gave mech-interp its first scalable way to extract interpretable features from frontier-scale models — Anthropic's Sonnet 3 has ~30M SAE features documented.
- **Steering.** Clamping feature $z_i$ to a high or low value at inference is the standard tool for **causal** interpretability experiments — "is this feature *the* knob for behavior X?"
- **Safety hopes.** Latent defenses propose using SAE features as monitors / interventions to detect or suppress unsafe behavior at the residual-stream level.

## Gotchas & tricks

- **Reconstruction–sparsity tradeoff.** Tighter sparsity = more monosemantic features but more reconstruction error; the L0/MSE Pareto curve is the standard health check.
- **Feature splitting / merging.** Larger SAE dictionaries split a single feature into specializations; smaller dictionaries merge unrelated concepts. There is no "right" $D$.
- **Reconstruction residual is not noise.** The part of $h$ the SAE *can't* explain still carries information — and downstream computations read it. This is the central failure mode behind the unreliability of SAE-based interventions (see [sae-interventions](sae-interventions.md)).
- **Distribution shift on intervention.** Clamping a feature pushes activations off the manifold the SAE was trained on. The dictionary is no longer accurate at the clamped point.
- **Comparison to standard basis.** Recent work (see [bag-of-dims](bag-of-dims.md)) argues the standard basis already encodes interpretable features via sign patterns — SAE papers should now report standard-basis baselines.

## Sources

- Paper: *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* — Anthropic, 2023.
- Paper: *Scaling and evaluating sparse autoencoders* — OpenAI (Gao et al.), 2024 — top-k SAEs at scale.
- Paper: *Sparse Autoencoders Find Highly Interpretable Features in Language Models* — Cunningham et al., 2023.
- Paper: *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* — Anthropic, 2024.
