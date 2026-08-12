# Sparse Autoencoders (SAEs)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a **sparse, overcomplete autoencoder** on a frozen LLM's activations. The encoder maps `d`-dim activations to a much larger sparse code (typically 8–64× wider) with an L1 or top-k sparsity penalty; the decoder reconstructs the original activation from the sparse code. The nonzero code dimensions on any given input tend to correspond to **monosemantic features** — human-interpretable concepts that fire on a semantically coherent set of tokens.

**Prereqs:** [README.md](README.md)
**Related:** [activation-steering.md](activation-steering.md), [inherently-interpretable-training.md](inherently-interpretable-training.md)

---

## What it is

Neurons in a trained transformer are **polysemantic**: a single neuron fires on multiple unrelated concepts because there are more useful features than there are neuron dimensions (superposition). SAEs are Anthropic's proposal (2023–2024) to un-mix superposition by projecting activations into a much wider, sparser space where each active dimension corresponds to a distinct feature.

## How it works

Given an activation `a ∈ R^d` from a frozen LLM layer:

```
z = TopK( W_enc · a + b_enc )         # z ∈ R^m,  m ≫ d,  only k nonzero
â = W_dec · z + b_dec                  # reconstruct
L = || a - â ||² + λ · sparsity_penalty(z)
```

Common variants:
- **L1-SAE.** `sparsity = ‖z‖₁` — encourages sparse but not exactly-zero activations.
- **Top-k SAE.** `z` is projected onto its top-`k` entries — hard sparsity, no L1 tuning.
- **JumpReLU / Gated SAE.** Learned per-feature thresholds to control which features fire.

After training, feature `i` in the SAE corresponds to the direction `W_dec[:,i]` in activation space. Feature interpretation: collect the tokens where feature `i` fires most strongly, look for a common pattern.

## Why it matters

- **De-superposition.** SAEs empirically decompose polysemantic neurons into monosemantic features — the primary interpretability win.
- **Steering primitive.** Once you know which feature encodes a concept, you can add or subtract its decoder vector to the model's activations to amplify or suppress the concept at inference time.
- **Circuit discovery.** SAE features are the units researchers use to map circuits (e.g. "which SAE features in layer L flow into which features in layer L+1").
- **Standard interpretability tool.** Anthropic's Claude 3 Sonnet SAE work (2024) established SAEs as the default lens for mechanistic interpretability at frontier scale.

## Gotchas & tricks

- **Reconstruction gap is real.** SAE reconstructions never perfectly match the original activations — the residual matters for downstream behavior, especially when steering.
- **Feature death.** Many SAE features never activate on any real input during training. Standard practice: reinitialize dead features periodically.
- **Feature splitting.** Larger SAEs split coarse features into finer sub-features — 8× wide might learn "capitals"; 64× wide learns "European capitals", "Asian capitals", etc. Choose width for the granularity you need.
- **Interpretation is manual.** Auto-generated feature labels from LLM judges help scale, but human review is still the ground truth. Feature 12874 "fires on tokens related to the concept of continuity" is only useful if a human agrees.
- **SAE training is expensive.** For a frontier model with hundreds of layers, training a full-model SAE library is a serious compute investment — often comparable to a small model pretraining run.
- **Alternative framings.** Inherently interpretable training (Steerling) tries to bake the disentanglement into the base model, sidestepping the SAE reconstruction gap entirely. Complementary, not replacements.

## Sources

- Paper: *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* — Bricken et al., Anthropic, 2023 — foundational SAE-on-LLM demonstration.
- Paper: *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* — Templeton et al., Anthropic, 2024 — frontier-scale SAE.
- Paper: *Gated SAEs* / *JumpReLU SAEs* — DeepMind, 2024 — sparsity mechanism improvements.
- Related: [inherently-interpretable-training.md](inherently-interpretable-training.md) for the train-time alternative.
