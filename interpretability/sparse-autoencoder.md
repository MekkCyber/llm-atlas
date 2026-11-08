# Sparse autoencoder (SAE) on LLM activations
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Fit a very wide autoencoder with an **L1 or top-K sparsity penalty** to an LLM's intermediate activations. The resulting dictionary of latents is far more human-interpretable than the raw neurons, and each latent fires on a narrow concept. SAEs have become the primary tool for mechanistic interpretability of LLM feature spaces.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** none

---

## What it is

An LLM activation vector `h ∈ R^d` is polysemantic — a single neuron can respond to many unrelated concepts. The *superposition hypothesis* is that the model packs many sparse features into fewer dense dimensions. An SAE tries to recover those sparse features.

Given activations `h`, train:

```
z = ReLU(W_e @ h + b_e)   # encoder — wide, sparse latent
h' = W_d @ z + b_d        # decoder — reconstruction
```

with reconstruction loss `||h − h'||²` plus a sparsity penalty (L1 on `z`, or a top-K hard constraint keeping only the K largest latents). `z` has dimension much larger than `d` (often 4–64× wider).

## How it works

- **Overcomplete dictionary.** The encoder matrix `W_e` has shape (M, d) with M ≫ d. Each row is a candidate feature direction.
- **Sparsity enforcement.** L1 pushes most latents to zero; top-K makes sparsity exact at the cost of a non-smooth gradient.
- **Decoder reconstructs** by summing active latents' decoder rows (`W_d` rows are "feature directions" in activation space).
- **Training data.** A large corpus of activations collected from the frozen base LLM at a chosen layer / site (residual stream after block N is typical).

After training, each latent `z_i` has an interpretable meaning: it fires on inputs that share a concept. Automated interpretation pipelines label latents by feeding activating examples to an LLM-as-judge.

## Why it matters

SAEs underpin modern mech-interp: feature extraction, circuit analysis, activation steering. They convert the black-box activation space into a sparse, labelled vocabulary of features. Recent variants (top-K, JumpReLU, gated SAE) address pathologies of L1 SAEs (dead latents, shrinkage).

The technique generalises: any fixed representation space with suspected superposition — including MoE outputs, image encoder features, and attention-head outputs — can be analysed this way (see [sae-moe-attribution](sae-moe-attribution.md)).

## Gotchas & tricks

- **Dead latents.** With L1, many latents never fire. Mitigations: auxiliary "resample" losses, warm-up, top-K variants.
- **Shrinkage.** L1 biases latent magnitudes toward zero; the reconstruction compensates by over-amplifying decoder rows. Top-K / JumpReLU avoid this.
- **Site matters.** SAEs at different layers / stream positions find different features. Residual-stream SAEs are the default; attention-output SAEs find head-specific features.
- **Interpretation is downstream.** Training an SAE produces latents; *interpreting* them is a separate step (auto-interp via LLM judges, manual inspection).

## Sources

- Paper: Towards Monosemanticity (Anthropic, 2023); Scaling Monosemanticity (Anthropic, 2024); JumpReLU SAEs (DeepMind, 2024); top-K SAEs (OpenAI, 2024).
