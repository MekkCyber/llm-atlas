# Sparse Autoencoders (SAEs)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An SAE is a shallow autoencoder trained on a transformer's internal activations with an **L1 (or top-k) sparsity constraint on the hidden layer**. The hidden units become a dictionary of *features* — directions in activation space that are each active rarely and separately. Used to decompose dense residual-stream activations into human-interpretable components you can inspect, label, intervene on, and compose.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [sae-additivity-residuals.md](./sae-additivity-residuals.md)

---

## What it is

Transformer activations are superposition-encoded: more concepts than neurons, each concept distributed across many directions. An SAE learns an overcomplete, sparse basis where features line up with distinct concepts.

Architecturally minimal: a linear encoder, a sparse hidden layer, a linear decoder. Trained to reconstruct the activation while most hidden units stay silent.

$$
h = \sigma(W_{\text{enc}}\, x + b_{\text{enc}}), \qquad \hat x = W_{\text{dec}}\, h + b_{\text{dec}}
$$

Loss: reconstruction + sparsity.

$$
\mathcal{L} = \|x - \hat x\|_2^2 + \lambda\,\|h\|_1 \qquad\text{(or top-k gating)}
$$

Trained on activations drawn from a specific site in a frozen transformer (post-MLP residual at some layer is the most common choice). Weights are not shared with the transformer; the transformer isn't retrained.

## How it works

1. **Collect activations.** Run many prompts through the frozen model and store activations at the chosen site.
2. **Train the SAE.** On that activation distribution, minimize reconstruction + sparsity. Hidden dimension typically $\gg$ model's residual dimension (overcomplete, 8–64× is common).
3. **Interpret features.** For each hidden unit $f_i$, find the inputs where $h_i$ fires most strongly, label the pattern.
4. **Intervene.** Add or subtract a feature's decoder direction to the residual stream to steer behavior ("activation steering" via SAE directions).

Two sparsity recipes:

- **L1 penalty.** Classic; smooth but gives dead features and a bias–variance knob on $\lambda$.
- **Top-k.** Keep only the top-$k$ activations per example; sharper separation, easier to reason about.

## Why it matters

- **Operationalizes mech-interp.** You can enumerate features, label them, and compose them — a concrete object to study instead of raw directions.
- **Steering handle.** Adding/subtracting feature decoder directions produces controlled behavior changes that transfer across prompts.
- **Shared vocabulary across layers and models.** Features at matched layers across model sizes often align, giving a cross-model unit of analysis.

## Gotchas & tricks

- **Features ≠ atomic concepts.** An SAE feature is a *useful fiction* — compression artifact + genuine structure. Expect polysemantic features and feature splitting across SAE training runs.
- **Reconstruction error is a knob.** A tiny $\lambda$ gives a near-identity SAE with no insight; a large one gives a sparse representation that loses information. Watch loss recovery (how much of the downstream model's loss the reconstructed activation preserves).
- **Compositionality is not guaranteed.** SAE features decode *linearly*, but behavioral effects of simultaneous feature presence need not be additive. See [sae-additivity-residuals.md](./sae-additivity-residuals.md) for a formal measurement.
- **Choice of site matters.** Post-MLP residual is standard; attention output projection or MLP output give different (sometimes finer) features.
- **Dead features.** Common with L1; several training tricks (ghost gradients, auxiliary losses, re-init) address them. Top-k largely sidesteps this.
- **Not causal yet.** SAE feature discovery is correlational — a feature "fires on X" doesn't mean deleting it will kill behavior X. Causal interventions require ablation / patching experiments.

## Sources

- Paper: *Sparse Autoencoders Find Highly Interpretable Features in Language Models* — Cunningham, Ewart, Riggs, Huben, Sharkey, 2023 — [arXiv:2309.08600](https://arxiv.org/abs/2309.08600).
- Paper: *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* — Templeton et al., Anthropic, 2024.
- Paper: *Scaling and Evaluating Sparse Autoencoders* — Gao et al., OpenAI, 2024 — top-k SAEs at scale.
