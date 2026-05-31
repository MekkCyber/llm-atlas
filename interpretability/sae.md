# Sparse Autoencoders (SAEs)

*Depth — feature decomposition of LLM activations into a large dictionary of sparse, interpretable units.*

**TL;DR:** SAEs train a wide autoencoder over a frozen LLM's residual stream activations with an $L_1$ (or top-$k$) sparsity penalty. The hidden layer is far wider than the model's hidden dimension (typically 8×–64×), but only a small fraction of units fire per token. Each surviving feature is hypothesized to encode one human-interpretable concept (a *monosemantic* feature). Used for inspecting what an LLM has learned, for steering its behavior, and now extended to other embedding spaces (e.g., dense retrievers).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [retrieval-feature-steering.md](retrieval-feature-steering.md)

---

## What it is

An autoencoder trained *on activations*, not on data. Given a frozen LLM and a chosen activation site (residual stream after some layer), collect activations on a large corpus and train:

$$
h(x) = \mathrm{ReLU}(W_{\text{enc}} x + b_{\text{enc}}), \quad \hat{x} = W_{\text{dec}} h + b_{\text{dec}}
$$

with a reconstruction loss + sparsity penalty:

$$
\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|h\|_1
$$

The encoder dimension is much wider than $x$'s — overcomplete by design, because the underlying claim is that the LLM stores many more *features* than its hidden dimension and represents them in superposition.

## How it works

1. **Collect activations.** Run the frozen LLM over a large text corpus; cache the chosen-layer residual-stream activations.
2. **Train the SAE.** $W_{\text{enc}} \in \mathbb{R}^{d \times d'}$, with $d' = 8d$ to $64d$. Adam, $L_1$ penalty on the hidden activations. Reconstruction loss anchors the geometry; sparsity penalty forces only a few features per token.
3. **Interpret features.** For each feature $i$, find the inputs that maximally activate it (top-activating examples), then label by inspection — a feature might encode "Python list comprehension," "second-person pronoun," or "Shakespearean register."
4. **Use the features.** Three main consumers:
   - **Inspection.** Which features fire on which inputs / behaviors.
   - **Steering.** Add or subtract a feature's decoder vector from the residual stream to amplify / suppress the behavior at inference.
   - **Causal analysis.** Patch features in/out across forward passes to test which feature causes which output behavior.

Variants: **top-$k$ SAE** replaces $L_1$ with a hard top-$k$ activation selector per token (cleaner sparsity, less collapse). **Gated SAE** adds a gate that decides which features can fire, separating "is this feature active" from "how much."

## Why it matters

- The most successful approach to *mechanistic interpretability* at scale. Reports from Anthropic (Claude), DeepMind, and academic labs show SAEs producing thousands of nameable features over frontier-class models.
- The discovered features support causal interventions — turning a feature up or down in the residual stream produces predictable behavioral changes. Bridges interpretability with control.
- Extends to non-LLM spaces. Recent work (Xetrieval) applies the same recipe to dense-retriever embeddings, producing per-feature explanations of retrieval decisions.

## Gotchas & tricks

- **Dead features.** A large fraction of SAE units never fire ("dead"). Top-$k$ SAEs sharply reduce dead-feature rates; resampling dead features during training also helps.
- **Feature splitting / merging.** Train at multiple widths $d'$ — a feature at $d' = 8d$ may split into 3 subfeatures at $d' = 64d$. The "right" width depends on what you want to inspect.
- **Polysemantic survivors.** Sparsity alone doesn't guarantee monosemanticity — some features still activate on unrelated concepts. Manual triage on top-activating examples is the diagnostic.
- **Steering ≠ true control.** Adding a feature direction to the residual stream sometimes produces clean behavior changes and sometimes produces incoherence — the LLM "fights back" against arbitrary edits. Calibrate the steering coefficient per feature.
- **Cost.** SAE training on a frontier model's activations is non-trivial — billions of tokens of activations cached, days of training on a wide encoder. Not as cheap as the small architecture suggests.

## Sources

- Paper: *Toy Models of Superposition* — Elhage et al., Anthropic, 2022 — formalizes superposition as the reason SAEs are needed.
- Paper: *Sparse Autoencoders Find Highly Interpretable Features in Language Models* — Cunningham et al., 2023, arXiv 2309.08600 — early SAE-on-LLM result.
- Paper: *Scaling Monosemanticity* — Templeton et al., Anthropic, 2024 — SAEs on Claude 3 Sonnet at very large dictionary sizes; named-feature catalog.
- Paper: *Scaling and evaluating sparse autoencoders (top-k SAEs)* — Gao et al., OpenAI, 2024.
- Paper: *Xetrieval* — 2026 — applies SAEs to dense-retriever embeddings; see [retrieval-feature-steering.md](retrieval-feature-steering.md).
