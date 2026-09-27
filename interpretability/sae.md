# Sparse Autoencoders (SAEs)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A sparse autoencoder trained on a frozen LLM's residual-stream activations decomposes them into a much larger, mostly-off dictionary of **features**. Each feature is a direction in activation space that fires on a narrow slice of inputs, and the sparsity constraint pushes those features toward monosemanticity — one feature, one concept — giving interp researchers a per-token, per-layer readout of what the model is "thinking".

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [README.md](README.md)

---

## What it is

A single-hidden-layer autoencoder `x → h = ReLU(W_enc x + b_enc) → x̂ = W_dec h + b_dec` trained to reconstruct activations `x` sampled from a fixed layer (usually the residual stream) of a frozen LLM. The dictionary `h ∈ ℝ^d` is **much wider than the activation** (`d ≫ dim(x)`, e.g. 32× the model width) and is forced to be sparse — only a handful of latents are nonzero per token.

The point is not compression but **decomposition**: the columns of `W_dec` become a learned overcomplete basis in which activations are (approximately) a sparse combination of interpretable atoms.

## How it works

Loss is reconstruction + a sparsity penalty:
$$\mathcal{L} = \|x - x̂\|_2^2 + \lambda \|h\|_1$$
with a per-feature L1 that (after normalising `W_dec` columns to unit norm) drives most latents to zero on any given token.

Newer variants replace L1 with hard-topk gating (**TopK-SAE**), a JumpReLU with a learned threshold, or a mixture — same idea, better active-set control.

Once trained, you interpret a feature `i` by (a) collecting the tokens that maximally activate it, (b) reading off a text description, and (c) running causal checks — ablation, activation patching, or steering by adding/subtracting `W_dec[:, i]` in the residual stream.

## Why it matters

SAEs are currently the most productive method for turning dense residual streams into per-token labels an interpretability researcher can *look at*. They ground steering vectors, circuit analysis and safety probes (e.g. deception features, refusal features) in objects with concrete geometry, not just intuitions.

They are also cheap relative to the base model: the SAE is small and can be trained after the fact on a fixed activations dump, so labs can ship interpretations of a frozen production checkpoint without touching training.

## Gotchas & tricks

- **Feature ↔ concept is not one-to-one.** *POS as Emergent Categories in SAE Latent Space* (2026) shows that even a canonical category like Part-of-Speech is carried by **compact groups** of latents, not single features — with substantial overlap between related tags. Assume distributed encoding unless you've verified atomicity.
- **Dead features.** A large fraction of latents can end up permanently off; resampling, TopK, or auxiliary losses (**aux-K**) are needed to keep the dictionary alive.
- **L0 and reconstruction fight.** More sparsity → worse reconstruction → the SAE is explaining less of the model. Report both numbers; a "beautiful" 5-feature SAE that reconstructs 30% of activations is not an interpretation.
- **Layer matters.** Early-layer features are lexical/syntactic; late-layer features are semantic/task-shaped. Don't generalise findings across layers.
- **Feature descriptions are hypotheses.** The text labels are LLM-generated interpretations of top-activating examples; they can be wrong or partial. Causal steering / ablation is the ground truth.

## Sources

- Paper: *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* — Bricken et al., Anthropic, 2023 — the modern L1-SAE recipe on a small transformer.
- Paper: *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* — Templeton et al., Anthropic, 2024 — production-scale SAEs.
- Paper: *Sparse Autoencoders Find Highly Interpretable Features in Language Models* — Cunningham et al., 2023, [arXiv 2309.08600](https://arxiv.org/abs/2309.08600) — parallel discovery, GPT-2 residual stream.
- Paper: *Parts-of-Speech as Emergent Categories in SAE Latent Space* — Bondielli, Passaro, Auriemma, Lenci, 2026, [arXiv 2609.29362](https://arxiv.org/abs/2609.29362) — evidence that even simple linguistic categories are distributed over groups of latents.
- Blog: *Sparse Autoencoders and the Feature Zoo* — Anthropic circuits thread.
