# Sparse Autoencoder (SAE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **wide, sparse dictionary** fit to transformer activations. Train an encoder $f$ and decoder $g$ so that for each activation $a$, $g(f(a)) \approx a$ with $f(a)$ sparse (most entries zero). Each learned dictionary direction tends to activate on a human-interpretable concept — a sparse code where each coordinate is approximately *monosemantic*. SAEs underpin modern mechanistic interpretability and increasingly serve as a general representation layer for steering, probing, and cross-model matching.

**Prereqs:** [attention.md](../fundamentals/attention.md)
**Related:** [lexical-reference-frames.md](lexical-reference-frames.md) · [sae-idf.md](sae-idf.md)

---

## What it is

Transformer activations are **polysemantic**: a single neuron responds to many unrelated concepts because the model packs more features than it has dimensions ("superposition"). A sparse autoencoder decomposes those activations into a wider, sparser code where each coordinate corresponds to roughly one concept.

Formally: given activations $a \in \mathbb{R}^d$ collected from some layer (residual stream, MLP output, attention output), train
- **encoder** $f : \mathbb{R}^d \to \mathbb{R}^D$ with $D \gg d$ (typical: $D = 8d$ to $64d$),
- **decoder** $g : \mathbb{R}^D \to \mathbb{R}^d$,
- a **sparsity-inducing loss** so that $f(a)$ has few active entries.

The learned columns of the decoder are the "features." Each feature can be labeled by looking at the top-activating examples in a reference corpus.

## How it works

**Loss.**

$$
L(a) = \| g(f(a)) - a \|_2^2 \;+\; \lambda \cdot \|f(a)\|_1 \qquad \text{(TopK variant replaces the L1 with a hard sparsity constraint)}
$$

The L1 encourages sparsity but introduces shrinkage; **TopK SAEs** instead keep only the top $K$ activations per input, which avoids the shrinkage and improves reconstruction at a given sparsity budget. Gated SAEs separate the "which features fire" decision from the "how much" decision for similar reasons.

**Training.** Activations are collected offline from a frozen target model, then the SAE is trained on them. Dictionary width $D$ and sparsity $K$ are the main hyperparameters. Dead features (never active) are periodically re-initialized.

**Reading.** For a new input, $f(a)$ gives a sparse code; non-zero coordinates name the active concepts. Labels for each coordinate come from inspecting the texts that most strongly activate it in a reference corpus.

## Why it matters

- **Approximate monosemanticity.** SAE features are easier to label than raw neurons: a given feature typically has a crisp description ("French prepositions", "code variable called `self`", "the Golden Gate Bridge").
- **Steering handle.** Add a feature direction to the residual stream and the model's output shifts predictably.
- **Shared coordinate system.** Because features have readable meaning, SAE codes are a natural bridge between models or symbolic systems (see [sae-idf](sae-idf.md)).
- **Downstream tasks.** Entity linking, concept matching, probing — SAE codes outperform plain hidden states on tasks where interpretability of the match matters.

## Gotchas & tricks

- **Features drift with training.** An SAE fit on checkpoint $t$ does not directly transfer to checkpoint $t+1$ — refit or use a "crosscoder" that spans checkpoints.
- **TopK beats L1 at matched sparsity.** The L1 is historical; most 2025 recipes use TopK or Gated SAEs.
- **Dead features are common.** Expect to re-init 1–10% of features periodically during training.
- **Reconstruction isn't everything.** A low-$L_2$ SAE with uninterpretable features is useless; always evaluate feature interpretability on top-activating examples.
- **Not a causal explanation.** Features correlate with concepts; proving a feature *causes* a behavior needs ablation / patching experiments.

## Sources

- Paper: *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* — Bricken et al., Anthropic, 2023.
- Paper: *Sparse Autoencoders Find Highly Interpretable Features in Language Models* — Cunningham et al., 2023.
- Paper: *Scaling and Evaluating Sparse Autoencoders* — Gao et al., OpenAI, 2024 — the TopK SAE paper.
- Paper: *Gated Sparse Autoencoders* — Rajamanoharan et al., DeepMind, 2024.
- Paper: *Sparse Autoencoders Are a Shared Coordinate System for Symbolic Matching* — Meng et al., 2026 — [arXiv:2610.19636](https://arxiv.org/abs/2610.19636) — SAE codes as a representation layer for cross-system matching.
