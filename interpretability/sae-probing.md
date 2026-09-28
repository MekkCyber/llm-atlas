# SAE Probing of Linguistic Categories
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use a well-defined linguistic category (parts-of-speech) as a controlled test case for what Sparse Autoencoder latents actually encode. The finding: morpho-syntactic categories are **highly recoverable** from SAE activations but **not one-to-one** with individual latents — each PoS is supported by a compact **group** of latents, stable on held-out data, with meaningful overlap between related categories.

**Prereqs:** [README.md](README.md)
**Related:** [superposition-linearity.md](superposition-linearity.md)

---

## What it is

The "monosemantic feature" ideal for SAEs promised one latent per interpretable concept. Empirical results have been mixed. This paper introduces a **discipline**: pick a category system with clear, controlled labels (PoS), then measure two things independently:

1. **Recoverability** — can you predict the category from SAE activations?
2. **Alignment** — is there a single latent per category?

Splitting these lets a distributed representation register as a *positive* interp finding rather than a *failed* monosemanticity claim.

## How it works

Given an SAE trained over an LLM's hidden states and a tagged corpus:

1. Run inputs through the LLM and the SAE, collecting per-token SAE activation vectors.
2. For each PoS tag `t`, train a lightweight classifier (linear / small MLP) to predict `t` from SAE activations. High accuracy = high **recoverability**.
3. Report **best single latent** per tag as a baseline. If it approaches the classifier's accuracy, the tag is close to a single latent; if it lags, the tag is *distributed*.
4. Identify the **compact group** of latents that contribute most to the classifier for each `t`; measure group stability on held-out data and group overlap between related tags.

For the paper: PoS distinctions are highly recoverable, single-latent baselines lag substantially, group sizes vary per tag, and the groups remain stable on held-out data. Open-class and closed-class PoS behave measurably differently.

## Why it matters

- Sharpens interp claims: "distributed but structured" becomes a testable, quantifiable property, not a hedge.
- Refutes the strong "atomic feature" reading of SAE latents on a category system where atomicity is *plausible* (PoS is discrete, well-defined) — if atomicity fails here, it fails almost everywhere.
- Points to compact-group as the operative unit of SAE interp, not the single feature.

## Gotchas & tricks

- Recoverability is not causation; a classifier can pick up on correlated latents that are not what actually drives the LLM's decisions.
- The findings depend on SAE hyperparameters (sparsity target, dictionary size); wildly different SAEs may show different atomicity profiles.
- PoS is a favorable test case because tags are near-mutually-exclusive; open-ended categories (semantic roles, discourse relations) may need a different measurement protocol.

## Sources

- Paper: *Parts-of-Speech as Emergent Categories in SAE Latent Space* — Bondielli, Passaro, Auriemma, Lenci — 2026 — [arXiv:2609.29362](https://arxiv.org/abs/2609.29362).
