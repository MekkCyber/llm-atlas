# Interpretability Lenses
*Taxonomy — ways to read a transformer's hidden state in token-space.*

**TL;DR:** A **lens** maps an internal hidden state $h^{(\ell)}_t$ to a distribution over vocabulary tokens. Lenses let researchers read "what the model is thinking" at intermediate layers. They differ in whether they are **fixed** (same mapping everywhere), **learned per layer** (one small probe per layer), or **model-native** (reusing the model's own downstream layers). Fixed lenses are cheapest; learned lenses are sharpest at the layers they were fit on; model-native lenses generalize across checkpoints and families without retraining.

**Related taxonomies:** _(none yet)_
**Depth files covered here:** [lexical-reference-frames](lexical-reference-frames.md)

---

## The problem

A transformer's residual stream carries information that is only *partially* in output frame at any given layer. At the last layer, the unembedding reads it directly. At an intermediate layer, the state is in some other representation that the output head cannot read correctly. Interpretability research needs a way to translate those intermediate states into token-space, both for inspection ("what is the model computing here?") and for intervention ("if I patch this state, what does the model think?").

## The shared pattern

All lenses:

1. Take a hidden state $h^{(\ell)}_t$ at some layer $\ell$, position $t$.
2. Apply a transformation $T$ to lift it into output-frame.
3. Multiply by the unembedding (or an equivalent) to get a vocabulary distribution.

What changes is what $T$ is: zero, a learned linear map, a learned mini-model, or the model's own remaining layers.

## Variants

| Technique | What $T$ is | Trained? | Context-aware? | When it wins |
| --- | --- | --- | --- | --- |
| **Logit Lens** | Identity (apply final LayerNorm + unembed directly) | No | No | Cheap sanity check at late layers |
| **Tuned Lens** | Learned affine per layer | Yes (one small probe per layer) | No | Mid-depth readings on a specific model |
| **Linear Probes** | Learned linear classifier for a target feature | Yes | No | Targeted feature detection |
| **[LexFrame\[k\]](lexical-reference-frames.md)** | Model's own layers $\ell\!+\!1\ldots L$, with $k$ extra attention rounds | No (parameter-free) | **Yes** (attention over positions) | Mid-depth readings, cross-checkpoint, cross-family |
| **Activation Patching Lens** | Replace one state and read the output head | No | Yes (full forward) | Causal attribution, not representation inspection |

## How to choose

- **Quick sanity check or late-layer reading** → Logit lens. Zero parameters, zero training, works where it works.
- **Need mid-depth readings on a known fixed model** → Tuned lens. Sharper than logit lens at the layers you fit, but you must refit per checkpoint.
- **Need mid-depth readings that generalize across checkpoints, scales, or families** → LexFrame[k] ([lexical-reference-frames.md](lexical-reference-frames.md)). Parameter-free; k=1 is the recommended default.
- **Target a specific feature (binary classification task)** → Linear probe. Simple, but needs labeled data and tells you about linear separability, not representation.
- **Need a causal statement about what a state computes** → Activation patching. Lenses alone give correlational readings.

## Adjacent but distinct

- **Sparse autoencoders ([sparse-autoencoder.md](sparse-autoencoder.md))** — decompose a hidden state into named features. Complements lenses: SAE tells you *what features are active*, lenses tell you *what the state looks like to the model's reader*.
- **Logit attribution / direct logit attribution (DLA)** — attributes the logit for a specific token back to individual layer contributions. A different direction (per-token attribution, not per-state readout).
- **Circuit discovery** — identifies which heads/neurons participate in a computation. Operates on causal graphs, not on single-state readouts.

## Sources

- Logit Lens — nostalgebraist, 2020 (blog post).
- Tuned Lens — Belrose, Furman, Smith, et al., 2023 — [arXiv:2303.08112](https://arxiv.org/abs/2303.08112).
- LexFrame — Anonymous, 2026 — [arXiv:2610.22169](https://arxiv.org/abs/2610.22169).
- Activation patching for lenses — Meng et al., ROME (2022).

---

## Conventions

- **Filename:** `_lenses.md` (leading underscore — taxonomy).
- **Folder placement:** `interpretability/` root, alongside the depth files it covers.
- **Scope:** techniques that produce a vocabulary-space readout from an internal state. Causal attribution and feature decomposition belong elsewhere.
