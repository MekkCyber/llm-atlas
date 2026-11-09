# Sparse autoencoder (SAE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A sparse autoencoder trained on a model's hidden activations learns an over-complete *dictionary* of features such that each activation is a sparse combination of dictionary atoms. SAEs are the main mech-interp tool for turning dense, polysemantic neurons into (mostly) monosemantic features that correspond to human-interpretable concepts — "San Francisco," "contradiction," "deception-related reasoning," etc. Prereq for most modern interpretability papers.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** none

---

## What it is

Given a frozen LLM and a stream of activations $x \in \mathbb{R}^d$ at some layer (residual stream, MLP output, attention output), an SAE is a shallow autoencoder:

$$z = \sigma(W_{\text{enc}} x + b_{\text{enc}}), \quad \hat x = W_{\text{dec}} z + b_{\text{dec}}$$

with $z \in \mathbb{R}^{Kd}$ for some expansion factor $K$ (typically 8–64), and a training loss that balances reconstruction with sparsity:

$$\mathcal{L} = \|x - \hat x\|_2^2 + \lambda \cdot \text{sparsity}(z)$$

The sparsity term pushes most entries of $z$ to zero on any given input, so each $x$ decomposes into a small set of active atoms (features). Rows of $W_{\text{dec}}$ are the learned feature directions.

## How it works

Variants differ mainly in the sparsity formulation:

- **L1-SAE** (Anthropic 2023, Bricken et al.): sparsity is $\lambda \|z\|_1$; the activation $\sigma$ is ReLU.
- **TopK-SAE** (Gao et al., OpenAI): select top-$K$ entries of the pre-activation per sample; others are zeroed. No L1 penalty, so no shrinkage bias.
- **Jump-ReLU / gated SAEs** — decouple "is this feature active" from "how strongly" to reduce shrinkage.

Training:
- Dataset: activations collected from passing a large text corpus through the frozen model.
- A **resampling** step revives dead atoms (features that never activate).
- Evaluation: reconstruction loss, L0 (average active features per sample), and manual interpretability of top activations.

## Why it matters

- **Superposition.** Neurons in LLMs are polysemantic because the network packs more features than it has dimensions. SAEs *unpack* those features into a wider, sparser basis.
- **Enables feature-level analysis.** Once features are identified, you can steer them (activation patching), monitor them (safety probing), or ablate them (causal interventions).
- **Downstream papers depend on it.** SAE features are the units of analysis in most 2024–2026 mech-interp work; MoE-expert attribution (2611.08089) is one recent example.

## Gotchas & tricks

- **Dead features.** A large fraction of atoms never activate; resampling is essential. Ignore anyone who reports "n features" without reporting live-feature count.
- **Feature interpretability degrades at higher layers.** Mid-layer residual stream is the usual sweet spot; attention output and MLP output are harder to interpret.
- **Splitting and merging.** SAEs trained at different expansion factors reveal different granularity — a single "emotion" feature at $K=8$ often splits into "anger / sadness / fear" at $K=64$.
- Reconstruction loss alone is not enough; evaluate feature quality by hand or with automated interpretability pipelines.

## Sources

- Paper: Towards Monosemanticity: Decomposing Language Models With Dictionary Learning — Bricken et al., 2023 — https://transformer-circuits.pub/2023/monosemantic-features
- Paper: Scaling and evaluating sparse autoencoders — Gao et al., OpenAI, 2024 — https://arxiv.org/abs/2406.04093
- Paper: SAE Interprets MoE — Zhuo et al., 2026 — [arXiv:2611.08089](https://arxiv.org/abs/2611.08089)
