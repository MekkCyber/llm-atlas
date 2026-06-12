# ICA Lens
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use **Independent Component Analysis (ICA)** — a classical, non-trained decomposition — to find interpretable directions in LLM activation space, instead of training a sparse autoencoder (SAE). ICA assumes the observed activations are a linear mixture of statistically independent latent sources, and recovers the unmixing matrix via maximum non-Gaussianity. The "ICA Lens" paper packages ICA with **GPU-parallel fitting** and a systematic evaluation harness, then shows it recovers interpretable directions across GPT-2 Small, Gemma 2 2B, and Qwen 3.5 2B Base — competitive with SAEs on standard interpretability benchmarks, with no separate dictionary training.

**Prereqs:** [README.md](README.md)
**Related:** [README.md](README.md)

---

## What it is

The dominant interpretability tool in 2024–2026 has been the **sparse autoencoder**: train a small encoder/decoder pair on LLM activations with a sparsity penalty, and read off interpretable features from the decoder columns. SAEs work — but they require their own training run, their own hyperparameters (dictionary size, sparsity penalty), and their own loss-curve debugging. A persistent question is whether the SAE's dictionary actually finds the model's features or just *some* sparse basis that fits the activations.

ICA Lens revisits a classical alternative. ICA assumes the activation vector $x \in \mathbb{R}^d$ is a linear mixture $x = A s$ of statistically independent latent sources $s$. The goal is to recover the unmixing matrix $W = A^{-1}$ so $s = W x$ extracts the sources from the activations. Unlike PCA (which finds directions of maximum *variance*), ICA finds directions of maximum *non-Gaussianity* — typically heavy-tailed, often interpretable as semantic features.

The pitch: ICA is **convex / deterministic**, has well-understood convergence, and doesn't need a separate training run.

---

## How it works

### The decomposition

Given a matrix of activations $X \in \mathbb{R}^{n \times d}$ sampled from a layer (or residual stream) of an LLM:

1. **Whiten** $X$ (center, decorrelate, scale to unit covariance).
2. **Find rotations** of the whitened space that maximize non-Gaussianity of the projected coordinates. Standard solvers: FastICA, InfoMax.
3. **Read off components** — each row of the unmixing matrix is an interpretable direction in activation space; each column of the mixing matrix is the corresponding "feature."

The non-Gaussianity criterion is what makes ICA find *features* rather than just orthogonal axes: by central-limit-theorem reasoning, mixtures of independent sources tend toward Gaussian, so isolating non-Gaussian projections recovers the original sources.

### GPU-parallel fitting

Classical ICA solvers are CPU-bound. ICA Lens reimplements FastICA on GPU using batched matrix ops over the LLM's full activation tensor — fitting takes minutes per layer instead of hours, making the method practical at LLM-activation scale.

### Evaluation tooling

ICA Lens ships a systematic evaluation harness — feature-purity metrics, downstream-probe accuracy, and concept-level probes — so ICA directions can be benchmarked against SAE features on the same activations.

---

## Why it matters

- **No training run.** ICA is a fixed-point iteration with a clear convergence criterion. No loss curves to babysit, no dictionary-size sweep.
- **Deterministic and reproducible.** Same activations → same components (up to permutation and sign). SAEs depend on initialization, optimizer, and stopping.
- **Stronger baseline for SAE work.** Previous "SAEs find features" claims often compared against PCA, which is a weak baseline (variance, not non-Gaussianity). ICA is a stronger reference point.
- **Cheap to deploy.** Once GPU-fitted, applying ICA at inference is one matrix multiplication. The interp pipeline simplifies dramatically.

---

## Gotchas & tricks

- **Linearity assumption.** ICA assumes a linear mixing model. The "features-as-linear-directions" hypothesis (the superposition view) makes this reasonable for LLM residual streams, but it's an assumption — non-linearly mixed features won't be recovered.
- **Component count = activation dim.** ICA produces at most $d$ components (the activation dimension). SAEs can use much larger overcomplete dictionaries (e.g. $16d$). For models where features genuinely outnumber dimensions, ICA caps short of SAE coverage.
- **Permutation indeterminacy.** ICA components come out in arbitrary order with arbitrary signs. You need a downstream stability step (e.g. matching components across runs by correlation) for reproducibility across models or layers.
- **Whitening can wash out small features.** The pre-whitening step normalizes variance, which can suppress rare but interpretable features. Per-layer tuning helps.
- **Doesn't give you sparsity.** SAEs by construction produce sparse activations; ICA produces dense ones. If your downstream use case requires sparsity (sparse probes, sparse circuit analysis), ICA isn't a drop-in replacement.

---

## Sources

- Paper: *ICA Lens: Interpreting Language Models Without Training Another Dictionary* — anon. authors, 2026 — [arXiv 2606.11722](https://arxiv.org/abs/2606.11722).
- Classical: Hyvärinen & Oja, *Independent Component Analysis: Algorithms and Applications*, 2000 — the FastICA reference algorithm.
- Concept: Sparse autoencoders for interpretability — see [README.md](README.md) for context.
