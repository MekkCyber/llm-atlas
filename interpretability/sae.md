# Sparse Autoencoders (SAEs) for Interpretability
*Depth — a dictionary-learning probe for neural-network hidden states.*

**TL;DR:** A **sparse autoencoder** is a one-hidden-layer autoencoder with an overcomplete hidden dim and an L1 (or top-K) sparsity penalty, trained to reconstruct a model's hidden states. The learned hidden units act as a *dictionary of features*: each feature tends to activate on a human-interpretable concept. SAEs have become the default interpretability probe for frontier LLMs because they *scale* — one SAE per residual-stream layer gives millions of interpretable features without hand-engineering.

**Prereqs:** [attention](../fundamentals/attention.md), [transformer-block](../architectures/transformer-block.md)
**Related:** [sae-moe-probing](./sae-moe-probing.md)

---

## What it is

Given hidden states $x \in \mathbb{R}^d$ (typically the residual stream after some layer), train:

$$
f = \mathrm{relu}(W_{\text{enc}} x + b_{\text{enc}}), \qquad \hat x = W_{\text{dec}} f + b_{\text{dec}}
$$

- $W_{\text{enc}} \in \mathbb{R}^{m \times d}$ with $m \gg d$ (typically $m / d \in [8, 64]$).
- Loss $= \lVert x - \hat x \rVert_2^2 + \lambda \lVert f \rVert_1$ (standard "L1 SAE"), or an L0 constraint via **top-K SAE** (keep the $K$ largest $f_i$, zero the rest). Top-K is now preferred because the sparsity level is explicit and shrinkage-free.

Each unit $f_i$ is a *feature*. "Interpret this feature" = pick the top-activating dataset examples and read what they have in common.

## How it works

1. Run the base model on a large corpus; collect residual-stream activations $\{x^{(t)}\}$ at the chosen hook (one SAE per hook).
2. Train the SAE on these activations (frozen base model). Batches of millions of tokens, hours of H100 for a 1B base model.
3. For each feature $i$, gather the top-K samples with the largest $f_i$. Label the feature by inspection or by an auto-interp LLM.
4. Downstream uses: (a) feature-level steering (clamp $f_i$ to a value, decode), (b) attribution (which features drive a behaviour), (c) auditing (search for features that match a concept of concern).

## Why it matters

- **Scalable interpretability.** Pre-SAE methods (probing classifiers, logit lens, ablations) require you to *know what you are looking for*. SAEs produce a dictionary first, labels second. This is the dominant mode of mech-interp in 2025 and used to interpret MoE routing ([sae-moe-probing](./sae-moe-probing.md)), compare attention heads, and build feature-level safety monitors.
- **Monosemanticity, at scale.** Anthropic's Sonnet SAE work showed millions of features, many of which are human-interpretable when the dictionary is large enough and sparse enough.
- **Composable with other techniques.** Features can be fed into activation steering, used as regressors in probing, or used as inputs to CoT monitors.

## Gotchas & tricks

- **Dead features.** A large fraction of SAE units never activate above zero on the training distribution. Modern recipes use "resampling" (re-init dead features from high-loss activations) or top-K SAEs (sparsity is enforced per-sample, no dead units).
- **Shrinkage with L1.** L1 pulls $f$ toward 0, biasing reconstructions. Top-K SAEs eliminate shrinkage but introduce a hard $K$ hyperparameter.
- **Interpretability is not identifiability.** A feature that *looks* monosemantic on top-activations may still polysemantically fire in the long tail. Always check middle- and low-activation regions.
- **Width / sparsity tradeoff.** Wider SAE → finer features but more dead units; sparser SAE → more monosemantic but worse reconstruction. $m/d \approx 16$, $K \approx 50$ is a common starting point for frontier models.
- **One SAE per layer per model.** SAEs are *not* transferable across checkpoints — a small finetune can invalidate the dictionary. Expensive to maintain for a shipped model.

## Sources

- Paper: *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning* — Bricken et al., Anthropic, 2023.
- Paper: *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* — Templeton et al., Anthropic, 2024.
- Paper: *Scaling and evaluating sparse autoencoders* — Gao et al., OpenAI, 2024 — introduces top-K SAEs.
- Paper: *SAE Interprets MoE* — Zhuo et al., ByteDance Seed, 2026 — applies SAE features to probe MoE expert specialization ([sae-moe-probing](./sae-moe-probing.md)).
