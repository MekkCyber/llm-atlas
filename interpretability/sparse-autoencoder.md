# Sparse Autoencoder (SAE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A wide, sparse, overcomplete autoencoder trained on an LLM's internal activations. Its hidden units are hypothesized to **align one-feature-per-direction** with human-interpretable concepts (monosemanticity), providing a basis in which the model's forward pass can be read as a sum of named features. The standard primitive for 2023–2026 mechanistic-interp work on frontier LLMs; still being characterized — e.g., SAE Additivity Residuals (SAR) now provides a direct way to measure how compositional learned SAE coordinates actually are.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Pick a transformer activation site $h \in \mathbb{R}^d$ (post-attention residual stream, MLP output, etc.) and train an autoencoder

$$
z = \mathrm{ReLU}(W_{\text{enc}} h + b_{\text{enc}}) \in \mathbb{R}^D, \quad \hat{h} = W_{\text{dec}} z + b_{\text{dec}}
$$

with $D \gg d$ (overcomplete) and a sparsity penalty on $z$:

$$
\mathcal{L} = \|h - \hat{h}\|_2^2 + \lambda \|z\|_1
$$

(or an $L_0$ surrogate / top-$k$ sparsity in modern variants). The hypothesis: when $D$ is large enough and $\lambda$ is tuned, each row of $W_{\text{dec}}$ ends up pointing in the direction of a single interpretable *feature* (a concept, a syntactic role, a behavior), and the activation $h$ decomposes as a sparse sum $h \approx \sum_i z_i \cdot W_{\text{dec}, i}$.

---

## How it works

1. **Collect activations.** Run the base model over a large text corpus and dump $h$ from one site per token. For frontier LLMs this is on the order of billions of activation samples.
2. **Train the SAE.** Standard reconstruction + sparsity loss. Variants:
   - L1 penalty with learned feature biases ("gated" SAEs).
   - Top-$k$ activations instead of L1 (Anthropic 2024).
   - Jump-ReLU for cleaner thresholding.
3. **Interpret features.** For each column of $W_{\text{dec}}$, find activating examples (tokens where $z_i$ is high), inspect them, label the feature.
4. **Use the features.** Three downstream patterns:
   - **Probing** — does feature $i$ fire on tokens matching a predicate?
   - **Steering** — add $\alpha \cdot W_{\text{dec}, i}$ to the residual stream to amplify or suppress behavior $i$.
   - **Circuit analysis** — compose features across layers to trace the computation of a specific capability.

---

## Why it matters

- **Standard interp primitive.** From Anthropic's *Towards Monosemanticity* through 2026 (Gemma-Scope, GPT-4 SAE suites, open-weight SAEs for Llama/Qwen), SAEs are the default way to extract an interpretable basis from a frontier LLM.
- **Steering works.** Activation-steering experiments using SAE directions produce reproducible behavior changes (reduced sycophancy, suppressed refusal, elevated honesty-adjacent features). This is strong operational evidence that the directions carry meaning.
- **Compositionality is now measurable.** SAE Additivity Residuals (Mei & Sala 2026) quantify how much behavioral variance is explained by sums vs higher-order interactions of SAE features. In 18 model/behavior settings, learned SAEs have *smaller* first-order residuals than sum-preserving random rotations of the same subspace — evidence that SAE coordinates are more compositional than a random basis change, though pairwise terms still improve rank fidelity (0.5874 → 0.7790 in a combined-steering experiment).

---

## Gotchas & tricks

- **Monosemanticity is a hypothesis, not a guarantee.** Many SAE features are polysemantic in practice; manual labeling catches this.
- **Width matters.** Too-narrow SAEs ($D \sim d$) collapse multiple concepts per unit. Too-wide SAEs waste capacity and fragment features into near-duplicates.
- **Dead features.** Units that never activate after some training step. Standard mitigations: resample dead units from reconstruction errors, use top-$k$ instead of L1.
- **Reconstruction loss is a proxy, not the thing.** Low reconstruction loss doesn't imply interpretable features — tuning $\lambda$ against the auto-interp task (Open-ended Automated Interpretability) is more informative.
- **Site matters.** SAEs on the residual stream, MLP output, and attention output extract different features — residual-stream SAEs are the common default.
- **Compositionality is partial.** Expect sums of SAE directions to recover most behavior but not all; higher-order interactions matter for cases where features compose non-linearly.

---

## Sources

- Paper: *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* — Anthropic, 2023 — the first large-scale SAE on an LLM.
- Paper: *Scaling Monosemanticity* — Anthropic, 2024 — SAEs on Claude 3 Sonnet.
- Paper: *Gemma Scope* — DeepMind, 2024 — open-weight SAE suite on Gemma 2.
- Paper: *Behavioral Compositionality of Sparse Autoencoder Representations* — Mei & Sala, UW–Madison, 2026 — arXiv [2610.21261](https://arxiv.org/abs/2610.21261) — introduces SAE Additivity Residuals.
