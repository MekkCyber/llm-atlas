# Sparse autoencoders (SAEs)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a wide, overcomplete autoencoder on the **activations** of a layer in a frozen target model, with an L1 (or top-k) penalty that forces each example to activate only a few hidden units. The learned units — "features" — tend to be more **monosemantic** than the raw residual-stream directions and can be auto-labeled with natural-language concepts. SAEs are the current workhorse of mechanistic interpretability and are starting to appear as deployment-time audit layers on top of production classifiers.

**Prereqs:** [README.md](README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../safety/README.md](../safety/README.md) — SAEs are a key audit mechanism for safety-critical deployments.

---

## What it is

Transformer activations are **superposed**: a single residual-stream direction mixes many unrelated concepts because the model has more features than dimensions. An SAE is a learned decomposition that "spreads out" these mixtures into a wider, sparser space where (ideally) each hidden unit corresponds to one human-interpretable concept.

## How it works

Given a frozen target model $M$, pick an activation site (e.g., residual stream at layer $\ell$). For a batch of activations $x \in \mathbb{R}^d$:

- **Encoder:** $z = \mathrm{ReLU}(W_\text{enc}\, x + b_\text{enc})$, with $z \in \mathbb{R}^D$ and $D \gg d$ (e.g., $D = 16 d$ to $128 d$).
- **Decoder:** $\hat{x} = W_\text{dec}\, z + b_\text{dec}$.
- **Loss:** $\|x - \hat{x}\|_2^2 + \lambda \|z\|_1$ — reconstruction plus sparsity. Modern variants replace L1 with **top-k** (keep only the $k$ largest activations per example) which gives cleaner features and avoids L1-shrinkage pathologies.

Train the SAE on activations collected from running $M$ over a large text corpus. The target model is **not** updated.

### Reading features

- Each hidden unit $i$ has a **decoder direction** $W_\text{dec}[:, i]$ in the model's activation space.
- Find top-activating examples for feature $i$ in the corpus; summarize what they share. This is the raw material for *auto-labeling* features — the common pattern is to feed top-activating examples to a stronger LLM and keep only labels whose predicted activation pattern matches the observed one above some fidelity threshold.
- Features can be *steered* by directly adding $\alpha \cdot W_\text{dec}[:, i]$ to the residual stream at generation time — a cheap intervention.

## Why it matters

- **Monosemanticity, mostly.** Compared to raw neurons or residual-stream directions, SAE features are far more likely to correspond to a single human concept — the empirical finding that launched the current wave of mech-interp work (Anthropic 2023, OpenAI 2024, Google DeepMind 2024).
- **Scales to frontier models.** SAEs have been successfully trained on Claude, GPT-4, and open models, with feature counts in the millions.
- **Deployment lever.** Beyond pure research, SAE features can replace opaque classifier heads with auditable, concept-level classifiers — the pattern used in auditable AI-text detection and similar high-stakes decisions.

## Gotchas & tricks

- **Dead features.** Many hidden units never fire; monitor and resample them. Top-k SAEs are more robust to this than L1 SAEs.
- **Width is a dial.** Wider SAEs find more fine-grained features but take more compute and more features to interpret. There is no principled stopping point; "enough" is set by the downstream task.
- **Reconstruction is not understanding.** A low reconstruction loss means activations can be reassembled, not that each feature is semantically clean. Monosemanticity has to be checked separately (via feature labeling and probing).
- **Features are site-specific.** An SAE trained on layer 12 says nothing about layer 20. Multi-layer analysis requires one SAE per site, or a shared cross-layer SAE (still a research topic).
- **Not causal by default.** A feature correlated with concept $C$ is not necessarily *used* for $C$ downstream. Causal validation needs activation patching or intervention experiments.

## Sources

- Paper: *Toy Models of Superposition* — Elhage et al., Anthropic, 2022.
- Paper: *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* — Templeton et al., Anthropic, 2024.
- Paper: *Scaling and Evaluating Sparse Autoencoders* — Gao et al., OpenAI, 2024 — top-k SAEs.
- Paper: *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders* — Rajamanoharan et al., Google DeepMind, 2024.
- Paper: *Building Auditable AI Text Detectors with Sparse Autoencoders* — 2026 — https://arxiv.org/abs/2610.30006 — SAEs as a deployment audit layer.
