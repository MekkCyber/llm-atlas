# Sparse Autoencoders (SAEs)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A wide, single-hidden-layer autoencoder trained to **reconstruct a transformer's hidden states from a sparse code**. The encoder projects $d$-dim activations into a much wider $D$-dim feature space (often $D \gg 16d$) with an L1 / TopK / JumpReLU sparsity constraint; the decoder reconstructs the activation as a sparse combination of feature directions. Yields a **dictionary of approximately monosemantic features** — empirically more interpretable than individual neurons. Used for feature discovery, circuit analysis, and increasingly as the substrate for **steering** (manipulate latents rather than raw activations).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [activation-steering.md](activation-steering.md)

---

## What it is

Transformer neurons are **polysemantic**: a single neuron lights up for many unrelated concepts because the model is in the *superposition* regime — encoding more features than it has dimensions. The SAE hypothesis is that the model represents $D$ semi-discrete features in $d$-dim activations as **sparsely active linear combinations**. Train a model that **decodes the features back out** of the activations and you get a labelable basis.

Architecture: $h \in \mathbb{R}^d \mapsto z = \sigma(W_e h + b_e) \in \mathbb{R}^D \mapsto \hat h = W_d z + b_d$. The encoder is a single matrix + nonlinearity; the decoder is a single matrix. The columns of $W_d$ are the **feature directions** — one per latent.

## How it works

Loss: reconstruction MSE + sparsity penalty.
$$
\mathcal{L} = \|h - \hat h\|_2^2 + \lambda \cdot \mathrm{sparsity}(z)
$$
Variants of the sparsity term:
- **L1** (Anthropic 2023): $\lambda \|z\|_1$. Simple, biased toward small activations.
- **TopK**: keep top-$k$ latents per token, zero the rest. Removes the L1 shrinkage bias.
- **JumpReLU** / Gated SAE: thresholded ReLU with a learnable per-feature threshold. Sparser at matched reconstruction.

Train on activations sampled from a frozen base model — typically the residual stream after a chosen layer, or the MLP output. $D$ is usually $8 \times d$ to $64 \times d$. Training is one-shot (no curriculum) but expensive — wide $D$, billions of tokens.

After training, **interpret** by: showing the top-activating examples for each feature, ablating the feature and measuring downstream change, or correlating features with known concepts via a learned probe.

**SAE-based steering** (Whisper-hallucination paper, Cunningham et al., others): identify a feature aligned with an unwanted behavior, then **zero or subtract** that latent's contribution to $\hat h$ before continuing the forward pass. More targeted than raw-activation steering — only the relevant feature direction is touched.

## Why it matters

- **Polysemantic neurons are the main blocker for mechanistic interpretability.** SAEs convert "what does neuron 4729 do?" into "what does feature 18,432 do?" — and the answer is usually a single concept.
- **Steering becomes feature-level.** The Whisper paper shows SAE-targeted steering can drop ASR hallucination rate from 86.9% to 27.3% with minimal accuracy loss — far cleaner than additive raw-activation steering.
- **Cross-modal applicability.** The technique transfers from text LLMs (Anthropic, OpenAI) to ASR (Whisper), vision, and multimodal models — same recipe, different activations.

## Gotchas & tricks

- **Dead features.** A large fraction of $D$ latents end up never firing. Common fixes: re-initialize dead features to high-loss examples (Anthropic's "resampling"), or TopK / JumpReLU which avoid the L1 shrinkage trap.
- **Reconstruction-vs-sparsity tradeoff.** Sparser codes lose reconstruction quality; better reconstruction takes more active features. The Pareto frontier is the right metric, not a single point.
- **Feature splitting / merging.** Same real-world concept can be split across multiple SAE features (Asia + East Asia + China), or several concepts can be merged into one feature. The dictionary is not canonical.
- **Cross-checkpoint instability.** SAEs trained on different checkpoints of the same model can produce different feature partitions — comparison across training steps requires alignment.
- **Layer choice.** Residual stream vs MLP output vs attention output. Residual stream is the canonical default since it accumulates everything.

## Sources

- Paper: *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* — Anthropic Bricken et al., 2023 — the modern SAE-on-LLM template.
- Paper: *Scaling Monosemanticity* — Anthropic Templeton et al., 2024 — frontier-scale SAEs on Claude 3 Sonnet.
- Paper: *Scaling and Evaluating Sparse Autoencoders* — Gao et al., OpenAI, 2024 — TopK SAEs.
- Paper: *Whisper Hallucination Detection and Mitigation via Hidden Representation Steering and Sparse AutoEncoders* — Popov et al., 2026 — arXiv 2606.07473 — SAE steering on ASR.
