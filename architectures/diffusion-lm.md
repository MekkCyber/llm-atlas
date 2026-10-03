# Diffusion Language Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A family of LMs that generates text by iteratively denoising a sequence rather than sampling tokens left-to-right. Two main branches: **discrete** diffusion (operates on tokens through a mask/absorbing or categorical corruption process) and **continuous** diffusion (denoises a continuous latent that is decoded to tokens at the end). Both aim at parallel generation, bidirectional context, and controllable editing — properties autoregressive (AR) LMs don't have cleanly.

**Prereqs:** [transformer-block.md](transformer-block.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [moe-as-latent.md](moe-as-latent.md) · [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Autoregressive LMs factor $p(x_{1:T}) = \prod_t p(x_t | x_{<t})$ and sample one token at a time. Diffusion LMs instead define a forward *corruption* process $q(x^k | x^{k-1})$ that gradually turns clean text into noise/masks, and train a *reverse* network to denoise. At inference you start from pure noise/masks and apply the reverse process for $K$ steps.

Two main regimes:

- **Discrete / masked diffusion LMs (MDMs).** The corruption masks tokens with increasing probability; the reverse model predicts the masked positions. Each step can unmask many positions in parallel. Examples: D3PM, SEDD, MaskGIT (vision), MDLM, LLaDA.
- **Continuous diffusion LMs (CDLMs).** The corruption is Gaussian noise added to the sequence's embedding; the denoiser is trained on standard DDPM/EDM-style losses in embedding space; a decoder (often a frozen T5-family encoder's inverse, learned nearest-neighbor, or an auxiliary softmax head) maps the final latent back to tokens. Examples: Diffusion-LM, SSD-LM, Plaid.
- **Hierarchical / hybrid.** Keep a continuous latent as the persistent generative state but *also* emit discrete tokens at every step that feed back into the next latent update (HC-DLM, 2026). Combines CDLM's dependency structure with MDM's grounded token readout.

## How it works

The common training template:

1. Sample $k \sim \mathrm{Uniform}(1, K)$ and apply the forward corruption to a clean sequence $x_0$ to get $x^k$.
2. Train a transformer to predict either $x_0$ (reconstruction) or the per-step denoising distribution.
3. At inference, start from $x^K$ (all mask or pure Gaussian) and iterate the reverse network $K$ times.

Key design choices:

- **How many tokens to unmask per step.** MDMs trade speed vs quality here: unmask many → fast but position-independent samples; unmask few → close to AR cost.
- **Latent space.** CDLMs are extremely sensitive to the embedding they diffuse in. Stronger teachers help, but overly discriminative teachers push embeddings too far apart and the diffuser can't hit them; soft-label *distillation* of the teacher gives a more connected, "diffusible" latent (2026).
- **Coupling.** HC-DLM makes the latent carry the full trajectory while tokens act as scaffolds, giving a VAE-flavored training objective derived from a variational bound on token likelihood.

## Why it matters

- **Parallel decoding.** $K \ll T$ steps for comparable quality means wall-clock advantages when inference FLOPs are bounded by latency, not throughput.
- **Bidirectional reasoning.** Each denoising step sees the entire partially-filled sequence, which helps on global-constraint tasks (Sudoku, Countdown, structured planning) where AR left-to-right ordering fights the problem shape.
- **Editing-friendly.** Masking a span and running a few reverse steps is a natural infilling / editing primitive.

## Gotchas & tricks

- **MDM few-step quality is weak** because factorized reverse processes sever cross-position dependencies at each step. Mixture-of-experts latents (E-MoE, 2026) and continuous-latent couplings (HC-DLM) are two recent fixes.
- **CDLMs are embedding-bound.** The latent space has to be smooth enough that noisy denoising lands near valid tokens; raw strong encoder embeddings are often too discriminative.
- **Benchmarks lag.** DLMs are compared on LM1B and small reasoning puzzles far more than on OpenWebText perplexity or MMLU, which makes cross-paper comparison noisy.
- **Scaling story is immature.** No DLM has been trained at frontier scale as of late 2026; most interest is still at the ≤1B parameter range.

## Sources

- Paper: *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM) — Austin et al., 2021.
- Paper: *Diffusion-LM Improves Controllable Text Generation* — Li et al., 2022 — early continuous DLM.
- Paper: *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* (SEDD) — Lou et al., 2023.
- Paper: *Hierarchical Continuous Diffusion Language Models* — Ren et al., UIUC, 2026 — [arXiv:2610.02193](https://arxiv.org/abs/2610.02193).
- Paper: *E-MoE: Enhanced Mixture-of-Experts for Non-Factorized Diffusion Language Models* — Ivanov et al., 2026 — [arXiv:2609.37533](https://arxiv.org/abs/2609.37533).
- Paper: *Scaling and Distilling Text Embeddings for Better Diffusibility* — Zhang et al., U. Michigan, 2026 — [arXiv:2610.01016](https://arxiv.org/abs/2610.01016).
