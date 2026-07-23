# Diffusion-LM latent time probing (Subliminal Clocks)
*Depth — probing and steering the implicit denoising-timestep signal in diffusion language models.*

**TL;DR:** Diffusion language models (DLMs) are not explicitly conditioned on a timestep, unlike image diffusion models. Yet they encode a **latent representation of denoising progress** in their residual streams, extractable by linear probes across layers. Steering along a low-dimensional subspace associated with the inferred timestep produces **predictable, monotone shifts in generation confidence and entropy** — a clean causal demonstration that the signal is real and used downstream.

**Prereqs:** *(none)*
**Related:** [jacobian-lens.md](./jacobian-lens.md), [README.md](./README.md)

---

## What it is

Diffusion image models are given the noise level `t` at each step; DLMs are usually not. That raises a natural question: do DLMs *reinvent* an internal notion of denoising progress? The paper answers yes, and shows it's linearly decodable and low-dimensional — a "subliminal clock" the model reads even though no one hands it one.

## How it works

1. **Linear probe for timestep.** Fit a linear map from residual-stream activations at each layer to the true (training-time) timestep `t`. Report probe accuracy across layers.
2. **Subspace identification.** SVD of the probe's weight matrix isolates the low-dimensional subspace along which timestep is encoded.
3. **Activation steering.** During generation, add a scaled vector along the identified subspace to the residual stream, biasing the model toward an inferred earlier/later timestep than the actual one.
4. **Measure downstream effects.** Steering along the subspace produces **monotone changes in per-token entropy and confidence**, matching the qualitative behavior of a model that thinks it's at a different denoising step than it actually is.

The paper also analyzes the geometry of the identified subspace and finds structured, interpretable properties in activation space.

## Why it matters

- **First clean interp result on diffusion LMs.** The class of models keeps threatening to displace AR LLMs for generation, but had no substantive mech-interp yet.
- **Direct transfer of AR-LLM interp tooling.** Linear probing + activation steering, both born on autoregressive models, work here with minimal modification.
- **A lever for controllable generation.** If entropy and confidence are steerable via a known subspace, the same mechanism enables inference-time diversity/temperature control without touching sampling hyperparameters.

## Gotchas & tricks

- Probe accuracy varies by layer — the useful subspace lives in a specific middle range, not everywhere.
- The steering vector is directional; scale matters — too large and generation degrades outside the "timestep confusion" regime the paper studies.
- Whether the timestep signal is *causally required* for the model to denoise, or merely a byproduct that steering happens to affect, is a subtler question the paper touches but doesn't fully resolve.

## Sources

- Paper: *Subliminal Clocks: Latent Time Modelling in Diffusion Language Models* — Rulli et al., 2026 — [arXiv:2607.01774](https://arxiv.org/abs/2607.01774)
