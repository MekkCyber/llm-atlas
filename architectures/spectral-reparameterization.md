# Spectral Reparameterization for Linear-Attention Long-Context Extension
*Depth — reshaping the transition-matrix decay spectrum at init to extend the context of gated linear-attention models (Gated DeltaNet).*

**TL;DR:** Extending the context of a linear-attention model by continued pretraining alone systematically underuses the layer — its state-transition matrix decays at rates that don't match the new dependency lengths. **SpectralShift** (Liu et al. 2026) identifies two spectral conditions for long-range retrieval: a **sufficiently broad slow spectral band** aligned with the target dependency length, and **preservation of fast-decaying modes** for state clearing / context switching. It reparameterizes the α-projection initialization to shift the decay spectrum and adds a per-α learning-rate scaling. Demonstrated on Gated DeltaNet.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [README](README.md) · [multi-head-attention](multi-head-attention.md)

---

## What it is

Linear-attention and SSM layers (Mamba, Gated DeltaNet) represent long-range dependency through a **state-transition matrix** whose eigenvalues determine how the layer decays past information. The set of eigenvalues is the *decay spectrum*:

- **Slow modes** (eigenvalues near 1) preserve information over many tokens — necessary for long-range retrieval.
- **Fast modes** (small eigenvalues) forget quickly — necessary for state clearing between contexts.

At pretraining time these spectra are set by the initialization of the α-projection (the layer's gating parameter). When the base context is short (e.g. 4k), the spectrum is calibrated for 4k-scale dependencies. **Continued pretraining does not retune the spectrum**; it re-optimizes weights while leaving the eigenvalue distribution near its initial values, so a 4k-trained GDN extended to 128k underuses its layers.

## How it works

SpectralShift makes two changes to continued pretraining:

1. **Reparameterize α initialization.** Adjust the α-projection init so the *slow band* covers the target dependency length, while preserving *fast-decaying modes* for context switching. Concretely: shift the distribution of transition eigenvalues toward 1 by a fixed reparameterization — a change of coordinates, not new parameters.
2. **Scale the learning rate for α.** The α projection is treated as its own optimization group with a scaled LR, letting the spectrum move during training even when the rest of the layer moves slowly.

The rest of the model (attention heads, FFN, embeddings) is trained normally.

## Why it matters

- **Cheap fix for a real ceiling.** Linear-attention / SSM layers are increasingly used in hybrid stacks (Mamba/DeltaNet + attention) as a cheap long-context path. SpectralShift is a near-free intervention at continued-pretraining time.
- **Right frame for extension.** Treats long-context extension as a *spectral* problem, not a data-scaling one. The frame generalizes across linear-attention layers even where the specific reparameterization does not.
- **Consistent long-context gains.** Reported to consistently improve long-context capabilities over standard extension throughout training on Gated DeltaNet.

## Gotchas & tricks

- **Don't destroy the fast band.** Shifting all eigenvalues toward 1 breaks state clearing. The reparameterization *preserves* fast modes — verify empirically after applying by checking eigenvalue histograms per head.
- **Layer-specific.** Derived for Gated DeltaNet's transition structure. Porting to Mamba / RWKV / GLA requires re-deriving the spectral map — the *principle* transfers, the *formula* does not.
- **LR scaling is coupled with the reparameterization.** Skipping the LR scaling leaves the α-projection stuck at init; skipping the reparameterization leaves the base LR too small to move the spectrum meaningfully.
- **Not a replacement for RoPE-style extrapolation** in attention layers. Applies to the linear-attention / SSM portion of hybrid stacks only.

## Sources

- Paper: *SpectralShift: Effective Context Window Extension of Gated DeltaNet via Spectral Reparameterization* — Liu, Hu, Dong, Xie, Zhao, Ding, Tao, Dai, Renmin University of China, 2026, arXiv 2609.14320.
- Related: *Gated Delta Networks* — Yang et al., 2024 (base architecture).
