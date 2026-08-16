# Massive Activations
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A small set of hidden-state dimensions in trained transformers carry activations *orders of magnitude* larger than the rest of the tensor. They matter for quantization (they blow calibration windows), interpretability (they are attention-sink anchors), and architecture design. In **hybrid linear-attention** LLMs they take two architecture-aligned shapes: **Pre-Attention Spikes** immediately before every full-attention layer, and **Inter-Spike Plateaus** that persist across intervening linear-attention layers.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [README.md](README.md)
**Related:** [../architectures/mla.md](../architectures/mla.md), [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

In a trained transformer, look at the hidden state at some layer for some token. A handful of dimensions — typically fewer than 10 out of thousands — hold values 100–1000× the median magnitude. These are **massive activations** (MAs). They're not noise: they're stable across inputs, tied to specific token positions (often the BOS token or newlines), and load-bear attention sinks and normalization behavior.

The 2026 hybrid-linear-attention (HLA) study is the first systematic look at MAs in models that *interleave* linear-attention layers with full-attention layers (Jamba-family, hybrid Mamba+attention, etc.). It reports two morphologies that don't exist in pure dense transformers.

## How it works

**Pre-Attention Spike (PAS).** In an HLA model, MAs consistently spike *immediately before* every full-attention layer. The spike is generated in the layers feeding into the full-attention block and consumed by the attention operation itself — a form of "just-in-time" magnitude allocation tied to when it's needed.

**Inter-Spike Plateau (ISP).** Between two full-attention layers, when intervening layers are linear-attention (which cannot host attention sinks in the same way), MAs *persist at plateau level* rather than decaying. The linear-attention layers propagate the plateau without amplifying or fully absorbing it.

The paper proposes a **lifecycle model**: MAs arise when a "generation" process (a specific FFN or attention layer producing large-magnitude features) outpaces the "cancellation" process (a downstream residual write that would normalize them away). In HLA, cancellation is timing-locked to full-attention layer positions, giving rise to the spike-and-plateau structure. The morphologies emerge early in training and respond distinctly to output gating — gating alters which layers cancel MAs.

## Why it matters

- **Quantization.** MAs are the outliers that break naive weight/activation quantization. Knowing *which layers* host them (spikes before full-attention) tells the quant designer exactly where to keep higher precision — instead of protecting every layer uniformly.
- **Kernel design.** HLA kernels that assume roughly uniform activation magnitude across layers under-provision numeric range for the spike positions. The PAS/ISP structure gives a principled precision map.
- **Interpretability.** MAs correlate with attention-sink behavior — the mechanism the model uses to "park" attention weight when it has nothing important to attend to. Understanding where and why they form is a step toward mechanistic accounts of attention in hybrid architectures.
- **Architecture guidance.** If ISPs across linear-attention layers are load-bearing, replacing a full-attention layer with a linear-attention layer is *not* free even ignoring capacity — the cancellation timing changes.

## Gotchas & tricks

- **Not the same as dense-transformer MAs.** The dense-transformer literature (Sun et al., 2024) reports MAs as roughly persistent through the whole model; in HLA, they're concentrated around full-attention boundaries. Ports of dense-transformer quant strategies to HLA will misallocate precision.
- **Output gating changes the pattern.** The study reports that turning gating on/off shifts the spike/plateau structure — any analysis has to fix the gating configuration.
- **Emerge early, don't disappear.** MAs form in early pretraining and are stable across the rest of training. Post-hoc surgery ("just clamp them") tends to hurt validation.

## Sources

- Paper: *Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus* — Su et al., 2026 — [arXiv:2608.12149](https://arxiv.org/abs/2608.12149).
- Prior: *Massive Activations in Large Language Models* — Sun et al., 2024 — the dense-transformer analog.
- Related: attention-sink phenomenon from Xiao et al., 2023 (streaming LLM).
