# Massive Activations
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A small number of hidden-state entries in trained transformers develop **magnitudes orders of magnitude larger** than the rest — the "massive activations" (MAs) that power softmax sinks, dominate quantization error, and largely determine attention structure. In hybrid linear-attention LLMs (Mamba+attention style stacks) the phenomenon takes an architecture-aligned shape: **pre-attention spikes** appear right before every full-attention layer and **inter-spike plateaus** carry that magnitude through the intervening linear layers.

**Prereqs:** [multi-head-attention](../architectures/multi-head-attention.md), [transformer-block](../architectures/transformer-block.md)
**Related:** [fp8](../quantization/fp8.md), [mla](../architectures/mla.md)

---

## What it is

Well-trained transformers exhibit a small set of dimensions in their residual stream whose activation magnitudes are 100–1000× the median. These MAs are:

- **Sparse in the channel axis** — a few tens of dims at most.
- **Concentrated on specific tokens** — often the BOS token or high-frequency delimiters (the "attention sink").
- **Load-bearing** — zeroing them collapses model quality; they carry the attention-sink signal that stabilizes softmax when no token deserves attention.

In pure-attention transformers this had been catalogued layer-by-layer. The open question was: what happens in **hybrid** architectures that interleave linear-attention layers (Mamba, Gated DeltaNet, linear-recurrent) with a smaller number of full-attention layers?

## How it works

The Su et al. (2026) study — the first systematic look at MAs in hybrid linear-attention LLMs up to 397B — finds two architecture-aligned morphologies:

**Pre-attention spikes (PAS).** MA magnitude peaks in the residual stream **immediately before every full-attention layer** and drops sharply right after. Full attention requires high-magnitude keys/queries to establish clean attention sinks; the surrounding layers *manufacture* that magnitude on demand.

**Inter-spike plateaus (ISP).** Between two full-attention layers, the intervening linear-attention layers **preserve** the spike as a plateau rather than dissipating it. The magnitude is transported forward, then cancelled by the next full-attention layer's output.

The lifecycle is therefore governed by **full-attention timing**:

1. **Birth** — layers immediately before full attention amplify a chosen dim.
2. **Plateau** — linear-attention layers carry the magnitude forward.
3. **Cancellation** — the full-attention layer subtracts the spike out (via its output projection) once it has served its role.

Output gating on the linear-attention layers changes the numerical magnitudes but not the layerwise organization: PAS/ISP are structural, not scale artifacts.

## Why it matters

- **Quantization.** MAs are what break naïve INT8 / FP8 quantization — they blow the per-tensor range. Knowing exactly *where* they live (right before full attention) tells you which layers need per-channel scales or mixed precision.
- **Interpretability of hybrid stacks.** Massive activations were the entry point to attention-sink theory in pure transformers; PAS/ISP extend the framework to Mamba-Attention hybrids that are now shipping in frontier models.
- **Layer-schedule design.** If MA lifecycle is governed by full-attention timing, the schedule of full-attention layers isn't only a compute-vs-quality knob — it also shapes the numerics of the residual stream in ways that matter for training stability and downstream quantization.

## Gotchas & tricks

- **Zeroing MAs is destructive.** Ablation-first instinct is misleading — they are load-bearing, not artifacts.
- **PAS ≠ outliers you already knew about.** SmoothQuant-style outlier taxonomies from pure-Transformer literature don't transfer cleanly; the hybrid morphology is different because linear-attention layers *transport* rather than cancel.
- **Gate values move magnitudes, not organization.** Don't over-rely on gate statistics as a proxy for MA location.
- **Scale-invariant up to 397B.** The PAS/ISP structure is stable across model scales in the study — you can profile a small proxy and expect the pattern.

## Sources

- Paper: *Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus* — Zunhai Su, Bohan Sun, Xialie Zhuang, Shuibai Zhang, He Xiao, Jing Xiong, Hengyuan Zhang, Zhongzhu Zhou, Tiantian Zhang, Ngai Wong, Chuan-Wei Kuo, 2026 — [arXiv:2608.12149](https://arxiv.org/abs/2608.12149).
- Predecessor: *Massive Activations in Large Language Models* — Sun et al., 2024 — the pure-Transformer version of the phenomenon.
- Predecessor: *Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2023 — the attention-sink account MAs power.
