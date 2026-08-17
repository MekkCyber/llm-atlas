# Massive activations

*Depth — the emergent per-token activation spikes in transformer residual streams that dominate quantization outlier problems and now show architecture-aligned morphologies in hybrid-linear models.*

**TL;DR:** **Massive activations (MAs)** are a small number of channels in the residual stream whose magnitudes explode to 10³–10⁵× the median, appearing consistently at specific tokens and layers. In full-attention LLMs they act as **attention sinks** — depositing "attend to me by default" bias for the softmax — and they are the single biggest obstacle to low-bit quantization. In hybrid linear-attention models (2026), MAs organize architecturally: **pre-attention spikes (PAS)** appear immediately before every full-attention layer, and **inter-spike plateaus (ISP)** carry the magnitude across intervening linear-attention layers. Understanding their lifecycle gives quantization designers positional rules rather than empirical heuristics.

**Prereqs:** [../architectures/transformer-block.md](../architectures/transformer-block.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/hybrid-linear-attention.md](../architectures/hybrid-linear-attention.md), [../quantization/_number-formats.md](../quantization/_number-formats.md), [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

Under a normal token, most residual-stream channels sit in a ~unit-magnitude range set by the model's normalization. Massive activations are the outliers: a handful of channels (often the same handful across many tokens) whose values spike orders of magnitude above the median. They are **not** noise — they are *learned*, appear reproducibly, and, if you zero them out, model performance collapses.

Functionally, in full-attention transformers, MAs serve as **attention sinks**: the model dumps enormous key/query norms into a fixed sink position (often BOS or an early token) so that the softmax has somewhere reasonable to route "no information to attend to right now" attention. Remove the sink → softmax over garbage → performance collapse.

In hybrid linear-attention (HLA) models, MAs organize differently, in two morphologies discovered by the 2026 systematic study:

- **Pre-attention spike (PAS)** — an MA reliably appears **immediately before** every full-attention layer.
- **Inter-spike plateau (ISP)** — the magnitude *persists* through intervening linear-attention layers, forming a plateau that connects successive PAS.

As full-attention density → 1, PAS and ISP fuse into the stable global MA morphology of full-attention transformers. As density → 0, they disappear along with the model's usable capacity.

## How it works

**Formation lifecycle (write-sink-cancel):** the paper argues MAs follow a shared three-step lifecycle governed by *when* cancellation happens.

1. **Write.** A layer writes a large activation into a specific residual-stream channel.
2. **Sink.** Downstream layers use that large activation as a sink for a specific computation (attention distribution, softmax normalization).
3. **Cancel.** A later layer subtracts (cancels) the large activation, returning the residual stream to normal magnitude.

The difference between PAS and ISP is *when cancellation runs*:

- **PAS** = localized write → sink at the immediately-following full-attention layer → cancel right after. The whole cycle fits between one full-attention layer's input and the next block.
- **ISP** = write occurs, but cancellation is *delayed* until the next full-attention layer runs. In between, the large activation persists as a plateau visible through every linear-attention layer.

Full-attention output gating **attenuates** absolute MA magnitudes without disrupting the *layerwise organization* — the pattern of "where do MAs live" is preserved, just at lower amplitude.

## Why it matters

- **Quantization design.** MAs are the outlier problem that FP8 / INT4 / MXFP4 formats spend their dynamic range on. Knowing that they are architecture-aligned in hybrids means quantization can be **positional**: aggressive quantization through the linear-attention stretches, protected precision around the pre-attention layer.
- **Attention-sink theory generalizes.** The full-attention "attention sink" story extends to hybrids via PAS. Removing the sink still breaks the model; the sink just lives in a positionally-predictable place.
- **Recurring behavior across scales and configs.** The 2026 paper confirms the pattern across 5 linear architectures, 6 hybrid configs, 5 data domains, and 1.2B → 397B models — this is not a small-model artifact.
- **Training-time levers.** Controlled 1.3B GDN-hybrid pretraining shows both morphologies emerge early; removing GDN output gates yields modest amplification, whereas full-attention output gating strongly attenuates. So the gate location on the *full* layer is the biggest lever.

## Gotchas & tricks

- **Do not zero out MAs.** Every removal experiment on well-trained models degrades quality dramatically. They are load-bearing.
- **Global outlier statistics mislead.** In hybrids, MA density varies enormously by layer position. Per-layer outlier stats matter; global histograms hide the structure.
- **Watch the layer *feeding* the full-attention layer.** That's where PAS is written. Its output distribution is where your quantization budget must go.
- **Long-context can rearrange MA channels.** Some channels stay stable across positions; some move. Recalibrate for the deployment context length, not just the training length.
- **Output-gate tuning is a lever.** Sometimes what looks like a quantization-friendliness issue is actually an unusually large amplitude on a specific gate; a small architectural tweak can be cheaper than an elaborate quantization recipe.

## Sources

- Paper: *Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus* — Su, Sun, Zhuang, Zhang, Xiao, Xiong, Zhang, Zhou, Zhang, Wong, Kuo, 2026, [arXiv:2608.12149](https://arxiv.org/abs/2608.12149) — the HLA-specific morphology and lifecycle.
- Paper: *Massive Activations in Large Language Models* — Sun et al., 2024 — the original full-attention MA characterization.
- Paper: *Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2023, arXiv 2309.17453 — attention-sink account of MAs in full attention.
