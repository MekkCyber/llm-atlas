# Asymmetric Mixture-of-Transformers
*Depth — per-modality expert widths in a multimodal Mixture-of-Transformers.*

**TL;DR:** In a multimodal MoT, forcing every expert (per-modality) to the same width wastes compute — the modality with the most raw signal (usually video) needs the widest expert, while cheap-to-model outputs (actions, tactile) do fine with much slimmer ones. **Asymmetric MoT** pairs a full-width expert for the heavy modality with slim experts for the others, keeping the model fast enough for real-time control while still absorbing large multimodal pretraining data.

**Prereqs:** [_moe.md](./_moe.md)
**Related:** [transformer-block.md](./transformer-block.md)

---

## What it is

A Mixture-of-Transformers architecture where the expert corresponding to each modality has a **different hidden width**. This diverges from standard MoT/MoE where experts share dimensions. The width is chosen per-modality according to the signal density that modality carries and the latency budget of the modality's output.

## How it works

- **Routing** is modality-based, not learned per-token: the video pathway always uses the video expert, actions always use the action expert, etc.
- **Widths** are chosen empirically:
  - **Full-width** expert for the modality that dominates parameter budget (typically vision/video). Enough capacity to learn rich world dynamics.
  - **Slim** experts for output modalities (actions, tactile features) — a fraction of the width, enough for the modality's intrinsic dimensionality.
- **Shared attention** (or shared normalization / cross-attention paths) links the experts so cross-modal information flows even though the widths differ.

The result is a model whose *inference* latency is dominated by only the pathway you actually need at inference time (in robotics: the slim action expert), while training benefits from the wide video expert.

## Why it matters

- Enables real-time deployment of models that would otherwise be too big — the action expert stays fast even when the video expert is huge.
- Cleaner parameter-scaling behavior: you can grow the heavy modality without paying that cost on the outputs the model *emits*.
- Composable with standard sparse-MoE ideas (adding more experts *within* a modality).

## Gotchas & tricks

- Cross-expert dimension mismatch is the load-bearing detail. Projection layers between experts must be sized carefully to avoid becoming the new bottleneck.
- Balancing training signal across experts of very different widths requires per-expert learning rates or per-expert weight decay.
- Compared to full MoE per-token routing, modality routing is simpler but leaves cross-modal token-level specialization on the table — a hybrid is possible.

## Sources

- Paper: *N₀-TWAM: Scaling Tactile-Native World-Action Model for Contact-Rich Manipulation* — NeoteAI, 2026 — [arXiv:2607.23783](https://arxiv.org/abs/2607.23783)
